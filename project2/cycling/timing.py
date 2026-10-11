"""Independent duration/HR quality audit; traffic attribution is not elapsed truth."""

import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .recordings import parse_recording

VERSION = "timing-2"


def finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def timing_quality(meta, points, recorded=None, encounters=()):
    recorded = recorded or {}
    start, end = points.time.min(), points.time.max()
    span = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else None
    candidates = {
        "strava": finite(meta.get("summary_elapsed_s")),
        "fit": finite(recorded.get("recorded_elapsed_s", meta.get("recorded_elapsed_s"))),
        "gps": finite(span),
    }
    valid = {k: v for k, v in candidates.items() if v is not None and 30 <= v <= 259200}
    elapsed = valid.get("strava", valid.get("fit", valid.get("gps")))
    reasons, notes = [], []
    if elapsed is None:
        reasons.append("missing_or_implausible_duration")
    tolerance = max(60, (elapsed or 0) * 0.02)
    agreeing = [k for k, v in valid.items() if abs(v - (elapsed or 0)) <= tolerance]
    fit_start = pd.to_datetime(
        recorded.get("recorded_start_utc", meta.get("recorded_start_utc")), utc=True, errors="coerce"
    )
    activity_start = start
    if pd.notna(fit_start):
        if abs((fit_start - start).total_seconds()) <= max(120, (elapsed or 0) * 0.05):
            activity_start = fit_start
        else:
            reasons.append("activity_start_conflicts_with_gps")
    if elapsed is not None and len(agreeing) < 2:
        reasons.append("duration_sources_disagree_or_single_source")
    if "gps" in valid and elapsed is not None and abs(valid["gps"] - elapsed) > max(120, elapsed * 0.05):
        reasons.append("recording_span_conflicts_with_activity")
    distance = finite(meta.get("distance_m")) or float(points.ds_m.sum())
    summary_distance = finite(meta.get("summary_distance_m"))
    if summary_distance and abs(distance / summary_distance - 1) > 0.15:
        reasons.append("recording_distance_mismatch")
    summary_moving = finite(meta.get("summary_moving_s"))
    if summary_moving is not None and (summary_moving < 0 or summary_moving > (elapsed or 0) + 5):
        reasons.append("invalid_summary_moving_duration")
    # Keep the same moving definition as the preserved riding model. The Strava
    # moving/timer definitions are retained separately, never treated as pauses.
    moving = float(points.moving_s.sum())
    if elapsed is not None and (moving > elapsed + tolerance or distance / elapsed * 3.6 > 65):
        reasons.append("implausible_movement_or_duration")
    tail = 0.0
    for p in reversed(list(points.itertuples())):
        if p.moving_s > 0 or p.unobserved_s > 10:
            break
        tail += p.stationary_s
    if tail > max(1800, (elapsed or 0) * 0.15):
        reasons.append("long_stationary_recording_tail")
    if float(points.unobserved_s.sum()) > max(60, (elapsed or 0) * 0.03):
        notes.append("unobserved_intervals_inside_trustworthy_activity_duration")
    if summary_moving and abs(moving - summary_moving) > max(120, 0.1 * summary_moving):
        notes.append("moving_definitions_differ")
    total_stopped = max(0, (elapsed or 0) - moving)
    traffic = sum(
        float(o.get("stationary_s") or 0)
        for o in encounters
        if o.get("reliable") and o.get("kind") != "corner" and finite(o.get("stationary_s")) is not None
    )
    if traffic > total_stopped + 10:
        notes.append("traffic_attribution_exceeds_stationary_ledger")
    traffic = min(traffic, total_stopped)
    measured = points.hr.between(35, 230) & points.moving_s.gt(0)
    coverage = float(points.loc[measured, "moving_s"].sum() / max(1, moving))
    sample_hr = (
        float(np.average(points.loc[measured, "hr"], weights=points.loc[measured, "moving_s"]))
        if measured.any()
        else None
    )
    hr = sample_hr if coverage >= 0.7 else None
    hr_source = (
        "moving-time-weighted samples" if hr is not None else "unavailable / insufficient samples"
    )
    summary_hr = finite(meta.get("summary_avg_hr")) or finite(
        recorded.get("recorded_avg_hr", meta.get("recorded_avg_hr"))
    )
    if (
        hr is None
        and not measured.any()
        and summary_hr
        and 35 <= summary_hr <= 230
        and meta.get("hr_source") == "summary"
    ):
        hr, hr_source = summary_hr, "recorded summary; sample coverage unavailable"
    progress = np.zeros(10)
    stationary = points.stationary_s.to_numpy(dtype=float).copy()
    # Traffic encounter windows are already exclusive. Remove the observed wait
    # proportionally from its recorded window before learning broad progress bins.
    for o in encounters:
        if not o.get("reliable") or o.get("kind") == "corner":
            continue
        center = finite(o.get("distance_m"))
        if center is None:
            continue
        mask = abs(points.distance_m.to_numpy() - center) <= 100
        budget = float(stationary[mask].sum())
        if budget:
            stationary[mask] *= max(0, 1 - float(o.get("stationary_s") or 0) / budget)
    bins = np.minimum(9, (points.distance_m.to_numpy() / max(1, distance) * 10).astype(int))
    np.add.at(progress, bins, stationary)
    # Unlocalized pauses are distributed, never assigned invented café locations.
    unattributed = max(0, total_stopped - traffic)
    if progress.sum() < unattributed:
        progress += (unattributed - progress.sum()) / 10
    result = {
        "timing_version": VERSION,
        "activity_start_utc": activity_start.isoformat(),
        "activity_end_utc": (activity_start + pd.Timedelta(seconds=elapsed)).isoformat()
        if elapsed
        else end.isoformat(),
        "elapsed_truth_s": elapsed,
        "elapsed_quality": "reliable" if not reasons else "partial" if elapsed else "unreliable",
        "elapsed_reliable": not reasons,
        "elapsed_sources": agreeing,
        "elapsed_quality_reasons": sorted(set(reasons)),
        "timing_notes": sorted(set(notes)),
        "truth_moving_s": moving,
        "total_stationary_s": total_stopped,
        "identified_traffic_s": traffic,
        "unattributed_stationary_s": unattributed,
        "hr_recommendation_value": hr,
        "hr_recommendation_source": hr_source,
        "hr_moving_coverage": coverage,
        "hr_reliable": hr is not None,
        "recording_tail_s": tail,
        "break_progress_weights": progress.tolist(),
        "paved_fraction": float(points.loc[points.surface.eq("paved"), "ds_m"].sum() / max(1, distance))
        if "surface" in points
        else None,
    }
    result.update(
        {
            k: v
            for k, v in recorded.items()
            if k.startswith("recorded_") or k in ("timer_s", "timer_event_count")
        }
    )
    return result


def audit_history(store, settings, archive=None, progress=None):
    """Read cached points and, once, original FIT metadata. Do not rewrite traces."""
    rides = store.rides()
    source = zipfile.ZipFile(archive) if archive else None
    entries = {n.replace("\\", "/"): n for n in source.namelist()} if source else {}
    try:
        for i, ride in enumerate(rides.itertuples()):
            meta = ride._asdict()
            if meta.get("timing_version") == VERSION:
                continue
            recorded = {}
            name = meta.get("recording", "")
            if (
                source
                and name in entries
                and ".fit" in name.lower()
                and "recorded_elapsed_s" not in meta
            ):
                _, recorded = parse_recording(source.read(entries[name]), name, settings)
            updates = timing_quality(
                meta, store.points(ride.id), recorded, store.get_state(f"encounters:{ride.id}", [])
            )
            segments = store.segments([ride.id])
            if "air_temperature_c" in segments:
                good = segments.air_temperature_c.notna()
                updates["mean_temperature_c"] = (
                    float(
                        np.average(
                            segments.loc[good, "air_temperature_c"],
                            weights=segments.loc[good, "distance_m"],
                        )
                    )
                    if good.any()
                    else None
                )
            store.update_metadata(ride.id, updates)
            if progress:
                progress((i + 1) / len(rides), f"Activity timing/HR audit {i + 1}/{len(rides)}")
    finally:
        if source:
            source.close()
    rides = store.rides()
    counts = {}
    for values in rides.elapsed_quality_reasons:
        for reason in values:
            counts[reason] = counts.get(reason, 0) + 1
    report = {
        "rides": len(rides),
        "reliable_elapsed": int(rides.elapsed_reliable.sum()),
        "reliable_measured_hr": int(rides.hr_reliable.sum()),
        "excluded_reasons": counts,
        "version": VERSION,
        "moving_definition": "preserved trace movement; summary/timer retained as diagnostics",
    }
    store.set_state("timing_quality", report)
    (Path(store.root) / "reports/timing-quality.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
