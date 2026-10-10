"""Bounded ZIP streaming and duplicate-aware Strava metadata matching."""

import csv
from datetime import datetime, timezone
import hashlib
import gzip
from io import BytesIO, StringIO
import re
from pathlib import Path, PurePosixPath
import stat
import zipfile

import numpy as np
import pandas as pd

from .geometry import clean_points
from .recordings import parse_recording


def canonical(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def number(value):
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


def read_activity_csv(data):
    reader = csv.reader(StringIO(data.decode("utf-8-sig")))
    headers = next(reader, [])
    normalized = [canonical(h) for h in headers]
    rows = []
    for values in reader:

        def get(*aliases, last=True):
            for alias in aliases:
                positions = [
                    i
                    for i, col in enumerate(normalized)
                    if col == canonical(alias) and i < len(values) and values[i].strip()
                ]
                if positions:
                    return values[positions[-1 if last else 0]]
            return None

        # Strava duplicates Distance: the first is display km, the second raw metres.
        distance_positions = [i for i, h in enumerate(normalized) if h == "distance" and i < len(values)]
        explicit = number(get("Distance (m)", "Distance meters", "Distance metres"))
        km = number(get("Distance (km)", "Distance kilometers"))
        raw_distance = number(get("Distance"))
        distance = (
            explicit
            if explicit is not None
            else km * 1000
            if km is not None
            else (
                raw_distance * (1 if len(distance_positions) > 1 else 1000)
                if raw_distance is not None
                else None
            )
        )
        date = pd.to_datetime(
            get("Activity Date", "Start Date", "Date", "Start Time"), utc=True, errors="coerce"
        )
        rows.append(
            {
                "id": get("Activity ID", "id"),
                "name": get("Activity Name", "name") or "Activity",
                "sport": get("Activity Type", "Sport Type", "sport", "type") or "",
                "date": date.isoformat() if pd.notna(date) else None,
                "filename": (get("Filename", "File", "Activity File") or "").replace("\\", "/"),
                "bike": get("Activity Gear", "Gear Name", "Bike", "Gear") or "unknown",
                "summary_distance_m": distance,
                "summary_moving_s": number(get("Moving Time", "Moving Time (s)")),
                "summary_elapsed_s": number(get("Elapsed Time", "Elapsed Time (s)")),
                "summary_avg_hr": number(get("Average Heart Rate", "Average HR", "avg_hr")),
            }
        )
    return rows


def validate_archive(archive, settings):
    entries = archive.infolist()
    if len(entries) > settings.max_archive_entries:
        raise ValueError("ZIP contains too many entries.")
    if sum(e.file_size for e in entries) > settings.max_uncompressed_bytes:
        raise ValueError("ZIP decompressed size exceeds the configured limit.")
    for entry in entries:
        name = entry.filename.replace("\\", "/")
        path = PurePosixPath(name)
        mode = entry.external_attr >> 16
        if (
            path.is_absolute()
            or ".." in path.parts
            or re.match(r"^[A-Za-z]:", name)
            or stat.S_ISLNK(mode)
        ):
            raise ValueError(f"Unsafe ZIP entry: {name[:100]}")
        if entry.flag_bits & 1:
            raise ValueError("Encrypted ZIP archives are not supported.")
        if entry.file_size > settings.max_entry_bytes:
            raise ValueError(f"ZIP entry exceeds size limit: {name[:100]}")
        if entry.file_size > 10_000_000 and entry.file_size / max(1, entry.compress_size) > 300:
            raise ValueError("ZIP compression ratio is excessive; possible archive bomb.")


def import_archive(source, store, settings, progress=None):
    if isinstance(source, bytes):
        if len(source) > settings.max_archive_bytes:
            raise ValueError("ZIP exceeds the configured upload limit.")
        source = BytesIO(source)
    elif isinstance(source, (str, Path)):
        if Path(source).stat().st_size > settings.max_archive_bytes:
            raise ValueError("ZIP exceeds the configured upload limit.")
    report = {"imported": 0, "excluded": 0, "duplicates": 0, "partial": 0, "failed": 0, "items": []}
    try:
        archive = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise ValueError("This file is not a valid ZIP archive.") from exc
    with archive:
        validate_archive(archive, settings)
        entries = {e.filename.replace("\\", "/"): e for e in archive.infolist() if not e.is_dir()}
        csv_names = [n for n in entries if PurePosixPath(n).name.lower() == "activities.csv"]
        if not csv_names:
            raise ValueError("No activities.csv found. Upload the full Strava account export.")
        metadata = read_activity_csv(archive.read(entries[csv_names[0]]))
        by_path = {r["filename"].lower(): r for r in metadata if r["filename"]}
        by_basename = {PurePosixPath(r["filename"]).name.lower(): r for r in metadata if r["filename"]}
        by_id = {str(r["id"]): r for r in metadata if r["id"]}
        candidates = [
            n
            for n in entries
            if n.lower().endswith((".fit", ".fit.gz", ".tcx", ".tcx.gz", ".gpx", ".gpx.gz"))
        ]
        # Prefer original FIT, then TCX, then GPX if multiple copies identify the same activity.
        candidates.sort(key=lambda n: (0 if ".fit" in n.lower() else 1 if ".tcx" in n.lower() else 2, n))
        matched = set()
        for index, name in enumerate(candidates):
            if progress:
                progress(
                    index / max(1, len(candidates)), f"Reading recording {index + 1}/{len(candidates)}"
                )
            stem = PurePosixPath(name).name.split(".")[0]
            meta = (
                by_path.get(name.lower())
                or by_basename.get(PurePosixPath(name).name.lower())
                or by_id.get(stem)
            )
            meta = dict(meta) if meta else {"id": stem, "name": stem, "bike": "unknown", "sport": ""}
            ride_id = str(meta.get("id") or stem)
            matched.add(ride_id)
            item = {"id": ride_id, "file": name}
            sport = canonical(meta.get("sport", ""))
            if sport and (
                sport not in settings.allowed_sports
                or sport == "ebikeride"
                and not settings.include_ebikes
            ):
                report["excluded"] += 1
                report["items"].append(
                    {**item, "status": "excluded", "reason": f"Sport: {meta['sport']}"}
                )
                continue
            try:
                recording = archive.read(entries[name])
                # Hash the uncompressed recording inside .gz as well, so wrapper changes do not duplicate it.
                if name.lower().endswith(".gz"):
                    with gzip.GzipFile(fileobj=BytesIO(recording)) as stream:
                        fingerprint_data = stream.read(settings.max_entry_bytes + 1)
                    if len(fingerprint_data) > settings.max_entry_bytes:
                        raise ValueError("Gzip decompression limit exceeded.")
                else:
                    fingerprint_data = recording
                fingerprint = hashlib.sha256(fingerprint_data).hexdigest()
                if store.contains(ride_id, fingerprint):
                    report["duplicates"] += 1
                    report["items"].append({**item, "status": "duplicate", "reason": "Already imported"})
                    continue
                raw, recorded = parse_recording(recording, name, settings)
                rec_sport = canonical(recorded.get("sport", ""))
                if rec_sport and rec_sport not in ("cycling", "biking", "ride"):
                    report["excluded"] += 1
                    report["items"].append(
                        {**item, "status": "excluded", "reason": f"Recorded sport: {rec_sport}"}
                    )
                    continue
                if canonical(recorded.get("sub_sport", "")) in (
                    "indoorcycling",
                    "virtualactivity",
                    "virtualcycling",
                    "spin",
                ):
                    report["excluded"] += 1
                    report["items"].append(
                        {**item, "status": "excluded", "reason": "Indoor/virtual recording"}
                    )
                    continue
                if not sport and not rec_sport:
                    report["excluded"] += 1
                    report["items"].append(
                        {
                            **item,
                            "status": "excluded",
                            "reason": "Unmatched file has no reliable cycling sport metadata",
                        }
                    )
                    continue
                points, diagnostics = clean_points(raw, settings)
                if diagnostics["distance_m"] < 200 or diagnostics["moving_s"] < 30:
                    report["excluded"] += 1
                    report["items"].append(
                        {
                            **item,
                            "status": "excluded",
                            "reason": "Insufficient outdoor movement for route-speed learning",
                        }
                    )
                    continue
                measured = points.hr.notna() & (points.moving_s > 0)
                sample_hr = (
                    float(
                        np.average(points.loc[measured, "hr"], weights=points.loc[measured, "moving_s"])
                    )
                    if measured.any()
                    else None
                )
                summary_hr = meta.get("summary_avg_hr") or recorded.get("recorded_avg_hr")
                summary_hr = float(summary_hr) if summary_hr and 35 <= float(summary_hr) <= 230 else None
                avg_hr = sample_hr if sample_hr is not None else summary_hr
                meta.update(
                    {
                        "id": ride_id,
                        "date": points.time.min().isoformat(),
                        "sport": meta.get("sport") or "Ride",
                        "recording": name,
                        "measured_avg_hr": avg_hr,
                        "hr_source": "samples" if sample_hr else "summary" if summary_hr else "missing",
                        "hr_sample_fraction": float(measured.sum() / len(points)),
                        **diagnostics,
                    }
                )
                meta["timestamp_reliable"] = diagnostics["unobserved_s"] <= max(
                    30, diagnostics["elapsed_s"] * 0.03
                )
                reasons = []
                if diagnostics["unobserved_s"] > 30:
                    reasons.append("Sparse/paused intervals; some stopped time cannot be identified")
                if diagnostics["elevation_missing_fraction"] > 0.25:
                    reasons.append("Missing elevation")
                if diagnostics["gps_jump_edges"]:
                    reasons.append("Implausible GPS edges excluded")
                summary_distance = meta.get("summary_distance_m")
                if summary_distance and abs(diagnostics["distance_m"] / summary_distance - 1) > 0.15:
                    reasons.append("Recording distance differs from Strava summary by >15%")
                summary_elapsed = meta.get("summary_elapsed_s")
                if summary_elapsed and abs(diagnostics["elapsed_s"] - summary_elapsed) > max(
                    60, 0.05 * summary_elapsed
                ):
                    meta["timestamp_reliable"] = False
                    reasons.append("Recording elapsed time differs from Strava summary")
                store.add_ride(meta, fingerprint, points)
                report["imported"] += 1
                report["partial"] += bool(reasons)
                report["items"].append(
                    {
                        **item,
                        "status": "partial" if reasons else "imported",
                        "reason": "; ".join(reasons),
                    }
                )
            except Exception as exc:
                report["failed"] += 1
                report["items"].append(
                    {**item, "status": "failed", "reason": f"{type(exc).__name__}: {str(exc)[:220]}"}
                )
        for meta in metadata:
            if str(meta["id"]) not in matched:
                report["excluded"] += 1
                report["items"].append(
                    {
                        "id": str(meta["id"]),
                        "file": meta["filename"],
                        "status": "excluded",
                        "reason": "No supported original recording in archive",
                    }
                )
    report["date"] = datetime.now(timezone.utc).isoformat()
    store.set_state("last_import", report)
    if progress:
        progress(1.0, "Import complete")
    return report


def impute_ride_hr(rides):
    """Use only the supplied training subset; never compute this on a future test ride."""
    result = rides.copy()
    measured = pd.to_numeric(
        result.get("measured_avg_hr", pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    measured = measured.where(measured.between(35, 230))
    median = float(measured.median()) if measured.notna().any() else None
    result["hr_imputed"] = measured.isna()
    result["average_hr"] = measured.fillna(median) if median is not None else np.nan
    return result, median
