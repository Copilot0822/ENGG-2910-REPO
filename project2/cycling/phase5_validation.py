"""Reproducible expanding-prefix physical forecasts and earlier-only elapsed selection."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .accuracy import percentage_rows
from .config import Settings
from .elapsed_models import DurationCorrection, historical_baseline, select_candidate
from .hr import recommend_hr
from .journey_validation import _fit_prefix, _row, distance_group, DISTANCE_LABELS, elapsed_metrics
from .prediction import predict_route
from .stopping import StoppingModel, route_features
from .storage import Store
from .tail_calibration import TailCalibration
from .two_stage import HurdleStopping
from .validation import held_out_route
from .weather import HistoricalWeather


STOP_GRID = {
    f"hurdle-{half}-{method}": {"half_life": half, "conditional": method}
    for half in (180, 365)
    for method in ("winsor_mean", "median")
}
BIAS_GRID = {
    f"{kind}-{strength}": {"kind": kind, "parameter": strength}
    for kind in ("bias", "distance_bias")
    for strength in (0.5, 1.0)
}
HYBRID_GRID = {
    **{f"blend-{w}": {"kind": "blend", "parameter": w} for w in (0.25, 0.5, 0.75)},
    "smooth_blend": {"kind": "smooth_blend", "parameter": 0.5},
    **{f"route_relative-{w}": {"kind": "route_relative", "parameter": w} for w in (0.25, 0.5, 0.75)},
    **{f"ridge-{alpha}": {"kind": "ridge", "parameter": alpha} for alpha in (20, 100)},
}
COMBINED_GRID = {
    f"combined-{stop}-{w}": {"stop": stop, "kind": "blend", "parameter": w}
    for stop in STOP_GRID
    for w in (0.25, 0.5, 0.75)
}


def _initialize(root, settings, segments, observations, halves, hr_half, residuals, stop_grid):
    global _context
    _context = root, settings, segments, observations, halves, hr_half, residuals, stop_grid
    threadpool_limits(limits=3)


def _raw_fold(task):
    split, train, test = task
    root, settings, segments, observations, halves, hr_half, residuals, stop_grid = _context
    store = Store(root)
    token = hashlib.sha256(
        (Path(__file__).read_bytes() + Path(__file__).with_name("two_stage.py").read_bytes())
        + json.dumps([train.id.tolist(), str(test.date), stop_grid, halves, hr_half]).encode()
    ).hexdigest()
    cache = store.root / "journey/phase5" / f"{token}.joblib"
    if cache.exists():
        cached = joblib.load(cache)
        for row in cached["rows"]:
            row["split"] = split
        return cached
    points = store.points(test.id)
    route, events = held_out_route(points, test, settings)
    features = route_features(route, test.bike)
    surfaces = route.groupby("surface").distance_m.sum()
    composition = (surfaces / max(1, route.distance_m.sum())).to_dict()
    suggestion = recommend_hr(train, features["distance_km"], test.date, hr_half, features)
    provider = HistoricalWeather(store.root, replace(settings, weather_offline=True), details=False)
    provider.stations = json.loads((store.root / "weather/stations.json").read_text(encoding="utf-8"))
    destination = float(points.lat.iloc[-1]), float(points.lon.iloc[-1])
    osm = float(points.loc[points.map_confidence.ge(0.5), "ds_m"].sum() / max(1, points.ds_m.sum()))
    rows, paths = [], {}
    for variant in ("C", "CW"):
        model, path = _fit_prefix(store, train, test, segments, observations, halves[variant], variant)
        paths[variant] = str(path)
        result = predict_route(
            model,
            route,
            events,
            suggestion["suggested_hr"],
            test.bike,
            "observed",
            departure=test.date,
            weather_provider=provider if variant == "CW" else None,
            break_mode="minimal",
        )
        rows.append(_row(result, test, variant + "-auto", suggestion, osm, destination))
        if variant == "CW":
            old_pool = [r for r in residuals if pd.Timestamp(r["date"]) < pd.Timestamp(test.date)]
            variants = {
                "CW+stops-auto": StoppingModel.fit(train, "residual", 180, old_pool),
                **{key: HurdleStopping.fit(train, **params) for key, params in stop_grid.items()},
            }
            for key, stops in variants.items():
                pred = predict_route(
                    model,
                    route,
                    events,
                    suggestion["suggested_hr"],
                    test.bike,
                    "observed",
                    departure=test.date,
                    weather_provider=provider,
                    stopping_model=stops,
                    break_mode="historical",
                )
                row = _row(pred, test, key, suggestion, osm, destination)
                estimate = pred["break_estimate"] or {}
                row.update(
                    stop_probability=estimate.get("probability_of_stopping"),
                    conditional_stop_s=estimate.get("conditional_duration_s"),
                    stop_effective_samples=estimate.get("effective_samples"),
                )
                rows.append(row)
            full = predict_route(
                model,
                route,
                events,
                suggestion["suggested_hr"],
                test.bike,
                "full_stop",
                departure=test.date,
                weather_provider=provider,
                break_mode="minimal",
            )
            delta = full["elapsed_s"] - result["elapsed_s"]
    baseline = historical_baseline(train, features["distance_km"], test.date)
    direct = {
        **result,
        "elapsed_s": baseline,
        "overall_kmh": features["distance_km"] * 3600 / max(1, baseline),
        "baseline_only": True,
    }
    rows.append(_row(direct, test, "simple-elapsed", suggestion, osm, destination))
    for row in rows:
        row.update(
            split=split,
            baseline_s=baseline,
            gain_per_km=features["gain_per_km"],
            route_paved_fraction=features["paved_fraction"],
            full_stop_increment_s=delta,
            statistical_correction_s=0.0,
            physical_s=row["predicted_elapsed_s"],
            prediction_training_last=str(train.date.max()),
            surface_composition=composition,
        )
        if row["variant"] == "simple-elapsed":
            row["physical_s"] = None
    # Direct event-level assessment is restricted to reliable actually measured full stops.
    observed = [
        o
        for o in observations
        if str(o["ride_id"]) == str(test.id)
        and o["kind"] == "stop"
        and o["reliable"]
        and o["behavior"] == "full_stop"
    ]
    measured = []
    for event in observed:
        prediction = model.behavior.predict(
            event,
            max(1, float(event.get("approach_mps", 5))),
            "full_stop",
            pd.Timestamp(test.date).hour,
            pd.Timestamp(test.date).dayofweek,
        )
        measured.append(
            {
                "id": str(test.id),
                "actual_wait_s": event["stationary_s"],
                "predicted_wait_s": prediction["stationary_s"],
            }
        )
    assert provider.client.requests == 0
    item = {"rows": rows, "paths": paths, "measured_full_stops": measured}
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(item, cache, compress=3)
    return item


def raw_backtests(tasks, store, settings, old, stop_grid, progress=None):
    segments = store.segments()
    residuals = [
        r for r in old["residual_rows"] if pd.Timestamp(r["date"]) < pd.Timestamp(old["final_first"])
    ]
    results = []
    with ProcessPoolExecutor(
        max_workers=settings.comparison_workers,
        initializer=_initialize,
        initargs=(
            str(store.root),
            settings,
            segments,
            store.get_state("observations", []),
            old["speed_halves"],
            old["selected_hr_half_life"],
            residuals,
            stop_grid,
        ),
    ) as pool:
        futures = {pool.submit(_raw_fold, t): t for t in tasks}
        for future in as_completed(futures):
            results.append(future.result())
            if progress:
                progress(
                    len(results) / len(tasks),
                    f"Phase 5 physical {tasks[0][0]} forecasts {len(results)}/{len(tasks)}",
                )
    return sorted(results, key=lambda item: pd.Timestamp(item["rows"][0]["date"]))


def corrected_row(base, total_s, detail, variant):
    row = {
        **base,
        "variant": variant,
        "predicted_elapsed_s": total_s,
        "statistical_correction_s": total_s - base["predicted_elapsed_s"],
        "correction_details": detail,
    }
    row["elapsed_error_s"] = total_s - row["actual_elapsed_s"]
    row["absolute_elapsed_error_s"] = abs(row["elapsed_error_s"])
    row["predicted_overall_kmh"] = row["distance_km"] * 3600 / max(1, total_s)
    if row.get("cutoff"):
        row["predicted_before_cutoff"] = pd.Timestamp(row["date"]) + pd.Timedelta(
            seconds=total_s
        ) <= pd.Timestamp(row["cutoff"])
    return row


def correction_record(row):
    return {
        "id": row["id"],
        "date": row["date"],
        "distance_km": row["distance_km"],
        "physical_s": row["predicted_elapsed_s"],
        "baseline_s": row["baseline_s"],
        "actual_elapsed_s": row["actual_elapsed_s"],
        "gain_per_km": row["gain_per_km"],
        "paved_fraction": row["route_paved_fraction"],
    }


def development_candidates(items):
    rows, pool = [], []
    for index, item in enumerate(items):
        current = {r["variant"]: r.copy() for r in item["rows"]}
        cw = current["CW-auto"]
        split = "seed" if index < 18 else "selection"
        for row in current.values():
            row["split"] = split
            rows.append(row)
        for name, params in {**BIAS_GRID, **HYBRID_GRID, **COMBINED_GRID}.items():
            base = current[params["stop"]] if "stop" in params else cw
            past = pool if "stop" not in params else []
            correction = DurationCorrection(params["kind"], params["parameter"], rows=past)
            total, detail = correction.predict(
                base["predicted_elapsed_s"],
                cw["baseline_s"],
                cw["distance_km"],
                cw["date"],
                {"gain_per_km": cw["gain_per_km"], "paved_fraction": cw["route_paved_fraction"]},
            )
            row = corrected_row(base, total, detail, name)
            row["split"] = split
            rows.append(row)
        pool.append(correction_record(cw))
    return rows, pool


def choose_families(rows):
    grids = {
        "two-stage": STOP_GRID,
        "bias-corrected": BIAS_GRID,
        "hybrid": HYBRID_GRID,
        "hybrid-stops": COMBINED_GRID,
    }
    families, scores = {}, {}
    for family, grid in grids.items():
        selected, candidates = select_candidate(rows, grid)
        families[family] = {"candidate": selected["candidate"], **selected["parameters"]}
        scores[family] = candidates
    eligible = {
        key: scores[key][
            next(i for i, s in enumerate(scores[key]) if s["candidate"] == families[key]["candidate"])
        ]
        for key in families
    }
    eligible["CW-auto"] = {
        "elapsed_mape": elapsed_metrics(
            [r for r in rows if r["split"] == "selection" and r["variant"] == "CW-auto"]
        )["elapsed_mape"]
    }
    winner = min(eligible, key=lambda k: eligible[k]["elapsed_mape"])
    return families, scores, winner


def paired_difference(rows, candidate, reference="CW-auto", repeats=4000):
    frame = percentage_rows(rows)
    a = frame[frame.variant.eq(candidate)].sort_values("date").set_index("id")
    b = frame[frame.variant.eq(reference)].set_index("id")
    ids = a.index.intersection(b.index)
    delta = (a.loc[ids].absolute_percentage_error - b.loc[ids].absolute_percentage_error).to_numpy()
    if not len(delta):
        return {"rides": 0}
    # Circular contiguous blocks retain short-range chronological dependence.
    rng = np.random.default_rng(2910)
    block = min(4, len(delta))
    starts = rng.integers(0, len(delta), (repeats, int(np.ceil(len(delta) / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(delta)).reshape(repeats, -1)[
        :, : len(delta)
    ]
    boot = delta[indices].mean(axis=1)
    return {
        "rides": len(delta),
        "mape_difference_pp": float(delta.mean()),
        "block_length": block,
        "bootstrap_95_pp": np.quantile(boot, [0.025, 0.975]).tolist(),
        "note": "paired circular-block bootstrap; descriptive uncertainty, not a guarantee",
    }


def calibrated_rows(rows, seed=()):
    pools = {}
    for row in seed:
        pools.setdefault(row["variant"], []).append(row)
    result = []
    for row in sorted(rows, key=lambda r: (pd.Timestamp(r["date"]), r["variant"])):
        row = row.copy()
        pool = pools.setdefault(row["variant"], [])
        estimate = TailCalibration(pool).estimate(
            row["predicted_elapsed_s"], row["distance_km"], row["date"]
        )
        row["interval_training_ids"] = [
            r["id"] for r in pool if pd.Timestamp(r["date"]) < pd.Timestamp(row["date"])
        ]
        for c in (80, 90):
            bounds = estimate["intervals_s"].get(str(c))
            row[f"interval_{c}_covered"] = (
                bool(bounds[0] <= row["actual_elapsed_s"] <= bounds[1]) if bounds else None
            )
            row[f"interval_{c}_lower_s"] = bounds[0] if bounds else None
            row[f"interval_{c}_upper_s"] = bounds[1] if bounds else None
        row["conservative_s"] = estimate["conservative_s"]
        row["conservative_before_cutoff"] = (
            (
                pd.Timestamp(row["date"]) + pd.Timedelta(seconds=estimate["conservative_s"])
                <= pd.Timestamp(row["cutoff"])
            )
            if estimate["conservative_s"] is not None and row.get("cutoff")
            else None
        )
        result.append(row)
        pool.append(row)
    return result, pools


def summary(rows):
    variants = sorted({r["variant"] for r in rows})
    comparisons, groups = [], {}
    for key in variants:
        subset = [r for r in rows if r["variant"] == key]
        metric = elapsed_metrics(subset)
        usable = [
            r
            for r in subset
            if r.get("conservative_before_cutoff") is not None
            and r.get("actual_before_cutoff") is not None
        ]
        metric.update(
            conservative_daylight_rides=len(usable),
            conservative_optimistic_errors=sum(
                bool(r["conservative_before_cutoff"]) and not bool(r["actual_before_cutoff"])
                for r in usable
            ),
        )
        comparisons.append({"variant": key, **metric})
        groups[key] = {
            band: elapsed_metrics([r for r in subset if distance_group(r["distance_km"]) == band])
            for band in DISTANCE_LABELS
        }
    return {"comparisons": comparisons, "groups": groups}


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.bool_, np.integer)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def ledger_discrepancy(row):
    """Largest accounting discrepancy, without claiming personal-stop causation."""
    if row.get("predicted_moving_s") is None or pd.isna(row.get("predicted_moving_s")):
        return "direct total only; no component attribution"
    values = {
        "moving-model discrepancy": row.get("moving_error_s"),
        "traffic-attribution discrepancy": row.get("traffic_attribution_error_s"),
        "unattributed stopping allowance discrepancy": row.get("unattributed_error_s"),
        "separate statistical adjustment": row.get("statistical_correction_s", 0),
    }
    valid = {key: abs(value) for key, value in values.items() if value is not None and pd.notna(value)}
    return max(valid, key=valid.get) if valid else "unavailable"


def evaluate_phase5(store, settings, progress=None):
    old = store.get_state("journey_validation")
    rides = store.rides()
    rides["date"] = rides.activity_start_utc.fillna(rides.date)
    ordered = rides.sort_values("date").reset_index(drop=True)
    boundary = pd.Timestamp(old["final_first"])
    early = ordered[pd.to_datetime(ordered.date, utc=True) < boundary]
    # Fixed before opening new candidate errors. Retain all three early long rides.
    indices = sorted(set(range(60, len(early), 2)) | set(early[early.distance_m.ge(100000)].index))
    tasks = [
        (
            "development",
            ordered[pd.to_datetime(ordered.date, utc=True) < pd.Timestamp(ordered.iloc[i].date)],
            ordered.iloc[i],
        )
        for i in indices
        if ordered.iloc[i].elapsed_reliable
    ]
    protocol = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "development_ids": [str(t.id) for _, _, t in tasks],
        "freeze_before": str(boundary),
        "stop_grid": STOP_GRID,
        "bias_grid": BIAS_GRID,
        "hybrid_grid": HYBRID_GRID,
        "combined_grid": COMBINED_GRID,
        "release_gate": "Earlier-selected winner needs >1 percentage point MAPE improvement over CW on fresh later rides, block bootstrap upper bound <0; otherwise retain existing default. No post-test retuning.",
        "benchmark": "32 Phase 4 rides are reused assessment only. Later complementary rides have no prior elapsed-validation predictions; they are mostly short. Historical models and datasets have already been inspected.",
    }
    protocol_path = store.root / "reports/phase5-protocol.json"
    if not protocol_path.exists():
        protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    items = raw_backtests(tasks, store, settings, old, STOP_GRID, progress)
    development, correction_pool = development_candidates(items)
    families, scores, winner = choose_families(development)
    frozen = {
        "families": families,
        "winner": winner,
        "scores": scores,
        "frozen_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "selection_last": max(r["date"] for r in development),
        "selection_rides": len({r["id"] for r in development if r["split"] == "selection"}),
    }
    (store.root / "reports/phase5-frozen-selection.json").write_text(
        json.dumps(clean_json(frozen), indent=2, allow_nan=False), encoding="utf-8"
    )
    # Nested chronological outer blocks select solely from completed earlier folds.
    nested = []
    ids = [item["rows"][0]["id"] for item in items]
    for start in range(28, len(ids), 8):
        prefix = [r for r in development if r["id"] in set(ids[:start])]
        fs, _, selected = choose_families(prefix)
        candidate = fs[selected]["candidate"] if selected in fs else selected
        for row in development:
            if row["id"] in ids[start : start + 8] and row["variant"] == candidate:
                nested.append({**row, "variant": "nested-selected", "selected_candidate": candidate})
    later = ordered[pd.to_datetime(ordered.date, utc=True) >= boundary]
    final_tasks = [
        ("assessment", ordered[pd.to_datetime(ordered.date, utc=True) < pd.Timestamp(t.date)], t)
        for _, t in later.iterrows()
        if t.elapsed_reliable
    ]
    needed_stops = {families["two-stage"]["candidate"], families["hybrid-stops"]["stop"]}
    final_items = raw_backtests(
        final_tasks, store, settings, old, {k: STOP_GRID[k] for k in needed_stops}, progress
    )

    def family_rows(item, pool):
        current = {r["variant"]: r for r in item["rows"]}
        cw = current["CW-auto"]
        output = [current[k].copy() for k in ("C-auto", "CW-auto", "CW+stops-auto", "simple-elapsed")]
        output.append({**current[families["two-stage"]["candidate"]], "variant": "two-stage"})
        for family in ("bias-corrected", "hybrid", "hybrid-stops"):
            params = families[family]
            base = current[params["stop"]] if "stop" in params else cw
            correction = DurationCorrection(
                params["kind"], params["parameter"], rows=pool if "stop" not in params else []
            )
            total, detail = correction.predict(
                base["predicted_elapsed_s"],
                cw["baseline_s"],
                cw["distance_km"],
                cw["date"],
                {"gain_per_km": cw["gain_per_km"], "paved_fraction": cw["route_paved_fraction"]},
            )
            output.append(corrected_row(base, total, detail, family))
        output.append(
            {
                **next(r for r in output if r["variant"] == winner),
                "variant": "automatic",
                "selected_candidate": winner,
            }
        )
        return output

    # Fixed families; the correction fit itself may learn from completed earlier
    # OOF rides, exactly as it would in prospective use. No future errors enter.
    final_rows = []
    for item in final_items:
        final_rows.extend(family_rows(item, correction_pool))
        correction_pool.append(
            correction_record(next(r for r in item["rows"] if r["variant"] == "CW-auto"))
        )
    development_family = []
    earlier_pool = []
    for item in items:
        development_family.extend(family_rows(item, earlier_pool))
        earlier_pool.append(
            correction_record(next(r for r in item["rows"] if r["variant"] == "CW-auto"))
        )
    calibrated, calibrations = calibrated_rows(final_rows, development_family)
    for row in calibrated:
        row["largest_ledger_discrepancy"] = ledger_discrepancy(row)
    benchmark_ids = set(r["id"] for r in old["rows"])
    reused = [r for r in calibrated if r["id"] in benchmark_ids]
    fresh = [r for r in calibrated if r["id"] not in benchmark_ids]
    fresh_delta = paired_difference(fresh, "automatic")
    supported = (
        winner != "CW-auto"
        and fresh_delta.get("mape_difference_pp", 0) < -1
        and fresh_delta.get("bootstrap_95_pp", [0, 1])[1] < 0
    )
    report = {
        "status": "complete",
        "protocol": protocol,
        "frozen_selection": frozen,
        "selected_method": winner,
        "recommended_method": winner if supported else "CW+stops-auto",
        "default_changed": bool(supported),
        "fresh_difference": fresh_delta,
        "reused_difference": paired_difference(reused, "automatic"),
        "nested_selection": summary(nested),
        "nested_rows": nested,
        "development": summary([r for r in development if r["split"] == "selection"]),
        "fresh": summary(fresh),
        "reused": summary(reused),
        "all_later": summary(calibrated),
        "rows": percentage_rows(calibrated).to_dict("records"),
        "calibration_rows": calibrations,
        "correction_rows": correction_pool,
        "development_rows": development,
        "measured_full_stops": [r for item in final_items for r in item["measured_full_stops"]],
    }
    report = clean_json(report)
    (store.root / "reports/phase5-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(report["rows"]).to_csv(store.root / "reports/phase5-validation.csv", index=False)
    store.set_state("phase5_validation", report)
    return report


if __name__ == "__main__":
    settings = Settings.load()
    report = evaluate_phase5(Store(settings.data_dir), settings, lambda v, m: print(m, flush=True))
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "selected_method",
                    "recommended_method",
                    "fresh_difference",
                    "reused_difference",
                )
            },
            indent=2,
        ),
        flush=True,
    )
