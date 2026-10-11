"""Reanalyse owned recordings without repeating OSM matching; behavioral audit stays private."""

import json

import numpy as np
import pandas as pd

from .events import observe_events, route_events

VERSION = "behavior-2"


def refresh_behavior(store, progress=None):
    rides, observations = store.rides(), []
    for i, ride in enumerate(rides.itertuples()):
        points = store.points(ride.id)
        events = route_events(points)
        observed = observe_events(points, events, str(ride.id))
        observations.extend(observed)
        store.set_state(f"encounters:{ride.id}", observed)
        store.update_metadata(
            ride.id,
            {
                "behavior_version": VERSION,
                "traffic_stationary_s": sum(
                    o["stationary_s"]
                    for o in observed
                    if o["reliable"] and o["kind"] in ("stop", "signal", "railway")
                ),
            },
        )
        if progress:
            progress((i + 1) / len(rides), f"Traffic speed-trace analysis: {i + 1}/{len(rides)} rides")
    store.set_state("observations", observations)
    store.invalidate_model()
    report = traffic_report(observations)
    store.set_state("traffic_behavior", report)
    (store.root / "reports" / "traffic-behavior.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    pd.DataFrame(observations).to_parquet(
        store.root / "reports" / "traffic-encounters.parquet", index=False
    )
    return report


def traffic_report(observations):
    report = {"version": VERSION, "total_encounters": len(observations), "kinds": {}}
    for kind in ("stop", "signal", "railway", "yield", "roundabout", "corner"):
        all_rows = [o for o in observations if o["kind"] == kind]
        rows = [o for o in all_rows if o["reliable"] and o["behavior"] != "unknown"]
        waits = [o["stationary_s"] for o in rows if o["behavior"] == "full_stop"]
        counts = {b: sum(o["behavior"] == b for o in rows) for b in ("full_stop", "rolling", "minimal")}
        report["kinds"][kind] = {
            "detected": len(all_rows),
            "usable": len(rows),
            "excluded": len(all_rows) - len(rows),
            "counts": counts,
            "fractions": {b: n / len(rows) if rows else None for b, n in counts.items()},
            "stop_probability": counts["full_stop"] / len(rows) if rows else None,
            "conditional_wait_mean_s": float(np.mean(waits)) if waits else None,
            "conditional_wait_median_s": float(np.median(waits)) if waits else None,
            "wait_samples": len(waits),
            "unique_directional_intersections": len({o["identity"] for o in rows}),
        }
    report["exclusion_reasons"] = {}
    for o in observations:
        for reason in o.get("reliability_reasons", []):
            report["exclusion_reasons"][reason] = report["exclusion_reasons"].get(reason, 0) + 1
    report["note"] = (
        "Stops require consecutive recorded near-zero speed close to the control. Sparse/gapped/auto-paused and long-dwell encounters are excluded; percentages apply to usable recordings only."
    )
    return report
