"""Distance-weighted OSM quality reports, calculated from persisted annotations."""

import numpy as np
import pandas as pd


def trace_coverage(points):
    total = float(points.ds_m.sum())
    matched = points.map_confidence.ge(0.5)
    surface = points.groupby("surface", observed=True).ds_m.sum()
    sources = points.groupby("surface_source", observed=True).ds_m.sum()
    controls = points[points.control_kind.ne("")]
    return {
        "total_km": total / 1000,
        "matched_km": float(points.loc[matched, "ds_m"].sum() / 1000),
        "high_confidence_km": float(points.loc[points.map_confidence.ge(0.75), "ds_m"].sum() / 1000),
        "low_confidence_km": float(
            points.loc[points.map_confidence.between(0.00001, 0.499999), "ds_m"].sum() / 1000
        ),
        "surfaces": {str(k): float(v / 1000) for k, v in surface.items()},
        "sources": {str(k): float(v / 1000) for k, v in sources.items()},
        "controls": {str(k): int(v) for k, v in controls.control_kind.value_counts().items()},
        "unique_controls": {
            str(k): controls[controls.control_kind.eq(k)].control_id.unique().tolist()
            for k in controls.control_kind.unique()
        },
        "uncertain_controls": int(points.control_uncertain.sum()),
        "median_offset_m": float(points.loc[matched, "map_distance_m"].median())
        if matched.any()
        else None,
    }


def history_coverage(store):
    rides = store.rides()
    totals = {"total_km": 0.0, "matched_km": 0.0, "high_confidence_km": 0.0, "low_confidence_km": 0.0}
    surfaces, sources, controls, unique, regions = {}, {}, {}, {}, []
    uncertain = 0
    for ride in rides.itertuples():
        points = store.points(ride.id)
        if "map_confidence" not in points:
            continue
        report = trace_coverage(points)
        for k in totals:
            totals[k] += report[k]
        for key, dest in (("surfaces", surfaces), ("sources", sources), ("controls", controls)):
            for k, value in report[key].items():
                dest[k] = dest.get(k, 0) + value
        for k, values in report["unique_controls"].items():
            unique.setdefault(k, set()).update(values)
        uncertain += report["uncertain_controls"]
        weak = points[points.map_confidence.lt(0.5)]
        if len(weak):
            cells = pd.DataFrame(
                {
                    "lat_cell": np.floor(weak.lat * 10) / 10,
                    "lon_cell": np.floor(weak.lon * 10) / 10,
                    "km": weak.ds_m / 1000,
                }
            )
            regions.append(cells.groupby(["lat_cell", "lon_cell"]).km.sum().reset_index())
    total = max(1e-9, totals["total_km"])
    observations = store.get_state("observations", [])
    segments = store.segments()
    matched_segments = segments[segments.map_confidence.ge(0.5)] if len(segments) else segments
    segment_quality = {
        "matched_segments": len(matched_segments),
        "high_confidence_segments": int(matched_segments.map_confidence.ge(0.75).sum())
        if len(matched_segments)
        else 0,
        "high_confidence_segment_percent": float(matched_segments.map_confidence.ge(0.75).mean() * 100)
        if len(matched_segments)
        else 0.0,
    }
    region_rows = []
    if regions:
        frame = (
            pd.concat(regions).groupby(["lat_cell", "lon_cell"]).km.sum().sort_values(ascending=False)
        )
        region_rows = [
            {"latitude_cell": float(a), "longitude_cell": float(b), "unmatched_km": float(v)}
            for (a, b), v in frame.head(30).items()
        ]
    return {
        **totals,
        **segment_quality,
        "rides": len(rides),
        "matched_percent": totals["matched_km"] / total * 100,
        "high_confidence_percent_of_matched": totals["high_confidence_km"]
        / max(1e-9, totals["matched_km"])
        * 100,
        "surfaces": [{"surface": k, "km": v, "percent": v / total * 100} for k, v in surfaces.items()],
        "surface_provenance": [
            {"source": k, "km": v, "percent": v / total * 100} for k, v in sources.items()
        ],
        "control_encounters": controls,
        "unique_directional_controls": {k: len(v) for k, v in unique.items()},
        "unique_control_locations": {
            k: len({identity.split(":")[1] for identity in v}) for k, v in unique.items()
        },
        "event_encounters": {
            kind: sum(o["kind"] == kind for o in observations)
            for kind in sorted({o["kind"] for o in observations})
        },
        "uncertain_controls": uncertain,
        "analyzed_encounters": len(observations),
        "reliable_encounters": sum(bool(o["reliable"]) for o in observations),
        "reliable_traffic_encounters": sum(
            bool(o["reliable"]) and o["kind"] != "corner" for o in observations
        ),
        "stop_behaviors": {
            behavior: sum(
                o["kind"] == "stop" and o["reliable"] and o["behavior"] == behavior for o in observations
            )
            for behavior in ("full_stop", "rolling", "minimal")
        },
        "weak_regions": region_rows,
    }
