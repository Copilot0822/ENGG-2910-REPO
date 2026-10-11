"""Per-ride elapsed percentage errors; no fitting or interval calibration."""

import numpy as np
import pandas as pd

PRIMARY_VARIANTS = ("C-auto", "CW-auto", "CW+stops-auto", "simple-elapsed")
MODEL_LABELS = {
    "C-auto": "C",
    "CW-auto": "C+Weather",
    "CW+stops-auto": "C+Weather + stopping",
    "simple-elapsed": "Simple historical baseline",
    "CW-known": "C+Weather / known HR",
    "CW+stops-known": "C+Weather + stopping / known HR",
}
DURATION_LABELS = ("Under 1 hour", "1–2 hours", "2–4 hours", "4–8 hours", "8+ hours")


def duration_group(seconds):
    return DURATION_LABELS[int(np.searchsorted([3600, 7200, 14400, 28800], seconds, side="right"))]


def percentage_rows(rows):
    """Retain excluded rows and explain why their percentage cannot be scored."""
    frame = pd.DataFrame(rows).copy().reset_index(drop=True)
    actual = pd.to_numeric(
        frame.get("actual_elapsed_s", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    predicted = pd.to_numeric(
        frame.get("predicted_elapsed_s", pd.Series(index=frame.index, dtype=float)), errors="coerce"
    )
    reliable = frame.get("elapsed_reliable", pd.Series(False, index=frame.index)).map(
        lambda value: value is True or isinstance(value, (bool, np.bool_)) and bool(value)
    )
    reasons = {
        "unreliable_elapsed": ~reliable,
        "invalid_actual_elapsed": ~(np.isfinite(actual) & actual.gt(0)),
        "invalid_prediction": ~(np.isfinite(predicted) & predicted.ge(0)),
    }
    eligible = ~pd.DataFrame(reasons).any(axis=1)
    signed = 100 * (predicted - actual) / actual.where(eligible)
    frame["percentage_eligible"] = eligible
    frame["percentage_exclusion_reason"] = [
        "; ".join(reason for reason, mask in reasons.items() if mask.loc[index]) for index in frame.index
    ]
    frame["signed_percentage_error"] = signed
    frame["absolute_percentage_error"] = signed.abs()
    frame["late_arrival_percentage_error"] = (-signed).clip(lower=0)
    frame["prediction_direction"] = np.select(
        [signed.lt(0), signed.gt(0), eligible],
        ["Underestimated", "Overestimated", "Exact"],
        default="Excluded",
    )
    start = pd.to_datetime(
        frame.get("date", pd.Series(index=frame.index, dtype=object)), utc=True, errors="coerce"
    )
    for name, seconds in (("actual_finish_utc", actual), ("predicted_finish_utc", predicted)):
        finite = seconds.where(np.isfinite(seconds), 0).fillna(0)
        finish = start + pd.to_timedelta(finite, unit="s", errors="coerce")
        frame[name] = finish.map(lambda value: value.isoformat() if pd.notna(value) else None).where(
            np.isfinite(seconds) & seconds.ge(0), None
        )
    return frame


def percentage_metrics(rows):
    frame = percentage_rows(rows)
    valid = frame[frame.percentage_eligible]
    absolute, signed = valid.absolute_percentage_error, valid.signed_percentage_error
    under = -signed[signed.lt(0)]

    def mean(values):
        return float(values.mean()) if len(values) else None

    def median(values):
        return float(values.median()) if len(values) else None

    def p90(values):
        return float(np.quantile(values, 0.9, method="linear")) if len(values) else None

    counts = {}
    for reasons in frame.loc[~frame.percentage_eligible, "percentage_exclusion_reason"]:
        for reason in reasons.split("; "):
            counts[reason] = counts.get(reason, 0) + 1
    return {
        "rides": len(valid),
        "input_rides": len(frame),
        "excluded_rides": len(frame) - len(valid),
        "percentage_exclusion_counts": counts,
        "elapsed_mape": mean(absolute),
        "elapsed_mdape": median(absolute),
        "elapsed_mspe": mean(signed),
        "elapsed_p90_ape": p90(absolute),
        "underpredicted_rides": len(under),
        "underprediction_percentage": 100 * len(under) / len(valid) if len(valid) else None,
        "underprediction_mape": mean(under),
        "underprediction_median_percentage": median(under),
        "underprediction_max_percentage": float(under.max()) if len(under) else None,
        "underprediction_p90_percentage": p90(under),
        "late_arrival_p90_percentage": p90(valid.late_arrival_percentage_error),
    }


def paired_percentage_rows(rows, variants=PRIMARY_VARIANTS):
    """Compare every primary model on the intersection of valid ride IDs."""
    frame = percentage_rows(rows)
    if frame.empty:
        frame["paired_comparison_eligible"] = pd.Series(dtype=bool)
        return frame, {"candidate_rides": 0, "evaluated_rides": 0, "ride_ids": [], "excluded_rides": []}
    main = frame[frame.variant.isin(variants)].copy()
    main["id"] = main.id.astype(str)
    if main.duplicated(["id", "variant"]).any():
        raise ValueError("Duplicate ride/model predictions cannot count as separate held-out rides.")
    eligible_ids, exclusions = [], []
    for ride_id, group in main.groupby("id", sort=True):
        reasons = []
        present = set(group.variant)
        for variant in variants:
            if variant not in present:
                reasons.append(f"missing prediction: {variant}")
        for row in group.itertuples():
            if not row.percentage_eligible:
                reasons.append(f"{row.variant}: {row.percentage_exclusion_reason}")
        actual = pd.to_numeric(group.actual_elapsed_s, errors="coerce").to_numpy()
        if np.isfinite(actual).all() and not np.allclose(actual, actual[0], rtol=0, atol=1e-6):
            reasons.append("inconsistent actual elapsed duration across models")
        if reasons:
            exclusions.append({"id": str(ride_id), "reasons": reasons})
        else:
            eligible_ids.append(str(ride_id))
    frame["paired_comparison_eligible"] = (
        frame.variant.isin(variants)
        & frame.id.astype(str).isin(eligible_ids)
        & frame.percentage_eligible
    )
    return frame, {
        "candidate_rides": int(main.id.nunique()),
        "evaluated_rides": len(eligible_ids),
        "ride_ids": eligible_ids,
        "excluded_rides": exclusions,
    }
