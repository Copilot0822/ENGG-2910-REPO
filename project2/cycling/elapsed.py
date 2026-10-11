"""Asymmetric empirical duration-error calibration, without Gaussian assumptions."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ElapsedCalibration:
    rows: list = field(default_factory=list)
    validated_coverage_80: float | None = None
    validated_coverage_90: float | None = None

    def estimate(
        self, predicted_s, distance_km, departure=None, forecast=False, break_mode="historical"
    ):
        rows = pd.DataFrame(self.rows)
        if departure is not None and len(rows):
            rows = rows[pd.to_datetime(rows.date, utc=True) < pd.Timestamp(departure)]
        warnings = []
        if len(rows):
            group = rows[rows.distance_km.ge(100) == (distance_km >= 100)]
            if len(group) >= 20:
                rows = group
            else:
                warnings.append(
                    "Sparse distance-specific elapsed calibration; pooled ride errors are used."
                )
        n = len(rows)
        supported = n >= 6
        if supported:
            errors = (
                (rows.actual_elapsed_s - rows.predicted_elapsed_s)
                / np.maximum(1800, rows.predicted_elapsed_s)
            ).to_numpy()
            durations = np.maximum(0, predicted_s + errors * max(1800, predicted_s))

            def quantile(q):
                return float(np.quantile(durations, q, method="higher"))

            intervals = {"80": [quantile(0.1), quantile(0.9)], "90": [quantile(0.05), quantile(0.95)]}
            median, conservative = quantile(0.5), max(predicted_s, quantile(0.9))
        else:
            durations, intervals, median, conservative = np.array([]), {}, None, None
            warnings.append(
                "Insufficient held-out elapsed errors for a conservative statistical estimate."
            )
        long_support = int(rows.distance_km.ge(150).sum()) if n else 0
        provisional = (
            n < 30
            or forecast
            or break_mode != "historical"
            or (distance_km >= 150 and long_support < 15)
        )
        coverage_ok = (
            self.validated_coverage_80 is not None
            and self.validated_coverage_80 >= 0.8
            and self.validated_coverage_90 is not None
            and self.validated_coverage_90 >= 0.9
        )
        if not coverage_ok:
            provisional = True
            warnings.append(
                "Held-out interval coverage has not met both nominal targets; finishing-time intervals remain provisional."
            )
        if forecast:
            warnings.append(
                "Issued-forecast elapsed errors have not been calibrated; observed-weather intervals are provisional."
            )
        if break_mode != "historical":
            warnings.append(
                "Elapsed calibration used usual historical breaks, not this changed break scenario."
            )
        if distance_km >= 150 and long_support < 15:
            warnings.append(
                "Very limited long-distance elapsed validation; avoid a confident before-dark claim."
            )
        return {
            "samples": n,
            "intervals_s": intervals,
            "median_s": median,
            "conservative_s": conservative,
            "provisional": provisional,
            "duration_samples_s": durations.tolist(),
            "probability_supported": n >= 60 and not provisional,
            "method": "earlier out-of-sample asymmetric errors scaled by predicted duration",
            "warnings": warnings,
        }


def cutoff_probability(uncertainty, available_s):
    if not uncertainty.get("probability_supported"):
        return None
    values = np.asarray(uncertainty["duration_samples_s"], float)
    return float(np.mean(values <= available_s)) if len(values) else None
