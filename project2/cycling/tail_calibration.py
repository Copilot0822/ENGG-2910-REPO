"""Finite-rank asymmetric duration tails, evaluated prequentially."""

import math

import numpy as np
import pandas as pd

from .elapsed import ElapsedCalibration


class TailCalibration(ElapsedCalibration):
    def estimate(
        self,
        predicted_s,
        distance_km,
        departure=None,
        forecast=False,
        break_mode="historical",
        scenario="observed",
        manual_hr=False,
    ):
        result = super().estimate(predicted_s, distance_km, departure, forecast, break_mode)
        rows = pd.DataFrame(self.rows)
        if len(rows) and departure is not None:
            rows = rows[pd.to_datetime(rows.date, utc=True, format="mixed") < pd.Timestamp(departure)]
        if len(rows):
            local = rows[rows.distance_km.ge(100) == (distance_km >= 100)]
            if len(local) >= 20:
                rows = local
        if len(rows) >= 6:
            scores = np.sort(
                (rows.actual_elapsed_s - rows.predicted_elapsed_s)
                / np.maximum(1800, rows.predicted_elapsed_s)
            )
            n = len(scores)

            def quantile(q, upper):
                rank = math.ceil((n + 1) * q) if upper else math.floor((n + 1) * q)
                error = scores[np.clip(rank - 1, 0, n - 1)]
                return float(max(0, predicted_s + error * max(1800, predicted_s)))

            result["intervals_s"] = {
                "80": [quantile(0.1, False), quantile(0.9, True)],
                "90": [quantile(0.05, False), quantile(0.95, True)],
            }
            result["conservative_s"] = max(predicted_s, quantile(0.9, True))
            result["median_s"] = quantile(0.5, True)
        result["method"] = (
            "finite-order chronological asymmetric scaled residuals; no exchangeability guarantee"
        )
        result["calibration_scenario"] = (
            "observed traffic, automatic suggested HR, usual historical breaks, observed weather"
        )
        if scenario != "observed":
            result["provisional"] = True
            result["warnings"].append(
                "Full-stop mode is a hypothetical scenario; observed-behavior error intervals are reference estimates, not full-stop validation."
            )
        if manual_hr:
            result["provisional"] = True
            result["warnings"].append(
                "Manual target HR is a planning scenario, not retrospective knowledge of actual recorded HR; automatic-HR errors provide only provisional reference intervals."
            )
        result["probability_supported"] = result["probability_supported"] and not result["provisional"]
        return result
