"""Small-data hurdle allowance for unattributed observed stopping, not stop intent."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hr import weighted_quantile
from .model import recency_weight
from .stopping import route_features


@dataclass
class HurdleStopping:
    rides: pd.DataFrame = field(default_factory=pd.DataFrame)
    half_life: float = 180
    conditional: str = "winsor_mean"
    method: str = "two-stage unattributed stops"

    @classmethod
    def fit(cls, rides, half_life=180, conditional="winsor_mean"):
        reliable = rides.get("elapsed_reliable", pd.Series(False, index=rides.index)).fillna(False)
        frame = rides[reliable].copy()
        # Missing trace time is uncertain elapsed time, never a fabricated break label.
        raw = pd.to_numeric(frame.unattributed_stationary_s, errors="coerce").fillna(0)
        gaps = pd.to_numeric(frame.get("unobserved_s", 0), errors="coerce")
        frame["stop_target_s"] = np.maximum(0, raw - np.nan_to_num(gaps))
        # <=1 min is weak evidence; 1–5 min is ambiguous; >=5 min is a
        # meaningful unattributed dwell. None of these labels identifies intent.
        frame["stop_label"] = ((frame.stop_target_s - 60) / 240).clip(0, 1)
        return cls(
            frame.reset_index(drop=True),
            half_life,
            conditional,
            "two-stage robust " + conditional + " unattributed dwell proxy",
        )

    def predict(self, frame, moving_s, departure, bike=None):
        features = route_features(frame, bike)
        moment = pd.Timestamp(departure)
        if moment.tzinfo is None:
            moment = moment.tz_localize("UTC")
        empty = {
            "expected_s": 0.0,
            "median_s": None,
            "upper_s": None,
            "probability_of_stopping": None,
            "conditional_duration_s": None,
            "samples": 0,
            "effective_samples": 0.0,
            "evidence": "unavailable",
            "method": self.method,
            "progress_weights": [1.0] * 10,
            "warnings": ["No earlier reliable stopping history; allowance is unknown."],
        }
        if self.rides.empty:
            return empty
        history = self.rides[pd.to_datetime(self.rides.date, utc=True, format="mixed") < moment].copy()
        if history.empty:
            return empty
        distance = history.distance_m.to_numpy() / 1000
        delta = np.log(np.maximum(0.1, distance) / max(0.1, features["distance_km"]))
        age = (moment - pd.to_datetime(history.date, utc=True)).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-0.5 * (delta / 0.45) ** 2) * recency_weight(age, self.half_life)
        effective = weights.sum() ** 2 / max(1e-30, (weights**2).sum())
        if effective < 4:
            weights = np.exp(-0.5 * (delta / 0.8) ** 2) * recency_weight(age, self.half_life)
        if weights.sum() < 1e-20:
            weights = np.ones(len(history))
        effective = weights.sum() ** 2 / (weights**2).sum()
        values, labels = history.stop_target_s.to_numpy(), history.stop_label.to_numpy()
        p = float(np.average(labels, weights=weights))
        positive_weights = weights * labels
        conditional = 0.0
        if positive_weights.sum() > 0:
            median = weighted_quantile(values, positive_weights, 0.5)
            cap = min(weighted_quantile(values, positive_weights, 0.9), max(300, 3 * median))
            conditional = (
                median
                if self.conditional == "median"
                else float(np.average(np.minimum(values, cap), weights=positive_weights))
            )
        # The mean is explicitly winsorized, not the uncapped population expectation.
        # The median candidate is a robust location proxy evaluated by MAPE.
        expected = p * conditional
        warnings = [
            "Meaningful unattributed stops use uncertain dwell labels; unmapped traffic and mechanical delays cannot be distinguished from personal breaks. Unobserved gaps are excluded from stop labels.",
            "Point allowance uses a robust conditional duration; rare long stops remain in the uncertainty distribution.",
        ]
        if features["distance_km"] > distance.max() * 1.05:
            warnings.append("Stopping extrapolation beyond prior distance support; low confidence.")
            expected *= min(2, features["distance_km"] / distance.max())
            conditional *= min(2, features["distance_km"] / distance.max())
        if effective < 8:
            warnings.append("Sparse comparable stopping evidence; allowance is provisional.")
        return {
            **empty,
            "expected_s": float(expected),
            "median_s": weighted_quantile(values, weights, 0.5),
            "upper_s": weighted_quantile(values, weights, 0.9),
            "probability_of_stopping": p,
            "conditional_duration_s": conditional,
            "conditional_estimator": self.conditional,
            "samples": int((weights >= 0.05 * weights.max()).sum()),
            "effective_samples": float(effective),
            "evidence": "moderate"
            if effective >= 8 and features["distance_km"] <= distance.max()
            else "low",
            "warnings": warnings,
            "zero_mass": float(1 - p),
            "distribution_s": values.tolist(),
            "distribution_weights": (weights / weights.sum()).tolist(),
        }
