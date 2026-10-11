"""Empirical personalized distributions of residual stationary time."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hr import weighted_quantile
from .model import recency_weight


def route_features(frame, bike=None):
    distance = float(frame.distance_m.sum())
    gain = float(frame.gain_m.max()) if "gain_m" in frame else 0.0
    paved = float(frame.loc[frame.surface.eq("paved"), "distance_m"].sum() / max(1, distance))
    temp = pd.to_numeric(frame.get("air_temperature_c", pd.Series(dtype=float)), errors="coerce")
    return {
        "distance_km": distance / 1000,
        "gain_per_km": gain / max(0.1, distance / 1000),
        "paved_fraction": paved,
        "bike": bike,
        "temperature_c": float(temp.mean()) if temp.notna().any() else None,
    }


@dataclass
class StoppingModel:
    rides: pd.DataFrame = field(default_factory=pd.DataFrame)
    method: str = "context"
    half_life: float = 180

    @classmethod
    def fit(cls, rides, method="context", half_life=180, residual_rows=()):
        reliable = rides.get("elapsed_reliable", pd.Series(False, index=rides.index)).fillna(False)
        frame = rides[reliable].copy()
        frame["stop_target_s"] = frame.get(
            "unattributed_stationary_s", pd.Series(0.0, index=frame.index)
        )
        if method == "residual" and len(residual_rows) >= 4:
            residual = pd.DataFrame(residual_rows)
            frame = frame.merge(residual[["id", "residual_stop_s", "predicted_moving_s"]], on="id")
            frame["stop_target_s"] = frame.residual_stop_s.clip(lower=0)
            frame["truth_moving_s"] = frame.predicted_moving_s
        elif method == "residual":
            method = "context"
        return cls(frame.reset_index(drop=True), method, half_life)

    def predict(self, frame, moving_s, departure, bike=None):
        features = route_features(frame, bike)
        empty = {
            "expected_s": 0.0,
            "median_s": None,
            "upper_s": None,
            "samples": 0,
            "effective_samples": 0.0,
            "evidence": "unavailable",
            "method": self.method,
            "warnings": [
                "No reliable stopping history: personal stopping allowance is unknown, not zero."
            ],
            "progress_weights": [1.0] * 10,
        }
        if self.rides.empty:
            return empty
        moment = pd.Timestamp(departure)
        moment = moment.tz_localize("UTC") if moment.tzinfo is None else moment.tz_convert("UTC")
        history = self.rides[pd.to_datetime(self.rides.date, utc=True) < moment].copy()
        if history.empty:
            return empty
        distance = history.distance_m.to_numpy() / 1000
        log_delta = np.log(distance / max(0.1, features["distance_km"]))
        age = (moment - pd.to_datetime(history.date, utc=True)).dt.total_seconds().to_numpy() / 86400
        weights = np.exp(-0.5 * (log_delta / 0.45) ** 2) * recency_weight(age, self.half_life)
        if self.method != "distance" and len(history) >= 12:
            duration = history.truth_moving_s.to_numpy()
            weights *= np.clip(
                np.exp(-abs(np.log(np.maximum(60, duration) / max(60, moving_s))) / 1.5), 0.5, 1
            )
            if features["bike"]:
                weights *= np.where(history.bike.eq(bike), 1, 0.8)
            gain = history.elevation_gain_m.fillna(0).to_numpy() / np.maximum(0.1, distance)
            weights *= np.clip(np.exp(-abs(gain - features["gain_per_km"]) / 40), 0.7, 1)
            if "paved_fraction" in history:
                weights *= np.clip(
                    1 - abs(history.paved_fraction.fillna(0.5) - features["paved_fraction"]), 0.7, 1
                )
            hour = pd.to_datetime(history.date, utc=True).dt.hour.to_numpy()
            hour_delta = abs(hour - moment.hour)
            weights *= np.clip(1 - np.minimum(hour_delta, 24 - hour_delta) / 24, 0.7, 1)
            weekend = pd.to_datetime(history.date, utc=True).dt.dayofweek.to_numpy() >= 5
            weights *= np.where(weekend == (moment.dayofweek >= 5), 1, 0.9)
            if features["temperature_c"] is not None and "mean_temperature_c" in history:
                delta = abs(
                    history.mean_temperature_c.fillna(features["temperature_c"])
                    - features["temperature_c"]
                )
                weights *= np.clip(np.exp(-delta / 30), 0.7, 1)
        effective = weights.sum() ** 2 / max(1e-30, float((weights**2).sum()))
        if effective < 4:
            weights = np.exp(-0.5 * (log_delta / 0.8) ** 2) * recency_weight(age, self.half_life)
            effective = weights.sum() ** 2 / max(1e-30, float((weights**2).sum()))
        if weights.sum() <= 1e-20:
            weights = np.ones(len(history))
        values = history.stop_target_s.to_numpy(dtype=float)
        # A distribution, including zero-stop rides; arithmetic expectation for
        # the ledger, robust median and upper quantile for transparent diagnostics.
        expected = float(np.average(values, weights=weights))
        progress = np.zeros(10)
        for w, row in zip(weights, history.itertuples()):
            p = np.asarray(getattr(row, "break_progress_weights", [1.0] * 10), float)
            if len(p) == 10 and p.sum() > 0:
                progress += w * p / p.sum()
        progress = progress / progress.sum() if progress.sum() else np.ones(10) / 10
        progress = 0.4 * progress + 0.6 / 10
        selected = weights >= weights.max() * 0.05
        warnings = [
            "Unattributed stationary time can include unmapped traffic and mechanical delays; stop reasons and locations are uncertain."
        ]
        if self.method == "residual":
            warnings.append(
                "The residual allowance can also absorb riding-model bias; it is not a pure measurement of personal break time."
            )
        if features["distance_km"] > distance.max() * 1.05:
            warnings.append(
                "Stopping allowance is extrapolated beyond historical ride length; confidence is low."
            )
            expected *= min(2, features["distance_km"] / distance.max())
        if effective < 8 or selected.sum() < 8:
            warnings.append("Sparse comparable stopping history; the distribution is provisional.")
        return {
            "expected_s": expected,
            "median_s": weighted_quantile(values, weights, 0.5),
            "upper_s": weighted_quantile(values, weights, 0.9),
            "samples": int(selected.sum()),
            "effective_samples": float(effective),
            "method": self.method,
            "evidence": "moderate"
            if effective >= 8 and features["distance_km"] <= distance.max()
            else "low",
            "progress_weights": progress.tolist(),
            "warnings": warnings,
        }


def allocate_breaks(frame, total_s, progress_weights=None):
    if total_s <= 0:
        return np.zeros(len(frame))
    progress = (frame.start_m.to_numpy() + frame.end_m.to_numpy()) / 2 / max(1, frame.distance_m.sum())
    weights = np.asarray(progress_weights if progress_weights is not None else np.ones(10), float)
    density = np.interp(progress, np.linspace(0.05, 0.95, len(weights)), weights)
    density = np.maximum(0, density) * frame.distance_m.to_numpy()
    if density.sum() <= 0:
        density = np.ones(len(frame))
    return total_s * density / density.sum()
