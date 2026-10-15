"""Transparent total-duration corrections fitted only to earlier OOF predictions."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .hr import weighted_quantile
from .model import recency_weight


LABELS = {
    "physical": "Riding and traffic with the selected break allowance",
    "C-auto": "Model C",
    "CW-auto": "C + Weather, no breaks",
    "CW+stops-auto": "Existing residual stops",
    "simple-elapsed": "Historical elapsed baseline",
    "two-stage": "Two-stage personalized stopping",
    "bias-corrected": "Bias-corrected C + Weather",
    "hybrid": "Hybrid elapsed duration",
    "hybrid-stops": "Hybrid with two-stage stopping",
    "automatic": "Earlier-selected candidate (assessment)",
}


def historical_baseline(rides, distance_km, departure, half_life=180):
    moment = pd.Timestamp(departure)
    h = rides[rides.elapsed_reliable & (pd.to_datetime(rides.date, utc=True) < moment)]
    if h.empty:
        return None
    age = (moment - pd.to_datetime(h.date, utc=True)).dt.total_seconds().to_numpy() / 86400
    return float(
        distance_km
        * weighted_quantile(
            h.elapsed_truth_s / (h.distance_m / 1000).clip(lower=0.1),
            recency_weight(age, half_life),
            0.5,
        )
    )


def correction_features(row):
    return np.array(
        [
            np.log(max(0.1, row["distance_km"]) / 50),
            np.log(max(1, row["baseline_s"]) / max(1, row["physical_s"])),
            np.clip(row.get("gain_per_km", 0) / 20, 0, 3),
            np.clip(row.get("paved_fraction", 0.5), 0, 1),
        ]
    )


@dataclass
class DurationCorrection:
    kind: str = "blend"
    parameter: float = 0.5
    half_life: float = 180
    rows: list = field(default_factory=list)

    def predict(self, physical_s, baseline_s, distance_km, departure, features=None):
        moment = pd.Timestamp(departure)
        past = pd.DataFrame(self.rows)
        if len(past):
            past = past[pd.to_datetime(past.date, utc=True, format="mixed") < moment]
        current = {
            "physical_s": physical_s,
            "baseline_s": baseline_s or physical_s,
            "distance_km": distance_km,
            **(features or {}),
        }
        if baseline_s is None:
            return physical_s, {"kind": "physical fallback", "samples": 0}
        if self.kind in ("blend", "smooth_blend"):
            w = self.parameter
            if self.kind == "smooth_blend":
                w = 0.25 + 0.5 / (1 + np.exp(-np.log(max(0.1, distance_km) / 100) / 0.7))
            value = w * physical_s + (1 - w) * baseline_s
            detail = {"kind": self.kind, "advanced_weight": float(w), "samples": len(past)}
        elif len(past) < 6:
            return physical_s, {
                "kind": "physical fallback: fewer than six earlier OOF errors",
                "samples": len(past),
            }
        else:
            age = (
                moment - pd.to_datetime(past.date, utc=True, format="mixed")
            ).dt.total_seconds().to_numpy() / 86400
            weights = recency_weight(age, self.half_life)
            ratios = past.actual_elapsed_s.to_numpy() / past.physical_s.to_numpy()
            if self.kind == "distance_bias":
                local = np.exp(-0.5 * (np.log(past.distance_km / max(0.1, distance_km)) / 0.7) ** 2)
                # Pooled pseudo-observations regularize sparse long-distance evidence.
                weights = weights * (local + 0.25)
            # MAPE-optimal weighted ratio median: per-ride percentage denominator.
            ratio = np.clip(weighted_quantile(ratios, weights / np.maximum(0.01, ratios), 0.5), 0.7, 1.6)
            if self.kind in ("bias", "distance_bias"):
                factor = 1 + self.parameter * (ratio - 1)
                value = physical_s * factor
                detail = {"kind": self.kind, "factor": float(factor), "samples": len(past)}
            elif self.kind == "route_relative":
                gamma = self.parameter
                fitted = past.baseline_s * (past.physical_s / past.baseline_s) ** gamma
                factor = weighted_quantile(
                    past.actual_elapsed_s / fitted, weights * fitted / past.actual_elapsed_s, 0.5
                )
                value = baseline_s * (physical_s / baseline_s) ** gamma * np.clip(factor, 0.7, 1.6)
                detail = {
                    "kind": self.kind,
                    "route_exponent": gamma,
                    "factor": float(factor),
                    "samples": len(past),
                }
            else:
                if len(past) < 12:
                    return physical_s, {
                        "kind": "physical fallback: sparse ridge calibration",
                        "samples": len(past),
                    }
                x = np.array([correction_features(r) for r in past.to_dict("records")])
                y = np.log(np.clip(ratios, 0.6, 2))
                ridge = Ridge(alpha=self.parameter).fit(x, y, sample_weight=weights)
                factor = np.clip(
                    np.exp(ridge.predict(correction_features(current).reshape(1, -1))[0]), 0.7, 1.8
                )
                value = physical_s * factor
                detail = {
                    "kind": "regularized OOF residual",
                    "factor": float(factor),
                    "samples": len(past),
                }
        return float(max(0, value)), {
            **detail,
            "training_ids": past.id.astype(str).tolist() if len(past) else [],
        }


def apply_duration_correction(result, total_s, detail):
    """Keep every physical component; adjustment is a separately named total correction."""
    result = result.copy()
    physical = result["elapsed_s"]
    result["physical_elapsed_s"] = physical
    result["statistical_correction_s"] = float(total_s - physical)
    result["duration_correction"] = detail
    result["elapsed_s"] = float(total_s)
    result["overall_kmh"] = result["distance_m"] / max(1, total_s) * 3.6
    result["finish_time_utc"] = (
        pd.Timestamp(result["departure_utc"]) + pd.Timedelta(seconds=total_s)
    ).isoformat()
    result["warnings"] = list(result["warnings"]) + [
        "Total-duration statistical adjustment includes historical interruption and riding bias. It is not an observed break or traffic wait. Segment progress retains the physical ledger; adjustment is unallocated."
    ]
    return result


def select_candidate(rows, candidates):
    """Explicit selection split; final or current/future rows cannot choose parameters."""
    from .accuracy import percentage_metrics

    scores = []
    for name, params in candidates.items():
        subset = [r for r in rows if r.get("split") == "selection" and r["variant"] == name]
        metric = percentage_metrics(subset)
        if metric["rides"]:
            scores.append({"candidate": name, "parameters": params, **metric})
    if not scores:
        raise ValueError("No earlier selection predictions")
    chosen = min(scores, key=lambda s: (s["elapsed_mape"], s["candidate"]))
    return chosen, scores
