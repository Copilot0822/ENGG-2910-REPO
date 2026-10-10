"""Ride-balanced, recency-weighted cruising models with measured-only HR adjustments."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import os
from pathlib import Path

# Small tabular models run faster with a bounded thread pool; avoid saturating a user's PC.
os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from .events import BehaviorModel
from .importer import impute_ride_hr

_THREAD_LIMIT = threadpool_limits(limits=4)

NUMERIC = [
    "gradient_pct",
    "planned_km",
    "completed_km",
    "progress",
    "gain_m",
    "moving_hours",
    "observation_day",
    "day_of_year",
    "elevation_missing",
]
CATEGORICAL = ["surface", "surface_raw", "smoothness", "tracktype", "highway", "bike"]


def recency_weight(age_days, half_life_days=120):
    if half_life_days <= 0:
        raise ValueError("Half-life must be positive.")
    return np.exp2(-np.maximum(0, np.asarray(age_days, dtype=float)) / half_life_days)


def training_weights(segments, reference_date, half_life_days):
    dates = pd.to_datetime(segments.date, utc=True)
    ages = (pd.Timestamp(reference_date) - dates).dt.total_seconds().to_numpy() / 86400
    weights = recency_weight(ages, half_life_days)
    # Every ride has the same pre-recency budget, regardless of its number of GPS segments.
    counts = segments.groupby("ride_id").ride_id.transform("count").to_numpy()
    weights /= np.maximum(counts, 1)
    # Keep some support for rare terrain / long-distance observations without a fitness-age floor.
    rare = segments.groupby("surface").ride_id.transform("nunique").to_numpy()
    weights *= np.where(rare < 5, 1.5, 1.0)
    return weights / max(1e-12, weights.mean())


@dataclass
class FeatureEncoder:
    categories: dict = field(default_factory=dict)
    ranges: dict = field(default_factory=dict)

    def fit(self, frame):
        for col in CATEGORICAL:
            self.categories[col] = list(frame[col].fillna("unknown").value_counts().head(60).index)
        for col in NUMERIC:
            values = pd.to_numeric(frame[col], errors="coerce")
            self.ranges[col] = (float(values.min()), float(values.max()))
        return self

    def transform(self, frame, hr=None):
        columns = []
        for col in NUMERIC:
            value = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            # Do not extend an unlearned fatigue or fitness trend indefinitely past observed support.
            lo, hi = self.ranges.get(col, (-np.inf, np.inf))
            if col in ("planned_km", "completed_km", "gain_m", "moving_hours", "observation_day"):
                value = np.clip(value, lo, hi)
            columns.append(value)
        for col in CATEGORICAL:
            mapping = {value: i for i, value in enumerate(self.categories[col])}
            columns.append(frame[col].fillna("unknown").map(mapping).fillna(-1).to_numpy(dtype=float))
        if hr is not None:
            columns.append(np.asarray(hr, dtype=float))
        return np.column_stack(columns)

    @property
    def categorical_mask(self):
        return [False] * len(NUMERIC) + [True] * len(CATEGORICAL)


def profile_features(frame):
    gradient = np.clip(frame.gradient_pct.to_numpy(dtype=float), -15, 15)
    return np.column_stack(
        [
            gradient,
            np.maximum(0, gradient),
            np.clip(frame.moving_hours.to_numpy(dtype=float), 0, 15),
            np.clip(frame.completed_km.to_numpy(dtype=float), 0, 300) / 100,
        ]
    )


@dataclass
class CyclingModel:
    encoder: FeatureEncoder
    median_hr: float | None
    half_life: float
    base: object = None
    hr_model: object = None
    hr_profile: object = None
    behavior: BehaviorModel = field(default_factory=BehaviorModel)
    baseline_bins: dict = field(default_factory=dict)
    fallback_kmh: float = 0.0
    rolling_baseline_kmh: float = 0.0
    hr_range: tuple = (None, None)
    train_ids: list = field(default_factory=list)
    support: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    fitness_adjustment: float = 1.0
    trained_at: str = ""
    kind: str = "personal historical baseline"

    @property
    def hr_available(self):
        return self.hr_model is not None and self.hr_profile is not None

    def predict_cruise(self, frame, hr=None, use_fitness=False):
        if self.base is None:
            result = np.asarray(
                [
                    self.baseline_bins.get(int(np.clip(np.floor(g / 3), -8, 8)), self.fallback_kmh)
                    for g in frame.gradient_pct
                ],
                dtype=float,
            )
        else:
            x = self.encoder.transform(frame)
            log_speed = self.base.predict(x)
            if hr is not None and self.hr_available:
                supplied = np.clip(np.asarray(hr), *self.hr_range)
                reference = np.full(len(frame), self.median_hr)
                delta = self.hr_model.predict(
                    self.encoder.transform(frame, supplied)
                ) - self.hr_model.predict(self.encoder.transform(frame, reference))
                # HR residuals are restrained to the range of measured relationships.
                log_speed += np.clip(delta, -0.35, 0.35)
            result = np.exp(log_speed)
        if use_fitness:
            result *= self.fitness_adjustment
        low, high = self.support.get("speed_bounds", (0.8, 90.0))
        return np.clip(result, max(0.8, low), min(100.0, high))

    def baseline_predict(self, frame, gradient=False):
        if not gradient:
            return np.full(len(frame), self.rolling_baseline_kmh)
        return np.asarray(
            [
                self.baseline_bins.get(int(np.clip(np.floor(g / 3), -8, 8)), self.fallback_kmh)
                for g in frame.gradient_pct
            ],
            dtype=float,
        )

    def save(self, path):
        path = Path(path)
        temporary = path.with_suffix(".tmp")
        joblib.dump(self, temporary, compress=3)
        temporary.replace(path)


def fit_model(rides, segments, observations, half_life=120, reference_date=None):
    if len(rides) == 0 or len(segments) == 0:
        raise ValueError("Import and process at least one outdoor cycling ride before training.")
    ids = set(rides.id.astype(str))
    segments = segments[segments.ride_id.astype(str).isin(ids)].copy()
    rides, median_hr = impute_ride_hr(rides)
    eligible = segments.is_cruise & segments.speed_kmh.between(3, 95) & segments.distance_m.ge(15)
    cruise = segments[eligible].copy()
    if len(cruise) < 10:
        raise ValueError("Too few reliable cruising segments. Import longer, timestamped outdoor rides.")
    reference = pd.Timestamp(reference_date or rides.date.max())
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    encoder = FeatureEncoder().fit(cruise)
    weights = training_weights(cruise, reference, half_life)
    speeds = cruise.speed_kmh.to_numpy()
    dates = pd.to_datetime(rides.date, utc=True)
    ride_weights = recency_weight((reference - dates).dt.total_seconds() / 86400, half_life)
    ride_speeds = rides.distance_m / rides.moving_s * 3.6
    rolling_baseline = float(np.average(ride_speeds, weights=ride_weights))
    # Weighted median is used for the declared historical-median baseline.
    order = np.argsort(ride_speeds)
    cum = np.cumsum(ride_weights[order])
    rolling_baseline = float(ride_speeds.iloc[order[np.searchsorted(cum, cum[-1] / 2)]])
    baseline_bins = {}
    for key, indices in cruise.groupby(np.floor(cruise.gradient_pct / 3).clip(-8, 8)).groups.items():
        selected = cruise.loc[indices]
        if len(selected) >= 8:
            baseline_bins[int(key)] = float(selected.speed_kmh.median())
    model = CyclingModel(
        encoder,
        median_hr,
        half_life,
        baseline_bins=baseline_bins,
        fallback_kmh=float(np.median(speeds)),
        rolling_baseline_kmh=rolling_baseline,
        train_ids=sorted(ids),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    model.behavior = BehaviorModel([o for o in observations if str(o["ride_id"]) in ids])
    model.support = {
        "rides": len(rides),
        "cruise_segments": len(cruise),
        "measured_hr_rides": int((~rides.hr_imputed).sum()),
        "imputed_hr_rides": int(rides.hr_imputed.sum()),
        "max_distance_km": float(rides.distance_m.max() / 1000),
        "long_rides": int((rides.distance_m >= 100_000).sum()),
        "speed_bounds": (
            float(np.quantile(speeds, 0.005) * 0.65),
            float(np.quantile(speeds, 0.995) * 1.2),
        ),
        "last_ride": str(rides.date.max()),
        "first_ride": str(rides.date.min()),
        "surfaces": sorted(cruise.surface.unique().tolist()),
        "bikes": sorted(rides.bike.unique().tolist()),
    }
    if len(rides) >= 4 and len(cruise) >= 100:
        model.base = HistGradientBoostingRegressor(
            max_iter=90,
            max_leaf_nodes=15,
            min_samples_leaf=max(15, min(70, len(cruise) // 80)),
            l2_regularization=8,
            learning_rate=0.07,
            categorical_features=encoder.categorical_mask,
            early_stopping=False,
            random_state=42,
        )
        x = encoder.transform(cruise)
        y = np.log(speeds)
        model.base.fit(x, y, sample_weight=weights)
        model.kind = "personal segment gradient-boosted model"
        averages = rides.set_index("id").measured_avg_hr.to_dict()
        measured = (
            cruise.hr_measured
            & cruise.measured_hr.between(45, 220)
            & cruise.ride_id.map(averages).notna()
        )
        hr_data = cruise[measured]
        if (
            hr_data.ride_id.nunique() >= 4
            and len(hr_data) >= 100
            and hr_data.measured_hr.std() >= 4
            and median_hr is not None
        ):
            low, high = np.quantile(hr_data.measured_hr, [0.02, 0.98])
            model.hr_range = (float(low), float(high))
            hw = training_weights(hr_data, reference, half_life)
            residual = np.log(hr_data.speed_kmh) - model.base.predict(encoder.transform(hr_data))
            monotonic = [0] * (len(NUMERIC) + len(CATEGORICAL)) + [1]
            model.hr_model = HistGradientBoostingRegressor(
                max_iter=65,
                max_leaf_nodes=8,
                min_samples_leaf=max(20, min(80, len(hr_data) // 60)),
                l2_regularization=15,
                learning_rate=0.06,
                categorical_features=encoder.categorical_mask + [False],
                monotonic_cst=monotonic,
                early_stopping=False,
                random_state=43,
            )
            model.hr_model.fit(
                encoder.transform(hr_data, hr_data.measured_hr), residual, sample_weight=hw
            )
            within_ride = hr_data.measured_hr - hr_data.ride_id.map(averages)
            model.hr_profile = Ridge(alpha=100)
            model.hr_profile.fit(profile_features(hr_data), within_ride, sample_weight=hw)
    # Measured-only HR adjustment means imputed rides retain their full terrain-learning weight.
    return model


def load_model(store):
    if not store.model_path().exists():
        return None
    # Only load this app's own local fitted models. Never deserialize an uploaded joblib file.
    return joblib.load(store.model_path())
