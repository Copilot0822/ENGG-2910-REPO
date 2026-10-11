import numpy as np
import pandas as pd
import pytest

from cycling.accuracy import percentage_metrics
from cycling.elapsed_models import (
    DurationCorrection,
    apply_duration_correction,
    historical_baseline,
    select_candidate,
)
from cycling.journey import plan_ride
from cycling.model import fit_model
from cycling.phase5_validation import calibrated_rows, paired_difference
from cycling.stopping import StoppingModel
from cycling.tail_calibration import TailCalibration
from cycling.two_stage import HurdleStopping
from cycling.validation import held_out_route


def history(values, distance=25, gaps=0):
    n = len(values)
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(n)],
            "date": pd.date_range("2024-01-01", periods=n, tz="UTC"),
            "distance_m": distance * 1000.0,
            "elapsed_reliable": True,
            "unattributed_stationary_s": values,
            "unobserved_s": gaps,
            "truth_moving_s": distance * 180.0,
            "elevation_gain_m": 100.0,
            "bike": "bike",
            "elapsed_truth_s": distance * 180 + np.asarray(values),
        }
    )


def route(distance):
    return pd.DataFrame(
        {
            "distance_m": [distance * 1000.0],
            "start_m": [0.0],
            "end_m": [distance * 1000.0],
            "gain_m": [100.0],
            "surface": ["paved"],
        }
    )


@pytest.mark.parametrize("values", [[0] * 12, [20] * 12, [60] * 12])
def test_zero_break_probability_without_meaningful_dwell(values):
    result = HurdleStopping.fit(history(values)).predict(route(25), 4500, "2025-01-01T00:00Z")
    assert result["probability_of_stopping"] == 0
    assert result["expected_s"] == 0
    assert result["zero_mass"] == 1


@pytest.mark.parametrize("conditional", ["median", "winsor_mean"])
def test_conditional_duration_and_uncertain_labels(conditional):
    h = history([0] * 6 + [180] * 6 + [600] * 6)
    result = HurdleStopping.fit(h, conditional=conditional).predict(route(25), 4500, "2025-01-01T00:00Z")
    assert 0 < result["probability_of_stopping"] < 1
    assert result["conditional_duration_s"] > 0
    assert result["expected_s"] == pytest.approx(
        result["probability_of_stopping"] * result["conditional_duration_s"]
    )
    assert any("uncertain" in w for w in result["warnings"])


def test_gaps_do_not_become_fabricated_breaks():
    result = HurdleStopping.fit(history([1800] * 12, gaps=1800)).predict(
        route(25), 4500, "2025-01-01T00:00Z"
    )
    assert result["expected_s"] == 0
    assert result["upper_s"] == 0


def test_empty_history_is_unknown_not_a_confident_zero():
    r = HurdleStopping().predict(route(25), 4500, "2025-01-01T00:00Z")
    assert r["samples"] == 0
    assert r["probability_of_stopping"] is None
    assert r["warnings"] and r["evidence"] == "unavailable"


def test_extrapolated_conditional_duration_accounts_for_point_allowance():
    r = HurdleStopping.fit(history([600] * 12)).predict(route(100), 20000, "2025-01-01T00:00Z")
    assert r["expected_s"] == pytest.approx(r["probability_of_stopping"] * r["conditional_duration_s"])
    assert any("extrapolation" in w for w in r["warnings"])


def test_skew_outlier_does_not_dominate_point_allowance():
    a = HurdleStopping.fit(history([0] * 8 + [600] * 12 + [3600])).predict(
        route(25), 4500, "2025-01-01T00:00Z"
    )
    b = HurdleStopping.fit(history([0] * 8 + [600] * 12 + [86400])).predict(
        route(25), 4500, "2025-01-01T00:00Z"
    )
    assert b["expected_s"] < 1.1 * a["expected_s"]
    assert b["expected_s"] < 600
    assert 86400 in b["distribution_s"]
    assert 0 < b["zero_mass"] < 1


def test_long_ride_with_multiple_breaks_and_smooth_distance():
    model = HurdleStopping.fit(pd.concat([history([0] * 25, 25), history([3600] * 12, 180)]))
    assert model.predict(route(180), 30000, "2025-01-01T00:00Z")["expected_s"] > 3000
    values = [
        model.predict(route(d), d * 180, "2025-01-01T00:00Z")["expected_s"]
        for d in [49.9, 50.0, 50.1, 99.9, 100.0, 100.1]
    ]
    assert abs(values[0] - values[2]) < 10
    assert abs(values[3] - values[5]) < 10


def test_stops_only_earlier_reliable_observations():
    h = history([0] * 12 + [10000])
    h.loc[12, "elapsed_reliable"] = False
    h.loc[11, "date"] = pd.Timestamp("2026-01-01T00:00Z")
    result = HurdleStopping.fit(h).predict(route(25), 4500, "2025-01-01T00:00Z")
    assert result["expected_s"] == 0
    assert result["samples"] == 11


def records(n=12):
    return [
        {
            "id": str(i),
            "date": str(pd.Timestamp("2024-01-01T00:00Z") + pd.Timedelta(days=i)),
            "distance_km": 25,
            "physical_s": 4000.0,
            "baseline_s": 5000.0,
            "actual_elapsed_s": 5000.0,
            "gain_per_km": 4,
            "paved_fraction": 1,
        }
        for i in range(n)
    ]


@pytest.mark.parametrize(
    "kind,param",
    [
        ("blend", 0.5),
        ("smooth_blend", 0.5),
        ("bias", 1.0),
        ("distance_bias", 1.0),
        ("route_relative", 0.5),
        ("ridge", 20),
    ],
)
def test_hybrid_and_bias_never_use_current_or_future_truth(kind, param):
    before = records()
    future = {**before[0], "id": "future", "date": "2025-01-01T00:00Z", "actual_elapsed_s": 1000000}
    a, detail = DurationCorrection(kind, param, rows=before).predict(4000, 5000, 25, "2025-01-01T00:00Z")
    b, other = DurationCorrection(kind, param, rows=before + [future]).predict(
        4000, 5000, 25, "2025-01-01T00:00Z"
    )
    assert a == pytest.approx(b)
    assert "future" not in other["training_ids"]
    assert 4000 <= a <= 5500
    assert detail["samples"] == 12


def test_blend_calculation_and_selection_cannot_see_final_errors():
    value, detail = DurationCorrection("blend", 0.75).predict(4000, 8000, 50, "2025-01-01T00:00Z")
    assert value == 5000
    assert detail["advanced_weight"] == 0.75
    rows = [
        {
            "id": str(i),
            "variant": v,
            "split": split,
            "actual_elapsed_s": 1000,
            "predicted_elapsed_s": p,
            "elapsed_reliable": True,
        }
        for i in range(8)
        for split in ["selection", "final"]
        for v, p in [
            ("a", 1100 if split == "selection" else 100000),
            ("b", 1300 if split == "selection" else 1000),
        ]
    ]
    selected, _ = select_candidate(rows, {"a": {}, "b": {}})
    assert selected["candidate"] == "a"
    assert select_candidate(list(reversed(rows)), {"b": {}, "a": {}})[0]["candidate"] == "a"


def test_residual_target_is_not_independent_break_truth():
    h = history([300] * 8)
    residuals = [{"id": str(i), "residual_stop_s": 900, "predicted_moving_s": 3900} for i in range(8)]
    fitted = StoppingModel.fit(h, "residual", 180, residuals)
    assert fitted.rides.stop_target_s.eq(900).all()
    new = HurdleStopping.fit(h)
    assert new.rides.stop_target_s.eq(300).all()


def errors(n=45):
    return [
        {
            "id": str(i),
            "date": str(pd.Timestamp("2024-01-01T00:00Z") + pd.Timedelta(days=i)),
            "distance_km": 30,
            "predicted_elapsed_s": 3600.0,
            "actual_elapsed_s": 3600 * (1 + (i % 10) / 50),
        }
        for i in range(n)
    ]


def test_tail_calibration_asymmetric_and_only_earlier():
    prior = errors()
    a = TailCalibration(prior, 0.9, 0.95).estimate(7200, 100, "2025-01-01T00:00Z")
    current = {**prior[0], "date": "2025-01-01T00:00Z", "actual_elapsed_s": 999999}
    b = TailCalibration(prior + [current], 0.9, 0.95).estimate(7200, 100, "2025-01-01T00:00Z")
    assert a["intervals_s"] == b["intervals_s"]
    assert a["conservative_s"] > 7200
    assert a["intervals_s"]["90"][0] >= 7200


@pytest.mark.parametrize(
    "kwargs",
    [
        {"manual_hr": True},
        {"scenario": "full_stop"},
        {"break_mode": "custom"},
        {"break_mode": "minimal"},
        {"forecast": True},
    ],
)
def test_scenarios_do_not_claim_validated_probability(kwargs):
    result = TailCalibration(errors(70), 0.9, 0.95).estimate(3600, 30, "2025-01-01T00:00Z", **kwargs)
    assert result["provisional"]
    assert not result["probability_supported"]
    assert result["warnings"]


def test_sparse_long_distance_support_is_provisional():
    result = TailCalibration(errors(70), 0.9, 0.95).estimate(40000, 200, "2025-01-01T00:00Z")
    assert result["provisional"] and not result["probability_supported"]
    assert any("long-distance" in w for w in result["warnings"])


def test_prequential_interval_coverage_and_current_exclusion():
    rows = [
        {
            **r,
            "variant": "x",
            "elapsed_reliable": True,
            "actual_before_cutoff": True,
            "cutoff": "2025-01-01T23:00Z",
        }
        for r in errors(30)
    ]
    calibrated, _ = calibrated_rows(rows)
    assert calibrated[0]["interval_90_covered"] is None
    assert calibrated[-1]["interval_90_covered"] is True
    assert calibrated[-1]["id"] not in calibrated[-1]["interval_training_ids"]
    assert len(calibrated[-1]["interval_training_ids"]) == 29


def test_percentage_late_statistics_and_paired_bootstrap():
    rows = [
        {
            "id": str(i),
            "date": str(pd.Timestamp("2024-01-01T00:00Z") + pd.Timedelta(days=i)),
            "variant": v,
            "actual_elapsed_s": 1000,
            "predicted_elapsed_s": p,
            "elapsed_reliable": True,
        }
        for i in range(12)
        for v, p in [("a", 1100), ("CW-auto", 800)]
    ]
    metrics = percentage_metrics([r for r in rows if r["variant"] == "CW-auto"])
    assert metrics["elapsed_mape"] == 20
    assert metrics["elapsed_mspe"] == -20
    assert metrics["underprediction_percentage"] == 100
    assert metrics["underprediction_max_percentage"] == 20
    assert paired_difference(rows, "a")["bootstrap_95_pp"] == [-10.0, -10.0]


def test_planning_manual_hr_breaks_and_corrected_ledger(synthetic_dataset, settings):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    model.variant = "CW"

    class MeasuredEffortResponse:
        def predict(self, x):
            return (x[:, -1] - 140) * 0.002

    model.hr_model = MeasuredEffortResponse()
    model.hr_range = (120, 180)
    s, e = held_out_route(traces["9"], rides.iloc[-1], settings)
    r = {"points": traces["9"], "segments": s, "events": e}
    h = rides.copy()
    h["elapsed_reliable"] = True
    h["elapsed_truth_s"] = h.moving_s + 600
    h["unattributed_stationary_s"] = 600
    h["truth_moving_s"] = h.moving_s
    planning = {
        "phase5": True,
        "recommended_method": "hybrid",
        "stopping": HurdleStopping.fit(h),
        "corrections": {"hybrid": DurationCorrection("blend", 0.5)},
        "calibrations": {
            "hybrid": TailCalibration(errors(70), 0.9, 0.95),
            "CW+stops-known": TailCalibration(errors(70), 1.0, 1.0),
        },
    }
    result = plan_ride(
        model,
        r,
        h,
        settings,
        planning,
        140,
        departure="2025-07-01T12:00Z",
        scenario="observed",
        manual_hr=True,
    )
    assert result["elapsed_uncertainty"]["calibration_key"] == "hybrid"
    assert result["elapsed_uncertainty"]["provisional"]
    assert result["elapsed_s"] == pytest.approx(
        result["moving_s"]
        + result["stationary_s"]
        + result["personal_break_s"]
        + result["statistical_correction_s"]
    )
    for mode, expected in [("custom", 1200), ("minimal", 0)]:
        p = plan_ride(
            model,
            r,
            h,
            settings,
            planning,
            140,
            departure="2025-07-01T12:00Z",
            break_mode=mode,
            break_minutes=20,
        )
        assert p["personal_break_s"] == pytest.approx(expected)
        assert p["statistical_correction_s"] == 0
        assert p["elapsed_s"] == pytest.approx(p["moving_s"] + p["stationary_s"] + expected)
    faster = plan_ride(
        model,
        r,
        h,
        settings,
        planning,
        155,
        departure="2025-07-01T12:00Z",
        scenario="observed",
        manual_hr=True,
    )
    assert faster["requested_hr"] == 155
    assert faster["elapsed_s"] != result["elapsed_s"]
    assert historical_baseline(h, 10, "2020-01-01T00:00Z") is None
    adjusted = apply_duration_correction(result, result["elapsed_s"] + 300, {"kind": "test"})
    assert adjusted["personal_break_s"] == result["personal_break_s"]
    assert (
        pd.Timestamp(adjusted["finish_time_utc"]) - pd.Timestamp(adjusted["departure_utc"])
    ).total_seconds() == pytest.approx(adjusted["elapsed_s"], abs=1e-6)
