from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from cycling.accuracy import (
    PRIMARY_VARIANTS,
    paired_percentage_rows,
    percentage_metrics,
    percentage_rows,
)
from cycling.journey_validation import (
    percentage_report,
    refresh_percentage_reporting,
    select_stopping_method,
)


def row(actual, predicted, ride_id="1", variant="CW+stops-auto", distance=30, **extra):
    return {
        "id": ride_id,
        "date": "2025-06-01T12:00:00Z",
        "actual_elapsed_s": actual,
        "predicted_elapsed_s": predicted,
        "elapsed_reliable": True,
        "variant": variant,
        "distance_km": distance,
        **extra,
    }


def report_fixture():
    rows = []
    for variant, multiplier in zip(PRIMARY_VARIANTS, [0.8, 0.9, 1.0, 1.05]):
        rows.extend(
            [
                row(3600, 4320 if variant == "CW+stops-auto" else 3600 * multiplier, variant=variant),
                row(
                    36000, 32400 if variant == "CW+stops-auto" else 36000 * multiplier, "2", variant, 120
                ),
            ]
        )
    return {
        "rows": rows,
        "selected_stopping_method": "residual",
        "selected_hr_half_life": 365,
        "final_rides": 2,
        "selection_rides": 12,
        "selection_last": "2025-05-01T00:00Z",
        "final_first": "2025-06-01T12:00:00Z",
        "calibration_rows": {"CW+stops-auto": [{"sentinel": "unchanged"}]},
        "residual_rows": [{"sentinel": "unchanged"}],
        "weather_evaluation": "Mocked retrospective weather evaluation.",
        "probability_note": "Probabilities are withheld; calibration is insufficient.",
        "hr_metrics": {"mae_bpm": 8, "global_median_mae_bpm": 9, "rides": 2},
        "hr_groups": {},
        "stopping_candidates": [{"method": "residual", "elapsed_mae_minutes": 10}],
        "hr_candidates": [],
    }


def test_five_hour_prediction_of_five_and_half_hour_ride():
    result = percentage_rows([row(5.5 * 3600, 5 * 3600)]).iloc[0]
    assert result.signed_percentage_error == pytest.approx(-9.0909090909)
    assert result.absolute_percentage_error == pytest.approx(9.0909090909)
    assert result.prediction_direction == "Underestimated"
    assert pd.Timestamp(result.predicted_finish_utc) < pd.Timestamp(result.actual_finish_utc)


def test_each_ride_is_normalized_before_averaging_not_ratio_of_mae():
    values = [row(3600, 5400), row(36000, 32400, "2")]
    metrics = percentage_metrics(values)
    assert metrics["elapsed_mape"] == 30  # (50% + 10%) / 2
    assert metrics["elapsed_mspe"] == 20  # (50% - 10%) / 2
    wrong_ratio = 100 * ((1800 + 3600) / 2) / ((3600 + 36000) / 2)
    assert metrics["elapsed_mape"] != pytest.approx(wrong_ratio)
    percentage_improvement_in_mae = 100 * (3000 - 2700) / 3000
    assert metrics["elapsed_mape"] != percentage_improvement_in_mae


def test_opposite_signed_errors_do_not_cancel_absolute_errors():
    metrics = percentage_metrics([row(100, 90), row(200, 220, "2")])
    assert metrics["elapsed_mape"] == 10
    assert metrics["elapsed_mdape"] == 10
    assert metrics["elapsed_mspe"] == 0
    assert metrics["elapsed_p90_ape"] == 10


def test_median_and_linear_p90_are_manually_verifiable():
    metrics = percentage_metrics(
        [row(100, predicted, str(i)) for i, predicted in enumerate([100, 110, 120, 130, 150])]
    )
    assert metrics["elapsed_mape"] == 22
    assert metrics["elapsed_mdape"] == 20
    assert metrics["elapsed_p90_ape"] == 42  # linear interpolation: 30 + .6*(50-30)


def test_underprediction_frequency_and_conditional_magnitudes():
    values = [row(100, p, str(i)) for i, p in enumerate([95, 90, 80, 120, 100])]
    metrics = percentage_metrics(values)
    assert metrics["underpredicted_rides"] == 3
    assert metrics["underprediction_percentage"] == 60
    assert metrics["underprediction_mape"] == pytest.approx(35 / 3)
    assert metrics["underprediction_median_percentage"] == 10
    assert metrics["underprediction_max_percentage"] == 20
    assert metrics["underprediction_p90_percentage"] == 18
    assert metrics["late_arrival_p90_percentage"] == 16
    assert metrics["elapsed_mspe"] == -3


@pytest.mark.parametrize("actual", [0, -1, None, np.nan, np.inf, "invalid"])
def test_invalid_actual_durations_are_explained_and_not_divided(actual):
    values = [row(actual, 100), row(100, 150, "2")]
    metrics = percentage_metrics(values)
    assert metrics["rides"] == 1 and metrics["input_rides"] == 2 and metrics["excluded_rides"] == 1
    assert metrics["elapsed_mape"] == 50
    assert metrics["percentage_exclusion_counts"] == {"invalid_actual_elapsed": 1}
    excluded = percentage_rows(values).iloc[0]
    assert pd.isna(excluded.signed_percentage_error)
    assert excluded.prediction_direction == "Excluded"


def test_unreliable_and_invalid_predictions_are_excluded_but_zero_prediction_is_scored():
    values = [
        row(100, 100, "1", elapsed_reliable=False),
        row(100, np.inf, "2"),
        row(100, -1, "3"),
        row(100, 0, "4"),
    ]
    metrics = percentage_metrics(values)
    assert metrics["rides"] == 1 and metrics["elapsed_mape"] == 100
    assert metrics["percentage_exclusion_counts"] == {"unreliable_elapsed": 1, "invalid_prediction": 2}


def test_empty_and_no_underpredictions_do_not_invent_metrics():
    empty = percentage_metrics([])
    assert empty["rides"] == 0 and empty["elapsed_mape"] is None
    assert empty["underprediction_percentage"] is None
    metrics = percentage_metrics([row(100, 110)])
    assert metrics["underprediction_percentage"] == 0
    assert metrics["underprediction_mape"] is None
    assert metrics["underprediction_p90_percentage"] is None
    assert metrics["late_arrival_p90_percentage"] == 0


def test_same_ride_intersection_and_explicit_exclusions_keep_large_errors():
    values = [row(100, 2000, variant=variant) for variant in PRIMARY_VARIANTS]
    values += [row(100, 100, "2", variant) for variant in PRIMARY_VARIANTS if variant != "CW-auto"]
    values += [row(0, 100, "3", variant) for variant in PRIMARY_VARIANTS]
    frame, cohort = paired_percentage_rows(values)
    assert cohort["candidate_rides"] == 3 and cohort["evaluated_rides"] == 1
    assert cohort["ride_ids"] == ["1"]
    assert len(cohort["excluded_rides"]) == 2
    assert any("missing prediction: CW-auto" in r["reasons"] for r in cohort["excluded_rides"])
    report = percentage_report({"rows": values})
    assert all(r["rides"] == 1 and r["elapsed_mape"] == 1900 for r in report["comparisons"])
    assert frame[frame.paired_comparison_eligible].id.nunique() == 1
    assert len(report["rows"]) == len(values)  # difficult and excluded records are retained


def test_inconsistent_truth_and_duplicates_do_not_create_unfair_comparisons():
    values = [row(100 if i else 200, 100, variant=variant) for i, variant in enumerate(PRIMARY_VARIANTS)]
    _, cohort = paired_percentage_rows(values)
    assert cohort["evaluated_rides"] == 0
    assert "inconsistent actual elapsed duration across models" in cohort["excluded_rides"][0]["reasons"]
    with pytest.raises(ValueError, match="Duplicate"):
        paired_percentage_rows(values + [values[0]])


def test_group_counts_and_percentage_ranking_do_not_change_selection_or_calibration():
    original = report_fixture()
    copy = deepcopy(original)
    report = percentage_report(original)
    assert original == copy
    for key in [
        "selected_stopping_method",
        "selected_hr_half_life",
        "final_first",
        "selection_last",
        "calibration_rows",
        "residual_rows",
        "stopping_candidates",
    ]:
        assert report[key] == original[key]
    groups = report["groups"]["CW+stops-auto"]
    assert groups["under 50 km"]["rides"] == 1
    assert groups["100–150 km"]["rides"] == 1
    assert groups["50–100 km"]["rides"] == 0
    assert groups["50–100 km"]["elapsed_mape"] is None
    durations = report["duration_groups"]["CW+stops-auto"]
    assert durations["1–2 hours"]["rides"] == 1 and durations["8+ hours"]["rides"] == 1
    assert report["stopping_selection_metric"] == "elapsed_mae_minutes"
    assert report["percentage_ranking"]["lowest_mape_variant"] == "simple-elapsed"


def test_future_selection_uses_earlier_mape_and_never_final_ranking():
    earlier = []
    for method, predicted in [
        ("context", [110, 1000]),
        ("distance", [100, 1020]),
        ("residual", [200, 2000]),
    ]:
        earlier += [
            row(actual, estimate, str(i), f"candidate-{method}", split="selection")
            for i, (actual, estimate) in enumerate(zip([100, 1000], predicted))
        ]
    selected, candidates = select_stopping_method(earlier)
    assert selected == "distance"  # 1% MAPE despite larger minute MAE than context
    assert min(candidates, key=lambda r: r["elapsed_mae_minutes"])["method"] == "context"
    final = [
        row(100, p, "future", f"candidate-{m}", split="calibration")
        for m, p in [("context", 100), ("distance", 100000), ("residual", 100)]
    ]
    assert select_stopping_method(earlier + final) == (selected, candidates)


def test_rescoring_saved_report_does_not_fit_or_modify_model_files(store):
    original = report_fixture()
    store.set_state("journey_validation", original)
    artifact = store.root / "models/journey.joblib"
    artifact.write_bytes(b"sentinel: model must not be opened or fitted")
    report = refresh_percentage_reporting(store)
    assert artifact.read_bytes() == b"sentinel: model must not be opened or fitted"
    assert report["selected_stopping_method"] == original["selected_stopping_method"]
    assert [(r["actual_elapsed_s"], r["predicted_elapsed_s"]) for r in report["rows"]] == [
        (r["actual_elapsed_s"], r["predicted_elapsed_s"]) for r in original["rows"]
    ]
    exported = pd.read_csv(store.root / "reports/journey-validation.csv")
    assert {
        "signed_percentage_error",
        "absolute_percentage_error",
        "actual_finish_utc",
        "predicted_finish_utc",
    } <= set(exported)


@pytest.mark.parametrize("invalid", [False, True])
def test_performance_page_percentage_metrics_chart_filters_and_unavailable(
    monkeypatch, store, settings, invalid
):
    monkeypatch.setenv("CYCLING_DATA_DIR", settings.data_dir)
    report = report_fixture()
    if invalid:
        for value in report["rows"]:
            value["actual_elapsed_s"] = 0
    store.set_state("journey_validation", report)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=60)
    app.run()
    app.sidebar.radio[0].set_value("Model Performance").run()
    assert not app.exception
    assert app.metric[0].label == "Total Elapsed-Time MAPE (%)"
    assert app.metric[0].value == ("Unavailable" if invalid else "15.0%")
    assert any(
        m.label == "Total evaluated rides" and m.value == ("0" if invalid else "2") for m in app.metric
    )
    view = next(d.value for d in app.dataframe if "Absolute error (%)" in d.value.columns)
    assert len(view) == 2
    if not invalid:
        assert view["Absolute error (%)"].tolist() == [20, 10]
        charts = app.get("plotly_chart")
        assert len(charts) >= 3
    next(s for s in app.selectbox if s.label == "Validation distance group").set_value(
        "100–150 km"
    ).run()
    assert not app.exception
    view = next(d.value for d in app.dataframe if "Absolute error (%)" in d.value.columns)
    assert len(view) == 1 and view["Ride distance (km)"].iloc[0] == 120
    next(s for s in app.selectbox if s.label == "Elapsed diagnostic model").set_value("CW-auto").run()
    assert not app.exception
    assert app.metric[0].value == ("Unavailable" if invalid else "10.0%")
