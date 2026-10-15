import hashlib
import json
from copy import deepcopy

import joblib
import pandas as pd
import pytest

from cycling.elapsed_models import DurationCorrection
from cycling.journey import load_planning, plan_ride
from cycling.journey_validation import elapsed_metrics
from cycling.model import fit_model
from cycling.release_policy import (
    LEGACY,
    PREFERRED,
    REFERENCE,
    operational_decision,
    reference_gate,
    refresh_operational_default,
    verify_two_stage,
)
from cycling.stopping import StoppingModel
from cycling.tail_calibration import TailCalibration
from cycling.two_stage import HurdleStopping
from cycling.validation import held_out_route


class EffortResponse:
    def predict(self, x):
        return (x[:, -1] - 140) * 0.002


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def installed(store, settings, synthetic_dataset):
    rides, segments, traces = synthetic_dataset
    model = fit_model(rides, segments, [])
    model.variant = "CW"
    model.hr_model = EffortResponse()
    model.hr_range = (120, 180)
    model.save(store.root / "models/model-CW.joblib")
    model.save(store.model_path())
    history = rides.copy()
    history["elapsed_reliable"] = True
    history["elapsed_truth_s"] = history.moving_s + 300
    history["truth_moving_s"] = history.moving_s
    history["unattributed_stationary_s"] = 300
    history["unobserved_s"] = 0
    for row in history.to_dict("records"):
        store.add_ride(row, row["id"], traces[row["id"]])
    route, events = held_out_route(traces["9"], history.iloc[-1], settings)
    route = {"points": traces["9"], "segments": route, "events": events, "filename": "synthetic.gpx"}
    route["warnings"] = []
    calibration = [
        {
            "id": f"cal-{i}",
            "date": str(pd.Timestamp("2024-01-01T00:00Z") + pd.Timedelta(days=i)),
            "distance_km": 30.0,
            "actual_elapsed_s": 1400.0 if i % 4 else 1100.0,
            "predicted_elapsed_s": 1200.0,
            "variant": PREFERRED,
        }
        for i in range(30)
    ]
    rows = []
    for i in range(20):
        for key, pred in [
            (LEGACY, 1600),
            (PREFERRED, 1100),
            (REFERENCE, 900),
            ("hybrid-stops", 1200),
            ("bias-corrected", 1050),
        ]:
            rows.append(
                {
                    "id": f"assess-{i}",
                    "date": str(pd.Timestamp("2024-03-01T00:00Z") + pd.Timedelta(days=i)),
                    "distance_km": 30.0,
                    "actual_elapsed_s": 1000.0,
                    "predicted_elapsed_s": pred,
                    "elapsed_reliable": True,
                    "variant": key,
                    "interval_80_covered": i % 4 != 0,
                    "interval_90_covered": i % 5 != 0,
                }
            )
    comparisons = [
        {"variant": key, **elapsed_metrics([r for r in rows if r["variant"] == key])}
        for key in [LEGACY, PREFERRED, REFERENCE, "hybrid-stops", "bias-corrected"]
    ]
    report = {
        "status": "complete",
        "selected_method": "hybrid-stops",
        "recommended_method": LEGACY,
        "default_changed": False,
        "fresh_difference": {"rides": 15, "mape_difference_pp": 1.0, "bootstrap_95_pp": [-2.0, 5.0]},
        "frozen_selection": {"families": {PREFERRED: {"half_life": 180, "conditional": "median"}}},
        "rows": rows,
        "calibration_rows": {PREFERRED: calibration},
        "all_later": {"comparisons": comparisons},
        "reused": {"comparisons": [{**m, "rides": 5} for m in comparisons]},
        "fresh": {"comparisons": [{**m, "rides": 15} for m in comparisons]},
        "operational_benchmark_ids": [f"assess-{i}" for i in range(5)],
    }
    metric = next(m for m in comparisons if m["variant"] == PREFERRED)
    legacy = history.copy()
    legacy["unattributed_stationary_s"] = 1800
    planning = {
        "phase5": True,
        "recommended_method": LEGACY,
        "trained_at": "2025-05-01T00:00Z",
        "hr_half_life": 365,
        "train_ids": history.id.tolist(),
        "stopping": StoppingModel.fit(legacy),
        "hurdle": HurdleStopping.fit(history, 180, "median"),
        "hybrid_hurdle": HurdleStopping.fit(history, 180, "median"),
        "calibrations": {
            PREFERRED: TailCalibration(calibration, metric["coverage_80"], metric["coverage_90"]),
            LEGACY: TailCalibration(
                [{**r, "variant": LEGACY, "actual_elapsed_s": 3000} for r in calibration], 0.9, 0.95
            ),
            "bias-corrected": TailCalibration(calibration, 0.75, 0.8),
        },
        "corrections": {"bias-corrected": DurationCorrection("blend", 0.5)},
    }
    store.set_state("phase5_validation", report)
    store.set_state("model_stale", False)
    store.set_state("phase5_stale", False)
    store.set_state("journey_stale", False)
    store.set_state(
        "journey_validation", {"rows": [{"id": i} for i in report["operational_benchmark_ids"]]}
    )
    joblib.dump(planning, store.root / "models/journey-phase5.joblib")
    (store.root / "reports/phase5-validation.json").write_text(json.dumps(report, indent=2))
    return {
        "store": store,
        "settings": settings,
        "model": model,
        "history": history,
        "route": route,
        "planning": planning,
        "report": report,
    }


@pytest.mark.parametrize(
    "difference,passed",
    [
        ({"mape_difference_pp": -2.0, "bootstrap_95_pp": [-4.0, -1.0]}, True),
        ({"mape_difference_pp": 1.0, "bootstrap_95_pp": [-3.0, 5.0]}, False),
    ],
)
def test_gate_reference_never_implicitly_selects_a_different_incumbent(difference, passed):
    gate = reference_gate("hybrid-stops", difference, LEGACY, REFERENCE)
    assert gate["passed"] is passed
    assert gate["reference_comparator"] == REFERENCE
    assert gate["current_deployed_incumbent"] == LEGACY
    assert "new_operational_default" not in gate


@pytest.mark.parametrize("original_passed", [True, False])
def test_original_gate_fact_is_distinct_from_explicit_operational_correction(installed, original_passed):
    report = deepcopy(installed["report"])
    report["default_changed"] = original_passed
    result = operational_decision(report, installed["planning"], LEGACY)
    assert result["new_operational_default"] == PREFERRED
    assert result["original_gate"]["passed"] is original_passed
    assert result["selected_development_candidate"] == "hybrid-stops"
    assert "already inspected" in result["decision_basis"]
    assert result["incumbent_comparison"]["bootstrap_95_pp"][1] < 0


def test_migration_preserves_weights_reports_and_restart_default_without_fitting(installed, monkeypatch):
    store = installed["store"]
    source = store.root / "models/journey-phase5.joblib"
    report_file = store.root / "reports/phase5-validation.json"
    before = [digest(source), digest(report_file), digest(store.root / "models/model-CW.joblib")]
    original_state = store.get_state("phase5_validation")

    def forbidden(*args, **kwargs):
        raise AssertionError("migration must not fit")

    monkeypatch.setattr(HurdleStopping, "fit", forbidden)
    new = refresh_operational_default(store)
    assert new["operational_default"] == PREFERRED
    pd.testing.assert_frame_equal(new["hurdle"].rides, installed["planning"]["hurdle"].rides)
    assert new["trained_at"] == installed["planning"]["trained_at"]
    path = store.root / "models/journey-phase5.1.joblib"
    migrated_hash = digest(path)
    assert load_planning(store)["operational_default"] == PREFERRED
    assert refresh_operational_default(store)["operational_default"] == PREFERRED
    assert digest(path) == migrated_hash
    assert before == [digest(source), digest(report_file), digest(store.root / "models/model-CW.joblib")]
    assert store.get_state("phase5_validation") == original_state
    assert store.get_state("phase5_1_release")["new_operational_default"] == PREFERRED


@pytest.mark.parametrize(
    "defect", ["wrong_labels", "wrong_predictions", "missing_calibration", "wrong_parameters"]
)
def test_mismatched_artifact_cannot_activate_two_stage_or_borrow_legacy_intervals(installed, defect):
    p = deepcopy(installed["planning"])
    if defect == "wrong_labels":
        p["calibrations"][PREFERRED] = p["calibrations"][LEGACY]
    elif defect == "wrong_predictions":
        p["calibrations"][PREFERRED].rows[0]["predicted_elapsed_s"] += 100
    elif defect == "missing_calibration":
        del p["calibrations"][PREFERRED]
    else:
        p["hurdle"].half_life = 365
    assert not verify_two_stage(p, installed["report"])["valid"]
    joblib.dump(p, installed["store"].root / "models/journey-phase5.joblib")
    planning = load_planning(installed["store"])
    assert planning["operational_default"] is None
    assert planning["operational_blocker"]
    with pytest.raises(ValueError, match="blocked"):
        plan_ride(
            installed["model"],
            installed["route"],
            installed["history"],
            installed["settings"],
            planning,
            departure="2025-07-01T12:00Z",
        )


def test_automatic_executes_hurdle_time_and_its_own_asymmetric_calibration(installed):
    p = refresh_operational_default(installed["store"])
    common = (installed["model"], installed["route"], installed["history"], installed["settings"], p)
    kwargs = {"target_hr": 140, "departure": "2025-07-01T12:00Z", "scenario": "observed"}
    automatic = plan_ride(*common, **kwargs)
    explicit = plan_ride(*common, **kwargs, journey_method=PREFERRED)
    legacy = plan_ride(*common, **kwargs, journey_method=LEGACY)
    assert automatic["journey_method"] == automatic["stopping_method"] == PREFERRED
    assert automatic["elapsed_s"] == pytest.approx(explicit["elapsed_s"])
    assert automatic["personal_break_s"] == pytest.approx(300)
    assert legacy["personal_break_s"] == pytest.approx(1800)
    assert automatic["elapsed_s"] != legacy["elapsed_s"]
    u = automatic["elapsed_uncertainty"]
    expected = p["calibrations"][PREFERRED].estimate(
        automatic["elapsed_s"], automatic["distance_m"] / 1000, kwargs["departure"]
    )
    wrong = p["calibrations"][LEGACY].estimate(
        automatic["elapsed_s"], automatic["distance_m"] / 1000, kwargs["departure"]
    )
    assert u["calibration_key"] == PREFERRED
    assert u["intervals_s"] == expected["intervals_s"] != wrong["intervals_s"]
    assert u["conservative_s"] == expected["conservative_s"]
    assert u["provisional"] and not u["probability_supported"]
    assert automatic["daylight"]["before_cutoff_probability"] is None
    assert any("nominal targets" in w for w in automatic["warnings"])
    assert (
        pd.Timestamp(automatic["daylight"]["sunset"]) - pd.Timestamp(automatic["daylight"]["cutoff"])
    ).total_seconds() == 1800


@pytest.mark.parametrize("mode,seconds", [("custom", 1200), ("minimal", 0)])
def test_custom_minimal_bypass_hurdle_and_total_corrections(installed, mode, seconds):
    p = refresh_operational_default(installed["store"])

    class MustNotRun:
        def predict(self, *args, **kwargs):
            raise AssertionError("historical stopping must be bypassed")

    p["hurdle"] = p["stopping"] = MustNotRun()
    result = plan_ride(
        installed["model"],
        installed["route"],
        installed["history"],
        installed["settings"],
        p,
        140,
        departure="2025-07-01T12:00Z",
        break_mode=mode,
        break_minutes=20,
    )
    assert result["personal_break_s"] == pytest.approx(seconds)
    assert result["statistical_correction_s"] == 0
    assert result["elapsed_s"] == pytest.approx(result["moving_s"] + result["stationary_s"] + seconds)


def test_explicit_advanced_method_and_manual_effort_remain_distinct(installed):
    p = refresh_operational_default(installed["store"])
    common = (installed["model"], installed["route"], installed["history"], installed["settings"], p)
    manual = plan_ride(
        *common, 140, departure="2025-07-01T12:00Z", journey_method="bias-corrected", manual_hr=True
    )
    fast = plan_ride(
        *common, 155, departure="2025-07-01T12:00Z", journey_method="bias-corrected", manual_hr=True
    )
    assert manual["journey_method"] == "bias-corrected"
    assert manual["elapsed_uncertainty"]["calibration_key"] == "bias-corrected"
    assert manual["hr_selection"] == "manual override"
    assert manual["elapsed_s"] != fast["elapsed_s"]
    assert p["operational_default"] == PREFERRED


def test_corrupt_or_stale_release_does_not_fall_back_to_legacy_automatic(installed):
    store = installed["store"]
    refresh_operational_default(store)
    store.set_state("phase5_stale", True)
    p = load_planning(store)
    assert p["operational_default"] is None and "stale" in p["operational_blocker"]
    store.set_state("phase5_stale", False)
    model = joblib.load(store.root / "models/model-CW.joblib")
    model.fitness_adjustment = 1.1
    model.save(store.root / "models/model-CW.joblib")
    p = load_planning(store)
    assert p["operational_default"] is None and "dependencies changed" in p["operational_blocker"]
    with pytest.raises(ValueError, match="riding model changed"):
        refresh_operational_default(store)


def test_two_stage_chronology_still_excludes_future_stops_and_errors(installed):
    p = refresh_operational_default(installed["store"])
    departure = "2025-07-01T12:00Z"
    frame = installed["route"]["segments"]
    before = p["hurdle"].predict(frame, 1000, departure)
    future = p["hurdle"].rides.iloc[-1].copy()
    future["date"] = pd.Timestamp(departure)
    future["stop_target_s"] = 999999
    p["hurdle"].rides = pd.concat([p["hurdle"].rides, pd.DataFrame([future])], ignore_index=True)
    assert p["hurdle"].predict(frame, 1000, departure)["expected_s"] == before["expected_s"]
    calibration = p["calibrations"][PREFERRED]
    original = calibration.estimate(5000, 50, departure)
    calibration.rows.append(
        {**calibration.rows[0], "id": "future", "date": departure, "actual_elapsed_s": 999999}
    )
    assert calibration.estimate(5000, 50, departure)["intervals_s"] == original["intervals_s"]


@pytest.mark.parametrize("defect", ["metadata", "stopping_history"])
def test_tampered_release_blocks_automatic_and_explicit_refresh_restores_verified_source(
    installed, defect
):
    store = installed["store"]
    p = refresh_operational_default(store)
    path = store.root / "models/journey-phase5.1.joblib"
    if defect == "metadata":
        p["recommended_method"] = LEGACY
    else:
        p["hurdle"].rides.loc[0, "stop_target_s"] += 10000
    joblib.dump(p, path)
    blocked = load_planning(store)
    assert blocked["operational_default"] is None and blocked["operational_blocker"]
    fixed = refresh_operational_default(store)
    assert fixed["operational_default"] == fixed["recommended_method"] == PREFERRED
    assert load_planning(store)["operational_default"] == PREFERRED
    pd.testing.assert_frame_equal(fixed["hurdle"].rides, installed["planning"]["hurdle"].rides)


def test_cli_uses_same_operational_default_and_historical_time(installed, monkeypatch, capsys):
    import sys
    from cycling import cli

    p = refresh_operational_default(installed["store"])
    gpx = installed["store"].root / "synthetic.gpx"
    gpx.write_text("synthetic input supplied by mocked route preparation")
    monkeypatch.setattr(cli.Settings, "load", lambda: installed["settings"])
    monkeypatch.setattr(cli, "prepare_route", lambda *args: installed["route"])
    monkeypatch.setattr(
        sys, "argv", ["cycling", "predict", str(gpx), "--departure", "2025-07-01T12:00Z", "--hr", "140"]
    )
    cli.main()
    output = json.loads(capsys.readouterr().out)
    direct = plan_ride(
        installed["model"],
        installed["route"],
        installed["history"],
        installed["settings"],
        p,
        140,
        departure="2025-07-01T12:00Z",
        manual_hr=True,
    )
    assert output["journey_method"] == PREFERRED
    assert output["elapsed_uncertainty"]["calibration_key"] == PREFERRED
    assert output["elapsed_s"] == pytest.approx(direct["elapsed_s"])
