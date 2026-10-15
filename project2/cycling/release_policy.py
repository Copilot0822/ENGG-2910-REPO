"""Separate historical gate facts from the explicitly authorised operational default."""

from copy import deepcopy
import hashlib
import json
import shutil

import joblib
import numpy as np
import pandas as pd

from .tail_calibration import TailCalibration
from .two_stage import HurdleStopping

POLICY_VERSION = "phase5.1-two-stage-v1"
LEGACY = "CW+stops-auto"
PREFERRED = "two-stage"
REFERENCE = "CW-auto"


def reference_gate(candidate, difference, incumbent, reference=REFERENCE):
    """A comparator gate records evidence; failure does not select another model."""
    bounds = difference.get("bootstrap_95_pp", [None, None])
    delta = difference.get("mape_difference_pp")
    passed = bool(
        candidate != reference
        and delta is not None
        and delta < -1
        and bounds[1] is not None
        and bounds[1] < 0
    )
    return {
        "current_deployed_incumbent": incumbent,
        "reference_comparator": reference,
        "selected_development_candidate": candidate,
        "passed": passed,
        "outcome": "passed" if passed else "failed",
        "difference": deepcopy(difference),
    }


def original_gate(report, incumbent):
    if report.get("reference_gate"):
        return deepcopy(report["reference_gate"])
    gate = reference_gate(report.get("selected_method"), report.get("fresh_difference", {}), incumbent)
    # Preserve the originally recorded decision, rather than retrospectively reselecting.
    gate["passed"] = bool(report.get("default_changed", False))
    gate["outcome"] = "passed" if gate["passed"] else "failed"
    gate["recorded_operational_default"] = report.get("recommended_method")
    return gate


def calibration_signature(rows):
    fields = ["id", "date", "distance_km", "actual_elapsed_s", "predicted_elapsed_s", "variant"]
    records = [{key: row.get(key) for key in fields} for row in rows]
    records.sort(key=lambda row: (str(row["date"]), str(row["id"])))
    return hashlib.sha256(json.dumps(records, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hurdle_signature(hurdle):
    fields = ["id", "date", "distance_m", "stop_target_s", "stop_label"]
    frame = hurdle.rides[fields].copy()
    frame["id"] = frame.id.astype(str)
    frame["date"] = pd.to_datetime(frame.date, utc=True, format="mixed").map(
        lambda value: value.isoformat()
    )
    return hashlib.sha256(
        frame.sort_values("id").to_json(orient="records", double_precision=15).encode()
    ).hexdigest()


def history_matches(hurdle, rides):
    reliable = rides.get("elapsed_reliable", pd.Series(False, index=rides.index)).fillna(False)
    expected = rides[reliable].copy()
    if "activity_start_utc" in expected:
        expected["date"] = expected.activity_start_utc.fillna(expected.date)
    raw = pd.to_numeric(expected.unattributed_stationary_s, errors="coerce").fillna(0)
    gaps = pd.to_numeric(expected.get("unobserved_s", 0), errors="coerce")
    expected["stop_target_s"] = np.maximum(0, raw - np.nan_to_num(gaps))
    expected["stop_label"] = ((expected.stop_target_s - 60) / 240).clip(0, 1)
    return hurdle_signature(
        HurdleStopping(expected, hurdle.half_life, hurdle.conditional)
    ) == hurdle_signature(hurdle)


def verify_two_stage(planning, report):
    """Check association and usable provisional tails; nominal coverage is not assumed."""
    errors = []
    hurdle = planning.get("hurdle")
    calibration = planning.get("calibrations", {}).get(PREFERRED)
    family = report.get("frozen_selection", {}).get("families", {}).get(PREFERRED, {})
    if report.get("status") != "complete" or not planning.get("phase5"):
        errors.append("Complete saved Phase 5 validation and planning are required.")
    if not isinstance(hurdle, HurdleStopping) or hurdle.rides.empty:
        errors.append("The saved two-stage stopping model is missing or empty.")
    elif hurdle.half_life != family.get("half_life") or hurdle.conditional != family.get("conditional"):
        errors.append("Stopping parameters do not match the frozen two-stage candidate.")
    if not isinstance(calibration, TailCalibration):
        errors.append(
            "The two-stage asymmetric calibration is missing; legacy calibration cannot substitute."
        )
    else:
        frame = pd.DataFrame(calibration.rows)
        required = {"id", "date", "distance_km", "actual_elapsed_s", "predicted_elapsed_s", "variant"}
        if len(frame) < 6 or not required.issubset(frame.columns):
            errors.append("Insufficient correctly labelled two-stage calibration records.")
        elif not frame.variant.eq(PREFERRED).all() or frame.duplicated(["id", "date"]).any():
            errors.append("Calibration is mixed, duplicated or labelled as another method.")
        elif (
            not np.isfinite(
                frame[["actual_elapsed_s", "predicted_elapsed_s", "distance_km"]].to_numpy(float)
            ).all()
            or not frame.actual_elapsed_s.gt(0).all()
            or not frame.predicted_elapsed_s.gt(0).all()
        ):
            errors.append("Invalid two-stage calibration durations.")
        else:
            expected = report.get("calibration_rows", {}).get(PREFERRED, [])
            if calibration_signature(calibration.rows) != calibration_signature(expected):
                errors.append("Calibration predictions differ from the saved two-stage evidence.")
            metric = next(
                (
                    m
                    for m in report.get("all_later", {}).get("comparisons", [])
                    if m["variant"] == PREFERRED
                ),
                {},
            )
            if calibration.validated_coverage_80 != metric.get(
                "coverage_80"
            ) or calibration.validated_coverage_90 != metric.get("coverage_90"):
                errors.append("Coverage metadata is not associated with the two-stage method.")
            dates = pd.to_datetime(frame.date, utc=True, format="mixed", errors="coerce")
            if dates.isna().any():
                errors.append("Invalid calibration dates.")
            else:
                probe = calibration.estimate(7200, 50, dates.max() + pd.Timedelta(days=1))
                bounds = probe["intervals_s"].get("90", [])
                if len(bounds) != 2 or not np.isfinite(bounds).all() or probe["conservative_s"] < 7200:
                    errors.append("No defensible finite conservative reference estimate.")
                below = metric.get("coverage_80", 0) < 0.8 or metric.get("coverage_90", 0) < 0.9
                if below and (not probe["provisional"] or probe["probability_supported"]):
                    errors.append("Subnominal interval coverage is incorrectly presented as supported.")
    return {
        "valid": not errors,
        "errors": errors,
        "calibration_key": PREFERRED,
        "calibration_samples": len(calibration.rows) if isinstance(calibration, TailCalibration) else 0,
    }


def operational_decision(report, planning, incumbent=LEGACY):
    """Phase 5.1 operational correction, using already inspected fixed predictions."""
    from .phase5_validation import paired_difference

    checks = verify_two_stage(planning, report)
    # Split by the saved benchmark membership, not a newly selected date/distance cutoff.
    # The report's rows cover 95 later rides; original benchmark IDs are supplied
    # by the immutable original journey-validation record at migration time.
    benchmark_ids = set(report.get("operational_benchmark_ids", report.get("benchmark_ids", [])))
    if not benchmark_ids:
        checks["errors"].append(
            "Saved benchmark membership is missing; fresh evidence cannot be inferred."
        )
        checks["valid"] = False
    fresh = [r for r in report.get("rows", []) if str(r["id"]) not in benchmark_ids]
    difference = paired_difference(fresh, PREFERRED, LEGACY)
    bounds = difference.get("bootstrap_95_pp", [None, None])
    evidence_ok = (
        difference.get("rides", 0) >= 6
        and difference.get("mape_difference_pp", 0) < 0
        and bounds[1] is not None
        and bounds[1] < 0
    )
    if not evidence_ok:
        checks["errors"].append(
            "Saved paired evidence does not verify improvement over the legacy stopping incumbent."
        )
        checks["valid"] = False
    return {
        "policy_version": POLICY_VERSION,
        "previous_operational_default": incumbent,
        "new_operational_default": PREFERRED if checks["valid"] else None,
        "reference_comparator": REFERENCE,
        "selected_development_candidate": report.get("selected_method"),
        "original_gate": original_gate(report, incumbent),
        "verification": checks,
        "incumbent_comparison": difference,
        "decision_basis": "Explicit Phase 5.1 operational correction using already inspected historical evidence; not a new preregistered or independently confirmed experiment.",
        "reason": "Replace the legacy stop allowance with the frozen two-stage model and its own asymmetric calibration. Failure of the hybrid gate against CW-auto does not select the legacy incumbent.",
        "already_inspected_cohorts": {
            "reused_benchmark": report.get("reused", {}).get("comparisons", [{}])[0].get("rides"),
            "fresh_complementary": report.get("fresh", {}).get("comparisons", [{}])[0].get("rides"),
        },
        "limitations": [
            "The fresh cohort is entirely below 100 km and has already influenced development.",
            "Two-stage underprediction is more frequent than with the old allowance; lower MAPE does not imply safer daylight decisions.",
            "Coverage below nominal, sparse long rides and changed forecast/HR/traffic/break scenarios keep confidence provisional; no unsupported before-dark probability.",
        ],
    }


def dependencies(store, report, source):
    rides = store.rides()
    columns = [
        c
        for c in [
            "id",
            "date",
            "activity_start_utc",
            "distance_m",
            "elapsed_reliable",
            "elapsed_truth_s",
            "unattributed_stationary_s",
            "unobserved_s",
            "hr_reliable",
            "hr_recommendation_value",
        ]
        if c in rides
    ]
    history = (
        rides.sort_values("id")[columns].to_json(orient="records", date_format="iso")
        if len(rides)
        else "[]"
    )
    model = store.root / "models/model-CW.joblib"
    return {
        "source_artifact": hashlib.sha256(source.read_bytes()).hexdigest(),
        "riding_model": hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None,
        "history": hashlib.sha256(history.encode()).hexdigest(),
        "calibration": calibration_signature(report.get("calibration_rows", {}).get(PREFERRED, [])),
        "selection": report.get("frozen_selection", {}).get("families", {}).get(PREFERRED),
    }


def archive_artifact(path):
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = path.parent / "archive" / f"{path.stem}-{digest}{path.suffix}"
        destination.parent.mkdir(exist_ok=True)
        if not destination.exists():
            shutil.copy2(path, destination)


def refresh_operational_default(store):
    """Create a separate versioned artifact; do not fit or mutate historical records."""
    source = store.root / "models/journey-phase5.joblib"
    if not source.exists():
        raise ValueError("No Phase 5 planning artifact to refresh.")
    if store.get_state("phase5_stale", False) or store.get_state("model_stale", False):
        raise ValueError(
            "Planning or riding artifacts are stale; the saved release evidence cannot activate Automatic."
        )
    original = joblib.load(source)
    report = deepcopy(store.get_state("phase5_validation", {}))
    report["operational_benchmark_ids"] = sorted(
        {str(r["id"]) for r in store.get_state("journey_validation", {}).get("rows", [])}
    )
    decision = operational_decision(report, original, original.get("recommended_method", LEGACY))
    if decision["verification"]["valid"] and not history_matches(original["hurdle"], store.rides()):
        decision["verification"]["valid"] = False
        decision["verification"]["errors"].append(
            "The saved two-stage history differs from current reliable history; refresh cannot silently refit it."
        )
        decision["new_operational_default"] = None
    model = store.root / "models/model-CW.joblib"
    if not model.exists() or getattr(joblib.load(model), "variant", None) != "CW":
        decision["verification"]["valid"] = False
        decision["verification"]["errors"].append(
            "The retained C+Weather riding artifact is missing or mismatched."
        )
        decision["new_operational_default"] = None
    if not decision["verification"]["valid"]:
        raise ValueError(
            "Operational correction blocked: " + " ".join(decision["verification"]["errors"])
        )
    decision["dependencies"] = dependencies(store, report, source)
    decision["hurdle_signature"] = hurdle_signature(original["hurdle"])
    decision["migrated_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    planning = dict(original)
    planning.update(
        operational_default=PREFERRED,
        recommended_method=PREFERRED,
        release_decision=decision,
        selection_reason=decision["reason"],
    )
    destination = store.root / "models/journey-phase5.1.joblib"
    if destination.exists():
        current = joblib.load(destination)
        previous_dependencies = current.get("release_decision", {}).get("dependencies", {})
        if (
            previous_dependencies.get("riding_model") != decision["dependencies"]["riding_model"]
            and previous_dependencies.get("source_artifact")
            == decision["dependencies"]["source_artifact"]
            and previous_dependencies.get("calibration") == decision["dependencies"]["calibration"]
        ):
            raise ValueError(
                "The riding model changed without new planning/calibration evidence; a default-only refresh cannot validate it."
            )
        if (
            current.get("release_decision", {}).get("dependencies") == decision["dependencies"]
            and current.get("release_decision", {}).get("policy_version") == POLICY_VERSION
            and current.get("release_decision", {}).get("hurdle_signature")
            == decision["hurdle_signature"]
            and current.get("operational_default")
            == current.get("recommended_method")
            == current.get("release_decision", {}).get("new_operational_default")
            == PREFERRED
            and verify_two_stage(current, report)["valid"]
            and hurdle_signature(current["hurdle"]) == decision["hurdle_signature"]
        ):
            return current
    archive_artifact(destination)
    temporary = destination.with_suffix(".tmp")
    joblib.dump(planning, temporary, compress=3)
    temporary.replace(destination)
    store.set_state("phase5_1_release", decision)
    (store.root / "reports/phase5.1-release.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False), encoding="utf-8"
    )
    return planning


def load_operational_planning(store):
    source = store.root / "models/journey-phase5.joblib"
    report = store.get_state("phase5_validation", {})
    destination = store.root / "models/journey-phase5.1.joblib"
    blocker = None
    if (
        destination.exists()
        and not store.get_state("phase5_stale", False)
        and not store.get_state("model_stale", False)
    ):
        current = joblib.load(destination)
        release = current.get("release_decision", {})
        if (
            source.exists()
            and release.get("policy_version") == POLICY_VERSION
            and release.get("dependencies") == dependencies(store, report, source)
            and current.get("operational_default")
            == current.get("recommended_method")
            == release.get("new_operational_default")
            == PREFERRED
        ):
            if verify_two_stage(current, report)["valid"]:
                if release.get("hurdle_signature") == hurdle_signature(current["hurdle"]):
                    return current
        blocker = "Saved operational artifact or its history/model/calibration dependencies changed; Automatic is unavailable until a verified planning refresh."
    if blocker:
        original = joblib.load(source) if source.exists() else {"stopping": None, "calibrations": {}}
        return {
            **original,
            "operational_default": None,
            "operational_blocker": blocker,
            "selection_reason": blocker,
        }
    try:
        return refresh_operational_default(store)
    except ValueError as exc:
        # Preserve explicit advanced choices, but never silently activate legacy Automatic.
        planning = joblib.load(source) if source.exists() else {"stopping": None, "calibrations": {}}
        return {
            **planning,
            "operational_default": None,
            "operational_blocker": str(exc),
            "selection_reason": str(exc),
        }
