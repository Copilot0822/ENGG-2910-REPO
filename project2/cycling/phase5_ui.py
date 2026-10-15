"""Paired accuracy and complete historical prediction inspector."""

import pandas as pd
import plotly.express as px
import streamlit as st

from .elapsed_models import LABELS


def phase5_performance(report, store):
    from .journey import load_planning
    from .phase5_validation import paired_difference

    st.title("Total ride-time accuracy")
    planning = load_planning(store)
    release = planning.get("release_decision", {})
    recommended = planning.get("operational_default", planning.get("recommended_method"))
    st.markdown(
        f"**Current operational default: {LABELS.get(recommended, recommended or 'Unavailable')}**"
    )
    st.caption(
        "Riding model: C+Weather. The operational correction uses already inspected evidence; it is not a newly preregistered or independent confirmation."
    )
    if planning.get("operational_blocker"):
        st.error(planning["operational_blocker"])
    if recommended:
        operational = []
        for label, key in [
            ("Reused benchmark (already inspected)", "reused"),
            ("Fresh complementary cohort (already inspected)", "fresh"),
        ]:
            metric = next((m for m in report[key]["comparisons"] if m["variant"] == recommended), None)
            if metric:
                operational.append(
                    {"Cohort": label, "Method": LABELS.get(recommended, recommended), **metric}
                )
        st.dataframe(
            pd.DataFrame(operational).reindex(
                columns=[
                    "Cohort",
                    "Method",
                    "rides",
                    "elapsed_mape",
                    "elapsed_mdape",
                    "elapsed_mspe",
                    "elapsed_p90_ape",
                    "underprediction_percentage",
                    "coverage_80",
                    "coverage_90",
                    "conservative_optimistic_errors",
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    cohort = st.selectbox(
        "Evaluation cohort", ["Fresh later comparisons", "Reused 32-ride benchmark", "All later rides"]
    )
    key = {
        "Fresh later comparisons": "fresh",
        "Reused 32-ride benchmark": "reused",
        "All later rides": "all_later",
    }[cohort]
    comparison = pd.DataFrame(report[key]["comparisons"])
    variants = comparison.variant.tolist()
    diagnostic = st.selectbox(
        "Elapsed diagnostic model",
        variants,
        index=variants.index(recommended) if recommended in variants else 0,
        format_func=lambda value: LABELS.get(value, value),
    )
    metric = comparison[comparison.variant.eq(diagnostic)].iloc[0]
    st.caption(f"Selected historical diagnostic: {LABELS.get(diagnostic, diagnostic)} · {cohort}.")
    st.metric("Average absolute total-time error", f"{metric.elapsed_mape:.1f}%")
    a, b, c = st.columns(3)
    a.metric("Median APE", f"{metric.elapsed_mdape:.1f}%")
    b.metric("Signed percentage bias", f"{metric.elapsed_mspe:+.1f}%")
    c.metric("P90 absolute percentage error", f"{metric.elapsed_p90_ape:.1f}%")
    a, b, c = st.columns(3)
    a.metric("Evaluated rides", int(metric.rides))
    b.metric("Underprediction rate", f"{metric.underprediction_percentage:.1f}%")
    c.metric(
        "Optimistic before-cutoff predictions",
        f"{int(metric.optimistic_daylight_errors)} / {int(metric.daylight_rides)}",
    )
    st.caption(
        "Historical per-ride percentages of actual complete elapsed duration; MAPE and P90 error are aggregate diagnostics, not route-specific uncertainty bounds. Negative bias means optimistic shorter rides."
    )
    st.warning(
        "The original 32 rides are a reused benchmark. The 63 fresh complementary rides were entirely below 100 km and have also been inspected and influenced this operational correction. They do not establish long-distance or prospective forecast accuracy. Long-distance and issued-forecast reliability remain provisional."
    )
    st.info(
        f"Earlier development candidate: {report['selected_method']}. Original gate against {release.get('reference_comparator', 'CW-auto')}: {release.get('original_gate', {}).get('outcome', 'passed' if report['default_changed'] else 'failed')}. Current operational default: {recommended or 'unavailable'}. The gate outcome and operational correction are separate decisions. Full-stop mode is a scenario; validation used observed traffic and automatic HR."
    )
    st.subheader("Old and new models · same rides")
    comparison["Model"] = comparison.variant.map(LABELS)
    columns = [
        "Model",
        "rides",
        "elapsed_mape",
        "elapsed_mdape",
        "elapsed_mspe",
        "elapsed_p90_ape",
        "elapsed_mae_minutes",
        "coverage_80",
        "coverage_90",
        "conservative_optimistic_errors",
    ]
    st.dataframe(comparison.reindex(columns=columns), hide_index=True, width="stretch")
    st.subheader("Error by distance")
    groups = pd.DataFrame(
        [{"Distance": band, **m} for band, m in report[key]["groups"][diagnostic].items()]
    )
    st.dataframe(
        groups.reindex(columns=["Distance", "rides", "elapsed_mape", "elapsed_mspe", "elapsed_p90_ape"]),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Each >150 km benchmark band contains only one ride; these categories do not establish generalization."
    )
    source = pd.DataFrame(report["rows"])
    old_ids = set(r["id"] for r in store.get_state("journey_validation", {}).get("rows", []))
    if key == "fresh":
        source = source[~source.id.isin(old_ids)]
    elif key == "reused":
        source = source[source.id.isin(old_ids)]
    difference = paired_difference(source.to_dict("records"), diagnostic)
    if difference.get("rides"):
        st.caption(
            f"{LABELS.get(diagnostic, diagnostic)} paired MAPE difference versus CW-auto in this cohort: {difference['mape_difference_pp']:+.2f} percentage points; 95% chronological block-bootstrap range {difference['bootstrap_95_pp']}. This is uncertainty in the historical comparison, not the selected model's route-specific finish interval."
        )
    if diagnostic == "bias-corrected":
        st.caption(
            "Bias correction remains an advanced alternative. Its fresh-cohort difference against CW-auto crossed zero; no parameters were retuned and it was not promoted by its lowest observed MAPE."
        )
    selected = source[source.variant.eq(diagnostic)].copy()
    selected["Actual elapsed (h)"] = selected.actual_elapsed_s / 3600
    selected["Predicted elapsed (h)"] = selected.predicted_elapsed_s / 3600
    for x, y, title in [
        ("Actual elapsed (h)", "Predicted elapsed (h)", "Actual versus predicted total duration"),
        ("distance_km", "signed_percentage_error", "Signed percentage error versus distance"),
        ("distance_km", "absolute_percentage_error", "Absolute percentage error versus distance"),
        (
            "actual_unattributed_s",
            "predicted_personal_s",
            "Predicted allowance versus actual unattributed stationary time",
        ),
    ]:
        st.plotly_chart(
            px.scatter(selected, x=x, y=y, hover_data=["date", "distance_km"], title=title),
            width="stretch",
        )
    st.plotly_chart(
        px.histogram(
            selected,
            x="signed_percentage_error",
            nbins=20,
            title="Overprediction and underprediction distribution",
        ),
        width="stretch",
    )
    stops = source[source.variant.isin(["CW+stops-auto", "two-stage"])]
    st.plotly_chart(
        px.scatter(
            stops,
            x="distance_km",
            y="predicted_personal_s",
            color="variant",
            title="Effect of replacing residual stops with two-stage stopping",
        ),
        width="stretch",
    )
    st.subheader("Daylight and interval reliability")
    st.dataframe(
        comparison.reindex(
            columns=[
                "Model",
                "underprediction_mape",
                "underprediction_median_percentage",
                "underprediction_max_percentage",
                "coverage_80",
                "coverage_80_rides",
                "coverage_90",
                "coverage_90_rides",
                "optimistic_daylight_errors",
                "conservative_optimistic_errors",
                "conservative_daylight_rides",
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Intervals and conservative duration use only earlier completed out-of-sample errors with asymmetric finite-order tails. Pooled errors regularize sparse distance support. Forecasts, manual HR, full stops and custom/minimal breaks use provisional reference intervals; no unsupported before-dark probability is shown."
    )
    selected = selected.sort_values("absolute_percentage_error", ascending=False)
    st.subheader("Sortable ride-by-ride predictions")
    st.dataframe(
        selected.reindex(
            columns=[
                "id",
                "date",
                "distance_km",
                "actual_elapsed_s",
                "predicted_elapsed_s",
                "signed_percentage_error",
                "absolute_percentage_error",
                "journey_method",
                "selected_candidate",
                "statistical_correction_s",
                "weather_coverage",
                "osm_coverage",
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download paired validation", selected.to_csv(index=False), "phase5-validation.csv", "text/csv"
    )
    st.subheader("Detailed prediction inspector")
    ride_id = st.selectbox(
        "Inspect historical ride",
        selected.id.tolist(),
        format_func=lambda value: (
            f"{value} · {selected[selected.id.eq(value)].distance_km.iloc[0]:.1f} km"
        ),
    )
    inspected = selected[selected.id.eq(ride_id)].iloc[0]
    fields = [
        "distance_km",
        "ascent_m",
        "surface_composition",
        "paved_fraction",
        "route_paved_fraction",
        "actual_measured_hr",
        "suggested_hr",
        "actual_elapsed_s",
        "actual_moving_s",
        "actual_stationary_s",
        "identified_traffic_s",
        "actual_unattributed_s",
        "predicted_cruising_s",
        "predicted_moving_slowdown_s",
        "predicted_moving_s",
        "largest_ledger_discrepancy",
        "predicted_traffic_s",
        "predicted_personal_s",
        "physical_s",
        "statistical_correction_s",
        "predicted_elapsed_s",
        "signed_percentage_error",
        "absolute_percentage_error",
        "stop_probability",
        "conditional_stop_s",
        "weather_coverage",
        "osm_coverage",
        "full_stop_increment_s",
        "prediction_training_last",
    ]
    st.dataframe(
        pd.DataFrame([{"Quantity": f, "Value": str(inspected.get(f, "unavailable"))} for f in fields]),
        hide_index=True,
        width="stretch",
    )
    detail = inspected.get("correction_details") or {}
    st.write(
        {
            "Method": inspected.get("selected_candidate") or diagnostic,
            "Correction": {k: v for k, v in detail.items() if k != "training_ids"},
        }
    )
    st.caption(
        "Stationary truth and traffic attribution are estimates; the remainder is unattributed and includes possible unmapped controls. Neither stop intent nor exact discretionary duration is independently observed."
    )
    st.subheader("Automatic heart-rate evidence")
    prior = store.get_state("journey_validation", {})
    if prior.get("hr_metrics"):
        st.write(prior["hr_metrics"])
        st.dataframe(pd.DataFrame(prior.get("hr_groups", [])), hide_index=True, width="stretch")
