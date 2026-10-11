import numpy as np
import pandas as pd

from cycling.hr import (
    recommend_hr,
    validate_hr,
    update_hr_state,
    mark_manual_hr,
    use_suggested_hr,
    measured_history,
)


def history(distances, heart_rates, ages=None):
    ages = ages or list(range(len(distances), 0, -1))
    return pd.DataFrame(
        {
            "id": list(map(str, range(len(distances)))),
            "distance_m": np.array(distances) * 1000,
            "date": [
                (pd.Timestamp("2026-01-01T00:00Z") - pd.Timedelta(days=a)).isoformat() for a in ages
            ],
            "measured_avg_hr": heart_rates,
            "hr_recommendation_value": heart_rates,
            "hr_reliable": True,
            "hr_moving_coverage": 1.0,
            "hr_source": "samples",
            "bike": "Road",
            "elevation_gain_m": np.array(distances) * 10,
        }
    )


def test_distance_suggestion_prioritizes_long_rides():
    rides = history([30] * 15 + [135, 145, 150, 155, 160, 170], [165] * 15 + [141] * 6)
    long = recommend_hr(rides, 150, "2026-01-01T12:00Z")
    short = recommend_hr(rides, 30, "2026-01-01T12:00Z")
    assert long["suggested_hr"] < 145
    assert short["suggested_hr"] > 160
    assert long["within_20_percent"] == 6 and long["distance_specific"]
    assert long["evidence"] == "moderate"


def test_distance_boundary_is_smooth():
    rides = history(
        [100, 120, 125, 130, 140, 150, 155, 160, 175, 180, 200],
        [150, 151, 149, 145, 144, 143, 141, 143, 140, 139, 139],
    )
    values = [recommend_hr(rides, d, "2026-01-01T12:00Z")["suggested_hr"] for d in [149, 150, 151]]
    assert max(values) - min(values) <= 1


def test_recency_has_its_own_effect():
    rides = history([100] * 12, [158] * 9 + [140] * 3, [120] * 9 + [1] * 3)
    recent = recommend_hr(rides, 100, "2026-01-01T12:00Z", half_life=30)
    slow = recommend_hr(rides, 100, "2026-01-01T12:00Z", half_life=365)
    assert recent["suggested_hr"] < slow["suggested_hr"] - 10


def test_imputed_and_poor_coverage_hr_do_not_count():
    rides = history([100] * 8, [145] * 4 + [220] * 4)
    rides["hr_imputed"] = [False] * 4 + [True] * 4
    rides.loc[0, "hr_moving_coverage"] = 0.1
    assert len(measured_history(rides)) == 3
    suggestion = recommend_hr(rides, 100, "2026-01-01T12:00Z")
    assert suggestion["suggested_hr"] == 145
    assert suggestion["evidence"] == "low"


def test_sparse_fallback_and_extrapolation():
    rides = history([25, 30], [140, 150])
    result = recommend_hr(rides, 200, "2026-01-01T12:00Z")
    assert result["method"] == "general measured-HR baseline"
    assert not result["distance_specific"] and result["evidence"] == "low"
    assert any("extrapolated" in w for w in result["warnings"])


def test_no_measured_hr_does_not_invent_target():
    rides = history([50, 100], [None, None])
    assert recommend_hr(rides, 100, "2026-01-01T12:00Z")["suggested_hr"] is None
    assert recommend_hr(rides.iloc[:0], 100, "2026-01-01T12:00Z")["suggested_hr"] is None


def test_hr_bound_and_manual_state_persist_on_route_change():
    rides = history([100] * 8, list(range(140, 156, 2)))
    suggestion = recommend_hr(rides, 100, "2026-01-01T12:00Z", hr_range=(142, 148))
    assert 142 <= suggestion["suggested_hr"] <= 148
    state = {}
    update_hr_state(state, "route-a", suggestion)
    assert state["target_hr"] == suggestion["suggested_hr"]
    state["target_hr"] = 135
    mark_manual_hr(state)
    update_hr_state(state, "route-a", {"suggested_hr": 150})
    update_hr_state(state, "route-b", {"suggested_hr": 155})
    assert state["target_hr"] == 135 and state["hr_recommendation"]["suggested_hr"] == 155
    use_suggested_hr(state)
    assert state["target_hr"] == 155 and not state["hr_manual"]


def test_chronological_recommendation_excludes_current_and_future():
    rides = history([100] * 12, [145] * 11 + [210])
    rows = validate_hr(rides)
    dates = rides.set_index("id").date.to_dict()
    for row in rows:
        assert row["id"] not in row["training_ids"]
        assert all(pd.Timestamp(dates[i]) < pd.Timestamp(row["date"]) for i in row["training_ids"])
    assert rows[-1]["suggested_hr"] == 145
