from pathlib import Path

from streamlit.testing.v1 import AppTest

from cycling.importer import import_archive
from cycling.pipeline import prepare_route, process_history, train
from cycling.prediction import predict_route
from conftest import gpx_bytes, strava_zip


def test_app_starts_and_empty_workflow_pages(monkeypatch, tmp_path):
    monkeypatch.setenv("CYCLING_DATA_DIR", str(tmp_path / "app-data"))
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    for page in ("Route Prediction", "Results", "Model Performance", "Historical Data"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception


def test_results_render_map_charts_and_segment_details(monkeypatch, store, settings):
    monkeypatch.setenv("CYCLING_DATA_DIR", settings.data_dir)
    import_archive(strava_zip(), store, settings)
    process_history(store, settings)
    model = train(store, settings, validate=False)
    route = prepare_route(gpx_bytes(timed=False), "route.gpx", store, settings)
    result = predict_route(model, route["segments"], route["events"], target_hr=145)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30)
    app.session_state["route"] = route
    app.session_state["result"] = result
    app.run()
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception
    assert len(app.metric) == 6
    assert len(app.get("plotly_chart")) >= 3
    assert len(app.get("deck_gl_json_chart")) == 1
