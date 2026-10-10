import gzip
import json

import requests

from cycling.osm import OSMProvider, read_local_osm
from test_geometry_osm import network_fixture, trace


def test_local_osm_xml_preserves_control_and_tags(tmp_path):
    xml = b"""<osm version="0.6"><node id="1" lat="43" lon="-80"/><node id="2" lat="43" lon="-79.99"><tag k="highway" v="stop"/><tag k="direction" v="backward"/></node><way id="10"><nd ref="1"/><nd ref="2"/><tag k="highway" v="residential"/><tag k="surface" v="asphalt"/></way></osm>"""
    for suffix, data in ((".osm", xml), (".osm.gz", gzip.compress(xml))):
        path = tmp_path / ("map" + suffix)
        path.write_bytes(data)
        payload = read_local_osm(path)
        way = next(e for e in payload["elements"] if e["type"] == "way")
        assert way["nodes"] == [1, 2]
        assert way["tags"]["surface"] == "asphalt"
        node = next(e for e in payload["elements"] if e["id"] == 2)
        assert node["tags"]["direction"] == "backward"


def test_valhalla_failure_retains_local_match(store, settings, tmp_path):
    points = trace([6] * 30, settings)
    path = tmp_path / "map.json"
    path.write_text(json.dumps(network_fixture(points)))
    settings.osm_provider = "local"
    settings.local_osm_path = str(path)
    settings.valhalla_url = "http://127.0.0.1:8002"

    class Unavailable:
        headers = {}

        def post(self, *args, **kwargs):
            raise requests.ConnectionError("offline test")

    provider = OSMProvider(store, settings, Unavailable())
    result = provider.annotate(points)
    assert result.surface.eq("paved").all()
    assert result.control_kind.eq("stop").sum() == 1
    assert result.lon.equals(points.lon)
    assert any("Valhalla failed" in warning for warning in provider.warnings)


def test_valhalla_confirmation_without_geometry_replacement(store, settings, tmp_path):
    points = trace([6] * 30, settings)
    path = tmp_path / "map.json"
    path.write_text(json.dumps(network_fixture(points)))
    settings.osm_provider, settings.local_osm_path = "local", str(path)
    settings.valhalla_url = "http://127.0.0.1:8002"

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "edges": [{"way_id": 99}],
                "matched_points": [
                    {"edge_index": 0, "distance_from_trace_point": 1, "type": "matched"}
                    for _ in range(30)
                ],
            }

    class LocalService:
        headers = {}

        def post(self, url, json, **kwargs):
            assert url.endswith("/trace_attributes")
            assert json["costing"] == "bicycle"
            return Response()

    result = OSMProvider(store, settings, LocalService()).annotate(points)
    assert result.map_confidence.ge(0.85).all()
    assert result.lon.equals(points.lon)
