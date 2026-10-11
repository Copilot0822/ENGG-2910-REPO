"""Cached OSM enrichment. Geometry is annotated, never silently rerouted."""

import gzip
import json
import math
from pathlib import Path
import time
import zlib

from defusedxml import ElementTree as ET
import requests


SURFACES = {
    "paved": {
        "asphalt",
        "concrete",
        "concrete:plates",
        "concrete:lanes",
        "paved",
        "paving_stones",
        "sett",
        "cobblestone",
        "metal",
        "wood",
    },
    "hardpack_gravel": {"compacted", "fine_gravel"},
    "loose_gravel": {"gravel", "pebblestone", "crushed_stone"},
    "rough_natural": {
        "dirt",
        "earth",
        "ground",
        "grass",
        "sand",
        "mud",
        "rock",
        "unpaved",
        "clay",
        "woodchips",
    },
}


def classify_surface(tags):
    raw = tags.get("surface", "").lower()
    for category, values in SURFACES.items():
        if raw in values:
            return category, "known", 1.0
    # Infer separately: highway type by itself is deliberately insufficient to assert pavement.
    track = tags.get("tracktype", "")
    inferred = {
        "grade1": "paved",
        "grade2": "hardpack_gravel",
        "grade3": "loose_gravel",
        "grade4": "rough_natural",
        "grade5": "rough_natural",
    }.get(track)
    return (inferred, "inferred", 0.45) if inferred else ("unknown", "unknown", 0.0)


def control_kind(tags):
    highway = tags.get("highway", "")
    if highway in ("stop", "traffic_signals", "give_way"):
        return {"traffic_signals": "signal", "give_way": "yield"}.get(highway, highway)
    if tags.get("railway") in ("level_crossing", "crossing"):
        return "railway"
    return ""


def control_applies(tags, forward, member=True, junction_shared=False):
    if not member:
        return False, "other road approach"
    kind = control_kind(tags)
    prefix = {"stop": "stop", "signal": "traffic_signals"}.get(kind, "")
    direction = tags.get(f"{prefix}:direction", tags.get("direction", ""))
    if direction in ("forward", "backward"):
        return (forward if direction == "forward" else not forward), "directional"
    if direction == "both" or tags.get("stop") == "all" or kind == "railway":
        return True, "both"
    if direction:
        return None, "unsupported directional tag"
    if junction_shared:
        return None, "shared junction has unspecified approach"
    return True, "road approach"


def read_local_osm(path):
    """Small XML/JSON/PBF fixtures; the bulk provider streams PBF through NetworkIndex."""
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix.lower() == ".pbf":
        try:
            import osmium
        except ImportError as exc:
            raise ValueError("PBF support requires: uv sync --extra pbf") from exc
        elements = []

        class Handler(osmium.SimpleHandler):
            def node(self, n):
                if n.location.valid() and len(n.tags):
                    elements.append(
                        {
                            "type": "node",
                            "id": n.id,
                            "lat": n.location.lat,
                            "lon": n.location.lon,
                            "tags": dict(n.tags),
                        }
                    )

            def way(self, w):
                if "highway" not in w.tags:
                    return
                nodes, geometry = [], []
                for n in w.nodes:
                    if n.location.valid():
                        nodes.append(n.ref)
                        geometry.append({"lat": n.location.lat, "lon": n.location.lon})
                elements.append(
                    {
                        "type": "way",
                        "id": w.id,
                        "nodes": nodes,
                        "geometry": geometry,
                        "tags": dict(w.tags),
                    }
                )

        Handler().apply_file(str(path), locations=True, idx="flex_mem")
        return {"elements": elements}
    elements = []
    stream = gzip.open(path, "rb") if path.suffix.lower() == ".gz" else path.open("rb")
    with stream:
        for _, node in ET.iterparse(stream, events=("end",)):
            if node.tag == "node":
                elements.append(
                    {
                        "type": "node",
                        "id": int(node.attrib["id"]),
                        "lat": float(node.attrib["lat"]),
                        "lon": float(node.attrib["lon"]),
                        "tags": {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")},
                    }
                )
                node.clear()
            elif node.tag == "way":
                tags = {t.attrib["k"]: t.attrib["v"] for t in node.findall("tag")}
                if "highway" in tags:
                    elements.append(
                        {
                            "type": "way",
                            "id": int(node.attrib["id"]),
                            "nodes": [int(n.attrib["ref"]) for n in node.findall("nd")],
                            "tags": tags,
                        }
                    )
                node.clear()
    return {"elements": elements}


class OSMProvider:
    def __init__(self, store, settings, session=None):
        self.store, self.settings = store, settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "PersonalCyclingPredictor/0.1 (local research app)"})
        self.requested = 0
        self.last_request = 0.0
        self.warnings = set()
        self.local = None
        self.loaded_tiles = {}
        self.unavailable = set()
        self.network = None

    def _tile(self, a, b):
        size = self.settings.osm_tile_degrees
        key = f"{self.settings.overpass_url}:{size}:{a}:{b}"
        if key in self.loaded_tiles:
            return self.loaded_tiles[key]
        if key in self.unavailable:
            return None
        with self.store.connect() as db:
            row = db.execute("SELECT fetched_at,payload FROM osm_tiles WHERE key=?", (key,)).fetchone()
        cached = json.loads(zlib.decompress(row[1])) if row else None
        if row and (
            self.settings.osm_provider == "offline"
            or time.time() - row[0] < self.settings.osm_cache_days * 86400
        ):
            self.loaded_tiles[key] = cached
            return cached
        if self.settings.osm_provider != "overpass" or self.requested >= self.settings.osm_max_tiles:
            self.unavailable.add(key)
            if self.settings.osm_provider == "overpass":
                self.warnings.add(
                    "OSM tile budget reached; uncovered roads remain unknown. Process another batch or use a local extract."
                )
            return cached
        south, west = a * size, b * size
        box = f"{south:.6f},{west:.6f},{south + size:.6f},{west + size:.6f}"
        query = f'[out:json][timeout:25];(way["highway"]({box});node["highway"~"^(stop|traffic_signals|give_way)$"]({box});node["railway"="level_crossing"]({box}););out body geom;>;out skel qt;'
        self.requested += 1
        for attempt in range(3):
            wait = max(0, self.settings.osm_min_interval_s - (time.monotonic() - self.last_request))
            if wait:
                time.sleep(wait)
            self.last_request = time.monotonic()
            try:
                response = self.session.post(
                    self.settings.overpass_url, data={"data": query}, timeout=(5, 35)
                )
                response.raise_for_status()
                payload = response.json()
                if "elements" not in payload or "remark" in payload:
                    raise ValueError(payload.get("remark", "Malformed OSM response"))
                with self.store.connect() as db:
                    db.execute(
                        "INSERT OR REPLACE INTO osm_tiles VALUES (?,?,?)",
                        (key, time.time(), zlib.compress(json.dumps(payload).encode())),
                    )
                self.loaded_tiles[key] = payload
                return payload
            except (requests.RequestException, ValueError) as exc:
                if attempt < 2:
                    time.sleep(2**attempt)
                else:
                    self.warnings.add(
                        f"OSM unavailable ({type(exc).__name__}); cached roads or unknown surfaces used."
                    )
        self.unavailable.add(key)
        return cached

    def data_for(self, points):
        if self.settings.osm_provider == "local":
            if self.network is None:
                try:
                    from .network import NetworkIndex, source_paths

                    self.network = NetworkIndex(
                        self.store.root, source_paths(self.settings, self.store.root)
                    )
                except (OSError, ValueError, ET.ParseError) as exc:
                    self.warnings.add(f"Local OSM load failed: {exc}")
                    return {"elements": []}
            return self.network.data_for(points)
        size = self.settings.osm_tile_degrees
        # Add immediate neighboring cells near borders, without downloading a whole ride bounding box.
        tiles = set()
        sampled = points.iloc[:: max(1, len(points) // 4000)]
        for lat, lon in zip(sampled.lat, sampled.lon):
            for dy, dx in ((0, 0), (0.0004, 0), (-0.0004, 0), (0, 0.0004), (0, -0.0004)):
                tiles.add((math.floor((lat + dy) / size), math.floor((lon + dx) / size)))
        elements = {}
        for a, b in sorted(tiles):
            payload = self._tile(a, b)
            if payload:
                for element in payload["elements"]:
                    key = (element["type"], element["id"])
                    previous = elements.get(key, {})
                    elements[key] = {
                        **previous,
                        **element,
                        "tags": {**previous.get("tags", {}), **element.get("tags", {})},
                    }
        return {"elements": list(elements.values())}

    def annotate(self, points):
        payload = self.data_for(points)
        from .matching import annotate_hmm

        result = annotate_hmm(
            points, payload, self.settings.map_match_radius_m, self.store.root / "osm" / "graphs"
        )
        if self.settings.valhalla_url:
            from urllib.parse import urlparse

            if urlparse(self.settings.valhalla_url).hostname not in ("localhost", "127.0.0.1", "::1"):
                self.warnings.add("Remote trace matching disabled: use a local Valhalla endpoint.")
            else:
                result = self._valhalla(result, payload)
        if not result.map_confidence.gt(0).any():
            self.warnings.add(
                "No OSM roads matched. Geometry retained; surfaces and traffic controls remain unknown."
            )
        return result

    def signature(self, points=None):
        from .matching import MATCH_VERSION

        if self.settings.osm_provider == "local":
            if self.network is None:
                from .network import NetworkIndex, source_paths

                try:
                    self.network = NetworkIndex(
                        self.store.root, source_paths(self.settings, self.store.root)
                    )
                except (OSError, ValueError) as exc:
                    self.warnings.add(f"Local OSM unavailable: {exc}")
                    return f"{MATCH_VERSION}:missing:{self.settings.local_osm_path}"
            source = self.network.signature(points)
        else:
            with self.store.connect() as db:
                revisions = db.execute("SELECT key,fetched_at FROM osm_tiles ORDER BY key").fetchall()
            source = f"{self.settings.osm_provider}:{self.settings.overpass_url}:{self.settings.osm_cache_days}:{revisions}"
        return f"{MATCH_VERSION}:{self.settings.map_match_radius_m}:{source}"

    def _valhalla(self, points, payload):
        # Request annotations, not a replacement route. Default configuration never sends traces externally.
        ways = {str(e["id"]): e for e in payload["elements"] if e["type"] == "way"}
        result = points.copy()
        for start in range(0, len(result), 499):
            subset = result.iloc[start : start + 500]
            if len(subset) < 2:
                continue
            body = {
                "shape": [{"lat": float(r.lat), "lon": float(r.lon)} for r in subset.itertuples()],
                "costing": "bicycle",
                "shape_match": "map_snap",
                "trace_options": {"search_radius": 40},
                "filters": {
                    "action": "include",
                    "attributes": [
                        "edge.way_id",
                        "matched.point",
                        "matched.type",
                        "matched.edge_index",
                        "matched.distance_from_trace_point",
                    ],
                },
            }
            try:
                response = self.session.post(
                    self.settings.valhalla_url.rstrip("/") + "/trace_attributes",
                    json=body,
                    timeout=(5, 35),
                )
                response.raise_for_status()
                data = response.json()
                edges = data.get("edges", [])
                for offset, match in enumerate(data.get("matched_points", [])):
                    if offset >= len(subset) or match.get("type") == "unmatched":
                        continue
                    edge_index = match.get("edge_index")
                    if edge_index is None or edge_index >= len(edges):
                        continue
                    way = ways.get(str(edges[edge_index].get("way_id")))
                    if (
                        way
                        and match.get("distance_from_trace_point", 999)
                        <= self.settings.map_match_radius_m
                    ):
                        ix = subset.index[offset]
                        # Local matcher retains directional control decisions; override only compatible way IDs.
                        if str(result.loc[ix, "way_id"]) == str(way["id"]):
                            result.loc[ix, "map_confidence"] = max(
                                0.85, result.loc[ix, "map_confidence"]
                            )
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                self.warnings.add(
                    f"Valhalla failed ({type(exc).__name__}); directional local matching used."
                )
                break
        return result


def annotate_network(points, payload, radius=40.0):
    """Local directed HMM matching, with unchanged recorded GPS geometry."""
    from .matching import annotate_hmm

    return annotate_hmm(points, payload, radius)
