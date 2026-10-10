"""Cached OSM enrichment. Geometry is annotated, never silently rerouted."""

from collections import defaultdict
import gzip
import json
import math
from pathlib import Path
import time
import zlib

from defusedxml import ElementTree as ET
import numpy as np
import requests
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .geometry import angle_difference, bearing, haversine

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
    """OSM XML/Overpass JSON; optional pyosmium reads local PBF without Docker."""
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
            if self.local is None:
                try:
                    self.local = read_local_osm(self.settings.local_osm_path)
                except (OSError, ValueError, ET.ParseError) as exc:
                    self.warnings.add(f"Local OSM load failed: {exc}")
                    self.local = {"elements": []}
            return self.local
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
        result = annotate_network(points, payload, self.settings.map_match_radius_m)
        if self.settings.valhalla_url:
            result = self._valhalla(result, payload)
        if not result.map_confidence.gt(0).any():
            self.warnings.add(
                "No OSM roads matched. Geometry retained; surfaces and traffic controls remain unknown."
            )
        return result

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
    result = points.copy()
    defaults = {
        "surface": "unknown",
        "surface_source": "unknown",
        "surface_confidence": 0.0,
        "surface_raw": "",
        "smoothness": "",
        "tracktype": "",
        "highway": "unknown",
        "bicycle": "",
        "junction": "",
        "way_id": "",
        "map_confidence": 0.0,
        "map_distance_m": np.nan,
        "forward": True,
        "osm_tags": "{}",
        "control_kind": "",
        "control_id": "",
        "control_uncertain": False,
    }
    for key, value in defaults.items():
        result[key] = value
    elements = payload.get("elements", [])
    nodes = {e["id"]: e for e in elements if e["type"] == "node"}
    roads = [e for e in elements if e["type"] == "way" and "highway" in e.get("tags", {})]
    if not roads:
        return result
    lat0, lon0 = float(points.lat.mean()), float(points.lon.mean())
    scale_x, scale_y = 111195 * np.cos(np.radians(lat0)), 111195.0

    def xy(lat, lon):
        return ((lon - lon0) * scale_x, (lat - lat0) * scale_y)

    lines, properties = [], []
    membership = defaultdict(set)
    for road in roads:
        ids = road.get("nodes", [])
        geometry = road.get("geometry") or [nodes[n] for n in ids if n in nodes]
        for n in ids:
            membership[n].add(str(road["id"]))
        for a, b in zip(geometry[:-1], geometry[1:]):
            line = LineString([xy(a["lat"], a["lon"]), xy(b["lat"], b["lon"])])
            if line.length < 0.1:
                continue
            lines.append(line)
            properties.append((road, float(bearing(a["lat"], a["lon"], b["lat"], b["lon"]))))
    if not lines:
        return result
    tree = STRtree(lines)
    previous_way = None
    for row in result.itertuples():
        point = Point(xy(row.lat, row.lon))
        candidates = tree.query(point.buffer(radius))
        if not len(candidates):
            previous_way = None
            continue
        scored = []
        for candidate in candidates:
            road, heading = properties[candidate]
            distance = lines[candidate].distance(point)
            diff = abs(float(angle_difference(row.heading, heading)))
            forward = diff <= 90
            parallel_diff = min(diff, 180 - diff)
            score = (
                distance
                + parallel_diff * 0.18
                + (3 if previous_way and str(road["id"]) != previous_way else 0)
            )
            oneway = road["tags"].get("oneway:bicycle", road["tags"].get("oneway", ""))
            if oneway in ("yes", "1") and not forward or oneway == "-1" and forward:
                score += 18
            scored.append((score, candidate, distance, forward, parallel_diff))
        scored.sort(key=lambda v: v[0])
        score, selected, distance, forward, diff = scored[0]
        road, _ = properties[selected]
        if distance > radius or diff > 70:
            continue
        tags = road["tags"]
        surface, source, confidence = classify_surface(tags)
        map_confidence = max(0.05, 1 - distance / radius) * max(0.2, 1 - diff / 90)
        if (
            len(scored) > 1
            and scored[1][0] - score < 3
            and properties[scored[1][1]][0]["id"] != road["id"]
        ):
            map_confidence *= 0.6
        ix = row.Index
        updates = {
            "surface": surface,
            "surface_source": source,
            "surface_confidence": confidence,
            "surface_raw": tags.get("surface", ""),
            "highway": tags.get("highway", "unknown"),
            "smoothness": tags.get("smoothness", ""),
            "tracktype": tags.get("tracktype", ""),
            "bicycle": tags.get("bicycle", ""),
            "junction": tags.get("junction", ""),
            "way_id": str(road["id"]),
            "map_confidence": map_confidence,
            "map_distance_m": distance,
            "forward": forward,
            "osm_tags": json.dumps(tags),
        }
        for key, value in updates.items():
            result.at[ix, key] = value
        previous_way = str(road["id"])
    # Only controls on the actual matched way can affect it. Spatial proximity alone is insufficient.
    controls = [n for n in nodes.values() if control_kind(n.get("tags", {}))]
    seen = {}
    for node in controls:
        member_ways = membership[node["id"]]
        eligible = result.way_id.isin(member_ways) & result.map_confidence.ge(0.25)
        if not eligible.any():
            continue
        distances = haversine(result.lat, result.lon, node["lat"], node["lon"])
        indices = np.flatnonzero(eligible.to_numpy() & (distances < 22))
        # Each traversal gets one encounter, even if repeated points sit beside the same sign.
        traversals = []
        for ix in indices:
            if (
                not traversals
                or result.distance_m.iloc[ix] - result.distance_m.iloc[traversals[-1][-1]] > 90
            ):
                traversals.append([ix])
            else:
                traversals[-1].append(ix)
        for group in traversals:
            ix = min(group, key=lambda i: distances[i])
            applies, reason = control_applies(
                node.get("tags", {}), bool(result.forward.iloc[ix]), True, len(member_ways) > 1
            )
            if applies is False:
                continue
            if applies is None:
                result.at[ix, "control_uncertain"] = True
                continue
            key = (node["id"], bool(result.forward.iloc[ix]))
            if key in seen and result.distance_m.iloc[ix] - seen[key] < 90:
                continue
            # Multiple co-located signal/stop nodes are consolidated to avoid charging twice.
            nearby = result.control_kind.ne("") & (
                abs(result.distance_m - result.distance_m.iloc[ix]) < 35
            )
            if nearby.any():
                continue
            result.at[ix, "control_kind"] = control_kind(node["tags"])
            result.at[ix, "control_id"] = f"{node['id']}:{'f' if result.forward.iloc[ix] else 'b'}"
            seen[key] = result.distance_m.iloc[ix]
    return result
