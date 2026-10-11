"""Bounded, directed, topology-aware HMM matching of an unchanged GPS trace.

OSM node IDs define connectivity: geometric crossings never join bridge/tunnel layers.
Distance/heading emissions and shortest-path transitions are evaluated together using
Viterbi and a backward pass. Ambiguity is exposed rather than resolved by road class.
"""

from collections import defaultdict
import hashlib
import heapq
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

from .geometry import angle_difference, bearing

MATCH_VERSION = "directed-hmm-6"
DEFAULTS = {
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
    "edge_id": "",
    "map_confidence": 0.0,
    "map_distance_m": np.nan,
    "forward": True,
    "osm_tags": "{}",
    "control_kind": "",
    "control_id": "",
    "control_uncertain": False,
    "control_confidence": 0.0,
    "matched_lat": np.nan,
    "matched_lon": np.nan,
    "matched_geometry": "[]",
    "matched_distance_m": 0.0,
    "match_status": "unmatched",
}


class RoadGraph:
    def __init__(self, payload):
        self.nodes = {e["id"]: e for e in payload.get("elements", []) if e["type"] == "node"}
        roads = [
            e
            for e in payload.get("elements", [])
            if e["type"] == "way" and "highway" in e.get("tags", {})
        ]
        geometry = [
            p
            for r in roads
            for p in (
                r.get("geometry") or [self.nodes[n] for n in r.get("nodes", []) if n in self.nodes]
            )
        ]
        self.lat0 = float(np.mean([p["lat"] for p in geometry])) if geometry else 0.0
        self.lon0 = float(np.mean([p["lon"] for p in geometry])) if geometry else 0.0
        self.sx, self.sy = 111195 * np.cos(np.radians(self.lat0)), 111195.0
        self.edges, self.lines, self.adj = [], [], defaultdict(list)
        self.membership = defaultdict(set)
        self.roads = {}
        for road in roads:
            tags = road["tags"]
            if tags.get("bicycle") in ("no", "dismount"):
                continue
            if tags.get("highway") in ("motorway", "motorway_link", "construction", "proposed"):
                if tags.get("bicycle") not in ("yes", "designated", "permissive"):
                    continue
            ids = road.get("nodes", [])
            geom = road.get("geometry") or [self.nodes[n] for n in ids if n in self.nodes]
            if len(ids) != len(geom):
                continue
            self.roads[str(road["id"])] = road
            for node in ids:
                self.membership[node].add(str(road["id"]))
            oneway = tags.get("oneway:bicycle", tags.get("oneway", ""))
            if tags.get("cycleway", "").startswith("opposite"):
                oneway = "no"
            if tags.get("junction") == "roundabout" and not oneway:
                oneway = "yes"
            for offset, (a, b) in enumerate(zip(geom[:-1], geom[1:])):
                line = LineString([self.xy(a["lat"], a["lon"]), self.xy(b["lat"], b["lon"])])
                if line.length < 0.1:
                    continue
                line_id = len(self.lines)
                self.lines.append(line)
                heading = float(bearing(a["lat"], a["lon"], b["lat"], b["lon"]))
                for forward in (True, False):
                    if (oneway in ("yes", "1", "true") and not forward) or (oneway == "-1" and forward):
                        continue
                    u, v = (ids[offset], ids[offset + 1]) if forward else (ids[offset + 1], ids[offset])
                    edge = {
                        "way": str(road["id"]),
                        "offset": offset,
                        "line": line_id,
                        "u": u,
                        "v": v,
                        "length": line.length,
                        "forward": forward,
                        "heading": heading if forward else (heading + 180) % 360,
                    }
                    index = len(self.edges)
                    self.edges.append(edge)
                    self.adj[u].append((v, line.length, index))
        self.by_line = defaultdict(list)
        for i, edge in enumerate(self.edges):
            self.by_line[edge["line"]].append(i)
        self.tree = STRtree(self.lines)
        # Signals on short connected approaches to one junction share an encounter identity.
        signals = {
            n
            for n, value in self.nodes.items()
            if value.get("tags", {}).get("highway") == "traffic_signals"
        }
        parents = {n: n for n in signals}

        def root(n):
            while parents[n] != n:
                parents[n] = parents[parents[n]]
                n = parents[n]
            return n

        for signal in signals:
            queue, visited = [(signal, 0.0)], {signal}
            for node, distance in queue:
                for nxt, length, _ in self.adj.get(node, []):
                    if nxt not in visited and distance + length <= 35:
                        visited.add(nxt)
                        queue.append((nxt, distance + length))
                        if nxt in signals:
                            a, b = root(signal), root(nxt)
                            parents[max(a, b)] = min(a, b)
        self.control_clusters = {n: root(n) for n in signals}

    def xy(self, lat, lon):
        return ((lon - self.lon0) * self.sx, (lat - self.lat0) * self.sy)

    def candidates(self, lat, lon, heading, radius):
        point = Point(self.xy(lat, lon))
        candidates = []
        for line_id in self.tree.query(point, predicate="dwithin", distance=radius):
            line = self.lines[line_id]
            distance = line.distance(point)
            for edge_id in self.by_line[int(line_id)]:
                edge = self.edges[edge_id]
                diff = abs(float(angle_difference(heading, edge["heading"])))
                if diff > 100:
                    continue
                position = line.project(point)
                if not edge["forward"]:
                    position = line.length - position
                emission = 0.5 * (distance / 9.0) ** 2 + 0.5 * (diff / 35.0) ** 2
                candidates.append((emission, edge_id, position, distance))
        return sorted(candidates)[:6]

    def shortest(self, source, targets, limit):
        distances, parents = {source: 0.0}, {}
        queue, remaining = [(0.0, source)], set(targets)
        expanded = 0
        while queue and remaining and expanded < 2500:
            distance, node = heapq.heappop(queue)
            if distance != distances[node]:
                continue
            remaining.discard(node)
            expanded += 1
            for nxt, length, edge in self.adj.get(node, []):
                new = distance + length
                if new <= limit and new < distances.get(nxt, np.inf):
                    distances[nxt] = new
                    parents[nxt] = (node, edge)
                    heapq.heappush(queue, (new, nxt))
        result = {}
        for target in targets:
            if target not in distances:
                continue
            path, cursor = [], target
            while cursor != source:
                cursor, edge = parents[cursor]
                path.append(edge)
            result[target] = (distances[target], list(reversed(path)))
        return result


def cached_graph(payload, directory=None):
    if directory is None:
        return RoadGraph(payload)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (MATCH_VERSION + json.dumps(payload, sort_keys=True, separators=(",", ":"))).encode()
    ).hexdigest()
    path = directory / f"{digest}.joblib"
    if path.exists():
        return joblib.load(path)
    graph = RoadGraph(payload)
    temp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    joblib.dump(graph, temp, compress=1)
    temp.replace(path)
    return graph


def annotate_hmm(points, payload, radius=40.0, cache_dir=None):
    from .osm import classify_surface, control_applies, control_kind

    result = points.copy()
    for key, value in DEFAULTS.items():
        result[key] = value
    graph = cached_graph(payload, cache_dir)
    if not graph.edges or len(points) < 2:
        return result
    updates = {key: result[key].to_numpy(copy=True) for key in DEFAULTS}
    lat, lon = points.lat.to_numpy(), points.lon.to_numpy()
    distance = points.distance_m.to_numpy()
    heading = points.heading.to_numpy()
    # Anchor spacing bounds work without assuming dense or regularly timed recordings.
    for _, subset in points.groupby("group", sort=False):
        indices = subset.index.to_numpy()
        anchors = [int(indices[0])]
        for index in indices[1:-1]:
            last = anchors[-1]
            if distance[index] - distance[last] >= 35 or (
                distance[index] - distance[last] >= 12
                and abs(float(angle_difference(heading[index], heading[last]))) > 40
            ):
                anchors.append(int(index))
        if anchors[-1] != indices[-1]:
            anchors.append(int(indices[-1]))
        candidates = [graph.candidates(lat[i], lon[i], heading[i], radius) for i in anchors]
        # Unmatched anchors split independent chains; a missing extract never invents a transition.
        start = 0
        while start < len(anchors):
            if not candidates[start]:
                start += 1
                continue
            end = start + 1
            while end < len(anchors) and candidates[end]:
                end += 1
            chain = candidates[start:end]
            aa = anchors[start:end]
            scores = [np.array([c[0] for c in chain[0]])]
            transitions, paths, parents = [], [], []
            for step in range(1, len(chain)):
                previous, current = chain[step - 1], chain[step]
                ds = distance[aa[step]] - distance[aa[step - 1]]
                maxpath = max(180.0, ds * 2.5 + 80.0)
                if "dt_s" in points and points.dt_s.iloc[aa[step - 1] + 1 : aa[step] + 1].sum() > 0:
                    dt = points.dt_s.iloc[aa[step - 1] + 1 : aa[step] + 1].sum()
                    maxpath = min(maxpath, max(180.0, dt * 35 + 40))
                matrix = np.full((len(previous), len(current)), 10000.0)
                step_paths = {}
                targets = {graph.edges[c[1]]["u"] for c in current}
                for a, p in enumerate(previous):
                    pe = graph.edges[p[1]]
                    routes = graph.shortest(pe["v"], targets, maxpath)
                    for b, c in enumerate(current):
                        ce = graph.edges[c[1]]
                        path = None
                        if p[1] == c[1] and c[2] >= p[2] - 8:
                            length, path = max(0.0, c[2] - p[2]), [p[1]]
                        elif ce["u"] in routes:
                            middle, edges = routes[ce["u"]]
                            length = pe["length"] - p[2] + middle + c[2]
                            path = [p[1], *edges, c[1]]
                        if path is not None and length <= maxpath:
                            matrix[a, b] = abs(length - ds) / max(18.0, ds * 0.25)
                            if pe["u"] == ce["v"] and pe["v"] == ce["u"]:
                                matrix[a, b] += 4.0  # improbable reversal without a recorded U-turn
                            step_paths[a, b] = path
                total = scores[-1][:, None] + matrix
                parents.append(total.argmin(axis=0))
                scores.append(total.min(axis=0) + np.array([c[0] for c in current]))
                transitions.append(matrix)
                paths.append(step_paths)
            selected = [int(scores[-1].argmin())]
            for parent in reversed(parents):
                selected.append(int(parent[selected[-1]]))
            selected.reverse()
            backward = [None] * len(chain)
            backward[-1] = np.zeros(len(chain[-1]))
            for step in range(len(chain) - 2, -1, -1):
                emissions = np.array([c[0] for c in chain[step + 1]])
                backward[step] = (
                    transitions[step] + emissions[None, :] + backward[step + 1][None, :]
                ).min(axis=1)
            confidence = []
            for step, choice in enumerate(selected):
                c = chain[step][choice]
                way = graph.edges[c[1]]["way"]
                total = scores[step] + backward[step]
                alternatives = [
                    total[j]
                    for j, other in enumerate(chain[step])
                    if graph.edges[other[1]]["way"] != way
                ]
                margin = max(0.0, min(alternatives) - total[choice]) if alternatives else 8.0
                conf = np.exp(-0.5 * (c[3] / 18.0) ** 2) * (0.3 + 0.7 * (1 - np.exp(-margin / 3)))
                if step and transitions[step - 1][selected[step - 1], choice] >= 10000:
                    conf = 0.0
                if step + 1 < len(chain) and transitions[step][choice, selected[step + 1]] >= 10000:
                    conf = 0.0
                confidence.append(float(conf))
            for step in range(max(1, len(chain) - 1)):
                left = aa[step]
                right = aa[min(step + 1, len(aa) - 1)]
                if len(chain) == 1:
                    right = left
                choice = selected[step]
                edge_ids = [chain[step][choice][1]]
                conf = confidence[step]
                if step + 1 < len(chain):
                    edge_ids = paths[step].get((choice, selected[step + 1]), [])
                    conf = min(conf, confidence[step + 1])
                if not edge_ids:
                    continue
                line_ids = [graph.edges[e]["line"] for e in edge_ids]
                local_tree = STRtree([graph.lines[e] for e in line_ids])
                for index in range(left, right + 1):
                    point = Point(graph.xy(lat[index], lon[index]))
                    local = int(local_tree.nearest(point))
                    edge_id = edge_ids[local]
                    edge = graph.edges[edge_id]
                    line = graph.lines[edge["line"]]
                    offset = line.distance(point)
                    if offset > radius:
                        continue
                    quality = min(conf, float(np.exp(-0.5 * (offset / 18.0) ** 2)))
                    tags = graph.roads[edge["way"]]["tags"]
                    projected = line.interpolate(line.project(point))
                    updates["way_id"][index] = edge["way"]
                    updates["edge_id"][index] = f"{edge['way']}:{edge['offset']}:{int(edge['forward'])}"
                    updates["forward"][index] = edge["forward"]
                    updates["map_distance_m"][index] = offset
                    updates["map_confidence"][index] = quality
                    updates["osm_tags"][index] = json.dumps(tags, separators=(",", ":"))
                    updates["matched_lat"][index] = projected.y / graph.sy + graph.lat0
                    updates["matched_lon"][index] = projected.x / graph.sx + graph.lon0
                    updates["matched_geometry"][index] = json.dumps(
                        [[x / graph.sx + graph.lon0, y / graph.sy + graph.lat0] for x, y in line.coords]
                    )
                    updates["match_status"][index] = "matched" if quality >= 0.5 else "low_confidence"
                    if quality >= 0.5:
                        surface, source, surface_conf = classify_surface(tags)
                        for key, value in {
                            "surface": surface,
                            "surface_source": source,
                            "surface_confidence": surface_conf * quality,
                            "surface_raw": tags.get("surface", ""),
                            "smoothness": tags.get("smoothness", ""),
                            "tracktype": tags.get("tracktype", ""),
                            "highway": tags.get("highway", "unknown"),
                            "bicycle": tags.get("bicycle", ""),
                            "junction": tags.get("junction", ""),
                            "matched_distance_m": points.ds_m.iloc[index],
                        }.items():
                            updates[key][index] = value
                # Nodes on the selected transition path, including sparsely sampled crossings.
                traversed = {graph.edges[e]["v"]: graph.edges[e] for e in edge_ids}
                for edge_id in edge_ids:
                    edge = graph.edges[edge_id]
                    traversed.setdefault(edge["u"], edge)
                for node_id, edge in traversed.items():
                    node = graph.nodes.get(node_id)
                    if node is None or not control_kind(node.get("tags", {})):
                        continue
                    tag = node["tags"]
                    # Pedestrian-only signals do not control the cyclist's matched approach.
                    if tag.get("traffic_signals") == "pedestrian" or tag.get("bicycle") == "no":
                        continue
                    applies, _ = control_applies(
                        tag, edge["forward"], True, len(graph.membership[node_id]) > 1
                    )
                    # A shared, explicitly signalized road junction controls connected approaches.
                    if (
                        applies is None
                        and control_kind(tag) == "signal"
                        and not any(k in tag for k in ("direction", "traffic_signals:direction"))
                    ):
                        applies = True
                    nodepoint = Point(graph.xy(node["lat"], node["lon"]))
                    # Reject endpoints of the candidate edge not actually passed by the trace.
                    distances = np.hypot(
                        (lon[left : right + 1] - node["lon"]) * graph.sx,
                        (lat[left : right + 1] - node["lat"]) * graph.sy,
                    )
                    nearest = left + int(distances.argmin())
                    rawline = LineString(
                        [graph.xy(lat[left], lon[left]), graph.xy(lat[right], lon[right])]
                    )
                    startxy = np.asarray(graph.xy(lat[left], lon[left]))
                    endxy = np.asarray(graph.xy(lat[right], lon[right]))
                    vector = endxy - startxy
                    span = float(np.linalg.norm(vector))
                    along = float(
                        np.dot(np.asarray(nodepoint.coords[0]) - startxy, vector) / max(1e-9, span)
                    )
                    if along < -8 or along > span + 8:
                        continue
                    if rawline.distance(nodepoint) > radius or conf < 0.5:
                        continue
                    if applies is None:
                        updates["control_uncertain"][nearest] = True
                        continue
                    if not applies:
                        continue
                    cluster = graph.control_clusters.get(node_id, node_id)
                    identity = f"osm:{cluster}:{int(edge['forward'])}"
                    previous = np.flatnonzero(updates["control_id"] == identity)
                    if len(previous) and distance[nearest] - distance[previous[-1]] < 100:
                        continue
                    # Consolidate signals only on the same connected approach, within 30m.
                    existing = np.flatnonzero(updates["control_kind"] != "")
                    if any(
                        abs(distance[nearest] - distance[j]) < 30
                        and updates["control_id"][j].split(":")[1] == str(cluster)
                        and updates["control_kind"][j] == control_kind(tag)
                        for j in existing
                    ):
                        continue
                    updates["control_kind"][nearest] = control_kind(tag)
                    updates["control_id"][nearest] = identity
                    updates["control_confidence"][nearest] = conf
            start = end
    for key, values in updates.items():
        result[key] = values
    return result
