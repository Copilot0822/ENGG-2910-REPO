"""Public regional extracts and persistent local SQLite/RTree road indexes.

Only public extract URLs leave the computer. Recorded coordinates are queried locally.
Indexes are built atomically, one per source, so an update affects only overlapping traces.
"""

from functools import lru_cache
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import requests

INDEX_VERSION = "roads-3"
REGIONS = ("ontario", "quebec", "new-brunswick", "prince-edward-island")


def download_region(root, region, progress=None):
    if region not in REGIONS:
        raise ValueError(
            f"Supported regions: {', '.join(REGIONS)}. Other extracts can be supplied locally."
        )
    directory = Path(root) / "osm" / "extracts"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{region}-latest.osm.pbf"
    url = f"https://download.geofabrik.de/north-america/canada/{region}-latest.osm.pbf"
    if target.exists():
        return target
    partial = target.with_suffix(".pbf.part")
    digest = hashlib.sha256()
    with requests.get(url, stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0))
        if total > 8_000_000_000:
            raise ValueError("Extract exceeds the 8 GB per-file download limit.")
        count = 0
        with partial.open("wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                count += len(chunk)
                if count > 8_000_000_000:
                    raise ValueError("Extract exceeds the download limit.")
                if progress:
                    progress(count / max(1, total), f"{region}: {count / 1e6:.0f}/{total / 1e6:.0f} MB")
        if total and count != total:
            raise ValueError("Incomplete OSM extract; rerun download.")
        partial.replace(target)
        target.with_suffix(".source.json").write_text(
            json.dumps(
                {
                    "url": url,
                    "downloaded_at": time.time(),
                    "bytes": count,
                    "last_modified": response.headers.get("Last-Modified"),
                    "sha256": digest.hexdigest(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return target


def source_paths(settings, root):
    configured = getattr(settings, "local_osm_paths", None) or []
    if settings.local_osm_path:
        configured = [*configured, settings.local_osm_path]
    paths = [Path(p).expanduser().resolve() for p in configured]
    if not paths:
        paths = sorted((Path(root) / "osm" / "extracts").glob("*.osm.pbf"))
    return list(dict.fromkeys(paths))


def _connect(path):
    return sqlite3.connect(path)


def build_index(source, target, progress=None):
    """Stream PBF highway ways and control nodes; never load a province into Python memory."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".building.sqlite")
    if temporary.exists():
        temporary.unlink()
    db = _connect(temporary)
    db.executescript("""
        PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
        CREATE TABLE ways(id INTEGER PRIMARY KEY, payload TEXT);
        CREATE VIRTUAL TABLE way_bounds USING rtree(id,minlon,maxlon,minlat,maxlat);
        CREATE TABLE controls(id INTEGER PRIMARY KEY,payload TEXT);
        CREATE VIRTUAL TABLE control_bounds USING rtree(id,minlon,maxlon,minlat,maxlat);
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
    """)
    ways, controls = [], []
    bounds = [180.0, -180.0, 90.0, -90.0]
    counts = [0, 0]

    def flush():
        for name, batch in (("ways", ways), ("controls", controls)):
            if not batch:
                continue
            singular = "way" if name == "ways" else "control"
            db.executemany(f"INSERT OR REPLACE INTO {name} VALUES (?,?)", [(v[0], v[1]) for v in batch])
            db.executemany(
                f"INSERT OR REPLACE INTO {singular}_bounds VALUES (?,?,?,?,?)",
                [(v[0], *v[2]) for v in batch],
            )
            batch.clear()
        db.commit()

    def add(element):
        from .osm import control_kind

        tags = element.get("tags", {})
        if element["type"] == "way" and tags.get("highway"):
            geometry = element.get("geometry", [])
            if len(geometry) < 2 or len(geometry) != len(element.get("nodes", [])):
                return
            lons, lats = [p["lon"] for p in geometry], [p["lat"] for p in geometry]
            box = [min(lons), max(lons), min(lats), max(lats)]
            ways.append((element["id"], json.dumps(element, separators=(",", ":")), box))
            bounds[:] = [
                min(bounds[0], box[0]),
                max(bounds[1], box[1]),
                min(bounds[2], box[2]),
                max(bounds[3], box[3]),
            ]
            counts[0] += 1
        elif element["type"] == "node" and control_kind(tags):
            box = [element["lon"], element["lon"], element["lat"], element["lat"]]
            controls.append((element["id"], json.dumps(element, separators=(",", ":")), box))
            counts[1] += 1
        if len(ways) + len(controls) >= 4000:
            flush()
            if progress:
                progress(0, f"Indexing {source.name}: {counts[0]:,} roads, {counts[1]:,} controls")

    try:
        if source.suffix.lower() == ".pbf":
            import osmium

            processor = (
                osmium.FileProcessor(str(source))
                .with_locations()
                .with_filter(osmium.filter.KeyFilter("highway", "railway"))
            )
            for item in processor:
                if item.is_way() and "highway" in item.tags:
                    if not all(n.location.valid() for n in item.nodes):
                        continue
                    add(
                        {
                            "type": "way",
                            "id": item.id,
                            "tags": dict(item.tags),
                            "nodes": [n.ref for n in item.nodes],
                            "geometry": [
                                {"lat": n.location.lat, "lon": n.location.lon} for n in item.nodes
                            ],
                        }
                    )
                elif item.is_node() and item.location.valid():
                    add(
                        {
                            "type": "node",
                            "id": item.id,
                            "tags": dict(item.tags),
                            "lat": item.location.lat,
                            "lon": item.location.lon,
                        }
                    )
        else:
            from .osm import read_local_osm

            elements = read_local_osm(source)["elements"]
            nodes = {v["id"]: v for v in elements if v["type"] == "node"}
            for element in elements:
                if element["type"] == "way" and not element.get("geometry"):
                    element["geometry"] = [
                        {"lat": nodes[n]["lat"], "lon": nodes[n]["lon"]}
                        for n in element["nodes"]
                        if n in nodes
                    ]
                add(element)
        flush()
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        metadata = {
            "version": INDEX_VERSION,
            "source": str(source),
            "size": source.stat().st_size,
            "mtime_ns": source.stat().st_mtime_ns,
            "sha256": digest,
            "bounds": bounds,
            "ways": counts[0],
            "controls": counts[1],
            "indexed_at": time.time(),
        }
        db.executemany(
            "INSERT INTO metadata VALUES (?,?)", [(k, json.dumps(v)) for k, v in metadata.items()]
        )
        db.commit()
        db.close()
        temporary.replace(target)
        return metadata
    except BaseException:
        db.close()
        raise


class NetworkIndex:
    def __init__(self, root, paths, progress=None):
        self.root = Path(root) / "osm"
        self.sources = []
        for source in paths:
            source = Path(source).resolve()
            if not source.is_file():
                raise ValueError(f"OSM extract does not exist: {source}")
            name = hashlib.sha256(str(source).encode()).hexdigest()[:20]
            index = self.root / "indexes" / f"{name}.sqlite"
            metadata = {}
            if index.exists():
                with closing(_connect(index)) as db:
                    metadata = {
                        k: json.loads(v) for k, v in db.execute("SELECT key,value FROM metadata")
                    }
            stat = source.stat()
            if (
                metadata.get("version") != INDEX_VERSION
                or metadata.get("size") != stat.st_size
                or metadata.get("mtime_ns") != stat.st_mtime_ns
            ):
                metadata = build_index(source, index, progress)
            self.sources.append((index, metadata))

    def signature(self, points=None):
        relevant = []
        for _, meta in self.sources:
            box = meta["bounds"]
            if points is None or (
                points.lon.max() >= box[0]
                and points.lon.min() <= box[1]
                and points.lat.max() >= box[2]
                and points.lat.min() <= box[3]
            ):
                relevant.append(meta["sha256"])
        return hashlib.sha256((INDEX_VERSION + ":" + ":".join(sorted(relevant))).encode()).hexdigest()

    @lru_cache(maxsize=128)
    def tile(self, a, b):
        # RTree is persisted. Buffered cells retain topology across cell boundaries.
        south, west = a * 0.05 - 0.003, b * 0.05 - 0.004
        north, east = (a + 1) * 0.05 + 0.003, (b + 1) * 0.05 + 0.004
        elements = {}
        for path, meta in self.sources:
            box = meta["bounds"]
            if east < box[0] or west > box[1] or north < box[2] or south > box[3]:
                continue
            with closing(_connect(path)) as db:
                for table, spatial in (("ways", "way_bounds"), ("controls", "control_bounds")):
                    rows = db.execute(
                        f"SELECT t.payload FROM {table} t JOIN {spatial} b ON t.id=b.id "
                        "WHERE b.minlon<=? AND b.maxlon>=? AND b.minlat<=? AND b.maxlat>=?",
                        (east, west, north, south),
                    )
                    for (payload,) in rows:
                        value = json.loads(payload)
                        elements[(value["type"], value["id"])] = value
        return list(elements.values())

    def data_for(self, points):
        tiles = set(
            (math.floor(lat / 0.05), math.floor(lon / 0.05)) for lat, lon in zip(points.lat, points.lon)
        )
        elements = {}
        for a, b in sorted(tiles):
            for value in self.tile(a, b):
                elements[(value["type"], value["id"])] = value
        return {"elements": list(elements.values())}

    def report(self):
        return [meta for _, meta in self.sources]
