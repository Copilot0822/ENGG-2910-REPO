"""FIT, TCX and GPX readers. All internal units are SI; timestamps are UTC."""

from bisect import bisect_right
from io import BytesIO
import gzip

from defusedxml import ElementTree as ET
from fitparse import FitFile
import gpxpy
import numpy as np
import pandas as pd


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return np.nan


def _extensions(point):
    values = {}
    mapping = {
        "hr": "hr",
        "heartrate": "hr",
        "cad": "cadence",
        "cadence": "cadence",
        "power": "power",
        "watts": "power",
        "atemp": "temperature",
        "speed": "speed",
    }
    for extension in point.extensions:
        for node in extension.iter():
            tag = node.tag.split("}")[-1].lower()
            if tag in mapping:
                values[mapping[tag]] = _number(node.text)
    return values


def parse_gpx(data, max_points=1_000_000):
    # Validate first with defusedxml; gpxpy then interprets only a safe XML document.
    ET.fromstring(data)
    gpx = gpxpy.parse(data.decode("utf-8-sig"))
    rows, group = [], 0
    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                if len(rows) >= max_points:
                    raise ValueError("Recording exceeds the configured point limit.")
                rows.append(
                    {
                        "lat": p.latitude,
                        "lon": p.longitude,
                        "elevation": p.elevation,
                        "time": p.time,
                        "track_group": group,
                        **_extensions(p),
                    }
                )
            group += 1
    if not rows:
        for route in gpx.routes:
            for p in route.points:
                if len(rows) >= max_points:
                    raise ValueError("Recording exceeds the configured point limit.")
                rows.append(
                    {
                        "lat": p.latitude,
                        "lon": p.longitude,
                        "elevation": p.elevation,
                        "time": p.time,
                        "track_group": group,
                        **_extensions(p),
                    }
                )
            group += 1
    return pd.DataFrame(rows), {"sport": None, "name": gpx.name or "GPX route"}


def parse_tcx(data, max_points=1_000_000):
    root = ET.fromstring(data)
    rows, group = [], 0
    sport = None
    for activity in root.findall(".//{*}Activity"):
        sport = activity.attrib.get("Sport", sport)
    for track in root.findall(".//{*}Track"):
        for point in track.findall("{*}Trackpoint"):
            if len(rows) >= max_points:
                raise ValueError("Recording exceeds the configured point limit.")

            def value(path):
                node = point.find(path)
                return node.text if node is not None else None

            row = {
                "lat": _number(value("{*}Position/{*}LatitudeDegrees")),
                "lon": _number(value("{*}Position/{*}LongitudeDegrees")),
                "time": value("{*}Time"),
                "elevation": _number(value("{*}AltitudeMeters")),
                "distance": _number(value("{*}DistanceMeters")),
                "hr": _number(value("{*}HeartRateBpm/{*}Value")),
                "cadence": _number(value("{*}Cadence")),
                "track_group": group,
            }
            for node in point.iter():
                tag = node.tag.split("}")[-1].lower()
                if tag in ("speed", "watts", "temperature"):
                    row[{"watts": "power"}.get(tag, tag)] = _number(node.text)
            rows.append(row)
        group += 1
    return pd.DataFrame(rows), {"sport": sport, "name": "TCX activity"}


def parse_fit(data, max_points=1_000_000):
    fit = FitFile(BytesIO(data), check_crc=True)
    rows, timer_events, metadata = [], [], {}
    for message in fit.get_messages():
        fields = {f.name: f.value for f in message}
        if message.name == "record":
            if len(rows) >= max_points:
                raise ValueError("Recording exceeds the configured point limit.")
            factor = 180 / 2**31
            lat, lon = fields.get("position_lat"), fields.get("position_long")
            rows.append(
                {
                    "lat": lat * factor if lat is not None else np.nan,
                    "lon": lon * factor if lon is not None else np.nan,
                    "time": fields.get("timestamp"),
                    "elevation": fields.get("enhanced_altitude", fields.get("altitude")),
                    "speed": fields.get("enhanced_speed", fields.get("speed")),
                    "distance": fields.get("distance"),
                    "hr": fields.get("heart_rate"),
                    "cadence": fields.get("cadence"),
                    "power": fields.get("power"),
                    "temperature": fields.get("temperature"),
                    "track_group": 0,
                }
            )
        elif message.name == "session":
            metadata.update(
                {
                    "sport": fields.get("sport"),
                    "sub_sport": fields.get("sub_sport"),
                    "recorded_avg_hr": fields.get("avg_heart_rate"),
                    "timer_s": fields.get("total_timer_time"),
                    "recorded_elapsed_s": fields.get("total_elapsed_time"),
                    "recorded_start_utc": str(fields.get("start_time") or ""),
                    "recorded_end_utc": str(fields.get("timestamp") or ""),
                }
            )
        elif message.name == "event" and fields.get("event") == "timer" and fields.get("timestamp"):
            kind = str(fields.get("event_type", ""))
            if kind == "start" or kind.startswith("stop"):
                timer_events.append(
                    (pd.Timestamp(fields["timestamp"]).tz_localize("UTC"), kind == "start")
                )
    result = pd.DataFrame(rows)
    if len(result) and timer_events:
        timer_events.sort()
        stamps, states = zip(*timer_events)
        times = pd.to_datetime(result.time, utc=True)
        result["timer_active"] = [
            states[max(0, bisect_right(stamps, t) - 1)] if pd.notna(t) else True for t in times
        ]
        pauses = []
        for previous, current in zip(times.shift(), times):
            stopped = 0.0
            if pd.notna(previous) and pd.notna(current) and current > previous:
                cuts = [previous] + [t for t in stamps if previous < t < current] + [current]
                for a, b in zip(cuts[:-1], cuts[1:]):
                    if not states[max(0, bisect_right(stamps, a) - 1)]:
                        stopped += (b - a).total_seconds()
            pauses.append(stopped)
        result["timer_pause_s"] = pauses
    metadata["timer_event_count"] = len(timer_events)
    return result, metadata


def parse_recording(data, filename, settings):
    name = filename.lower()
    if name.endswith(".gz"):
        with gzip.GzipFile(fileobj=BytesIO(data)) as stream:
            data = stream.read(settings.max_entry_bytes + 1)
        if len(data) > settings.max_entry_bytes:
            raise ValueError("Compressed recording exceeds the decompressed size limit.")
        name = name[:-3]
    if len(data) > settings.max_entry_bytes:
        raise ValueError("Recording exceeds the size limit.")
    if name.endswith(".fit"):
        return parse_fit(data, settings.max_points)
    if name.endswith(".gpx"):
        return parse_gpx(data, settings.max_points)
    if name.endswith(".tcx"):
        return parse_tcx(data, settings.max_points)
    raise ValueError("Unsupported recording; use FIT, FIT.GZ, TCX or GPX.")
