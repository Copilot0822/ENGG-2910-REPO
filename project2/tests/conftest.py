"""Synthetic fixtures only; never include personal recordings in the repository."""

from datetime import datetime, timedelta, timezone
from io import BytesIO, StringIO
import csv
import struct
import zipfile

from fitparse.records import Crc
import numpy as np
import pandas as pd
import pytest

from cycling.config import Settings
from cycling.events import route_events
from cycling.features import build_segments
from cycling.geometry import clean_points
from cycling.osm import annotate_network
from cycling.storage import Store


def gpx_bytes(count=180, hr=145, timed=True, elevation=True, route=False):
    start = datetime(2025, 6, 1, tzinfo=timezone.utc)
    points = []
    tag = "rtept" if route else "trkpt"
    for i in range(count):
        children = f"<ele>{100 + 4 * np.sin(i / 20):.3f}</ele>" if elevation else ""
        if timed:
            children += f"<time>{(start + timedelta(seconds=i * 5)).isoformat()}</time>"
        if hr:
            children += f"<extensions><hr>{hr}</hr><cad>80</cad></extensions>"
        points.append(f'<{tag} lat="43.0" lon="{-80 + i * 0.00035:.7f}">{children}</{tag}>')
    body = (
        "<rte>" + "".join(points) + "</rte>"
        if route
        else "<trk><trkseg>" + "".join(points) + "</trkseg></trk>"
    )
    return (
        '<?xml version="1.0"?><gpx version="1.1" creator="synthetic" xmlns="http://www.topografix.com/GPX/1/1">'
        + body
        + "</gpx>"
    ).encode()


def strava_zip(recording=None, filename="activities/999.gpx", ride_id="1", sport="Ride", hr=145):
    buffer = BytesIO()
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(
        [
            "Activity ID",
            "Activity Date",
            "Activity Name",
            "Activity Type",
            "Distance",
            "Filename",
            "Average Heart Rate",
            "Activity Gear",
        ]
    )
    writer.writerow(
        [
            ride_id,
            "Jun 1, 2025, 12:00:00 AM",
            "Synthetic ride",
            sport,
            "5.1",
            filename,
            hr or "",
            "Test bike",
        ]
    )
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("nested/activities.csv", csv_buffer.getvalue())
        archive.writestr(filename, recording if recording is not None else gpx_bytes(hr=hr))
    return buffer.getvalue()


def fit_bytes(count=30):
    # Minimal valid FIT file with a record definition, actual SI-scaled samples and CRCs.
    fields = [
        (253, 4, 0x86),
        (0, 4, 0x85),
        (1, 4, 0x85),
        (2, 2, 0x84),
        (3, 1, 0x02),
        (4, 1, 0x02),
        (5, 4, 0x86),
        (6, 2, 0x84),
        (7, 2, 0x84),
    ]
    data = bytes([0x40, 0, 0]) + struct.pack("<H", 20) + bytes([len(fields)])
    data += b"".join(bytes(f) for f in fields)
    seconds = int(
        (
            datetime(2025, 1, 1, tzinfo=timezone.utc) - datetime(1989, 12, 31, tzinfo=timezone.utc)
        ).total_seconds()
    )
    for i in range(count):
        data += bytes([0]) + struct.pack(
            "<IiiHBBIHH",
            seconds + 5 * i,
            round(43 * 2**31 / 180),
            round((-80 + i * 0.00035) * 2**31 / 180),
            3000,
            145,
            80,
            i * 2800,
            5600,
            180,
        )
    header = struct.pack("<BBHI4s", 14, 0x20, 2100, len(data), b".FIT")
    crc = Crc()
    crc.update(header)
    header += struct.pack("<H", crc.value)
    crc = Crc()
    crc.update(header + data)
    return header + data + struct.pack("<H", crc.value)


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "data"))


@pytest.fixture
def store(settings):
    return Store(settings.data_dir)


@pytest.fixture
def synthetic_dataset(settings):
    rides, frames, traces = [], [], {}
    for ride_number in range(10):
        n = 200
        dt = 5.0
        speed = 5.5 + 0.15 * ride_number
        step = speed * dt / (111195 * np.cos(np.radians(43)))
        date = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=ride_number * 12)
        raw = pd.DataFrame(
            {
                "lat": 43.0,
                "lon": -80 + np.arange(n) * step,
                "time": pd.date_range(date, periods=n, freq="5s"),
                "elevation": 100 + 4 * np.sin(np.arange(n) / 30),
                "speed": speed,
                "hr": 125 + ride_number * 3 + 8 * np.cos(np.arange(n) / 30),
            }
        )
        points, diag = clean_points(raw, settings)
        points = annotate_network(points, {"elements": []})
        meta = {
            "id": str(ride_number),
            "date": date.isoformat(),
            "bike": "Test bike",
            "measured_avg_hr": float(raw.hr.mean()),
            "timestamp_reliable": True,
            **diag,
        }
        events = route_events(points)
        segments = build_segments(points, meta, events)
        rides.append(meta)
        frames.append(segments)
        traces[str(ride_number)] = points
    return pd.DataFrame(rides), pd.concat(frames, ignore_index=True), traces
