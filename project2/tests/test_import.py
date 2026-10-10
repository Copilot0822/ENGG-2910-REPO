from io import BytesIO
import gzip
import zipfile

import pandas as pd
import pytest

from cycling.geometry import clean_points
from cycling.importer import impute_ride_hr, import_archive, read_activity_csv
from cycling.recordings import parse_recording
from conftest import fit_bytes, gpx_bytes, strava_zip


def test_duplicate_distance_columns_use_metres():
    data = b"Activity ID,Activity Type,Distance,Distance,Average Heart Rate,Filename\n7,Ride,12.50,12500,142,activities/9.fit.gz\n"
    row = read_activity_csv(data)[0]
    assert row["summary_distance_m"] == 12500
    assert row["summary_avg_hr"] == 142


def test_csv_aliases_and_display_kilometres():
    data = b"id,sport_type,Distance (km),avg_hr,File\n2,GravelRide,23.5,148,a.gpx\n"
    row = read_activity_csv(data)[0]
    assert row["id"] == "2" and row["summary_distance_m"] == 23500
    assert row["summary_avg_hr"] == 148


def test_fit_and_gz_units_and_timestamps(settings):
    for name, data in (("original.fit", fit_bytes()), ("original.fit.gz", gzip.compress(fit_bytes()))):
        points, _ = parse_recording(data, name, settings)
        assert points.lat.iloc[0] == pytest.approx(43, abs=1e-6)
        assert points.elevation.iloc[0] == 100
        assert points.speed.iloc[0] == 5.6
        assert points.distance.iloc[1] == 28
        assert points.hr.iloc[0] == 145
        assert points.power.iloc[0] == 180
        assert pd.Timestamp(points.time.iloc[0]).year == 2025


def test_gpx_samples_and_route_missing_elevation(settings):
    points, _ = parse_recording(gpx_bytes(), "a.gpx", settings)
    assert points.hr.iloc[0] == 145
    assert points.cadence.iloc[0] == 80
    raw, _ = parse_recording(gpx_bytes(timed=False, elevation=False, route=True), "route.gpx", settings)
    clean, diagnostics = clean_points(raw, settings, route=True)
    assert clean.elevation_missing.all()
    assert (clean.gradient_pct == 0).all()
    assert diagnostics["elevation_gain_m"] is None
    assert diagnostics["distance_m"] > 4000


def test_tcx_sensor_parsing(settings):
    data = b"""<TrainingCenterDatabase xmlns="http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2"><Activities><Activity Sport="Biking"><Lap><Track><Trackpoint><Time>2025-01-01T00:00:00Z</Time><Position><LatitudeDegrees>43</LatitudeDegrees><LongitudeDegrees>-80</LongitudeDegrees></Position><AltitudeMeters>100</AltitudeMeters><DistanceMeters>20</DistanceMeters><HeartRateBpm><Value>149</Value></HeartRateBpm><Cadence>81</Cadence><Extensions><TPX><Speed>7</Speed><Watts>210</Watts></TPX></Extensions></Trackpoint></Track></Lap></Activity></Activities></TrainingCenterDatabase>"""
    points, metadata = parse_recording(data, "original.tcx", settings)
    assert metadata["sport"] == "Biking"
    assert points.hr.iloc[0] == 149
    assert points.speed.iloc[0] == 7
    assert points.power.iloc[0] == 210


def test_incremental_duplicate_detection(store, settings):
    archive = strava_zip()
    first = import_archive(archive, store, settings)
    second = import_archive(archive, store, settings)
    assert first["imported"] == 1
    assert second["duplicates"] == 1
    assert len(store.rides()) == 1
    # Different Strava ID but byte-identical original must also be a duplicate.
    renamed = import_archive(strava_zip(ride_id="another"), store, settings)
    assert renamed["duplicates"] == 1


@pytest.mark.parametrize("sport", ["Run", "VirtualRide", "Swim", "EBikeRide"])
def test_exclude_irrelevant_sports(sport, store, settings):
    report = import_archive(strava_zip(sport=sport), store, settings)
    assert report["excluded"] == 1 and report["imported"] == 0


def test_median_hr_imputation_retains_flags():
    rides = pd.DataFrame({"measured_avg_hr": [130.0, 142.0, 150.0, None]})
    output, median = impute_ride_hr(rides)
    assert median == 142
    assert output.average_hr.iloc[-1] == 142
    assert output.hr_imputed.tolist() == [False, False, False, True]


def test_no_hr_does_not_fabricate_relationship(store, settings):
    report = import_archive(strava_zip(hr=None), store, settings)
    output, median = impute_ride_hr(store.rides())
    assert report["imported"] == 1
    assert median is None and output.average_hr.isna().all()
    assert store.points("1").hr.isna().all()


@pytest.mark.parametrize(
    "entry", ["../escape.gpx", "/absolute.gpx", "C:/escape.gpx", "a\\..\\escape.gpx"]
)
def test_zip_path_traversal_rejected(entry, store, settings):
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(entry, b"x")
    with pytest.raises(ValueError, match="Unsafe"):
        import_archive(buffer.getvalue(), store, settings)


def test_zip_bomb_limits(store, settings):
    settings.max_entry_bytes = 100
    with pytest.raises(ValueError, match="size limit"):
        import_archive(strava_zip(), store, settings)


def test_xml_entity_rejected(settings):
    malicious = b'<!DOCTYPE gpx [<!ENTITY private SYSTEM "file:///C:/secret">]><gpx>&private;</gpx>'
    with pytest.raises(Exception):
        parse_recording(malicious, "bad.gpx", settings)


def test_fit_reader_enforces_point_limit_during_parsing(settings):
    settings.max_points = 5
    with pytest.raises(ValueError, match="point limit"):
        parse_recording(fit_bytes(count=20), "large.fit", settings)
