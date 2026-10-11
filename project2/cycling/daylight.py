"""Destination sun events, explicit civil twilight and UTC-safe departure search."""

from datetime import datetime, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from astral import Observer
from astral.sun import dusk, elevation, sunrise, sunset
import pandas as pd
from timezonefinder import TimezoneFinder


@lru_cache(maxsize=1)
def timezone_lookup():
    return TimezoneFinder(in_memory=False)


def coordinate_timezone(lat, lon):
    return timezone_lookup().timezone_at(lat=float(lat), lng=float(lon)) or "UTC"


def local_departure(day, clock, timezone_name, occurrence=None):
    naive = pd.Timestamp(datetime.combine(day, clock))
    ambiguous = "raise" if occurrence is None else occurrence == "first"
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Unknown IANA departure timezone.") from exc
    try:
        return naive.tz_localize(zone, ambiguous=ambiguous, nonexistent="raise").isoformat()
    except Exception as exc:
        # pandas can raise non-ValueError DST exceptions on supported versions.
        raise ValueError(
            "Departure is ambiguous or nonexistent during a daylight-saving transition. Choose the first/second occurrence or another time."
        ) from exc


def daylight_plan(departure, elapsed_s, destination, reference="sunset", safety_minutes=30):
    if reference not in ("sunset", "civil_twilight") or safety_minutes < 0:
        raise ValueError("Choose sunset/civil_twilight and a nonnegative margin.")
    departure = pd.Timestamp(departure)
    if departure.tzinfo is None:
        raise ValueError("Departure must include a timezone.")
    lat, lon = destination
    timezone_name = coordinate_timezone(lat, lon)
    zone = ZoneInfo(timezone_name)
    local_start = departure.tz_convert(zone)
    finish = (departure.tz_convert("UTC") + pd.Timedelta(seconds=elapsed_s)).tz_convert(zone)
    observer = Observer(float(lat), float(lon))
    events, notes = {}, []
    # Use the departure's destination-calendar day, even for overnight arrival.
    for name, function in (("sunset", sunset), ("civil_twilight", dusk), ("sunrise", sunrise)):
        try:
            events[name] = pd.Timestamp(function(observer, date=local_start.date(), tzinfo=zone))
        except ValueError:
            events[name] = None
    cutoff = (
        events[reference] - pd.Timedelta(minutes=safety_minutes)
        if events[reference] is not None
        else None
    )
    if cutoff is None:
        notes.append(
            "No selected sun event occurs here on this date; polar daylight/darkness has no ordinary sunset cutoff."
        )
    if finish.date() != local_start.date():
        notes.append(
            "Arrival is on a different calendar day. The cutoff remains the departure day's sun event; this is an overnight plan."
        )
    if reference == "civil_twilight":
        notes.append("Civil twilight is reduced light, not full daylight.")
    remaining = (
        (events["sunset"] - finish).total_seconds() / 60 if events["sunset"] is not None else None
    )
    return {
        "departure_utc": departure.tz_convert("UTC").isoformat(),
        "departure_local": local_start.isoformat(),
        "finish_local": finish.isoformat(),
        "finish_utc": finish.tz_convert("UTC").isoformat(),
        "destination_timezone": timezone_name,
        "destination_lat": float(lat),
        "destination_lon": float(lon),
        "sunset": events["sunset"].isoformat() if events["sunset"] is not None else None,
        "civil_twilight": events["civil_twilight"].isoformat()
        if events["civil_twilight"] is not None
        else None,
        "cutoff": cutoff.isoformat() if cutoff is not None else None,
        "reference": reference,
        "safety_minutes": safety_minutes,
        "remaining_daylight_minutes": remaining,
        "cutoff_margin_minutes": (cutoff - finish).total_seconds() / 60 if cutoff is not None else None,
        "daylight_at_finish": elevation(observer, finish.to_pydatetime(warn=False)) >= -0.833,
        "notes": notes,
    }


def latest_departure(predictor, cutoff, earliest=None, iterations=8, criterion="conservative"):
    """Search re-predicted departures; check multiple brackets for weather effects."""
    cutoff = pd.Timestamp(cutoff)
    if criterion not in ("conservative", "median", "expected"):
        raise ValueError("Choose conservative, median or expected finishing time.")
    if cutoff.tzinfo is None:
        raise ValueError("The daylight cutoff must include a timezone.")
    cutoff = cutoff.tz_convert("UTC")
    earliest = (
        pd.Timestamp(earliest).tz_convert("UTC") if earliest is not None else cutoff - timedelta(days=2)
    )
    if earliest >= cutoff:
        raise ValueError("Departure search must start before the cutoff.")
    calls, provisional = 0, False

    def feasible(moment):
        nonlocal calls, provisional
        result = predictor(moment.isoformat())
        calls += 1
        uncertainty = result.get("elapsed_uncertainty") or {}
        conservative = (
            result["elapsed_s"]
            if criterion == "expected"
            else uncertainty.get("conservative_s" if criterion == "conservative" else "median_s")
        )
        provisional |= (
            bool(uncertainty.get("provisional", True))
            or conservative is None
            or criterion != "conservative"
        )
        duration = conservative if conservative is not None else result["elapsed_s"]
        return moment + pd.Timedelta(seconds=duration) <= cutoff

    grid = pd.date_range(earliest, cutoff, periods=9)
    good = [feasible(moment) for moment in grid]
    possible = [i for i, ok in enumerate(good) if ok]
    if not possible:
        return {
            "departure_utc": None,
            "provisional": True,
            "evaluations": calls,
            "note": "No feasible departure found inside the searched window.",
        }
    index = possible[-1]
    low, high = grid[index], grid[min(index + 1, len(grid) - 1)]
    for _ in range(iterations):
        if (high - low).total_seconds() <= 60:
            break
        mid = low + (high - low) / 2
        if feasible(mid):
            low = mid
        else:
            high = mid
    return {
        "departure_utc": low.isoformat(),
        "criterion": criterion,
        "provisional": provisional,
        "evaluations": calls,
        "note": "Latest sampled feasible bracket, refined to approximately one minute. Non-monotone weather/traffic and regional forecasts limit this estimate.",
    }
