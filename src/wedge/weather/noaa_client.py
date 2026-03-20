"""NOAA GEFS ensemble client.

Fetches 30-member GEFS ensemble data from NOMADS for a given city and target date.
GEFS data is typically available 1-2 hours earlier than Open-Meteo, providing
an information edge before market makers reprice.
"""
from __future__ import annotations

import asyncio
import io
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import eccodes
import httpx

from wedge.config import CityConfig
from wedge.log import get_logger

if TYPE_CHECKING:
    pass

log = get_logger("weather.noaa_client")

NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"

# GEFS has 30 ensemble members: gep01 .. gep30
NUM_MEMBERS = 30

# Processing delay: GEFS run data available ~3.5 hours after nominal run time
PROCESSING_DELAY_HOURS = 3.5

# Concurrency limit to avoid rate limiting
MAX_CONCURRENT = 5

# Small delay between member requests (seconds)
MEMBER_DELAY = 0.1


def _select_run_time(now_utc: datetime) -> tuple[datetime, int]:
    """Select the most recently available GEFS run.

    GEFS runs at 00/06/12/18 UTC. Each run takes ~3.5h to process.
    Returns (run_datetime, run_hour) where run_hour is 0/6/12/18.
    """
    # Walk backwards through run hours to find most recent available
    candidate = now_utc.replace(minute=0, second=0, microsecond=0)
    # Align to 6h boundary
    run_hour = (candidate.hour // 6) * 6
    candidate = candidate.replace(hour=run_hour)

    for _ in range(5):  # Try up to 5 previous runs (30h back)
        available_at = candidate + timedelta(hours=PROCESSING_DELAY_HOURS)
        if available_at <= now_utc:
            return candidate, candidate.hour
        candidate -= timedelta(hours=6)

    # Fallback: return 30h ago run
    return candidate, candidate.hour


def _forecast_hour(run_dt: datetime, target_date: date) -> int:
    """Compute the GEFS forecast hour for the target date's max temperature.

    We target 18 UTC (noon local for most US cities) as a proxy for daily max.
    Snaps to nearest 6h interval available in pgrb2s (f006, f012, ..., f384).
    """
    # Target: noon UTC on target_date is a reasonable proxy for daily max
    # (actual daily max varies by city but 18 UTC / 1pm ET works well)
    target_dt = datetime(
        target_date.year, target_date.month, target_date.day, 18, 0, 0, tzinfo=UTC
    )
    delta_hours = (target_dt - run_dt).total_seconds() / 3600
    # Snap to nearest 6h
    fhour = round(delta_hours / 6) * 6
    # Clamp to valid range [6, 384]
    fhour = max(6, min(384, fhour))
    return fhour


def _build_url(member: int, run_dt: datetime, fhour: int) -> str:
    """Build NOMADS filter URL for a single GEFS member."""
    date_str = run_dt.strftime("%Y%m%d")
    run_hour_str = f"{run_dt.hour:02d}"
    filename = f"gep{member:02d}.t{run_hour_str}z.pgrb2s.0p25.f{fhour:03d}"
    dir_path = f"/gefs.{date_str}/{run_hour_str}/atmos/pgrb2sp25"

    params = (
        f"file={filename}"
        "&var_TMP=on"
        "&lev_2_m_above_ground=on"
        "&subregion="
        "&leftlon=-130"
        "&rightlon=-60"
        "&toplat=55"
        "&bottomlat=20"
        f"&dir={dir_path}"
    )
    return f"{NOMADS_FILTER_URL}?{params}"


def _extract_temperature(grib_bytes: bytes, lat: float, lon: float) -> float | None:
    """Extract 2m temperature at nearest grid point from GRIB2 bytes.

    GEFS uses 0-360 longitude convention: west longitudes = 360 - abs(lon).
    Returns temperature in °F, or None if not found.
    """
    import os
    import tempfile

    # Convert longitude to 0-360 convention
    lon_360 = lon if lon >= 0 else 360.0 + lon

    temp_k: float | None = None

    # eccodes.codes_grib_new_from_file requires a real file (needs fileno())
    # Write bytes to a temp file, then parse
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    try:
        tmp.write(grib_bytes)
        tmp.flush()
        tmp.close()

        with open(tmp.name, "rb") as f:
            while True:
                try:
                    msg = eccodes.codes_grib_new_from_file(f)  # type: ignore[attr-defined]
                except Exception:
                    break
                if msg is None:
                    break

                try:
                    short_name = eccodes.codes_get(msg, "shortName")  # type: ignore[attr-defined]
                    if short_name != "2t":
                        continue

                    # Get grid info
                    lat_first = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")  # type: ignore[attr-defined]
                    lon_first = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")  # type: ignore[attr-defined]
                    lat_last = eccodes.codes_get(msg, "latitudeOfLastGridPointInDegrees")  # type: ignore[attr-defined]
                    lon_last = eccodes.codes_get(msg, "longitudeOfLastGridPointInDegrees")  # type: ignore[attr-defined]
                    ni = eccodes.codes_get(msg, "Ni")  # type: ignore[attr-defined]
                    nj = eccodes.codes_get(msg, "Nj")  # type: ignore[attr-defined]

                    # Compute grid resolution
                    d_lon = (lon_last - lon_first) / (ni - 1) if ni > 1 else 0.25
                    d_lat = (lat_first - lat_last) / (nj - 1) if nj > 1 else 0.25  # lat decreasing

                    # Find nearest grid indices
                    i = round((lon_360 - lon_first) / d_lon) if d_lon != 0 else 0
                    j = round((lat_first - lat) / d_lat) if d_lat != 0 else 0

                    # Clamp
                    i = max(0, min(ni - 1, i))
                    j = max(0, min(nj - 1, j))

                    idx = j * ni + i

                    values = eccodes.codes_get_array(msg, "values")  # type: ignore[attr-defined]
                    if idx < len(values):
                        temp_k = float(values[idx])

                finally:
                    eccodes.codes_release(msg)  # type: ignore[attr-defined]

                if temp_k is not None:
                    break

    except Exception as exc:
        log.warning("grib_parse_error", error=str(exc))
        return None
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if temp_k is None:
        return None

    # Convert K → °F
    return (temp_k - 273.15) * 9 / 5 + 32


async def _fetch_member(
    client: httpx.AsyncClient,
    member: int,
    url: str,
    lat: float,
    lon: float,
    semaphore: asyncio.Semaphore,
) -> float | None:
    """Download and parse a single GEFS member. Returns °F or None."""
    async with semaphore:
        await asyncio.sleep(MEMBER_DELAY * (member - 1))  # stagger requests
        for attempt in range(3):
            try:
                resp = await client.get(url, timeout=60.0)
                resp.raise_for_status()
                temp_f = _extract_temperature(resp.content, lat, lon)
                if temp_f is None:
                    log.warning("gefs_no_temp", member=member, attempt=attempt + 1)
                    return None
                log.debug("gefs_member_ok", member=member, temp_f=round(temp_f, 1))
                return temp_f
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
                wait = 2**attempt
                log.warning(
                    "gefs_member_retry",
                    member=member,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < 2:
                    await asyncio.sleep(wait)

    log.warning("gefs_member_failed", member=member)
    return None


async def fetch_ensemble_noaa(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: date,
) -> dict | None:
    """Fetch GEFS ensemble data from NOMADS for a city and target date.

    Returns an Open-Meteo-compatible dict with keys:
        {'daily': {'temperature_2m_max_member01': [t1], ..., 'temperature_2m_max_member30': [t30]}}

    Each member value is the temperature (°F) at the nearest grid point for target_date.
    Returns None on failure.
    """
    now_utc = datetime.now(UTC)
    run_dt, run_hour = _select_run_time(now_utc)
    fhour = _forecast_hour(run_dt, target_date)

    log.info(
        "gefs_fetch_start",
        city=city.name,
        target_date=str(target_date),
        run_dt=run_dt.strftime("%Y-%m-%d %H UTC"),
        fhour=fhour,
    )

    # Validate forecast hour makes sense (target date must be in the future relative to run)
    if fhour < 6:
        log.warning(
            "gefs_invalid_fhour",
            city=city.name,
            fhour=fhour,
            run_dt=str(run_dt),
            target_date=str(target_date),
        )
        return None

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        _fetch_member(
            client=client,
            member=m,
            url=_build_url(m, run_dt, fhour),
            lat=city.lat,
            lon=city.lon,
            semaphore=semaphore,
        )
        for m in range(1, NUM_MEMBERS + 1)
    ]

    results = await asyncio.gather(*tasks)

    # Build Open-Meteo compatible output
    daily: dict[str, list] = {}
    success_count = 0
    for i, temp_f in enumerate(results, start=1):
        key = f"temperature_2m_max_member{i:02d}"
        if temp_f is not None:
            daily[key] = [temp_f]
            success_count += 1
        else:
            # Use None placeholder so parse_distribution can detect missing members
            daily[key] = [None]

    if success_count == 0:
        log.warning("gefs_all_members_failed", city=city.name, target_date=str(target_date))
        return None

    if success_count < NUM_MEMBERS:
        log.warning(
            "gefs_partial_members",
            city=city.name,
            success=success_count,
            total=NUM_MEMBERS,
        )

    log.info(
        "gefs_fetch_done",
        city=city.name,
        target_date=str(target_date),
        members_ok=success_count,
    )

    return {"daily": daily}
