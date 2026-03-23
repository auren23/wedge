from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx
from eccodes import (
    codes_get_array,
    codes_grib_find_nearest,
    codes_grib_new_from_file,
    codes_release,
 )

from wedge.config import CityConfig
from wedge.log import get_logger

log = get_logger("weather.client")

NOMADS_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p25s.pl"
WUNDERGROUND_API_URL = "https://api.weather.com/v1/location/{station}:9:{country}/observations/historical.json"
AVIATIONWEATHER_METAR_URL = "https://aviationweather.gov/api/data/metar"
WUNDERGROUND_API_KEY = "e1f10a1e78da46f5b10a1e78da96f525"

# ICAO station prefix → country code (for Weather Company API)
_ICAO_TO_COUNTRY: dict[str, str] = {
    "K": "US",   # US stations
    "R": "KR",   # South Korea (RKSI = Incheon)
    "Z": "CN",   # China (ZSPD = Pudong)
    "E": "GB",   # UK (EGLL = Heathrow)
    "S": "AR",   # Argentina (SAEZ = Ezeiza)
    "N": "NZ",   # New Zealand (NZWN = Wellington)
}
_MEMBER_IDS = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))
_FORECAST_INTERVAL_HOURS = 3
_MAX_FORECAST_HOURS = 384
_MIN_MEMBER_COUNT = 2
_TRADABLE_MEMBER_COUNT = 20


@dataclass(slots=True)
class ReadinessProbeResult:
    run_date: date
    cycle_hour: int
    target_date: date
    forecast_hours: list[int]
    prefetched_temperatures: dict[str, list[float]]
    ready: bool
    reason: str
    checked_at: datetime
    attempts: int


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _resolve_latest_cycle(now: datetime | None = None) -> tuple[date, int]:
    current = now or _utc_now()
    cycle_hour = (current.hour // 6) * 6
    cycle_date = current.date()
    if current.hour < 4:
        cycle_date -= timedelta(days=1)
        cycle_hour = 18
    return cycle_date, cycle_hour


def _forecast_hours_for_target_date(
    target_date: date,
    city_timezone: str,
    run_date: date,
    run_hour: int,
) -> list[int]:
    city_tz = datetime.now().astimezone().tzinfo
    try:
        from zoneinfo import ZoneInfo

        city_tz = ZoneInfo(city_timezone)
    except Exception:
        city_tz = UTC

    run_dt_utc = datetime(run_date.year, run_date.month, run_date.day, run_hour, tzinfo=UTC)
    run_local_date = run_dt_utc.astimezone(city_tz).date()
    day_offset = (target_date - run_local_date).days
    if day_offset < 0:
        return []

    start = max(0, day_offset * 24)
    end = min(_MAX_FORECAST_HOURS, start + 21)
    hours = list(range(start, end + 1, _FORECAST_INTERVAL_HOURS))
    return hours


def _member_file(member_id: str, cycle_hour: int, forecast_hour: int) -> str:
    prefix = "gec00" if member_id == "c00" else f"gep{member_id[1:]}"
    return f"{prefix}.t{cycle_hour:02d}z.pgrb2s.0p25.f{forecast_hour:03d}"


def _extract_point_temperature_f(grib_bytes: bytes, city: CityConfig) -> float | None:
    import os
    import tempfile

    handle = None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".grib2") as tmp:
            tmp.write(grib_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            handle = codes_grib_new_from_file(fh)
        if handle is None:
            return None
        try:
            nearest = codes_grib_find_nearest(handle, city.lat, city.lon)
            if nearest and isinstance(nearest, (list, tuple)):
                candidate = nearest[0]
                value = candidate.get("value") if isinstance(candidate, dict) else None
                if value is not None and math.isfinite(value):
                    return float(value)
        except Exception:
            pass
        values = codes_get_array(handle, "values")
        if values is None or len(values) == 0:
            return None
        finite_values = [float(v) for v in values if math.isfinite(v)]
        if not finite_values:
            return None
        return finite_values[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("noaa_grib_parse_failed", city=city.name, error=str(exc))
        return None
    finally:
        if handle is not None:
            try:
                codes_release(handle)
            except Exception:  # pragma: no cover
                pass
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def parse_readiness_probe(
    *,
    run_date: date,
    cycle_hour: int,
    target_date: date,
    forecast_hours: list[int],
    prefetched_temperatures: dict[str, list[float]],
    ready: bool,
    reason: str,
    checked_at: datetime,
    attempts: int,
 ) -> ReadinessProbeResult:
    return ReadinessProbeResult(
        run_date=run_date,
        cycle_hour=cycle_hour,
        target_date=target_date,
        forecast_hours=list(forecast_hours),
        prefetched_temperatures={k: list(v) for k, v in prefetched_temperatures.items()},
        ready=ready,
        reason=reason,
        checked_at=checked_at,
        attempts=attempts,
    )


def _build_slice_params(
    *,
    city: CityConfig,
    dir_path: str,
    member_id: str,
    cycle_hour: int,
    forecast_hour: int,
 ) -> dict[str, str]:
    return {
        "file": _member_file(member_id, cycle_hour, forecast_hour),
        "lev_2_m_above_ground": "on",
        "var_TMP": "on",
        "subregion": "",
        "leftlon": str(city.lon),
        "rightlon": str(city.lon),
        "toplat": str(city.lat),
        "bottomlat": str(city.lat),
        "dir": dir_path,
    }


async def _fetch_slice_temperature(
    client: httpx.AsyncClient,
    *,
    city: CityConfig,
    member_id: str,
    forecast_hour: int,
    params: dict[str, str],
 ) -> tuple[str, int, float | None, bool]:
    try:
        resp = await client.get(NOMADS_FILTER_URL, params=params, timeout=30.0)
        resp.raise_for_status()
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
        log.warning(
            "noaa_fetch_failed",
            city=city.name,
            member=member_id,
            forecast_hour=forecast_hour,
            error=str(exc),
        )
        return member_id, forecast_hour, None, True

    value = _extract_point_temperature_f(resp.content, city)
    return member_id, forecast_hour, value, False


async def probe_cycle_readiness(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: date,
    *,
    now: datetime | None = None,
 ) -> ReadinessProbeResult:
    run_date, cycle_hour = _resolve_latest_cycle(now)
    forecast_hours = _forecast_hours_for_target_date(
        target_date, city.timezone, run_date, cycle_hour
    )
    checked_at = _utc_now()
    if not forecast_hours:
        return parse_readiness_probe(
            run_date=run_date,
            cycle_hour=cycle_hour,
            target_date=target_date,
            forecast_hours=[],
            prefetched_temperatures={},
            ready=False,
            reason="no_forecast_hours",
            checked_at=checked_at,
            attempts=0,
        )

    dir_path = f"/gefs.{run_date.strftime('%Y%m%d')}/{cycle_hour:02d}/atmos/pgrb2sp25"
    prefetched: dict[str, list[float]] = {}
    attempts = 0

    control_values: list[float] = []
    for forecast_hour in forecast_hours:
        attempts += 1
        _, _, value, had_error = await _fetch_slice_temperature(
            client,
            city=city,
            member_id="c00",
            forecast_hour=forecast_hour,
            params=_build_slice_params(
                city=city,
                dir_path=dir_path,
                member_id="c00",
                cycle_hour=cycle_hour,
                forecast_hour=forecast_hour,
            ),
        )
        if had_error:
            continue
        if value is not None and math.isfinite(value):
            control_values.append(value)

    if control_values:
        prefetched["c00"] = control_values
    else:
        return parse_readiness_probe(
            run_date=run_date,
            cycle_hour=cycle_hour,
            target_date=target_date,
            forecast_hours=forecast_hours,
            prefetched_temperatures=prefetched,
            ready=False,
            reason="control_member_missing",
            checked_at=checked_at,
            attempts=attempts,
        )

    first_horizon = forecast_hours[0]
    attempts += 1
    _, _, value, had_error = await _fetch_slice_temperature(
        client,
        city=city,
        member_id="p01",
        forecast_hour=first_horizon,
        params=_build_slice_params(
            city=city,
            dir_path=dir_path,
            member_id="p01",
            cycle_hour=cycle_hour,
            forecast_hour=first_horizon,
        ),
    )
    if not had_error and value is not None and math.isfinite(value):
        prefetched["p01"] = [value]

    perturb_values = prefetched.get("p01", [])
    ready = bool(perturb_values)
    reason = "ready" if ready else "perturbation_member_missing"

    return parse_readiness_probe(
        run_date=run_date,
        cycle_hour=cycle_hour,
        target_date=target_date,
        forecast_hours=forecast_hours,
        prefetched_temperatures=prefetched,
        ready=ready,
        reason=reason,
        checked_at=checked_at,
        attempts=attempts,
    )


async def _fetch_member_temperatures_parallel(
    client: httpx.AsyncClient,
    *,
    city: CityConfig,
    dir_path: str,
    cycle_hour: int,
    forecast_hours: list[int],
    seed_prefetch: dict[str, list[float]],
    max_concurrency: int,
 ) -> tuple[dict[str, list[float]], int]:
    semaphore = asyncio.Semaphore(max(1, max_concurrency))
    member_values = {member_id: list(values) for member_id, values in seed_prefetch.items()}
    error_count = 0
    tasks: list[asyncio.Task[tuple[str, int, float | None, bool]]] = []

    async def _run(
        member_id: str, forecast_hour: int, params: dict[str, str]
    ) -> tuple[str, int, float | None, bool]:
        async with semaphore:
            return await _fetch_slice_temperature(
                client,
                city=city,
                member_id=member_id,
                forecast_hour=forecast_hour,
                params=params,
            )

    for member_id in _MEMBER_IDS:
        existing = len(member_values.get(member_id, []))
        for forecast_hour in forecast_hours[existing:]:
            tasks.append(
                asyncio.create_task(
                    _run(
                        member_id,
                        forecast_hour,
                        _build_slice_params(
                            city=city,
                            dir_path=dir_path,
                            member_id=member_id,
                            cycle_hour=cycle_hour,
                            forecast_hour=forecast_hour,
                        ),
                    )
                )
            )

    for member_id, _, value, had_error in await asyncio.gather(*tasks):
        if had_error:
            error_count += 1
            continue
        if value is not None and math.isfinite(value):
            member_values.setdefault(member_id, []).append(value)

    return member_values, error_count


def _member_maxima(
    member_values: dict[str, list[float]],
 ) -> dict[str, float]:
    maxima: dict[str, float] = {}
    for member_id, values in member_values.items():
        if values:
            maxima[member_id] = max(values)
    return maxima

async def fetch_ensemble(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: date,
    *,
    probe: ReadinessProbeResult | None = None,
    parallel: bool = False,
    max_concurrency: int = 1,
    error_rate_threshold: float = 0.05,
 ) -> dict[str, object] | None:
    run_date, cycle_hour = (probe.run_date, probe.cycle_hour) if probe else _resolve_latest_cycle()
    forecast_hours = (
        list(probe.forecast_hours)
        if probe and probe.target_date == target_date
        else _forecast_hours_for_target_date(target_date, city.timezone, run_date, cycle_hour)
    )
    if not forecast_hours:
        log.warning(
            "noaa_no_forecast_hours",
            city=city.name,
            target_date=target_date.isoformat(),
        )
        return None

    dir_path = f"/gefs.{run_date.strftime('%Y%m%d')}/{cycle_hour:02d}/atmos/pgrb2sp25"
    fetch_mode = "sequential"
    error_count = 0
    prefetched = probe.prefetched_temperatures if probe else {}

    if parallel:
        member_values, error_count = await _fetch_member_temperatures_parallel(
            client,
            city=city,
            dir_path=dir_path,
            cycle_hour=cycle_hour,
            forecast_hours=forecast_hours,
            seed_prefetch=prefetched,
            max_concurrency=max_concurrency,
        )
        total_attempts = max(sum(len(v) for v in member_values.values()) + error_count, 1)
        if error_count / total_attempts > error_rate_threshold:
            fetch_mode = "parallel_degraded"
        else:
            fetch_mode = "parallel"
    else:
        member_values: dict[str, list[float]] = {
            member_id: list(values) for member_id, values in prefetched.items()
        }
        for member_id in _MEMBER_IDS:
            values = member_values.setdefault(member_id, [])
            for forecast_hour in forecast_hours[len(values) :]:
                _, _, value, had_error = await _fetch_slice_temperature(
                    client,
                    city=city,
                    member_id=member_id,
                    forecast_hour=forecast_hour,
                    params=_build_slice_params(
                        city=city,
                        dir_path=dir_path,
                        member_id=member_id,
                        cycle_hour=cycle_hour,
                        forecast_hour=forecast_hour,
                    ),
                )
                if had_error:
                    error_count += 1
                    continue
                if value is not None and math.isfinite(value):
                    values.append(value)

    member_temps_f = _member_maxima(member_values)
    member_count = len(member_temps_f)

    result = {
        "source": "noaa_gefs",
        "city": city.name,
        "target_date": target_date.isoformat(),
        "run_time": datetime(
            run_date.year,
            run_date.month,
            run_date.day,
            cycle_hour,
            tzinfo=UTC,
        ).isoformat(),
        "member_temps_f": member_temps_f,
        "member_count": member_count,
        "error_count": error_count,
        "fetch_mode": fetch_mode,
    }

    if member_count < _MIN_MEMBER_COUNT:
        log.error(
            "noaa_insufficient_members",
            city=city.name,
            target_date=target_date.isoformat(),
            members=member_count,
        )
        return None

    if parallel and member_count < _TRADABLE_MEMBER_COUNT:
        result["status"] = "partial_publication"
    else:
        result["status"] = "ready"

    return result

def _country_for_station(station: str) -> str | None:
    prefix = station[0] if station else ""
    return _ICAO_TO_COUNTRY.get(prefix)


async def fetch_actual_temperature(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: str,
) -> int | None:
    country = _country_for_station(city.station)
    if not country:
        log.error("unknown_station_country", station=city.station)
        return None
    url = WUNDERGROUND_API_URL.format(station=city.station, country=country)
    date_compact = target_date.replace("-", "")
    params = {
        "apiKey": WUNDERGROUND_API_KEY,
        "units": "e",
        "startDate": date_compact,
        "endDate": date_compact,
    }
    for attempt in range(3):
        try:
            resp = await client.get(url, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            observations = data.get("observations", [])
            if not observations:
                log.warning("wunderground_no_observations", city=city.name, date=target_date)
                return None
            first = observations[0]
            max_temp = first.get("max_temp")
            if max_temp is None:
                temps = [
                    o["temp"]
                    for o in observations
                    if isinstance(o.get("temp"), (int, float)) and o["temp"] is not None
                ]
                if not temps:
                    log.warning("wunderground_no_temps", city=city.name, date=target_date)
                    return None
                max_temp = max(temps)
            return int(max_temp)
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            wait = 2**attempt
            log.warning("wunderground_api_retry", city=city.name, attempt=attempt + 1, error=str(e))
            if attempt < 2:
                await asyncio.sleep(wait)
    log.error("wunderground_api_failed", city=city.name, date=target_date)
    return None


async def fetch_metar_observation(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: str,
) -> int | None:
    date_compact = target_date.replace("-", "")
    params = {
        "ids": city.station,
        "start": f"{date_compact}_0000",
        "end": f"{date_compact}_2359",
        "format": "json",
    }
    try:
        resp = await client.get(
            AVIATIONWEATHER_METAR_URL, params=params, timeout=30.0
        )
        resp.raise_for_status()
        reports = resp.json()
        if not isinstance(reports, list) or not reports:
            return None
        temps_c = [
            r["temp"]
            for r in reports
            if isinstance(r.get("temp"), (int, float)) and r["temp"] is not None
        ]
        if not temps_c:
            return None
        return round(max(temps_c) * 9.0 / 5.0 + 32.0)
    except Exception:
        return None
