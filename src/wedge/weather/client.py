from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import httpx

from wedge.config import CityConfig
from wedge.log import get_logger

if TYPE_CHECKING:
    from wedge.config import Settings

log = get_logger("weather.client")

ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


async def fetch_ensemble(
    client: httpx.AsyncClient,
    city: CityConfig,
    forecast_days: int = 7,
) -> dict | None:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "daily": "temperature_2m_max",
        "models": "gfs_seamless",
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
        "timezone": city.timezone,
    }

    for attempt in range(3):
        try:
            resp = await client.get(ENSEMBLE_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            return resp.json()
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            wait = 2 ** attempt
            log.warning("open_meteo_retry", city=city.name, attempt=attempt + 1, error=str(e))
            if attempt < 2:
                import asyncio
                await asyncio.sleep(wait)
    log.error("open_meteo_failed", city=city.name)
    return None


async def fetch_actual_temperature(
    client: httpx.AsyncClient,
    city: CityConfig,
    target_date: str,
) -> int | None:
    """Fetch observed daily max temperature for a specific date.

    Uses Open-Meteo Archive API. Returns rounded integer °F, or None on failure.
    """
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": target_date,
        "end_date": target_date,
        "daily": "temperature_2m_max",
        "temperature_unit": "fahrenheit",
        "timezone": city.timezone,
    }

    for attempt in range(3):
        try:
            resp = await client.get(ARCHIVE_URL, params=params, timeout=30.0)
            resp.raise_for_status()
            data = resp.json()
            temps = data.get("daily", {}).get("temperature_2m_max", [])
            if temps and temps[0] is not None:
                return round(temps[0])
            log.warning("no_actual_temp_data", city=city.name, date=target_date)
            return None
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            wait = 2 ** attempt
            log.warning("archive_api_retry", city=city.name, attempt=attempt + 1, error=str(e))
            if attempt < 2:
                import asyncio
                await asyncio.sleep(wait)
    log.error("archive_api_failed", city=city.name, date=target_date)
    return None


async def fetch_ensemble_auto(
    client: httpx.AsyncClient,
    city: CityConfig,
    settings: "Settings",
    target_date: date | None = None,
) -> dict | None:
    """Fetch ensemble data from the configured weather source.

    Routes to NOAA GEFS or Open-Meteo based on settings.weather_source.
    target_date is only used for NOAA; Open-Meteo returns a multi-day forecast.
    """
    if settings.weather_source == "noaa":
        if target_date is None:
            import datetime as _dt
            target_date = _dt.date.today()
        from wedge.weather.noaa_client import fetch_ensemble_noaa
        return await fetch_ensemble_noaa(client, city, target_date)
    else:
        return await fetch_ensemble(client, city)
