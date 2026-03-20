"""Tests for NOAA GEFS ensemble client."""
from __future__ import annotations

import io
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, patch

import eccodes
import httpx
import pytest

from wedge.config import CityConfig
from wedge.weather.noaa_client import (
    _build_url,
    _extract_temperature,
    _forecast_hour,
    _select_run_time,
    fetch_ensemble_noaa,
)

NYC = CityConfig(
    name="NYC",
    lat=40.7772,
    lon=-73.8726,
    timezone="America/New_York",
    station="KLGA",
)


# ---------------------------------------------------------------------------
# Helper: build a minimal GRIB2 message with uniform temperature
# ---------------------------------------------------------------------------

def _make_minimal_grib2(lat: float, lon: float, temp_k: float) -> bytes:
    """Create a minimal GRIB2 byte string with a 3x3 grid near (lat, lon).
    Raises eccodes.CodesInternalError if samples are not installed.
    """
    sample_id = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")

    eccodes.codes_set(sample_id, "edition", 2)
    eccodes.codes_set(sample_id, "centre", 7)  # NCEP
    eccodes.codes_set(sample_id, "shortName", "2t")
    eccodes.codes_set(sample_id, "typeOfLevel", "heightAboveGround")
    eccodes.codes_set(sample_id, "level", 2)

    step = 0.25
    eccodes.codes_set(sample_id, "Ni", 3)
    eccodes.codes_set(sample_id, "Nj", 3)
    eccodes.codes_set(sample_id, "latitudeOfFirstGridPointInDegrees", lat + step)
    eccodes.codes_set(sample_id, "latitudeOfLastGridPointInDegrees", lat - step)
    # GRIB2 uses 0-360 longitude
    lon360 = lon if lon >= 0 else 360.0 + lon
    eccodes.codes_set(sample_id, "longitudeOfFirstGridPointInDegrees", lon360 - step)
    eccodes.codes_set(sample_id, "longitudeOfLastGridPointInDegrees", lon360 + step)
    eccodes.codes_set(sample_id, "iDirectionIncrementInDegrees", step)
    eccodes.codes_set(sample_id, "jDirectionIncrementInDegrees", step)
    eccodes.codes_set_values(sample_id, [temp_k] * 9)

    buf = io.BytesIO()
    eccodes.codes_write(sample_id, buf)
    eccodes.codes_release(sample_id)
    return buf.getvalue()


def _grib2_available() -> bool:
    """Return True if eccodes GRIB2 samples are installed."""
    try:
        _make_minimal_grib2(40.0, -74.0, 295.0)
        return True
    except (eccodes.CodesInternalError, Exception):
        return False


requires_grib2 = pytest.mark.skipif(
    not _grib2_available(),
    reason="eccodes GRIB2 samples not available",
)


# ---------------------------------------------------------------------------
# _select_run_time
# ---------------------------------------------------------------------------

class TestSelectRunTime:
    def test_00z_available_at_0500(self):
        """At 05:00 UTC the 00Z run is ready (00Z + 3.5h = 03:30)."""
        now = datetime(2026, 3, 20, 5, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 0
        assert run_dt.date() == date(2026, 3, 20)

    def test_06z_not_ready_at_0700(self):
        """At 07:00 UTC the 06Z run is not yet ready (06Z + 3.5h = 09:30); falls back to 00Z."""
        now = datetime(2026, 3, 20, 7, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 0
        assert run_dt.date() == date(2026, 3, 20)

    def test_06z_available_at_1000(self):
        """At 10:00 UTC the 06Z run is ready (06Z + 3.5h = 09:30)."""
        now = datetime(2026, 3, 20, 10, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 6
        assert run_dt.date() == date(2026, 3, 20)

    def test_12z_available_at_1700(self):
        """At 17:00 UTC the 12Z run is ready (12Z + 3.5h = 15:30)."""
        now = datetime(2026, 3, 20, 17, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 12
        assert run_dt.date() == date(2026, 3, 20)

    def test_18z_available_at_2300(self):
        """At 23:00 UTC the 18Z run is ready (18Z + 3.5h = 21:30)."""
        now = datetime(2026, 3, 20, 23, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 18
        assert run_dt.date() == date(2026, 3, 20)

    def test_falls_back_to_previous_day_18z(self):
        """At 02:00 UTC today, 00Z is not ready; falls back to 18Z yesterday."""
        now = datetime(2026, 3, 20, 2, 0, tzinfo=UTC)
        run_dt, run_hour = _select_run_time(now)
        assert run_hour == 18
        assert run_dt.date() == date(2026, 3, 19)

    def test_result_is_always_a_6h_boundary(self):
        """run_hour is always 0, 6, 12, or 18."""
        for h in range(24):
            now = datetime(2026, 3, 20, h, 30, tzinfo=UTC)
            _, run_hour = _select_run_time(now)
            assert run_hour in (0, 6, 12, 18)


# ---------------------------------------------------------------------------
# _forecast_hour
# ---------------------------------------------------------------------------

class TestForecastHour:
    def test_next_day_from_00z(self):
        """18Z tomorrow from 00Z run = 42h."""
        run_dt = datetime(2026, 3, 20, 0, 0, tzinfo=UTC)
        assert _forecast_hour(run_dt, date(2026, 3, 21)) == 42

    def test_two_days_from_00z(self):
        run_dt = datetime(2026, 3, 20, 0, 0, tzinfo=UTC)
        assert _forecast_hour(run_dt, date(2026, 3, 22)) == 66

    def test_from_12z_run(self):
        """18Z tomorrow from 12Z run = 30h."""
        run_dt = datetime(2026, 3, 20, 12, 0, tzinfo=UTC)
        assert _forecast_hour(run_dt, date(2026, 3, 21)) == 30

    def test_result_is_multiple_of_6(self):
        run_dt = datetime(2026, 3, 20, 6, 0, tzinfo=UTC)
        fh = _forecast_hour(run_dt, date(2026, 3, 21))
        assert fh % 6 == 0


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------

class TestBuildUrl:
    def test_contains_member_file(self):
        run_dt = datetime(2026, 3, 20, 0, 0, tzinfo=UTC)
        url = _build_url(1, run_dt, 24)
        assert "gep01" in url
        assert "f024" in url
        assert "20260320" in url

    def test_contains_required_params(self):
        run_dt = datetime(2026, 3, 20, 0, 0, tzinfo=UTC)
        url = _build_url(1, run_dt, 24)
        assert "var_TMP=on" in url
        assert "lev_2_m_above_ground=on" in url

    def test_run_hour_zero_padded(self):
        run_dt = datetime(2026, 3, 20, 6, 0, tzinfo=UTC)
        url = _build_url(5, run_dt, 48)
        assert "t06z" in url
        assert "gep05" in url
        assert "f048" in url

    def test_member_30(self):
        run_dt = datetime(2026, 3, 20, 12, 0, tzinfo=UTC)
        url = _build_url(30, run_dt, 120)
        assert "gep30" in url
        assert "f120" in url


# ---------------------------------------------------------------------------
# _extract_temperature
# ---------------------------------------------------------------------------

class TestParseGrib2Temp:
    @requires_grib2
    def test_converts_kelvin_to_fahrenheit(self):
        temp_k = 295.15  # 71.6°F
        grib_bytes = _make_minimal_grib2(NYC.lat, NYC.lon, temp_k)
        result = _extract_temperature(grib_bytes, NYC.lat, NYC.lon)
        assert result is not None
        expected_f = (temp_k - 273.15) * 9 / 5 + 32
        assert abs(result - expected_f) < 1.0
        assert 0 < result < 150

    def test_empty_bytes_returns_none(self):
        assert _extract_temperature(b"", NYC.lat, NYC.lon) is None

    def test_invalid_bytes_returns_none(self):
        assert _extract_temperature(b"not grib data at all!!", NYC.lat, NYC.lon) is None

    @requires_grib2
    def test_freezing_temperature(self):
        temp_k = 273.15  # 32°F
        grib_bytes = _make_minimal_grib2(NYC.lat, NYC.lon, temp_k)
        result = _extract_temperature(grib_bytes, NYC.lat, NYC.lon)
        assert result is not None
        assert abs(result - 32.0) < 1.0

    @requires_grib2
    def test_hot_temperature(self):
        temp_k = 308.15  # ~95°F
        grib_bytes = _make_minimal_grib2(NYC.lat, NYC.lon, temp_k)
        result = _extract_temperature(grib_bytes, NYC.lat, NYC.lon)
        assert result is not None
        assert abs(result - 95.0) < 1.0


# ---------------------------------------------------------------------------
# fetch_ensemble_noaa (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchEnsembleNoaa:
    @requires_grib2
    @pytest.mark.asyncio
    async def test_returns_30_member_keys(self):
        """When all members succeed, result has 30 temperature_2m_max_memberNN keys."""
        grib_bytes = _make_minimal_grib2(NYC.lat, NYC.lon, 295.15)
        mock_resp = httpx.Response(
            200,
            content=grib_bytes,
            request=httpx.Request("GET", "https://example.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "wedge.weather.noaa_client._select_run_time",
            return_value=(datetime(2026, 3, 20, 0, 0, tzinfo=UTC), 0),
        ):
            result = await fetch_ensemble_noaa(mock_client, NYC, date(2026, 3, 22))

        assert result is not None
        daily = result["daily"]
        for i in range(1, 31):
            key = f"temperature_2m_max_member{i:02d}"
            assert key in daily
            assert isinstance(daily[key], list)
            assert len(daily[key]) == 1
            val = daily[key][0]
            if val is not None:
                assert 0 < val < 150

    @pytest.mark.asyncio
    async def test_returns_none_when_all_members_fail(self):
        """Returns None when every HTTP request fails."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with patch(
            "wedge.weather.noaa_client._select_run_time",
            return_value=(datetime(2026, 3, 20, 0, 0, tzinfo=UTC), 0),
        ):
            result = await fetch_ensemble_noaa(mock_client, NYC, date(2026, 3, 22))
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_past_target_date(self):
        """Returns None when forecast hour would be < 6 (target in the past relative to run)."""
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        # Run time is after target date
        with patch(
            "wedge.weather.noaa_client._select_run_time",
            return_value=(datetime(2026, 3, 22, 12, 0, tzinfo=UTC), 12),
        ):
            result = await fetch_ensemble_noaa(mock_client, NYC, date(2026, 3, 20))
        assert result is None

    @requires_grib2
    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_result(self):
        """If some members fail, still returns a result with at least one valid member."""
        grib_bytes = _make_minimal_grib2(NYC.lat, NYC.lon, 295.15)
        call_count = 0

        async def _flaky_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count % 5 == 0:  # every 5th request fails
                raise httpx.TimeoutException("timeout")
            return httpx.Response(
                200,
                content=grib_bytes,
                request=httpx.Request("GET", "https://example.com"),
            )

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = _flaky_get

        with patch(
            "wedge.weather.noaa_client._select_run_time",
            return_value=(datetime(2026, 3, 20, 0, 0, tzinfo=UTC), 0),
        ):
            result = await fetch_ensemble_noaa(mock_client, NYC, date(2026, 3, 22))

        assert result is not None
        assert "daily" in result

    @pytest.mark.asyncio
    async def test_http_error_response_returns_none_for_member(self):
        """HTTP 404 responses cause individual members to be None (not crash)."""
        mock_resp = httpx.Response(
            404,
            content=b"not found",
            request=httpx.Request("GET", "https://example.com"),
        )
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "wedge.weather.noaa_client._select_run_time",
            return_value=(datetime(2026, 3, 20, 0, 0, tzinfo=UTC), 0),
        ):
            result = await fetch_ensemble_noaa(mock_client, NYC, date(2026, 3, 22))

        # All members got 404 → all None → should return None
        assert result is None


# ---------------------------------------------------------------------------
# fetch_ensemble_auto routing
# ---------------------------------------------------------------------------

class TestFetchEnsembleAuto:
    @pytest.mark.asyncio
    async def test_routes_to_openmeteo(self):
        from wedge.config import Settings
        from wedge.weather.client import fetch_ensemble_auto

        settings = Settings(weather_source="openmeteo")
        mock_client = AsyncMock(spec=httpx.AsyncClient)

        with patch(
            "wedge.weather.client.fetch_ensemble",
            new_callable=AsyncMock,
            return_value={"daily": {}},
        ) as mock_om:
            result = await fetch_ensemble_auto(
                mock_client, NYC, settings, date(2026, 3, 22)
            )

        mock_om.assert_called_once_with(mock_client, NYC)
        assert result == {"daily": {}}

    @pytest.mark.asyncio
    async def test_routes_to_noaa(self):
        from wedge.config import Settings
        from wedge.weather.client import fetch_ensemble_auto
        import wedge.weather.noaa_client as noaa_mod

        settings = Settings(weather_source="noaa")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        expected = {"daily": {"temperature_2m_max_member01": [71.0]}}

        original_fn = noaa_mod.fetch_ensemble_noaa

        async def _fake_noaa(client, city, target_date):
            return expected

        noaa_mod.fetch_ensemble_noaa = _fake_noaa  # type: ignore[attr-defined]
        try:
            result = await fetch_ensemble_auto(
                mock_client, NYC, settings, date(2026, 3, 22)
            )
        finally:
            noaa_mod.fetch_ensemble_noaa = original_fn

        assert result == expected

    @pytest.mark.asyncio
    async def test_noaa_uses_today_when_target_date_none(self):
        """When target_date is None and source is noaa, uses today."""
        from wedge.config import Settings
        from wedge.weather.client import fetch_ensemble_auto
        import wedge.weather.noaa_client as noaa_mod

        settings = Settings(weather_source="noaa")
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        captured = {}

        original_fn = noaa_mod.fetch_ensemble_noaa

        async def _capture(client, city, target_date):
            captured["target_date"] = target_date
            return None

        noaa_mod.fetch_ensemble_noaa = _capture  # type: ignore[attr-defined]
        try:
            await fetch_ensemble_auto(mock_client, NYC, settings, None)
        finally:
            noaa_mod.fetch_ensemble_noaa = original_fn

        assert "target_date" in captured
        assert isinstance(captured["target_date"], date)
