"""Full coverage tests for wedge.pipeline."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from wedge.config import CityConfig, Settings
from wedge.db import Database
from wedge.execution.models import OrderResult
from wedge.market.models import MarketBucket, Position
from wedge.strategy.models import EdgeSignal
from wedge.weather.models import ForecastDistribution

# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def db(tmp_path):
    d = Database(str(tmp_path / "test.db"))
    await d.connect()
    yield d
    await d.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        mode="dry_run",
        bankroll=1000.0,
        max_bet=50.0,
        cities=[CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")],
        db_path=str(tmp_path / "test.db"),
    )


@pytest.fixture
def city_cfg():
    return CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")


@pytest.fixture
def forecast():
    return ForecastDistribution(
        city="NYC",
        date=date(2026, 3, 20),
        buckets={72: 0.30, 74: 0.45, 76: 0.25},
        ensemble_spread=2.5,
        member_count=51,
        updated_at=datetime.now(UTC),
    )


@pytest.fixture
def market_bucket(forecast):
    return MarketBucket(
        token_id="syn_NYC_2026-03-20_72",
        city="NYC",
        date=date(2026, 3, 20),
        temp_value=72,
        temp_unit="F",
        market_price=0.20,
        implied_prob=0.20,
    )


@pytest.fixture
def edge_signal(market_bucket):
    return EdgeSignal(
        city="NYC",
        date=date(2026, 3, 20),
        temp_value=72,
        temp_unit="F",
        token_id=market_bucket.token_id,
        p_model=0.30,
        p_market=0.20,
        edge=0.10,
        odds=4.0,
    )


@pytest.fixture
def position(market_bucket, edge_signal):
    return Position(
        bucket=market_bucket,
        size=10.0,
        entry_price=0.20,
        strategy="ladder",
        p_model=edge_signal.p_model,
        edge=edge_signal.edge,
    )


# ── _generate_synthetic_markets ────────────────────────────────────────────────


class TestGenerateSyntheticMarkets:
    def test_returns_one_bucket_per_forecast_bucket(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        markets = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        assert len(markets) == len(forecast.buckets)

    def test_token_id_format(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        markets = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        for m in markets:
            assert m.token_id.startswith("syn_NYC_2026-03-20_")

    def test_market_price_clamped(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        markets = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        for m in markets:
            assert 0.01 <= m.market_price <= 0.99

    def test_seeded_reproducible(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        m1 = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        m2 = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        prices1 = [m.market_price for m in m1]
        prices2 = [m.market_price for m in m2]
        assert prices1 == prices2

    def test_different_city_different_prices(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        m_nyc = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        m_chi = _generate_synthetic_markets(forecast, "CHI", date(2026, 3, 20))
        prices_nyc = [m.market_price for m in m_nyc]
        prices_chi = [m.market_price for m in m_chi]
        assert prices_nyc != prices_chi

    def test_noise_within_bounds(self, forecast):
        """Market price deviates by noise in [-0.05, 0.03]; after clamp still valid."""
        from wedge.pipeline import _generate_synthetic_markets

        markets = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        for m in markets:
            assert 0.01 <= m.implied_prob <= 0.99

    def test_implied_prob_equals_market_price(self, forecast):
        from wedge.pipeline import _generate_synthetic_markets

        markets = _generate_synthetic_markets(forecast, "NYC", date(2026, 3, 20))
        for m in markets:
            assert m.implied_prob == m.market_price


# ── _process_city ──────────────────────────────────────────────────────────────


class TestProcessCity:
    """Tests for _process_city via mocked dependencies."""

    def _make_executor(self, success=True):
        ex = AsyncMock()
        ex.place_order.return_value = OrderResult(success=success, order_id="ord1")
        return ex

    async def _call(self, db, settings, forecast, executor, city_cfg=None, **overrides):
        from wedge.pipeline import _process_city

        if city_cfg is None:
            city_cfg = CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")
        import httpx

        async with httpx.AsyncClient() as http_client:
            return await _process_city(
                http_client=http_client,
                settings=settings,
                db=db,
                executor=executor,
                city_cfg=city_cfg,
                target_date=date(2026, 3, 20),
                run_id="testrun",
                ladder_budget=700.0,
                **overrides,
            )

    @pytest.mark.asyncio
    async def test_no_weather_data_returns_zero(self, db, settings, forecast):
        executor = self._make_executor()
        with patch("wedge.pipeline.fetch_ensemble", return_value=None):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 0
        executor.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_distribution_returns_zero(self, db, settings, forecast):
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=None),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 0

    @pytest.mark.asyncio
    async def test_dry_run_generates_synthetic_markets(self, db, settings, forecast, position):
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline.detect_edges", return_value=[MagicMock()]),
            patch("wedge.pipeline.evaluate_ladder", return_value=[position]),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 1
        executor.place_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_markets_returns_zero(self, db, settings, forecast):
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline._generate_synthetic_markets", return_value=[]),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 0

    @pytest.mark.asyncio
    async def test_no_edges_returns_zero(self, db, settings, forecast, market_bucket):
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline._generate_synthetic_markets", return_value=[market_bucket]),
            patch("wedge.pipeline.detect_edges", return_value=[]),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 0

    @pytest.mark.asyncio
    async def test_failed_order_not_counted(self, db, settings, forecast, position):
        executor = self._make_executor(success=False)
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline.detect_edges", return_value=[MagicMock()]),
            patch("wedge.pipeline.evaluate_ladder", return_value=[position]),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 0

    @pytest.mark.asyncio
    async def test_live_mode_with_poly_client(self, db, forecast, market_bucket, position):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            cities=[
                CityConfig(
                    name="NYC",
                    lat=40.77,
                    lon=-73.87,
                    timezone="America/New_York",
                )
            ],
        )
        executor = self._make_executor()
        poly_client = MagicMock()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch(
                "wedge.pipeline.discover_weather_markets",
                new_callable=AsyncMock,
                return_value=([market_bucket], []),
            ),
            patch("wedge.pipeline.detect_edges", return_value=[MagicMock()]),
            patch("wedge.pipeline.evaluate_ladder", return_value=[position]),
        ):
            orders = await self._call(
                db, settings, forecast, executor, poly_client=poly_client
            )
        assert orders == 1

    @pytest.mark.asyncio
    async def test_live_mode_uses_configured_market_liquidity_params(self, db, forecast, market_bucket, position):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            market_min_volume=2345.0,
            market_min_open_interest=3456.0,
            market_max_spread=0.07,
            slippage_bet_size=12.5,
            cities=[
                CityConfig(
                    name="NYC",
                    lat=40.77,
                    lon=-73.87,
                    timezone="America/New_York",
                )
            ],
        )
        executor = self._make_executor()
        poly_client = MagicMock()
        raw = {"some": "data"}

        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch(
                "wedge.pipeline.discover_weather_markets",
                new_callable=AsyncMock,
                return_value=([market_bucket], []),
            ) as mock_scan,
            patch("wedge.pipeline.detect_edges", return_value=[MagicMock()]) as mock_detect,
            patch("wedge.pipeline.evaluate_ladder", return_value=[position]),
        ):
            orders = await self._call(
                db, settings, forecast, executor, poly_client=poly_client
            )

        assert orders == 1
        mock_scan.assert_awaited_once_with(
            poly_client,
            "NYC",
            date(2026, 3, 20),
            min_volume=2345.0,
            min_open_interest=3456.0,
            max_spread=0.07,
        )
        assert mock_detect.call_count == 1
        assert mock_detect.call_args.kwargs["slippage_bet_size"] == 12.5

    @pytest.mark.asyncio
    async def test_live_mode_without_poly_client_no_markets(self, db, forecast):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            cities=[CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")],
        )
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
        ):
            orders = await self._call(db, settings, forecast, executor, poly_client=None)
        assert orders == 0

    @pytest.mark.asyncio
    async def test_forecasts_inserted_into_db(self, db, settings, forecast, market_bucket):
        executor = self._make_executor()
        raw = {"some": "data"}
        await db.insert_run("testrun", datetime.now(UTC).isoformat())
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline._generate_synthetic_markets", return_value=[]),
        ):
            await self._call(db, settings, forecast, executor)
        cursor = await db.conn.execute("SELECT COUNT(*) FROM forecasts WHERE run_id='testrun'")
        row = await cursor.fetchone()
        assert row[0] == len(forecast.buckets)

    @pytest.mark.asyncio
    async def test_multiple_positions_all_counted(self, db, settings, forecast, position):
        executor = self._make_executor(success=True)
        pos2 = position.model_copy(
            update={
                "bucket": position.bucket.model_copy(update={"temp_value": 74, "token_id": "tok74"})
            }
        )
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline.detect_edges", return_value=[MagicMock()]),
            patch("wedge.pipeline.evaluate_ladder", return_value=[position, pos2]),
        ):
            orders = await self._call(db, settings, forecast, executor)
        assert orders == 2


    @pytest.mark.asyncio
    async def test_live_mode_limits_edge_detection_to_watchlist(self, db, forecast):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            market_watchlist_size=1,
            cities=[CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")],
        )
        executor = self._make_executor()
        poly_client = MagicMock()
        raw = {"some": "data"}
        all_markets = [
            MarketBucket(
                token_id="tok-74",
                city="NYC",
                date=date(2026, 3, 20),
                temp_value=74,
                temp_unit="F",
                market_price=0.21,
                implied_prob=0.21,
                volume_24h=4_000.0,
                open_interest=800.0,
                contract_type="daily",
            ),
            MarketBucket(
                token_id="tok-75",
                city="NYC",
                date=date(2026, 3, 20),
                temp_value=75,
                temp_unit="F",
                market_price=0.24,
                implied_prob=0.24,
                volume_24h=12_000.0,
                open_interest=4_000.0,
                contract_type="daily",
            ),
        ]

        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch(
                "wedge.pipeline.discover_weather_markets",
                new_callable=AsyncMock,
                return_value=(all_markets, []),
            ),
            patch("wedge.pipeline.detect_edges", return_value=[] ) as mock_detect,
        ):
            orders = await self._call(db, settings, forecast, executor, poly_client=poly_client)

        assert orders == 0
        ranked_markets = mock_detect.call_args.args[1]
        assert len(ranked_markets) == 1
        assert ranked_markets[0].temp_value == 75
        discoveries = await db.get_market_discoveries("NYC", "2026-03-20")
        assert len(discoveries) == 2
        assert discoveries[0]["temp_f"] == 75
        assert discoveries[0]["selected_for_watchlist"] == 1
        assert discoveries[0]["selection_reason"] == "watchlist_top_k"
        assert discoveries[1]["selected_for_watchlist"] == 0
        assert discoveries[1]["selection_reason"] == "ranked_out"

    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_live_mode_persists_rejected_discoveries_with_filter_reason(self, db, forecast):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            market_watchlist_size=1,
            cities=[CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")],
        )
        executor = self._make_executor()
        poly_client = MagicMock()
        raw = {"some": "data"}
        accepted_market = MarketBucket(
            token_id="tok-75",
            city="NYC",
            date=date(2026, 3, 20),
            temp_value=75,
            temp_unit="F",
            market_price=0.24,
            implied_prob=0.24,
            volume_24h=12_000.0,
            open_interest=4_000.0,
            contract_type="daily",
        )
        rejected_market = MarketBucket(
            token_id="tok-74",
            city="NYC",
            date=date(2026, 3, 20),
            temp_value=74,
            temp_unit="F",
            market_price=0.21,
            implied_prob=0.21,
            volume_24h=500.0,
            open_interest=200.0,
            contract_type="daily",
            filter_reason="low_volume",
        )

        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch(
                "wedge.pipeline.discover_weather_markets",
                new_callable=AsyncMock,
                return_value=([accepted_market], [rejected_market]),
            ),
            patch("wedge.pipeline.detect_edges", return_value=[]),
        ):
            orders = await self._call(db, settings, forecast, executor, poly_client=poly_client)

        assert orders == 0
        discoveries = await db.get_market_discoveries("NYC", "2026-03-20")
        assert len(discoveries) == 2
        assert discoveries[0]["selection_reason"] == "watchlist_top_k"
        assert discoveries[1]["filter_reason"] == "low_volume"


    async def test_dry_run_without_poly_client_does_not_write_market_discoveries(self, db, settings, forecast):
        executor = self._make_executor()
        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch("wedge.pipeline._generate_synthetic_markets", return_value=[]),
        ):
            orders = await self._call(db, settings, forecast, executor, poly_client=None)

        assert orders == 0
        discoveries = await db.get_market_discoveries("NYC", "2026-03-20")
        assert discoveries == []

# ── run_pipeline ───────────────────────────────────────────────────────────────


class TestRunPipeline:
    @pytest.mark.asyncio
    async def test_dry_run_creates_dry_run_executor(self, db, settings):
        from wedge.pipeline import run_pipeline

        with (
            patch("wedge.pipeline._process_city", new_callable=AsyncMock, return_value=0),
            patch("wedge.pipeline.DryRunExecutor") as mock_exec_cls,
        ):
            mock_exec_instance = AsyncMock()
            mock_exec_instance.get_balance.return_value = 1000.0
            mock_exec_instance.get_unrealized_pnl.return_value = 0.0
            mock_exec_cls.return_value = mock_exec_instance
            await run_pipeline(settings, db)

        mock_exec_cls.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_mode_creates_live_executor(self, db, tmp_path):
        settings = Settings(
            mode="live",
            bankroll=1000.0,
            cities=[CityConfig(name="NYC", lat=40.77, lon=-73.87, timezone="America/New_York")],
            db_path=str(tmp_path / "test.db"),
            polymarket_private_key="test_key",
            polymarket_api_key="test_api_key",
            polymarket_api_secret="test_api_secret",
        )
        from wedge.pipeline import run_pipeline

        mock_poly = AsyncMock()
        mock_exec = AsyncMock()
        mock_exec.get_balance.return_value = 1000.0
        mock_exec.get_unrealized_pnl.return_value = 0.0

        with (
            patch("wedge.pipeline.PolymarketClient", return_value=mock_poly),
            patch("wedge.pipeline.LiveExecutor", return_value=mock_exec),
            patch("wedge.pipeline._process_city", new_callable=AsyncMock, return_value=0),
        ):
            await run_pipeline(settings, db)

        mock_poly.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_inserts_run_and_snapshot(self, db, settings):
        from wedge.pipeline import run_pipeline

        mock_exec = AsyncMock()
        mock_exec.get_balance.return_value = 999.0
        mock_exec.get_unrealized_pnl.return_value = 0.0

        with (
            patch("wedge.pipeline.DryRunExecutor", return_value=mock_exec),
            patch("wedge.pipeline._process_city", new_callable=AsyncMock, return_value=2),
        ):
            await run_pipeline(settings, db)

        cursor = await db.conn.execute("SELECT COUNT(*) FROM runs")
        row = await cursor.fetchone()
        assert row[0] == 1

        cursor = await db.conn.execute("SELECT balance FROM bankroll_snapshots")
        row = await cursor.fetchone()
        assert row is not None
        assert abs(row[0] - 999.0) < 0.01

    @pytest.mark.asyncio
    async def test_city_exception_caught_continues(self, db, settings):
        from wedge.pipeline import run_pipeline

        mock_exec = AsyncMock()
        mock_exec.get_balance.return_value = 1000.0
        mock_exec.get_unrealized_pnl.return_value = 0.0

        call_count = 0

        async def boom(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        with (
            patch("wedge.pipeline.DryRunExecutor", return_value=mock_exec),
            patch("wedge.pipeline._process_city", side_effect=boom),
        ):
            # Should not raise despite city exception
            await run_pipeline(settings, db)

        assert call_count == len(settings.cities)

    @pytest.mark.asyncio
    async def test_no_notifier_no_error(self, db, settings):
        from wedge.pipeline import run_pipeline

        mock_exec = AsyncMock()
        mock_exec.get_balance.return_value = 1000.0
        mock_exec.get_unrealized_pnl.return_value = 0.0

        with (
            patch("wedge.pipeline.DryRunExecutor", return_value=mock_exec),
            patch("wedge.pipeline._process_city", new_callable=AsyncMock, return_value=0),
        ):
            await run_pipeline(settings, db)
    @pytest.mark.asyncio
    async def test_balance_restored_from_db(self, db, settings):
        """get_last_balance is called with bankroll as default."""
        from wedge.pipeline import run_pipeline

        mock_exec = AsyncMock()
        mock_exec.get_balance.return_value = 850.0
        mock_exec.get_unrealized_pnl.return_value = 0.0

        with (
            patch("wedge.pipeline.DryRunExecutor", return_value=mock_exec),
            patch("wedge.pipeline._process_city", new_callable=AsyncMock, return_value=0),
        ):
            await run_pipeline(settings, db)

        # Snapshot balance should be whatever executor returns
        cursor = await db.conn.execute("SELECT balance FROM bankroll_snapshots")
        row = await cursor.fetchone()
        assert abs(row[0] - 850.0) < 0.01


# ── run_single_scan ────────────────────────────────────────────────────────────


class TestRunSingleScan:
    @pytest.mark.asyncio
    async def test_invalid_city_returns_early(self, settings):
        from wedge.pipeline import run_single_scan

        with patch("wedge.pipeline.fetch_ensemble") as mock_fetch:
            await run_single_scan(settings, "NONEXISTENT")
        mock_fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_weather_data_returns_early(self, settings):
        from wedge.pipeline import run_single_scan

        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=None),
            patch("wedge.pipeline.parse_distribution") as mock_parse,
        ):
            await run_single_scan(settings, "NYC")
        mock_parse.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_distribution_returns_early(self, settings):
        from wedge.pipeline import run_single_scan

        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=None),
        ):
            await run_single_scan(settings, "NYC")

    @pytest.mark.asyncio
    async def test_successful_scan_logs_buckets(self, settings, forecast):
        from wedge.pipeline import run_single_scan

        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
        ):
            # Should complete without exception
            await run_single_scan(settings, "NYC")

    @pytest.mark.asyncio
    async def test_city_lookup_case_insensitive(self, settings, forecast):
        from wedge.pipeline import run_single_scan

        raw = {"some": "data"}
        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
        ):
            await run_single_scan(settings, "nyc")

    @pytest.mark.asyncio
    async def test_target_date_is_3_days_ahead(self, settings, forecast):
        """Target date passed to parse_distribution is local today + 3 days."""
        from wedge.pipeline import run_single_scan

        raw = {"some": "data"}
        captured_dates = []

        def capture_parse(raw_data, city, target_date):
            captured_dates.append(target_date)
            return forecast

        with (
            patch("wedge.pipeline.fetch_ensemble", return_value=raw),
            patch("wedge.pipeline.parse_distribution", side_effect=capture_parse),
        ):
            await run_single_scan(settings, "NYC")

        assert len(captured_dates) == 1
        from zoneinfo import ZoneInfo

        tz = ZoneInfo("America/New_York")
        expected = (datetime.now(tz) + timedelta(days=3)).date()
        assert captured_dates[0] == expected
