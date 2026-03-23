from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, patch

import pytest

from wedge.config import Settings


class TestSettingsLadderOnly:
    def test_settings_load_ignores_legacy_tail_and_telegram_fields(self):
        settings = Settings.load(
            mode="dry_run",
            bankroll=1000.0,
            ladder_edge=0.08,
            ladder_alloc=0.90,
            tail_edge=0.08,
            telegram_token="legacy-token",
        )

        assert settings.ladder_alloc == 0.90
        assert not hasattr(settings, "tail_edge")
        assert not hasattr(settings, "telegram_token")



    def test_settings_load_accepts_market_liquidity_overrides(self):
        settings = Settings.load(
            mode="dry_run",
            market_min_volume=2000.0,
            slippage_bet_size=25.0,
        )

        assert settings.market_min_volume == 2000.0
        assert settings.slippage_bet_size == 25.0

class TestCliLadderOnly:
    def test_run_command_exposes_only_ladder_options(self):
        from typer.main import get_command

        from wedge.cli import app

        cmd = get_command(app)
        run_cmd = cmd.commands["run"]
        param_names = {param.name for param in run_cmd.params}

        assert "tail_edge" not in param_names
        assert "telegram" not in param_names
        assert "ladder_edge" in param_names

    def test_calibration_command_removed(self):
        from typer.main import get_command

        from wedge.cli import app

        cmd = get_command(app)
        assert "calibration" not in cmd.commands


@pytest.mark.asyncio
class TestPipelineLadderOnly:
    async def test_process_city_places_only_ladder_orders(self, tmp_path):  # noqa: PLR0915
        from wedge.config import CityConfig, Settings
        from wedge.db import Database
        from wedge.market.models import MarketBucket
        from wedge.pipeline import _process_city
        from wedge.weather.models import ForecastDistribution

        city = CityConfig(
            name="NYC",
            lat=40.7772,
            lon=-73.8726,
            timezone="America/New_York",
            station="KLGA",
        )
        settings = Settings(mode="dry_run", db_path=str(tmp_path / "test.db"))
        db = Database(settings.db_path)
        await db.connect()

        forecast = ForecastDistribution(
            city="NYC",
            date=date(2026, 7, 1),
            buckets={80: 0.35, 81: 0.30, 82: 0.20, 83: 0.15},
            ensemble_spread=1.5,
            member_count=21,
            updated_at=datetime.now(UTC),
        )
        markets = [
            MarketBucket(
                token_id="tok80",
                city="NYC",
                date=date(2026, 7, 1),
                temp_value=80,
                temp_unit="F",
                market_price=0.20,
                implied_prob=0.20,
            )
        ]
        executor = AsyncMock()
        executor.place_order = AsyncMock(return_value=type("R", (), {"success": True})())

        with (
            patch(
                "wedge.pipeline.fetch_ensemble",
                new_callable=AsyncMock,
                return_value={"raw": True},
            ),
            patch("wedge.pipeline.parse_distribution", return_value=forecast),
            patch(
                "wedge.pipeline.scan_weather_markets",
                new_callable=AsyncMock,
                return_value=markets,
            ),
        ):
            orders = await _process_city(
                http_client=AsyncMock(),
                settings=settings,
                db=db,
                executor=executor,
                city_cfg=city,
                target_date=date(2026, 7, 1),
                run_id="run1",
                ladder_budget=200.0,
                poly_client=AsyncMock(),
            )

        assert orders >= 0
        for call in executor.place_order.await_args_list:
            request = call.args[0]
            assert request.strategy == "ladder"

        await db.close()
