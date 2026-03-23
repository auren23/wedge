"""Full coverage tests for wedge.cli."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import patch

from typer.testing import CliRunner

from wedge.cli import app

runner = CliRunner()


def _close_coro(coro):
    coro.close()
    return None


class TestRunCommand:
    def test_dry_run_default_exits_ok(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_dry_run_flag(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--dry-run"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_live_flag_exits_ok(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--live"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_live_flag_passes_live_mode_to_scheduler(self):
        with (
            patch("wedge.scheduler.run_scheduler"),
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--live"])
        assert result.exit_code == 0

    def test_custom_bankroll(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--bankroll", "5000.0"])
        assert result.exit_code == 0

    def test_custom_max_bet(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--max-bet", "200.0"])
        assert result.exit_code == 0

    def test_custom_kelly(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--kelly", "0.20"])
        assert result.exit_code == 0

    def test_custom_ladder_edge(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["run", "--ladder-edge", "0.07"])
        assert result.exit_code == 0

    def test_run_calls_setup_logging(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging") as mock_logging,
        ):
            runner.invoke(app, ["run"])
        mock_logging.assert_called_once()


class TestScanCommand:
    def test_default_city_exits_ok(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["scan"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_custom_city(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["scan", "--city", "Chicago"])
        assert result.exit_code == 0

    def test_scan_calls_setup_logging(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging") as mock_logging,
        ):
            runner.invoke(app, ["scan"])
        mock_logging.assert_called_once()

    def test_scan_calls_asyncio_run(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            runner.invoke(app, ["scan"])
        assert mock_run.call_count == 1


class TestStatsCommand:
    def test_default_days_exits_ok(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["stats"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_custom_days(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["stats", "--days", "7"])
        assert result.exit_code == 0

    def test_stats_short_flag(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["stats", "-d", "14"])
        assert result.exit_code == 0

    def test_stats_calls_setup_logging(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro),
            patch("wedge.cli.setup_logging") as mock_logging,
        ):
            runner.invoke(app, ["stats"])
        mock_logging.assert_called_once()


class TestHelpText:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "wedge" in result.output.lower() or "weather" in result.output.lower()

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output or "dry" in result.output.lower()

    def test_scan_help(self):
        result = runner.invoke(app, ["scan", "--help"])
        assert result.exit_code == 0
        assert "--city" in result.output

    def test_stats_help(self):
        result = runner.invoke(app, ["stats", "--help"])
        assert result.exit_code == 0
        assert "--days" in result.output or "-d" in result.output


class TestWatchlistCommand:
    def test_watchlist_default_exits_ok(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(app, ["watchlist"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_watchlist_passes_filters(self):
        with (
            patch("wedge.cli.asyncio.run", side_effect=_close_coro) as mock_run,
            patch("wedge.cli.setup_logging"),
        ):
            result = runner.invoke(
                app,
                ["watchlist", "--city", "NYC", "--date", "2026-03-20", "--all"],
            )
        assert result.exit_code == 0
        mock_run.assert_called_once()

    def test_watchlist_help(self):
        result = runner.invoke(app, ["watchlist", "--help"])
        assert result.exit_code == 0
        assert "--city" in result.output
        assert "--date" in result.output
        assert "--all" in result.output
        assert "--json" in result.output

    def test_watchlist_outputs_rows(self, tmp_path, monkeypatch):
        from wedge.config import Settings
        from wedge.db import Database

        db_path = tmp_path / "watchlist.db"

        async def seed() -> None:
            db = Database(str(db_path))
            await db.connect()
            try:
                await db.insert_run("run-watch", datetime.now(UTC).isoformat())
                await db.replace_market_discoveries(
                    run_id="run-watch",
                    city="NYC",
                    target_date="2026-03-20",
                    buckets=[
                        __import__("wedge.market.models", fromlist=["MarketBucket"]).MarketBucket(
                            token_id="tok-75",
                            city="NYC",
                            date=date(2026, 3, 20),
                            temp_value=75,
                            temp_unit="F",
                            market_price=0.41,
                            implied_prob=0.41,
                            volume_24h=9000.0,
                            open_interest=3000.0,
                            bid_price=0.40,
                            ask_price=0.42,
                            spread=0.02,
                            contract_type="daily",
                            liquidity_score=12.5,
                            selected_for_watchlist=True,
                            watchlist_rank=1,
                            selection_reason="watchlist_top_k",
                            filter_reason=None,
                        ),
                        __import__("wedge.market.models", fromlist=["MarketBucket"]).MarketBucket(
                            token_id="tok-74",
                            city="NYC",
                            date=date(2026, 3, 20),
                            temp_value=74,
                            temp_unit="F",
                            market_price=0.20,
                            implied_prob=0.20,
                            volume_24h=500.0,
                            open_interest=200.0,
                            contract_type="daily",
                            filter_reason="low_volume",
                        )
                    ],
                    discovered_at=datetime.now(UTC).isoformat(),
                )
            finally:
                await db.close()

        import asyncio as _asyncio
        _asyncio.run(seed())

        monkeypatch.setattr("wedge.cli.Settings.load", lambda **_: Settings(db_path=str(db_path)))
        result = runner.invoke(app, ["watchlist", "--city", "NYC", "--date", "2026-03-20", "--all"] )

        assert result.exit_code == 0
        assert "tok-75" in result.output
        assert "NYC" in result.output
        assert "75F" in result.output
        assert "watchlist_top_k" in result.output
        assert "low_volume" in result.output
    def test_watchlist_json_outputs_structured_rows(self, tmp_path, monkeypatch):
        from wedge.config import Settings
        from wedge.db import Database

        db_path = tmp_path / "watchlist-json.db"

        async def seed() -> None:
            db = Database(str(db_path))
            await db.connect()
            try:
                await db.insert_run("run-watch", datetime.now(UTC).isoformat())
                await db.replace_market_discoveries(
                    run_id="run-watch",
                    city="NYC",
                    target_date="2026-03-20",
                    buckets=[
                        __import__("wedge.market.models", fromlist=["MarketBucket"]).MarketBucket(
                            token_id="tok-75",
                            city="NYC",
                            date=date(2026, 3, 20),
                            temp_value=75,
                            temp_unit="F",
                            market_price=0.41,
                            implied_prob=0.41,
                            volume_24h=9000.0,
                            open_interest=3000.0,
                            contract_type="daily",
                            selected_for_watchlist=True,
                            watchlist_rank=1,
                            selection_reason="watchlist_top_k",
                        )
                    ],
                    discovered_at=datetime.now(UTC).isoformat(),
                )
            finally:
                await db.close()

        import asyncio as _asyncio
        _asyncio.run(seed())

        monkeypatch.setattr("wedge.cli.Settings.load", lambda **_: Settings(db_path=str(db_path)))
        result = runner.invoke(app, ["watchlist", "--json"])

        assert result.exit_code == 0
        assert '"city": "NYC"' in result.output
        assert '"selection_reason": "watchlist_top_k"' in result.output
        assert '"selected_for_watchlist": true' in result.output
