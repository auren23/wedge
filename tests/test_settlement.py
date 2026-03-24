from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from wedge.config import Settings
from wedge.db import Database


@pytest.fixture
async def db(tmp_path):
    db = Database(str(tmp_path / "test.db"))
    await db.connect()
    yield db
    await db.close()


@pytest.fixture
def settings():
    return Settings(mode="dry_run", bankroll=1000.0)


def _insert_trade(db, run_id, city, trade_date, temp_f, strategy="ladder"):
    """Helper to insert a trade synchronously via the event loop."""
    return db.insert_trade(
        run_id=run_id,
        city=city,
        date=trade_date,
        temp_f=temp_f,
        strategy=strategy,
        entry_price=0.20,
        size=10.0,
        p_model=0.25,
        p_market=0.20,
        edge=0.05,
        created_at=datetime.now(UTC).isoformat(),
    )


class TestGetUnsettledDates:
    @pytest.mark.asyncio
    async def test_no_unsettled(self, db):
        result = await db.get_unsettled_dates()
        assert result == []

    @pytest.mark.asyncio
    async def test_future_dates_excluded(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2099-12-31", 78)
        result = await db.get_unsettled_dates()
        assert result == []

    @pytest.mark.asyncio
    async def test_past_dates_included(self, db):
        await db.insert_run("run1", "2026-01-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-01-01", 78)
        result = await db.get_unsettled_dates()
        assert ("NYC", "2026-01-01") in result

    @pytest.mark.asyncio
    async def test_settled_trades_excluded(self, db):
        await db.insert_run("run1", "2026-01-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-01-01", 78)
        await db.settle_trades("NYC", "2026-01-01", actual_temp=78)
        result = await db.get_unsettled_dates()
        assert result == []

    @pytest.mark.asyncio
    async def test_distinct_pairs(self, db):
        await db.insert_run("run1", "2026-01-01T00:00:00")
        await db.insert_run("run2", "2026-01-01T00:00:00")
        # Two trades same city/date
        await _insert_trade(db, "run1", "NYC", "2026-01-01", 78)
        await _insert_trade(db, "run2", "NYC", "2026-01-01", 79)
        result = await db.get_unsettled_dates()
        assert len(result) == 1
        assert result[0] == ("NYC", "2026-01-01")


class TestSettleTrades:
    @pytest.mark.asyncio
    async def test_winning_trade_pnl(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-07-01", 78)
        count = await db.settle_trades("NYC", "2026-07-01", actual_temp=78)
        assert count == 1

        # Verify pnl: outcome=1.0, pnl = (1.0 - 0.20) * 10.0 / 0.20 = 40.0
        # Weather settlement should default to zero fees.
        cursor = await db.conn.execute(
            "SELECT settled, outcome, pnl FROM trades WHERE run_id='run1'"
        )
        row = await cursor.fetchone()
        assert row["settled"] == 1
        assert row["outcome"] == 1.0
        assert abs(row["pnl"] - 40.0) < 1e-9

    @pytest.mark.asyncio
    async def test_losing_trade_pnl(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-07-01", 78)
        count = await db.settle_trades("NYC", "2026-07-01", actual_temp=80)
        assert count == 1

        cursor = await db.conn.execute(
            "SELECT settled, outcome, pnl FROM trades WHERE run_id='run1'"
        )
        row = await cursor.fetchone()
        assert row["settled"] == 1
        assert row["outcome"] == 0.0
        # pnl = (0.0 - 0.20) * 10.0 / 0.20 = -10.0
        assert abs(row["pnl"] - (-10.0)) < 1e-9

    @pytest.mark.asyncio
    async def test_already_settled_not_resettled(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-07-01", 78)
        count1 = await db.settle_trades("NYC", "2026-07-01", actual_temp=78)
        assert count1 == 1
        count2 = await db.settle_trades("NYC", "2026-07-01", actual_temp=78)
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_settle_multiple_trades(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await db.insert_run("run2", "2026-07-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-07-01", 78)
        await _insert_trade(db, "run2", "NYC", "2026-07-01", 79)
        count = await db.settle_trades("NYC", "2026-07-01", actual_temp=78)
        assert count == 2


class TestUpdateForecastActual:
    @pytest.mark.asyncio
    async def test_updates_actual_temp(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await db.insert_forecast(
            run_id="run1",
            city="NYC",
            date="2026-07-01",
            temp_f=78,
            p_model=0.25,
            created_at="2026-07-01T00:00:00",
        )
        await db.update_forecast_actual("NYC", "2026-07-01", 80)

        cursor = await db.conn.execute(
            "SELECT actual_temp_f FROM forecasts WHERE city='NYC' AND date='2026-07-01'"
        )
        row = await cursor.fetchone()
        assert row["actual_temp_f"] == 80


class TestBrierScoreAfterSettlement:
    @pytest.mark.asyncio
    async def test_brier_populated_after_settlement(self, db):
        # Before settlement, Brier is None
        brier = await db.get_brier_score(days=30)
        assert brier is None

        await db.insert_run("run1", "2026-07-01T00:00:00")
        await db.insert_forecast(
            run_id="run1",
            city="NYC",
            date="2026-07-01",
            temp_f=78,
            p_model=0.25,
            created_at=datetime.now(UTC).isoformat(),
        )
        await db.insert_forecast(
            run_id="run1",
            city="NYC",
            date="2026-07-01",
            temp_f=79,
            p_model=0.40,
            created_at=datetime.now(UTC).isoformat(),
        )

        # Settle: actual was 79
        await db.update_forecast_actual("NYC", "2026-07-01", 79)

        brier = await db.get_brier_score(days=30)
        assert brier is not None
        assert brier > 0
        # For temp_value=78, temp_unit="F", p_model=0.25: (0.25 - 0)^2 = 0.0625
        # For temp_value=79, temp_unit="F", p_model=0.40: (0.40 - 1)^2 = 0.36
        # avg = (0.0625 + 0.36) / 2 = 0.21125
        assert abs(brier - 0.21125) < 1e-4

    @pytest.mark.asyncio
    async def test_perfect_brier_score(self, db):
        await db.insert_run("run1", "2026-07-01T00:00:00")
        await db.insert_forecast(
            run_id="run1",
            city="NYC",
            date="2026-07-01",
            temp_f=78,
            p_model=1.0,
            created_at=datetime.now(UTC).isoformat(),
        )
        await db.update_forecast_actual("NYC", "2026-07-01", 78)
        brier = await db.get_brier_score(days=30)
        assert brier is not None
        assert abs(brier) < 1e-9  # Perfect prediction → 0


class TestRunSettlement:
    @pytest.mark.asyncio
    async def test_settlement_pipeline_end_to_end(self, db, settings):
        from wedge.pipeline import run_settlement

        await db.insert_run("run1", "2026-01-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-01-01", 78)
        await db.insert_forecast(
            run_id="run1",
            city="NYC",
            date="2026-01-01",
            temp_f=78,
            p_model=0.25,
            created_at=datetime.now(UTC).isoformat(),
        )


        with patch(
            "wedge.pipeline.fetch_actual_temperature",
            return_value=78,
        ):
            settled = await run_settlement(settings, db)

        assert settled == 1

        # Verify brier is now computed
        brier = await db.get_brier_score(days=365)
        assert brier is not None

    @pytest.mark.asyncio
    async def test_settlement_no_pending(self, db, settings):
        from wedge.pipeline import run_settlement

        settled = await run_settlement(settings, db)
        assert settled == 0

    @pytest.mark.asyncio
    async def test_settlement_api_failure_skips(self, db, settings):
        from wedge.pipeline import run_settlement

        await db.insert_run("run1", "2026-01-01T00:00:00")
        await _insert_trade(db, "run1", "NYC", "2026-01-01", 78)

        with patch(
            "wedge.pipeline.fetch_actual_temperature",
            return_value=None,
        ):
            settled = await run_settlement(settings, db)

        assert settled == 0
        # Trade still unsettled
        unsettled = await db.get_unsettled_dates()
        assert len(unsettled) == 1

    @pytest.mark.asyncio
    async def test_settlement_unknown_city_skipped(self, db):
        from wedge.pipeline import run_settlement

        settings = Settings(mode="dry_run", cities=[])  # no cities configured

        await db.insert_run("run1", "2026-01-01T00:00:00")
        await _insert_trade(db, "run1", "UnknownCity", "2026-01-01", 78)

        settled = await run_settlement(settings, db)
        assert settled == 0
