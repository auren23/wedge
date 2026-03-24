"""Test dry-run P&L calculations for binary options."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from wedge.db import Database
from wedge.execution.dry_run import DryRunExecutor
from wedge.execution.models import OrderRequest
from wedge.market.models import MarketBucket


@pytest.fixture
async def db():
    """Create in-memory test database."""
    db = Database(":memory:")
    await db.connect()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_unrealized_pnl_calculation(db):
    """Test unrealized P&L uses correct binary option formula."""
    executor = DryRunExecutor(db, initial_balance=1000.0, max_bet=100.0)

    # Place order: $100 at price 0.40
    request = OrderRequest(
        run_id="test_run",
        token_id="test_token",
        city="NYC",
        date=date(2026, 3, 20),
        temp_value=70,
        temp_unit="F",
        strategy="ladder",
        limit_price=0.40,
        size=100.0,
        p_model=0.50,
        p_market=0.40,
        edge=0.10,
    )

    result = await executor.place_order(request)
    assert result.success

    # Simulate price movement to 0.50
    markets = [
        MarketBucket(
            token_id="test_token",
            city="NYC",
            date=date(2026, 3, 20),
            temp_value=70,
            temp_unit="F",
            market_price=0.50,
            implied_prob=0.50,
        )
    ]
    await executor.update_position_prices(markets)

    # Calculate unrealized P&L
    pnl = await executor.get_unrealized_pnl()

    # Expected: shares = 100 / 0.40 = 250
    # Current value = 250 * 0.50 = 125
    # P&L = 125 - 100 = 25
    # Formula: 100 * (0.50 - 0.40) / 0.40 = 25
    assert abs(pnl - 25.0) < 0.01, f"Expected P&L ~25.0, got {pnl}"


@pytest.mark.asyncio
async def test_settled_pnl_with_fee(db):
    """Test settled P&L defaults to zero fees for weather markets."""
    # Insert a winning trade
    await db.insert_run("test_run", datetime.now(UTC).isoformat())
    await db.insert_trade(
        run_id="test_run",
        city="NYC",
        date="2026-03-20",
        temp_f=70,
        strategy="ladder",
        entry_price=0.40,
        size=100.0,
        p_model=0.50,
        p_market=0.40,
        edge=0.10,
        token_id="test_token",
        order_id="test_order",
        created_at=datetime.now(UTC).isoformat(),
    )

    # Settle with winning outcome
    count = await db.settle_trades("NYC", "2026-03-20", actual_temp=70)
    assert count == 1

    # Get P&L
    summary = await db.get_pnl_summary(days=30)

    # Expected: (1.0 - 0.40) * 100 / 0.40 = 150
    expected_pnl = 150.0
    assert abs(summary["total_pnl"] - expected_pnl) < 0.01, (
        f"Expected P&L ~{expected_pnl}, got {summary['total_pnl']}"
    )


@pytest.mark.asyncio
async def test_settled_pnl_losing_trade_no_fee(db):
    """Test losing trades don't pay fees."""
    # Insert a losing trade
    await db.insert_run("test_run", datetime.now(UTC).isoformat())
    await db.insert_trade(
        run_id="test_run",
        city="NYC",
        date="2026-03-20",
        temp_f=70,
        strategy="ladder",
        entry_price=0.40,
        size=100.0,
        p_model=0.50,
        p_market=0.40,
        edge=0.10,
        token_id="test_token",
        order_id="test_order",
        created_at=datetime.now(UTC).isoformat(),
    )

    # Settle with losing outcome (different temp)
    count = await db.settle_trades("NYC", "2026-03-20", actual_temp=75)
    assert count == 1

    # Get P&L
    summary = await db.get_pnl_summary(days=30)

    # Expected: (0.0 - 0.40) * 100 / 0.40 = -100
    # No fee on losses
    expected_pnl = -100.0
    assert abs(summary["total_pnl"] - expected_pnl) < 0.01, (
        f"Expected P&L ~{expected_pnl}, got {summary['total_pnl']}"
    )


@pytest.mark.asyncio
async def test_position_persistence_across_runs(db):
    """Test positions are loaded from database across pipeline runs."""
    # First run: place order
    executor1 = DryRunExecutor(db, initial_balance=1000.0, max_bet=100.0)

    request = OrderRequest(
        run_id="run1",
        token_id="test_token",
        city="NYC",
        date=date(2026, 3, 20),
        temp_value=70,
        temp_unit="F",
        strategy="ladder",
        limit_price=0.40,
        size=100.0,
        p_model=0.50,
        p_market=0.40,
        edge=0.10,
    )

    await db.insert_run("run1", datetime.now(UTC).isoformat())
    result = await executor1.place_order(request)
    assert result.success

    # Second run: new executor should load positions from DB
    executor2 = DryRunExecutor(db, initial_balance=900.0, max_bet=100.0)

    # Update prices
    markets = [
        MarketBucket(
            token_id="test_token",
            city="NYC",
            date=date(2026, 3, 20),
            temp_value=70,
            temp_unit="F",
            market_price=0.50,
            implied_prob=0.50,
        )
    ]
    await executor2.update_position_prices(markets)

    # Should calculate P&L from loaded positions
    pnl = await executor2.get_unrealized_pnl()
    assert abs(pnl - 25.0) < 0.01, f"Expected P&L ~25.0, got {pnl}"
