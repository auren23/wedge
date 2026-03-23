from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wedge.db import Database


@pytest.mark.asyncio
async def test_cycle_marker_claim_is_persistent_across_reconnect(tmp_path):
    db_path = tmp_path / "phase2.db"
    claimed_at = datetime.now(UTC).isoformat()

    db = Database(str(db_path))
    await db.connect()
    try:
        claimed = await db.claim_cycle_marker(
            "gefs:20260320:00",
            trigger_mode="active",
            status="claimed",
            run_id="run-1",
            created_at=claimed_at,
        )
        assert claimed is True
    finally:
        await db.close()

    reopened = Database(str(db_path))
    await reopened.connect()
    try:
        marker = await reopened.get_cycle_marker("gefs:20260320:00")
        assert marker is not None
        assert marker["cycle_key"] == "gefs:20260320:00"
        assert marker["trigger_mode"] == "active"
        assert marker["status"] == "claimed"
        assert marker["run_id"] == "run-1"

        duplicate = await reopened.claim_cycle_marker(
            "gefs:20260320:00",
            trigger_mode="fallback",
            status="claimed",
            run_id="run-2",
            created_at=claimed_at,
        )
        assert duplicate is False
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_cycle_marker_status_can_be_updated(tmp_path):
    db = Database(str(tmp_path / "phase2.db"))
    await db.connect()
    try:
        now = datetime.now(UTC).isoformat()
        await db.claim_cycle_marker(
            "gefs:20260320:06",
            trigger_mode="active",
            status="claimed",
            run_id="run-1",
            created_at=now,
        )

        await db.update_cycle_marker_status(
            "gefs:20260320:06",
            status="completed",
            updated_at=now,
        )

        marker = await db.get_cycle_marker("gefs:20260320:06")
        assert marker is not None
        assert marker["status"] == "completed"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_insert_forecasts_batch_persists_all_rows(tmp_path):
    db = Database(str(tmp_path / "phase2.db"))
    await db.connect()
    try:
        run_id = "run-batch"
        await db.insert_run(run_id, datetime.now(UTC).isoformat())

        await db.insert_forecasts_batch(
            run_id=run_id,
            city="NYC",
            date="2026-03-20",
            buckets={70: 0.2, 71: 0.35, 72: 0.45},
            created_at=datetime.now(UTC).isoformat(),
        )

        cursor = await db.conn.execute(
            "SELECT temp_f, p_model FROM forecasts WHERE run_id=? ORDER BY temp_f",
            (run_id,),
        )
        rows = await cursor.fetchall()
        assert [(row["temp_f"], row["p_model"]) for row in rows] == [
            (70, 0.2),
            (71, 0.35),
            (72, 0.45),
        ]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_replace_market_discoveries_persists_watchlist_state(tmp_path):
    from datetime import date
    from wedge.market.models import MarketBucket
    db = Database(str(tmp_path / "phase2.db"))
    await db.connect()
    try:
        await db.replace_market_discoveries(
            run_id="run-watch",
            city="NYC",
            target_date="2026-03-20",
            buckets=[
                MarketBucket(
                    token_id="tok-75",
                    city="NYC",
                    date=date(2026, 3, 20),
                    temp_value=75,
                    temp_unit="F",
                    market_price=0.41,
                    implied_prob=0.41,
                    volume_24h=9_000.0,
                    open_interest=3_000.0,
                    contract_type="daily",
                    bid_price=0.40,
                    ask_price=0.42,
                    spread=0.02,
                    liquidity_score=12.5,
                    selected_for_watchlist=True,
                    watchlist_rank=1,
                    selection_reason="watchlist_top_k",
                    filter_reason=None,
                ),
                MarketBucket(
                    token_id="tok-74",
                    city="NYC",
                    date=date(2026, 3, 20),
                    temp_value=74,
                    temp_unit="F",
                    market_price=0.38,
                    implied_prob=0.38,
                    volume_24h=5_000.0,
                    open_interest=1_500.0,
                    contract_type="daily",
                    bid_price=0.31,
                    ask_price=0.45,
                    spread=0.14,
                    liquidity_score=8.5,
                    selected_for_watchlist=False,
                    watchlist_rank=None,
                    selection_reason="ranked_out",
                    filter_reason=None,
                ),
            ],
            discovered_at=datetime.now(UTC).isoformat(),
        )

        discoveries = await db.get_market_discoveries("NYC", "2026-03-20")

        assert len(discoveries) == 2
        assert discoveries[0]["temp_f"] == 75
        assert discoveries[0]["selected_for_watchlist"] == 1
        assert discoveries[0]["watchlist_rank"] == 1
        assert discoveries[1]["temp_f"] == 74
        assert discoveries[1]["selected_for_watchlist"] == 0
        assert discoveries[1]["watchlist_rank"] is None
        assert discoveries[0]["bid_price"] == 0.40
        assert discoveries[0]["ask_price"] == 0.42
        assert discoveries[0]["spread"] == 0.02
        assert discoveries[1]["spread"] == 0.14
        assert discoveries[0]["selection_reason"] == "watchlist_top_k"
        assert discoveries[0]["filter_reason"] is None
        assert discoveries[1]["selection_reason"] == "ranked_out"
    finally:
        await db.close()