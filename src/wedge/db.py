from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from wedge.log import get_logger

log = get_logger("db")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    city TEXT NOT NULL,
    date TEXT NOT NULL,
    temp_f INTEGER NOT NULL,
    temp_unit TEXT NOT NULL DEFAULT 'F',
    strategy TEXT NOT NULL,
    entry_price REAL NOT NULL,
    size REAL NOT NULL,
    p_model REAL NOT NULL,
    p_market REAL NOT NULL,
    edge REAL NOT NULL,
    token_id TEXT,
    order_id TEXT,
    settled INTEGER DEFAULT 0,
    outcome REAL,
    pnl REAL,
    fee_applied REAL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, city, date, temp_f, strategy)
);

CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    city TEXT NOT NULL,
    date TEXT NOT NULL,
    temp_f INTEGER NOT NULL,
    p_model REAL NOT NULL,
    actual_temp_f INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bankroll_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    balance REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cycle_markers (
    cycle_key TEXT PRIMARY KEY,
    trigger_mode TEXT NOT NULL,
    status TEXT NOT NULL,
    run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
 );

CREATE TABLE IF NOT EXISTS exit_tiers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    city TEXT NOT NULL,
    date TEXT NOT NULL,
    temp_f INTEGER NOT NULL,
    tier_index INTEGER NOT NULL,  
    exit_price REAL NOT NULL,
    shares_sold REAL NOT NULL,
    pnl REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS market_discoveries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    city TEXT NOT NULL,
    date TEXT NOT NULL,
    temp_f INTEGER NOT NULL,
    temp_unit TEXT NOT NULL DEFAULT 'F',
    token_id TEXT,
    contract_type TEXT NOT NULL,
    market_price REAL NOT NULL,
    implied_prob REAL NOT NULL,
    volume_24h REAL NOT NULL,
    open_interest REAL NOT NULL,
    bid_price REAL,
    ask_price REAL,
    spread REAL,
    liquidity_score REAL NOT NULL DEFAULT 0,
    selected_for_watchlist INTEGER NOT NULL DEFAULT 0,
    watchlist_rank INTEGER,
    selection_reason TEXT,
    filter_reason TEXT,
    discovered_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exit_tiers_pos ON exit_tiers(city, date, temp_f);

CREATE INDEX IF NOT EXISTS idx_trades_run ON trades(run_id);
CREATE INDEX IF NOT EXISTS idx_trades_city_date ON trades(city, date);
CREATE INDEX IF NOT EXISTS idx_forecasts_city_date ON forecasts(city, date);
CREATE INDEX IF NOT EXISTS idx_trades_settled ON trades(settled);
CREATE INDEX IF NOT EXISTS idx_cycle_markers_status ON cycle_markers(status);
CREATE INDEX IF NOT EXISTS idx_market_discoveries_city_date ON market_discoveries(city, date);
CREATE INDEX IF NOT EXISTS idx_market_discoveries_watchlist ON market_discoveries(selected_for_watchlist, date);
"""


class Database:
    def __init__(self, db_path: str = "wedge.db") -> None:
        self._path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_SCHEMA)
        # Migration: add columns if not exists (for backward compatibility)
        for migration_sql in [
            "ALTER TABLE trades ADD COLUMN temp_unit TEXT NOT NULL DEFAULT 'F'",
            "ALTER TABLE trades ADD COLUMN exit_price REAL",
            "ALTER TABLE trades ADD COLUMN exit_reason TEXT",
            "ALTER TABLE trades ADD COLUMN settled_at TEXT",
            "ALTER TABLE trades ADD COLUMN peak_p_model REAL",
            "ALTER TABLE trades ADD COLUMN remaining_size REAL",
            "ALTER TABLE market_discoveries ADD COLUMN bid_price REAL",
            "ALTER TABLE market_discoveries ADD COLUMN ask_price REAL",
            "ALTER TABLE market_discoveries ADD COLUMN spread REAL",
            "ALTER TABLE market_discoveries ADD COLUMN selection_reason TEXT",
            "ALTER TABLE market_discoveries ADD COLUMN filter_reason TEXT",
        ]:
            try:
                await self._conn.execute(migration_sql)
                await self._conn.commit()
            except aiosqlite.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    pass
                else:
                    raise
        # Initialize peak_p_model and remaining_size for existing rows
        await self._conn.execute(
            "UPDATE trades SET peak_p_model = p_model WHERE peak_p_model IS NULL"
        )
        await self._conn.execute(
            "UPDATE trades SET remaining_size = size WHERE remaining_size IS NULL"
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if not self._conn:
            raise RuntimeError("Database not connected")
        return self._conn

    async def insert_run(self, run_id: str, started_at: str) -> None:
        await self.conn.execute(
            "INSERT INTO runs (id, started_at, status) VALUES (?, ?, 'running')",
            (run_id, started_at),
        )
        await self.conn.commit()

    async def complete_run(self, run_id: str, completed_at: str, status: str = "completed") -> None:
        await self.conn.execute(
            "UPDATE runs SET completed_at = ?, status = ? WHERE id = ?",
            (completed_at, status, run_id),
        )
        await self.conn.commit()

    async def insert_trade(
        self,
        *,
        run_id: str,
        city: str,
        date: str,
        temp_f: int,
        temp_unit: str = "F",
        strategy: str,
        entry_price: float,
        size: float,
        p_model: float,
        p_market: float,
        edge: float,
        token_id: str | None = None,
        order_id: str | None = None,
        created_at: str,
    ) -> bool:
        """Insert trade idempotently. Returns True if inserted, False if duplicate."""
        try:
            await self.conn.execute(
                """INSERT INTO trades
                   (run_id, city, date, temp_f, temp_unit, strategy, entry_price, size,
                    p_model, p_market, edge, token_id, order_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    city,
                    date,
                    temp_f,
                    temp_unit,
                    strategy,
                    entry_price,
                    size,
                    p_model,
                    p_market,
                    edge,
                    token_id,
                    order_id,
                    created_at,
                ),
            )
            await self.conn.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def delete_trade(
        self,
        *,
        run_id: str,
        city: str,
        date: str,
        temp_f: int,
        strategy: str,
    ) -> None:
        """Delete a trade row by its unique key (used to rollback failed live executions)."""
        await self.conn.execute(
            "DELETE FROM trades WHERE run_id=? AND city=? AND date=? AND temp_f=? AND strategy=?",
            (run_id, city, date, temp_f, strategy),
        )
        await self.conn.commit()

    async def insert_forecast(
        self,
        *,
        run_id: str,
        city: str,
        date: str,
        temp_f: int,
        p_model: float,
        created_at: str,
    ) -> None:
        await self.conn.execute(
            """INSERT INTO forecasts (run_id, city, date, temp_f, p_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, city, date, temp_f, p_model, created_at),
        )
        await self.conn.commit()

    async def insert_forecasts_batch(
        self,
        *,
        run_id: str,
        city: str,
        date: str,
        buckets: dict[int, float],
        created_at: str,
    ) -> None:
        rows = [
            (run_id, city, date, temp_f, p_model, created_at)
            for temp_f, p_model in buckets.items()
        ]
        await self.conn.executemany(
            """INSERT INTO forecasts (run_id, city, date, temp_f, p_model, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            rows,
        )
        await self.conn.commit()

    async def replace_market_discoveries(
        self,
        *,
        run_id: str,
        city: str,
        target_date: str,
        buckets: list,
        discovered_at: str,
    ) -> None:
        await self.conn.execute(
            "DELETE FROM market_discoveries WHERE city=? AND date=?",
            (city, target_date),
        )
        rows = [
            (
                run_id,
                city,
                target_date,
                bucket.temp_value,
                bucket.temp_unit,
                bucket.token_id,
                bucket.contract_type,
                bucket.market_price,
                bucket.implied_prob,
                bucket.volume_24h,
                bucket.open_interest,
                bucket.bid_price,
                bucket.ask_price,
                bucket.spread,
                bucket.liquidity_score,
                1 if bucket.selected_for_watchlist else 0,
                bucket.watchlist_rank,
                bucket.selection_reason,
                bucket.filter_reason,
                discovered_at,
            )
            for bucket in buckets
        ]
        if rows:
            await self.conn.executemany(
                """INSERT INTO market_discoveries (
                   run_id, city, date, temp_f, temp_unit, token_id, contract_type,
                   market_price, implied_prob, volume_24h, open_interest, bid_price, ask_price, spread,
                   liquidity_score, selected_for_watchlist, watchlist_rank, selection_reason, filter_reason, discovered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
        await self.conn.commit()

    async def get_market_discoveries(self, city: str, target_date: str) -> list[dict]:
        cursor = await self.conn.execute(
            """SELECT city, date, temp_f, temp_unit, token_id, contract_type, market_price, implied_prob,
                      volume_24h, open_interest, bid_price, ask_price, spread,
                      liquidity_score, selected_for_watchlist, watchlist_rank, selection_reason, filter_reason, discovered_at
               FROM market_discoveries
               WHERE city=? AND date=?
               ORDER BY selected_for_watchlist DESC,
                        CASE WHEN watchlist_rank IS NULL THEN 999999 ELSE watchlist_rank END,
                        liquidity_score DESC, temp_f""",
            (city, target_date),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def list_market_discoveries(
        self,
        *,
        city: str | None = None,
        target_date: str | None = None,
        include_all: bool = False,
    ) -> list[dict]:
        where = []
        params: list[str] = []
        if city is not None:
            where.append("city=?")
            params.append(city)
        if target_date is not None:
            where.append("date=?")
            params.append(target_date)
        if not include_all:
            where.append("selected_for_watchlist=1")

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        cursor = await self.conn.execute(
            f"""SELECT city, date, temp_f, temp_unit, token_id, contract_type, market_price, implied_prob,
                      volume_24h, open_interest, bid_price, ask_price, spread,
                      liquidity_score, selected_for_watchlist, watchlist_rank, selection_reason, filter_reason, discovered_at
               FROM market_discoveries
               {where_sql}
               ORDER BY date DESC, city ASC,
                        selected_for_watchlist DESC,
                        CASE WHEN watchlist_rank IS NULL THEN 999999 ELSE watchlist_rank END,
                        liquidity_score DESC, temp_f""",
            params,
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def claim_cycle_marker(
        self,
        cycle_key: str,
        *,
        trigger_mode: str,
        status: str,
        run_id: str | None,
        created_at: str,
    ) -> bool:
        cursor = await self.conn.execute(
            """INSERT OR IGNORE INTO cycle_markers
               (cycle_key, trigger_mode, status, run_id, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cycle_key, trigger_mode, status, run_id, created_at, created_at),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def get_cycle_marker(self, cycle_key: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT cycle_key, trigger_mode, status, run_id, created_at, updated_at "
            "FROM cycle_markers WHERE cycle_key=?",
            (cycle_key,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def update_cycle_marker_status(
        self,
        cycle_key: str,
        *,
        status: str,
        updated_at: str,
        run_id: str | None = None,
    ) -> None:
        if run_id is None:
            await self.conn.execute(
                "UPDATE cycle_markers SET status=?, updated_at=? WHERE cycle_key=?",
                (status, updated_at, cycle_key),
            )
        else:
            await self.conn.execute(
                "UPDATE cycle_markers SET status=?, run_id=?, updated_at=? WHERE cycle_key=?",
                (status, run_id, updated_at, cycle_key),
            )
        await self.conn.commit()

    async def insert_bankroll_snapshot(
        self, balance: float, unrealized_pnl: float, created_at: str
    ) -> None:
        await self.conn.execute(
            "INSERT INTO bankroll_snapshots (balance, unrealized_pnl, created_at) VALUES (?, ?, ?)",
            (balance, unrealized_pnl, created_at),
        )
        await self.conn.commit()

    async def settle_trades(
        self,
        city: str,
        date: str,
        actual_temp: int,
        fee_rate: float = 0.02,
    ) -> int:
        """Settle all unsettled trades for a city/date. Returns count settled.

        Applies fee on profits only (not on losses).

        Args:
            city: City name
            date: Date string (ISO format)
            actual_temp: Actual temperature in Fahrenheit
            fee_rate: Fee rate on profits (default 2% for Polymarket)
        """
        cursor = await self.conn.execute(
            "SELECT id, temp_f, entry_price, size, remaining_size FROM trades "
            "WHERE city=? AND date=? AND settled=0",
            (city, date),
        )
        rows = await cursor.fetchall()
        count = 0
        for row in rows:
            outcome = 1.0 if row["temp_f"] == actual_temp else 0.0
            # Use remaining_size if set, else size for partial exits
            remaining = row["remaining_size"] if row["remaining_size"] is not None else row["size"]
            # Binary option P&L: (outcome - entry_price) * remaining / entry_price
            pnl = (outcome - row["entry_price"]) * remaining / row["entry_price"]

            # Apply fee on profits only
            if pnl > 0:
                pnl *= 1.0 - fee_rate

            # Note: fee_applied column may not exist in older databases
            # Use separate UPDATE for backward compatibility
            await self.conn.execute(
                "UPDATE trades SET settled=1, outcome=?, pnl=? WHERE id=?",
                (outcome, pnl, row["id"]),
            )
            # Optionally track fee applied (ignore error if column doesn't exist)
            try:
                await self.conn.execute(
                    "UPDATE trades SET fee_applied=? WHERE id=?",
                    (fee_rate if pnl > 0 else 0.0, row["id"]),
                )
            except aiosqlite.OperationalError as e:
                # Backward-compat: older DBs may not have fee_applied column yet.
                msg = str(e).lower()
                if "no such column" in msg and "fee_applied" in msg:
                    pass
                else:
                    raise
            count += 1
        await self.conn.commit()
        return count

    async def reconcile_positions(
        self,
        remote_positions: list[dict],
        city: str | None = None,
    ) -> dict:
        """Reconcile local positions with remote (Polymarket) positions.

        Args:
            remote_positions: List of remote positions with keys:
                - city, date, temp_f, size, entry_price
            city: Optional city filter

        Returns:
            Reconciliation report with:
                - matched: count of matched positions
                - local_only: positions only in local DB
                - remote_only: positions only in remote
                - discrepancies: positions with different size/price
        """
        # Get local unsettled positions
        if city:
            cursor = await self.conn.execute(
                """SELECT city, date, temp_f, size, entry_price
                   FROM trades WHERE settled=0 AND city=?""",
                (city,),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT city, date, temp_f, size, entry_price FROM trades WHERE settled=0"
            )

        local_rows = await cursor.fetchall()
        local_positions = [
            {
                "city": row["city"],
                "date": row["date"],
                "temp_f": row["temp_f"],
                "size": row["size"],
                "entry_price": row["entry_price"],
            }
            for row in local_rows
        ]

        # Create lookup keys
        def make_key(pos: dict) -> tuple:
            return (pos["city"], pos["date"], pos["temp_f"])

        local_by_key = {make_key(p): p for p in local_positions}
        remote_by_key = {make_key(p): p for p in remote_positions}

        # Find matches and discrepancies
        matched = []
        local_only = []
        remote_only = []
        discrepancies = []

        for key, local_pos in local_by_key.items():
            if key not in remote_by_key:
                local_only.append(local_pos)
            else:
                remote_pos = remote_by_key[key]
                # Check for discrepancies
                size_diff = abs(local_pos["size"] - remote_pos.get("size", 0))
                price_diff = abs(local_pos["entry_price"] - remote_pos.get("entry_price", 0))

                if size_diff > 0.01 or price_diff > 0.001:
                    discrepancies.append(
                        {
                            "key": key,
                            "local": local_pos,
                            "remote": remote_pos,
                            "size_diff": size_diff,
                            "price_diff": price_diff,
                        }
                    )
                else:
                    matched.append(local_pos)

        for key, remote_pos in remote_by_key.items():
            if key not in local_by_key:
                remote_only.append(remote_pos)

        return {
            "matched": len(matched),
            "local_only": local_only,
            "remote_only": remote_only,
            "discrepancies": discrepancies,
        }

    async def update_forecast_actual(self, city: str, date: str, actual_temp: int) -> None:
        await self.conn.execute(
            "UPDATE forecasts SET actual_temp_f=? WHERE city=? AND date=?",
            (actual_temp, city, date),
        )
        await self.conn.commit()

    async def get_last_balance(self, default: float = 1000.0) -> float:
        """Get balance from the most recent snapshot, or default if none exist."""
        cursor = await self.conn.execute(
            "SELECT balance FROM bankroll_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return row["balance"] if row else default

    async def get_last_balance_snapshot(self) -> tuple[float, float] | None:
        """Get (balance, unrealized_pnl) from the most recent snapshot."""
        cursor = await self.conn.execute(
            "SELECT balance, unrealized_pnl FROM bankroll_snapshots "
            "ORDER BY created_at DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return (row["balance"], row["unrealized_pnl"]) if row else None

    async def get_unsettled_dates(self) -> list[tuple[str, str]]:
        """Get distinct (city, date) pairs with unsettled trades where date <= today."""
        cursor = await self.conn.execute(
            """SELECT DISTINCT city, date FROM trades
               WHERE settled = 0 AND date <= date('now')"""
        )
        return [(row["city"], row["date"]) for row in await cursor.fetchall()]

    async def get_brier_score(self, days: int = 30) -> float | None:
        cursor = await self.conn.execute(
            """SELECT AVG((p_model - CASE WHEN actual_temp_f = temp_f THEN 1.0 ELSE 0.0 END)
               * (p_model - CASE WHEN actual_temp_f = temp_f THEN 1.0 ELSE 0.0 END))
               FROM forecasts
               WHERE actual_temp_f IS NOT NULL
               AND created_at >= datetime('now', ?)""",
            (f"-{days} days",),
        )
        row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    async def get_pnl_summary(self, days: int = 30) -> dict:
        cursor = await self.conn.execute(
            """SELECT
                 COUNT(*) as total_trades,
                 SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                 SUM(pnl) as total_pnl,
                 MIN(pnl) as worst_trade,
                 MAX(pnl) as best_trade
               FROM trades
               WHERE settled = 1
               AND created_at >= datetime('now', ?)""",
            (f"-{days} days",),
        )
        row = await cursor.fetchone()
        if not row or row["total_trades"] == 0:
            return {"total_trades": 0, "wins": 0, "total_pnl": 0, "win_rate": 0}
        return {
            "total_trades": row["total_trades"],
            "wins": row["wins"] or 0,
            "total_pnl": row["total_pnl"] or 0,
            "worst_trade": row["worst_trade"] or 0,
            "best_trade": row["best_trade"] or 0,
            "win_rate": (row["wins"] or 0) / row["total_trades"],
        }

    async def get_settled_trades(self, start_date, end_date) -> list[dict]:
        """Get all settled trades in date range for backtesting."""
        cursor = await self.conn.execute(
            """SELECT city, date, temp_f, strategy, entry_price, size,
                      p_model, p_market, edge, outcome, pnl, created_at
               FROM trades
               WHERE settled = 1
               AND date >= ?
               AND date <= ?
               ORDER BY date, city""",
            (str(start_date), str(end_date)),
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def close_position(
        self,
        city: str,
        date_str: str,
        temp_f: float,
        pnl: float,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        await self._conn.execute(
            """
            UPDATE trades
            SET settled=1, pnl=?, exit_price=?, exit_reason=?, settled_at=?
            WHERE city=? AND date=? AND temp_f=? AND settled=0
            """,
            (pnl, exit_price, exit_reason, now, city, date_str, temp_f),
        )
        await self._conn.commit()
        log.info(
            "db.close_position",
            city=city,
            date_str=date_str,
            temp_f=temp_f,
            pnl=pnl,
            exit_price=exit_price,
            exit_reason=exit_reason,
        )

    async def update_peak_p_model(self, city: str, date_str: str, temp_f: float, peak_p_model: float) -> None:
        """Update peak_p_model for trailing stop tracking."""
        await self.conn.execute(
            "UPDATE trades SET peak_p_model = ? WHERE city=? AND date=? AND temp_f=? AND settled=0",
            (peak_p_model, city, date_str, temp_f),
        )
        await self.conn.commit()

    async def record_tier_exit(
        self,
        *,
        city: str,
        date_str: str,
        temp_f: float,
        tier_index: int,
        exit_price: float,
        shares_sold: float,
        pnl: float,
        new_remaining_size: float,
    ) -> None:
        """Record a tier exit and update remaining_size."""
        now = datetime.now(UTC).isoformat()
        await self.conn.execute(
            """INSERT INTO exit_tiers (city, date, temp_f, tier_index, exit_price, shares_sold, pnl, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (city, date_str, temp_f, tier_index, exit_price, shares_sold, pnl, now),
        )
        await self.conn.execute(
            "UPDATE trades SET remaining_size = ? WHERE city=? AND date=? AND temp_f=? AND settled=0",
            (new_remaining_size, city, date_str, temp_f),
        )
        await self.conn.commit()

    async def get_tier_exits(self, city: str, date_str: str, temp_f: float) -> list[dict]:
        """Get all tier exits for a position."""
        cursor = await self.conn.execute(
            "SELECT * FROM exit_tiers WHERE city=? AND date=? AND temp_f=? ORDER BY tier_index",
            (city, date_str, temp_f),
        )
        return [dict(row) for row in await cursor.fetchall()]
    async def get_open_positions(self) -> list[dict]:
        """Get all unsettled positions."""
        cursor = await self.conn.execute(
            """SELECT city, date, temp_f AS temp_value, temp_unit,
                      strategy, entry_price, size, p_model, edge,
                      peak_p_model, remaining_size, created_at
               FROM trades
               WHERE settled = 0
               ORDER BY date, city, temp_f"""
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            if d.get("peak_p_model") is None:
                d["peak_p_model"] = d["p_model"]
            if d.get("remaining_size") is None:
                d["remaining_size"] = d["size"]
            # Skip fully exited positions (remaining_size <= 0)
            if d["remaining_size"] <= 0:
                continue
            result.append(d)
        return result

    async def has_open_position(self, city: str, date: str, temp_f: int) -> bool:
        """Check if there's already an open position for this market."""
        cursor = await self.conn.execute(
            "SELECT COUNT(*) FROM trades WHERE city=? AND date=? AND temp_f=? AND settled=0",
            (city, date, temp_f),
        )
        row = await cursor.fetchone()
        return row[0] > 0 if row else False
