from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from wedge.db import Database
from wedge.execution.executor import validate_order
from wedge.execution.models import OrderRequest, OrderResult
from wedge.log import get_logger
from wedge.market.models import MarketBucket, Position

log = get_logger("execution.dry_run")


class DryRunExecutor:
    def __init__(self, db: Database, initial_balance: float, max_bet: float = 100.0) -> None:
        self._db = db
        self._balance = initial_balance
        self._max_bet = max_bet
        self._positions: list[Position] = []
        self._order_ids: set[str] = set()
        self._positions_loaded = False
        self._load_lock = asyncio.Lock()  # Prevent concurrent position loading

    async def place_order(self, request: OrderRequest) -> OrderResult:
        error = validate_order(request, self._balance, self._max_bet)
        if error:
            log.warning("dry_run_order_rejected", reason=error, **request.model_dump(mode="json"))
            return OrderResult(success=False, error=error)

        order_id = f"dry_{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC).isoformat()

        inserted = await self._db.insert_trade(
            run_id=request.run_id,
            city=request.city,
            date=request.date.isoformat(),
            temp_f=request.temp_value,
            temp_unit=request.temp_unit,
            strategy=request.strategy,
            side=request.side,
            entry_price=request.limit_price,
            size=request.size,
            p_model=request.p_model,
            p_market=request.p_market,
            edge=request.edge,
            token_id=request.token_id,
            order_id=order_id,
            created_at=now,
        )

        if not inserted:
            log.info(
                "dry_run_duplicate_skipped",
                run_id=request.run_id,
                temp_value=request.temp_value,
            )
            return OrderResult(success=True, order_id=order_id, error="duplicate")

        self._balance -= request.size
        self._order_ids.add(order_id)

        self._positions.append(
            Position(
                bucket=MarketBucket(
                    token_id=request.token_id,
                    city=request.city,
                    date=request.date,
                    temp_value=request.temp_value,
                    temp_unit=request.temp_unit,
                    market_price=request.limit_price,
                    implied_prob=request.limit_price,
                ),
                size=request.size,
                entry_price=request.limit_price,
                strategy=request.strategy,
            )
        )

        log.info(
            "dry_run_order_placed",
            order_id=order_id,
            city=request.city,
            temp_value=request.temp_value,
            size=f"${request.size:.2f}",
            price=request.limit_price,
        )
        return OrderResult(
            success=True,
            order_id=order_id,
            filled_price=request.limit_price,
            filled_size=request.size,
        )

    async def cancel_order(self, order_id: str) -> bool:
        found = order_id in self._order_ids
        log.info("dry_run_cancel", order_id=order_id, found=found)
        return found

    async def get_positions(self) -> list[Position]:
        return list(self._positions)

    async def get_balance(self) -> float:
        return self._balance

    async def update_position_prices(self, markets: list[MarketBucket]) -> None:
        """Update position prices from current market data.

        This allows dry-run to track unrealized P&L based on real market prices.
        """
        # Load positions from database if not already loaded
        if not self._positions_loaded:
            await self._load_positions_from_db()

        market_map = {(m.city, m.date, m.temp_value): m.market_price for m in markets}

        for pos in self._positions:
            key = (pos.bucket.city, pos.bucket.date, pos.bucket.temp_value)
            if key in market_map:
                current_price = market_map[key]
                pos.bucket.market_price = current_price
                pos.bucket.implied_prob = current_price

    async def close_position(
        self,
        city: str,
        date_str: str,
        temp_f: float,
        exit_price: float,
        exit_reason: str,
        db: Database,
    ) -> float:
        """Close a position at exit_price. Returns realized pnl."""
        if not self._positions_loaded:
            await self._load_positions_from_db()

        # Find matching position
        matched = None
        for pos in self._positions:
            if (
                pos.bucket.city == city
                and pos.bucket.date.isoformat() == date_str
                and pos.bucket.temp_value == temp_f
            ):
                matched = pos
                break

        if matched is None:
            log.warning("dry_run_close_not_found", city=city, date=date_str, temp_f=temp_f)
            return 0.0

        # Binary option: shares = size / entry_price, pnl = shares * exit_price - size
        shares = matched.size / matched.entry_price
        pnl = shares * exit_price - matched.size
        self._balance += matched.size + pnl  # return cost basis + profit
        self._positions.remove(matched)

        log.info(
            "dry_run_close_position",
            city=city,
            date=date_str,
            temp_f=temp_f,
            entry_price=matched.entry_price,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl=round(pnl, 4),
        )

        await db.close_position(
            city=city,
            date_str=date_str,
            temp_f=temp_f,
            pnl=round(pnl, 4),
            exit_price=exit_price,
            exit_reason=exit_reason,
        )
        return pnl

    async def get_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L from current position values.

        For binary options: shares = size / entry_price
        Current value = shares * current_price = size * current_price / entry_price
        Unrealized P&L = current_value - size = size * (current_price - entry_price) / entry_price
        """
        # Load positions from database if not already loaded
        if not self._positions_loaded:
            await self._load_positions_from_db()

        total_pnl = 0.0
        for pos in self._positions:
            # Binary option P&L formula
            pnl = pos.size * (pos.bucket.market_price - pos.entry_price) / pos.entry_price
            total_pnl += pnl
        return total_pnl

    async def _load_positions_from_db(self) -> None:
        """Load open positions from database into memory.

        Thread-safe: uses asyncio.Lock to prevent concurrent loading.
        """
        async with self._load_lock:
            # Double-check after acquiring lock (another coroutine may have loaded)
            if self._positions_loaded:
                return

            open_positions = await self._db.get_open_positions()
            self._positions = []

            for pos_dict in open_positions:
                from datetime import date as date_type

                # Parse date string to date object
                date_obj = date_type.fromisoformat(pos_dict["date"])

                self._positions.append(
                    Position(
                        bucket=MarketBucket(
                            token_id=pos_dict.get("token_id", ""),
                            city=pos_dict["city"],
                            date=date_obj,
                            temp_value=pos_dict["temp_value"],
                            temp_unit=pos_dict.get("temp_unit", "F"),
                            market_price=pos_dict["entry_price"],
                            implied_prob=pos_dict["entry_price"],
                        ),
                        size=pos_dict["size"],
                        entry_price=pos_dict["entry_price"],
                        strategy=pos_dict["strategy"],
                        p_model=pos_dict.get("p_model", 0.0),
                        edge=pos_dict.get("edge", 0.0),
                    )
                )

            self._positions_loaded = True
