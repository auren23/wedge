from __future__ import annotations

import json
from typing import Any
from wedge.config import Settings
from wedge.db import Database


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "date": row["date"],
        "city": row["city"],
        "temp_f": row["temp_f"],
        "temp_unit": row["temp_unit"],
        "token_id": row["token_id"],
        "contract_type": row["contract_type"],
        "market_price": row["market_price"],
        "implied_prob": row["implied_prob"],
        "volume_24h": row["volume_24h"],
        "open_interest": row["open_interest"],
        "bid_price": row["bid_price"],
        "ask_price": row["ask_price"],
        "spread": row["spread"],
        "liquidity_score": row["liquidity_score"],
        "selected_for_watchlist": bool(row["selected_for_watchlist"]),
        "watchlist_rank": row["watchlist_rank"],
        "selection_reason": row.get("selection_reason"),
        "filter_reason": row.get("filter_reason"),
        "discovered_at": row["discovered_at"],
    }


async def show_watchlist(
    settings: Settings,
    *,
    city: str | None = None,
    target_date: str | None = None,
    include_all: bool = False,
    as_json: bool = False,
) -> None:
    """Print persisted market discoveries/watchlist rows as compact text or JSON."""
    db = Database(settings.db_path)
    await db.connect()
    try:
        rows = await db.list_market_discoveries(
            city=city,
            target_date=target_date,
            include_all=include_all,
        )
        normalized_rows = [_normalize_row(row) for row in rows]
        if not normalized_rows:
            if as_json:
                print("[]")
            else:
                print("No market discoveries found.")
            return

        if as_json:
            print(json.dumps(normalized_rows, indent=2, sort_keys=True))
            return

        header = (
            "date        city        rank watch temp px    bid   ask   sprd  vol24h   oi      score   sel_reason       filter_reason    token"
        )
        print(header)
        for row in normalized_rows:
            rank = row["watchlist_rank"] if row["watchlist_rank"] is not None else "-"
            watch = "Y" if row["selected_for_watchlist"] else "N"
            spread = f"{row['spread']:.2f}" if row["spread"] is not None else "-"
            bid = f"{row['bid_price']:.2f}" if row["bid_price"] is not None else "-"
            ask = f"{row['ask_price']:.2f}" if row["ask_price"] is not None else "-"
            token = row["token_id"] or "-"
            selection_reason = row["selection_reason"] or "-"
            filter_reason = row["filter_reason"] or "-"
            print(
                f"{row['date']:<10}  {row['city']:<10}  {str(rank):>4}  {watch:^5} "
                f"{row['temp_f']:>4}{row['temp_unit']:<1}  {row['market_price']:.2f}  "
                f"{bid:>5}  {ask:>5}  {spread:>5}  "
                f"{row['volume_24h']:>7.0f}  {row['open_interest']:>7.0f}  "
                f"{row['liquidity_score']:>7.3f}  {selection_reason:<15}  {filter_reason:<15}  {token}"
            )
    finally:
        await db.close()
