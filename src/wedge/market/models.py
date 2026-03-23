from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel


class MarketBucket(BaseModel):
    token_id: str
    city: str
    date: date
    temp_value: int  # Temperature value as shown on Polymarket
    temp_unit: str  # "F" or "C" - same unit as Polymarket market
    market_price: float  # 0-1
    implied_prob: float  # = market_price
    volume_24h: float = 2000.0  # 24h trading volume in USD
    open_interest: float = 0.0  # Total open interest
    bid_price: float | None = None
    ask_price: float | None = None
    spread: float | None = None
    contract_type: str = "daily"  # daily, weekly, monthly
    liquidity_score: float = 0.0
    selected_for_watchlist: bool = False
    watchlist_rank: int | None = None
    selection_reason: str | None = None
    filter_reason: str | None = None


class Position(BaseModel):
    bucket: MarketBucket
    side: Literal["buy", "sell"] = "buy"
    size: float  # USD amount
    entry_price: float
    strategy: Literal["ladder"]
    p_model: float = 0.0
    edge: float = 0.0