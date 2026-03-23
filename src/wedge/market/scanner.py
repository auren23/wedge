from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timedelta

from wedge.log import get_logger
from wedge.market.models import MarketBucket
from wedge.market.polymarket import PolymarketClient

log = get_logger("market.scanner")

_TEMP_PATTERN = re.compile(r"(\d+\.?\d*)\s*°?\s*([CF])", re.IGNORECASE)
_DATE_PATTERN = re.compile(
    r"(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)\s+(\d{1,2})",
    re.IGNORECASE,
)
_WEEK_PATTERN = re.compile(r"week(?:ly)?", re.IGNORECASE)
_MONTH_PATTERN = re.compile(
    r"month(?:ly)?|in\s+(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)",
    re.IGNORECASE,
)

_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_CITY_ALIASES = {
    "new york": "NYC",
    "nyc": "NYC",
    "miami": "Miami",
    "seoul": "Seoul",
    "london": "London",
    "shanghai": "Shanghai",
    "wellington": "Wellington",
}

# Map city names to Polymarket slug format (high liquidity markets only)
_CITY_TO_SLUG = {
    "NYC": "nyc",
    "Miami": "miami",
    "Seoul": "seoul",
    "London": "london",
    "Shanghai": "shanghai",
    "Wellington": "wellington",
}

# Liquidity thresholds
_MIN_VOLUME_24H = 2000.0  # Minimum $2K daily volume
_MIN_OPEN_INTEREST = 1000.0  # Minimum $1K open interest

_CONTRACT_TYPE_BONUS = {
    "daily": 0.30,
    "weekly": 0.15,
    "monthly": 0.05,
}


def _detect_contract_type(question: str) -> str:
    """Detect contract type from question text."""
    if _WEEK_PATTERN.search(question):
        return "weekly"
    elif _MONTH_PATTERN.search(question):
        return "monthly"
    return "daily"


def _extract_volume(market: dict) -> float:
    """Extract 24h volume from market data."""
    # Try various field names for volume (Gamma API uses volume24hr, not volume24h)
    for key in ["volume24hr", "volume24hrClob", "volume24h", "volume_24h", "volume", "notional24h"]:
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def _extract_open_interest(market: dict) -> float:
    """Extract open interest from market data."""
    for key in ["openInterest", "open_interest", "oi", "liquidity"]:
        val = market.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    return 0.0


def _extract_yes_bid_ask(market: dict, price: float) -> tuple[float | None, float | None, float | None]:
    """Extract yes-side bid/ask spread when available from Gamma market payloads."""
    bid = None
    ask = None

    for bid_key in ["bestBid", "best_bid", "bid"]:
        val = market.get(bid_key)
        if val is not None:
            try:
                bid = float(val)
                break
            except (ValueError, TypeError):
                pass

    for ask_key in ["bestAsk", "best_ask", "ask"]:
        val = market.get(ask_key)
        if val is not None:
            try:
                ask = float(val)
                break
            except (ValueError, TypeError):
                pass

    if bid is None and ask is None:
        return None, None, None
    if bid is None:
        bid = price
    if ask is None:
        ask = price
    spread = ask - bid
    if bid < 0 or ask < 0 or bid > 1 or ask > 1 or spread < 0:
        return None, None, None
    return bid, ask, spread


def _compute_liquidity_score(bucket: MarketBucket) -> float:
    """Score buckets by tradable liquidity within the weather-only universe."""
    volume_score = math.log10(bucket.volume_24h + 1.0)
    oi_score = math.log10(bucket.open_interest + 1.0)
    contract_bonus = _CONTRACT_TYPE_BONUS.get(bucket.contract_type, 0.0)
    spread_penalty = (bucket.spread or 0.0) * 4.0
    return round((volume_score * 0.7) + (oi_score * 0.3) + contract_bonus - spread_penalty, 6)


def _parse_json_field(value: str | list | dict | None, field_name: str = "") -> list | dict | None:
    """Parse a JSON field that may be string or already parsed."""
    if value is None:
        return None

    if isinstance(value, (list, dict)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            log.warning("invalid_json_field", field=field_name, value=value[:100])
            return None

    return None


def rank_market_buckets(
    buckets: list[MarketBucket],
    *,
    watchlist_size: int,
    rejected_buckets: list[MarketBucket] | None = None,
) -> list[MarketBucket]:
    """Rank buckets by liquidity and mark top-K as watchlist candidates."""
    ranked = sorted(
        (
            bucket.model_copy(update={"liquidity_score": _compute_liquidity_score(bucket)})
            for bucket in buckets
        ),
        key=lambda bucket: (bucket.liquidity_score, bucket.volume_24h, bucket.open_interest),
        reverse=True,
    )

    for idx, bucket in enumerate(ranked, start=1):
        selected = idx <= max(watchlist_size, 0)
        bucket.selected_for_watchlist = selected
        bucket.watchlist_rank = idx if selected else None
        bucket.selection_reason = "watchlist_top_k" if selected else "ranked_out"
        bucket.filter_reason = None

    if rejected_buckets is not None:
        for bucket in rejected_buckets:
            bucket.selected_for_watchlist = False
            bucket.watchlist_rank = None
            bucket.selection_reason = None

    return ranked


async def discover_weather_markets(
    client: PolymarketClient,
    city: str,
    target_date: date,
    *,
    min_volume: float = _MIN_VOLUME_24H,
    min_open_interest: float = _MIN_OPEN_INTEREST,
    max_spread: float | None = None,
    include_weekly: bool = True,
    include_monthly: bool = True,
) -> tuple[list[MarketBucket], list[MarketBucket]]:
    """Return accepted and rejected weather markets with reasons."""
    city_slug = _CITY_TO_SLUG.get(city)
    if not city_slug:
        log.warning("unsupported_city", city=city)
        return [], []

    month_name = target_date.strftime("%B").lower()
    day = target_date.day
    year = target_date.year

    slugs = []
    slugs.append(f"highest-temperature-in-{city_slug}-on-{month_name}-{day}-{year}")

    if include_weekly:
        week_start = target_date - timedelta(days=target_date.weekday())
        slugs.append(
            f"highest-temperature-in-{city_slug}-week-of-{week_start.strftime('%B').lower()}-{week_start.day}-{year}"
        )

    if include_monthly:
        slugs.append(f"highest-temperature-in-{city_slug}-in-{month_name}-{year}")

    events = []
    for slug in slugs:
        if hasattr(client, "get_event_by_slug"):
            event = await client.get_event_by_slug(slug)
            if event:
                events.append(event)
        else:
            log.warning("client_missing_get_event_by_slug", city=city)
            return [], []

    if not events:
        log.info("scan_complete", city=city, date=str(target_date), buckets_found=0)
        return [], []

    accepted: list[MarketBucket] = []
    rejected: list[MarketBucket] = []

    for event in events:
        for market in event.get("markets", []):
            try:
                question = market.get("question", "").lower()

                temp_match = _TEMP_PATTERN.search(question)
                if not temp_match:
                    continue

                try:
                    temp_value = int(temp_match.group(1))
                    temp_unit = temp_match.group(2).upper()
                except (ValueError, IndexError):
                    log.warning("invalid_temp_format", question=question)
                    continue

                contract_type = _detect_contract_type(question)
                volume_24h = _extract_volume(market)
                open_interest = _extract_open_interest(market)

                outcomes = _parse_json_field(market.get("outcomes"), "outcomes")
                if outcomes is None:
                    outcomes = market.get("outcomes", [])

                if not isinstance(outcomes, list) or len(outcomes) < 2:
                    continue

                prices = _parse_json_field(market.get("outcomePrices"), "outcomePrices")
                if prices is None:
                    prices = market.get("outcomePrices")

                yes_index = None
                for idx, outcome in enumerate(outcomes):
                    if isinstance(outcome, str):
                        if outcome.lower() == "yes":
                            yes_index = idx
                            break
                    elif isinstance(outcome, dict):
                        if outcome.get("outcome", "").lower() == "yes":
                            yes_index = idx
                            break

                if yes_index is None:
                    continue

                if prices:
                    if yes_index >= len(prices):
                        continue
                    try:
                        price = float(prices[yes_index])
                    except (ValueError, TypeError):
                        log.warning("invalid_price_value", price=prices[yes_index])
                        continue
                else:
                    yes_outcome = outcomes[yes_index]
                    if not isinstance(yes_outcome, dict):
                        continue
                    try:
                        price = float(yes_outcome.get("price", 0))
                    except (ValueError, TypeError):
                        log.warning("invalid_price_format", price=yes_outcome.get("price"))
                        continue

                if not (0 < price < 1):
                    continue

                clob_token_ids = _parse_json_field(market.get("clobTokenIds"), "clobTokenIds")
                if clob_token_ids is None:
                    clob_token_ids = market.get("clobTokenIds", [])

                token_id = ""
                if clob_token_ids and yes_index < len(clob_token_ids):
                    token_id = clob_token_ids[yes_index]

                bid_price, ask_price, spread = _extract_yes_bid_ask(market, price)
                bucket = MarketBucket(
                    token_id=token_id,
                    city=city,
                    date=target_date,
                    temp_value=temp_value,
                    temp_unit=temp_unit,
                    market_price=price,
                    implied_prob=price,
                    volume_24h=volume_24h,
                    open_interest=open_interest,
                    bid_price=bid_price,
                    ask_price=ask_price,
                    spread=spread,
                    contract_type=contract_type,
                )
                bucket.liquidity_score = _compute_liquidity_score(bucket)

                if volume_24h < min_volume:
                    bucket.filter_reason = "low_volume"
                    rejected.append(bucket)
                    log.debug(
                        "low_volume_filtered",
                        city=city,
                        question=question[:50],
                        volume_24h=volume_24h,
                    )
                    continue
                if open_interest < min_open_interest:
                    bucket.filter_reason = "low_open_interest"
                    rejected.append(bucket)
                    log.debug(
                        "low_open_interest_filtered",
                        city=city,
                        question=question[:50],
                        open_interest=open_interest,
                    )
                    continue
                if max_spread is not None and spread is not None and spread > max_spread:
                    bucket.filter_reason = "wide_spread"
                    rejected.append(bucket)
                    log.debug(
                        "wide_spread_filtered",
                        city=city,
                        question=question[:50],
                        spread=spread,
                        max_spread=max_spread,
                    )
                    continue

                accepted.append(bucket)
            except Exception as e:
                log.warning(
                    "market_parse_error",
                    market=market.get("question", "unknown"),
                    error=str(e),
                )
                continue

    log.info(
        "scan_complete",
        city=city,
        date=str(target_date),
        buckets_found=len(accepted),
        rejected_count=len(rejected),
        daily_contracts=len([b for b in accepted if b.contract_type == "daily"]),
        weekly_contracts=len([b for b in accepted if b.contract_type == "weekly"]),
        monthly_contracts=len([b for b in accepted if b.contract_type == "monthly"]),
    )
    return accepted, rejected


async def scan_weather_markets(
    client: PolymarketClient,
    city: str,
    target_date: date,
    *,
    min_volume: float = _MIN_VOLUME_24H,
    min_open_interest: float = _MIN_OPEN_INTEREST,
    max_spread: float | None = None,
    include_weekly: bool = True,
    include_monthly: bool = True,
) -> list[MarketBucket]:
    """Compatibility wrapper returning only accepted markets."""
    accepted, _ = await discover_weather_markets(
        client,
        city,
        target_date,
        min_volume=min_volume,
        min_open_interest=min_open_interest,
        max_spread=max_spread,
        include_weekly=include_weekly,
        include_monthly=include_monthly,
    )
    return accepted


def _extract_market_date(market: dict, year: int) -> date | None:
    """Try to extract date from market question or structured fields."""
    end_date = market.get("end_date_iso") or market.get("end_date")
    if end_date:
        try:
            return datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            pass

    question = market.get("question", "")
    match = _DATE_PATTERN.search(question)
    if match:
        month = _MONTH_MAP.get(match.group(1).lower())
        day = int(match.group(2))
        if month and 1 <= day <= 31:
            try:
                return date(year, month, day)
            except ValueError:
                pass
    return None