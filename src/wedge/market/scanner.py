from __future__ import annotations

import json
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


def _parse_json_field(value: str | list | dict | None, field_name: str = "") -> list | dict | None:
    """Parse a JSON field that may be string or already parsed.

    Args:
        value: Field value (may be JSON string or already parsed)
        field_name: Field name for logging

    Returns:
        Parsed value (list or dict) or None if parsing fails
    """
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


async def scan_weather_markets(
    client: PolymarketClient,
    city: str,
    target_date: date,
    *,
    min_volume: float = _MIN_VOLUME_24H,
    include_weekly: bool = True,
    include_monthly: bool = True,
) -> list[MarketBucket]:
    """Scan Polymarket for weather temperature contracts matching city and date.

    Supports daily, weekly, and monthly contracts.
    Filters out low-liquidity markets to avoid slippage.

    Args:
        client: Polymarket API client
        city: City name (e.g., "NYC", "London")
        target_date: Target date for daily contracts, or date within week/month
        min_volume: Minimum 24h volume in USD (default $2K)
        include_weekly: Include weekly contracts (default True)
        include_monthly: Include monthly contracts (default True)

    Returns:
        List of market buckets passing liquidity filters
    """
    city_slug = _CITY_TO_SLUG.get(city)
    if not city_slug:
        log.warning("unsupported_city", city=city)
        return []

    # Build slugs for different contract types
    month_name = target_date.strftime("%B").lower()
    day = target_date.day
    year = target_date.year

    slugs = []

    # Daily contract slug
    slugs.append(f"highest-temperature-in-{city_slug}-on-{month_name}-{day}-{year}")

    # Weekly contract: find week containing target_date
    if include_weekly:
        week_start = target_date - timedelta(days=target_date.weekday())  # Monday
        slugs.append(
            f"highest-temperature-in-{city_slug}-week-of-{week_start.strftime('%B').lower()}-{week_start.day}-{year}"
        )

    # Monthly contract
    if include_monthly:
        slugs.append(f"highest-temperature-in-{city_slug}-in-{month_name}-{year}")

    # Fetch all events
    events = []
    for slug in slugs:
        if hasattr(client, "get_event_by_slug"):
            event = await client.get_event_by_slug(slug)
            if event:
                events.append(event)
        else:
            log.warning("client_missing_get_event_by_slug", city=city)
            return []

    if not events:
        log.info("scan_complete", city=city, date=str(target_date), buckets_found=0)
        return []

    buckets: list[MarketBucket] = []

    for event in events:
        for market in event.get("markets", []):
            try:
                question = market.get("question", "").lower()

                # Extract temperature and unit directly from market question
                temp_match = _TEMP_PATTERN.search(question)
                if not temp_match:
                    continue

                try:
                    temp_value = int(temp_match.group(1))
                    temp_unit = temp_match.group(2).upper()
                except (ValueError, IndexError):
                    log.warning("invalid_temp_format", question=question)
                    continue

                # Detect contract type
                contract_type = _detect_contract_type(question)

                # Extract liquidity metrics
                volume_24h = _extract_volume(market)
                open_interest = _extract_open_interest(market)

                # Filter low-liquidity markets
                if volume_24h < min_volume:
                    log.debug(
                        "low_volume_filtered",
                        city=city,
                        question=question[:50],
                        volume_24h=volume_24h,
                    )
                    continue

                # Parse outcomes and prices using unified JSON parser
                outcomes = _parse_json_field(market.get("outcomes"), "outcomes")
                if outcomes is None:
                    outcomes = market.get("outcomes", [])

                if not isinstance(outcomes, list) or len(outcomes) < 2:
                    continue

                prices = _parse_json_field(market.get("outcomePrices"), "outcomePrices")
                if prices is None:
                    prices = market.get("outcomePrices")

                # Find "Yes" outcome
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

                # Get price
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

                # Get token_id using unified JSON parser
                clob_token_ids = _parse_json_field(market.get("clobTokenIds"), "clobTokenIds")
                if clob_token_ids is None:
                    clob_token_ids = market.get("clobTokenIds", [])

                token_id = ""
                if clob_token_ids and yes_index < len(clob_token_ids):
                    token_id = clob_token_ids[yes_index]

                buckets.append(
                    MarketBucket(
                        token_id=token_id,
                        city=city,
                        date=target_date,
                        temp_value=temp_value,
                        temp_unit=temp_unit,
                        market_price=price,
                        implied_prob=price,
                        volume_24h=volume_24h,
                        open_interest=open_interest,
                        contract_type=contract_type,
                    )
                )
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
        buckets_found=len(buckets),
        daily_contracts=len([b for b in buckets if b.contract_type == "daily"]),
        weekly_contracts=len([b for b in buckets if b.contract_type == "weekly"]),
        monthly_contracts=len([b for b in buckets if b.contract_type == "monthly"]),
    )
    return buckets


def _extract_market_date(market: dict, year: int) -> date | None:
    """Try to extract date from market question or structured fields."""
    # Try structured field first
    end_date = market.get("end_date_iso") or market.get("end_date")
    if end_date:
        try:
            return datetime.fromisoformat(end_date.replace("Z", "+00:00")).date()
        except (ValueError, AttributeError):
            pass

    # Fall back to parsing question text
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
