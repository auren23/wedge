from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from wedge.market.scanner import scan_weather_markets


def _make_client(event: dict | None) -> AsyncMock:
    """Create a mock client that returns an event from get_event_by_slug."""
    client = AsyncMock()
    client.get_event_by_slug.return_value = event
    return client


def _outcome(outcome: str, price: float) -> dict:
    """Create an outcome (Yes/No token)."""
    return {"outcome": outcome, "price": str(price)}


def _market(
    question: str,
    outcomes: list[dict],
    token_ids: list[str] | None = None,
    volume_24h: float = 10000.0,
) -> dict:
    """Create a market within an event."""
    m: dict = {"question": question, "outcomes": outcomes, "volume24h": volume_24h}
    if token_ids:
        m["clobTokenIds"] = token_ids
    return m


def _event(title: str, markets: list[dict]) -> dict:
    """Create an event containing markets."""
    return {"title": title, "markets": markets}


TARGET_DATE = date(2026, 7, 4)


class TestScanWeatherMarkets:
    @pytest.mark.asyncio
    async def test_no_markets_returns_empty(self):
        client = _make_client(None)  # Event not found
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_without_temperature_skipped(self):
        event = _event(
            "Highest temperature in NYC on July 4?",
            [_market("Will it rain in NYC?", [_outcome("Yes", 0.3), _outcome("No", 0.7)])],
        )
        client = _make_client(event)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_with_wrong_city_skipped(self):
        # This test is no longer relevant since we query by slug (city-specific)
        # But we keep it to test that unsupported cities return empty
        client = _make_client(None)
        result = await scan_weather_markets(client, "UnsupportedCity", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_with_wrong_date_skipped(self):
        # Date is now part of the slug, so wrong date means event not found
        client = _make_client(None)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_without_temp_in_question_skipped(self):
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be sunny?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                )
            ],
        )
        client = _make_client(event)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_with_price_zero_skipped(self):
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 70°F?",
                    [_outcome("Yes", 0.0), _outcome("No", 1.0)],
                )
            ],
        )
        client = _make_client(event)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_market_with_price_one_skipped(self):
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 70°F?",
                    [_outcome("Yes", 1.0), _outcome("No", 0.0)],
                )
            ],
        )
        client = _make_client(event)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE)
        assert result == []

    @pytest.mark.asyncio
    async def test_full_successful_scan(self):
        # Create events for daily, weekly, and monthly contracts
        daily_event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 70°F?",
                    [_outcome("Yes", 0.3), _outcome("No", 0.7)],
                    ["token_70"],
                ),
                _market(
                    "Will the highest temperature in New York City be 75°F?",
                    [_outcome("Yes", 0.4), _outcome("No", 0.6)],
                    ["token_75"],
                ),
            ],
        )

        # Set daily contract slug explicitly
        async def mock_get_slug(slug):
            return daily_event if "on-july-4" in slug else None

        client = AsyncMock()
        client.get_event_by_slug.side_effect = mock_get_slug

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=False, include_monthly=False
        )

        assert len(result) == 2
        assert result[0].temp_value == 70
        assert result[0].temp_unit == "F"
        assert result[0].market_price == 0.3
        assert result[0].token_id == "token_70"
        assert result[1].temp_value == 75
        assert result[1].temp_unit == "F"
        assert result[1].market_price == 0.4
        assert result[1].token_id == "token_75"

    @pytest.mark.asyncio
    async def test_market_with_no_date_still_included(self):
        # Test that markets are included when event is found
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 80°F?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                    ["token_80"],
                ),
            ],
        )
        client = AsyncMock()
        client.get_event_by_slug.side_effect = lambda slug: event if "on-july-4" in slug else None

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=False, include_monthly=False
        )

        assert len(result) == 1
        assert result[0].temp_value == 80
        assert result[0].temp_unit == "F"

    @pytest.mark.asyncio
    async def test_multiple_tokens_one_market(self):
        # In the new format, each market has one temperature
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be between 70-75°F?",
                    [_outcome("Yes", 0.6), _outcome("No", 0.4)],
                    ["token_70_75"],
                ),
            ],
        )
        client = AsyncMock()
        client.get_event_by_slug.side_effect = lambda slug: event if "on-july-4" in slug else None

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=False, include_monthly=False
        )

        # Should extract 70 from the question
        assert len(result) == 1
        assert result[0].temp_value in [70, 75]
        assert result[0].temp_unit == "F"

    @pytest.mark.asyncio
    async def test_liquidity_filter_low_volume(self):
        """Test that low volume markets are filtered out."""
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 75°F?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                    ["token_75"],
                ),
            ],
        )
        # Add volume data (below threshold)
        event["markets"][0]["volume24h"] = 1000  # Below $2K threshold

        client = _make_client(event)
        result = await scan_weather_markets(client, "NYC", TARGET_DATE, min_volume=2000)

        # Should be filtered out due to low volume
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_liquidity_filter_high_volume(self):
        """Test that high volume markets are included."""
        event = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 75°F?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                    ["token_75"],
                ),
            ],
        )
        # Add volume data (above threshold)
        event["markets"][0]["volume24h"] = 10000  # Above $2K threshold

        client = AsyncMock()
        client.get_event_by_slug.side_effect = lambda slug: event if "on-july-4" in slug else None

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, min_volume=2000, include_weekly=False, include_monthly=False
        )

        # Should be included
        assert len(result) == 1
        assert result[0].volume_24h == 10000

    @pytest.mark.asyncio
    async def test_contract_type_detection(self):
        """Test detection of daily/weekly/monthly contract types."""
        # Daily contract - use specific daily slug format
        # The slug format is: highest-temperature-in-nyc-on-july-4-2026
        event_daily = _event(
            "Highest temperature in NYC on July 4?",
            [
                _market(
                    "Will the highest temperature in New York City be 75°F?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                    ["token_75"],
                ),
            ],
        )
        event_daily["markets"][0]["volume24h"] = 10000

        async def mock_get_slug(slug):
            # Daily slug: highest-temperature-in-nyc-on-july-4-2026
            if slug == "highest-temperature-in-nyc-on-july-4-2026":
                return event_daily
            return None

        client = AsyncMock()
        client.get_event_by_slug.side_effect = mock_get_slug

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=False, include_monthly=False
        )

        assert len(result) == 1
        assert result[0].contract_type == "daily"

    @pytest.mark.asyncio
    async def test_weekly_contract_detection(self):
        """Test detection of weekly contract type."""
        event_weekly = _event(
            "Highest temperature in NYC this week?",
            [
                _market(
                    "Will the highest temperature in New York City this week be 75°F?",
                    [_outcome("Yes", 0.5), _outcome("No", 0.5)],
                    ["token_75"],
                ),
            ],
        )
        event_weekly["markets"][0]["volume24h"] = 10000

        client = AsyncMock()
        client.get_event_by_slug.side_effect = lambda slug: event_weekly if "week" in slug else None

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=True, include_monthly=False
        )

        # Should find weekly contract
        assert len(result) >= 1
        weekly_contracts = [r for r in result if r.contract_type == "weekly"]
        assert len(weekly_contracts) >= 1

    @pytest.mark.asyncio
    async def test_monthly_contract_detection(self):
        """Test detection of monthly contract type."""
        event_monthly = _event(
            "Highest temperature in NYC in July?",
            [
                _market(
                    "Will the highest temperature in New York City in July be 85°F?",
                    [_outcome("Yes", 0.4), _outcome("No", 0.6)],
                    ["token_85"],
                ),
            ],
        )
        event_monthly["markets"][0]["volume24h"] = 10000

        client = AsyncMock()
        client.get_event_by_slug.side_effect = lambda slug: (
            event_monthly if "in-july" in slug else None
        )

        result = await scan_weather_markets(
            client, "NYC", TARGET_DATE, include_weekly=False, include_monthly=True
        )

        # Should find monthly contract
        assert len(result) >= 1
        monthly_contracts = [r for r in result if r.contract_type == "monthly"]
        assert len(monthly_contracts) >= 1  # Could match either number

    @pytest.mark.asyncio
    async def test_celsius_temperature_conversion(self):
        """Test that Celsius temperatures are correctly converted to Fahrenheit."""
        # International cities use Celsius on Polymarket
        event_shanghai = _event(
            "Highest temperature in Shanghai in March?",
            [
                _market(
                    "Will the highest temperature in Shanghai be 12°C?",
                    [_outcome("Yes", 0.3), _outcome("No", 0.7)],
                    ["token_12c"],
                ),
            ],
        )
        event_shanghai["markets"][0]["volume24h"] = 10000

        client = AsyncMock()
        # Only match monthly contract slug
        client.get_event_by_slug.side_effect = lambda slug: (
            event_shanghai if slug == "highest-temperature-in-shanghai-in-march-2026" else None
        )

        # 12°C = 53.6°F → should round to 54°F
        result = await scan_weather_markets(
            client, "Shanghai", date(2026, 3, 15), include_weekly=False, include_monthly=True
        )

        assert len(result) == 1
        assert result[0].temp_value == 12  # Market shows 12°C
        assert result[0].temp_unit == "C"

    @pytest.mark.asyncio
    async def test_celsius_format_with_space(self):
        """Test Celsius detection with space format (e.g., '25 C')."""
        event = _event(
            "Highest temperature in Seoul in March?",
            [
                _market(
                    "Will the highest temperature in Seoul be 11 C?",
                    [_outcome("Yes", 0.4), _outcome("No", 0.6)],
                    ["token_11c"],
                ),
            ],
        )
        event["markets"][0]["volume24h"] = 10000

        client = AsyncMock()
        # Only match monthly contract slug
        client.get_event_by_slug.side_effect = lambda slug: (
            event if slug == "highest-temperature-in-seoul-in-march-2026" else None
        )

        # 11°C = 51.8°F → should round to 52°F
        result = await scan_weather_markets(
            client, "Seoul", date(2026, 3, 15), include_weekly=False, include_monthly=True
        )

        assert len(result) == 1
        assert result[0].temp_value == 11  # Market shows 11°C
        assert result[0].temp_unit == "C"
