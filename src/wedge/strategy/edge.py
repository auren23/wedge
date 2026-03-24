from __future__ import annotations

from datetime import UTC, date

from wedge.market.models import MarketBucket
from wedge.strategy.models import EdgeSignal
from wedge.weather.models import ForecastDistribution

_EPS = 1e-6
_DEFAULT_FEE_RATE = 0.0


def estimate_slippage(volume_24h: float, bet_size: float) -> float:
    """Estimate slippage cost as fraction of bet size."""
    if volume_24h <= 0:
        return 0.05

    if volume_24h < 1000:
        base_slippage = 0.05
    elif volume_24h < 5000:
        base_slippage = 0.02
    elif volume_24h < 25000:
        base_slippage = 0.01
    else:
        base_slippage = 0.005

    size_ratio = bet_size / max(volume_24h, 1)
    size_multiplier = 1.0 + (size_ratio * 10)
    return min(base_slippage * size_multiplier, 0.10)


def calculate_ev(
    p_model: float,
    market_price: float,
    fee_rate: float = _DEFAULT_FEE_RATE,
    slippage: float = 0.0,
) -> float:
    """Calculate expected value of a binary option bet.

    Bet $1 at price market_price. Win 1/market_price total if event happens.
    EV = p_model * (1 - fee_rate) / market_price - 1 - slippage
    """
    if not (_EPS < market_price < 1 - _EPS):
        return 0.0
    return p_model * (1 - fee_rate) / market_price - 1.0 - slippage


def calculate_ev_short(
    p_model: float,
    market_price: float,
    fee_rate: float = _DEFAULT_FEE_RATE,
    slippage: float = 0.0,
) -> float:
    """Calculate expected value of buying No (shorting Yes).

    Buy No at price (1 - market_price). Win if event doesn't happen.
    ev = (1 - p_model) * (1 - fee) * p_market / (1 - p_market) - p_model - slippage
    """
    no_price = 1.0 - market_price
    if not (_EPS < no_price < 1 - _EPS):
        return 0.0
    odds_no = market_price / no_price
    win_ev = (1.0 - p_model) * (1 - fee_rate) * odds_no
    loss_ev = p_model
    return win_ev - loss_ev - slippage


def detect_edges(
    forecast: ForecastDistribution,
    markets: list[MarketBucket],
    ladder_threshold: float = 0.05,
    fee_rate: float = _DEFAULT_FEE_RATE,
    slippage_bet_size: float = 50.0,
) -> list[EdgeSignal]:
    """Find buckets where model probability differs from market pricing.

    Long (buy Yes): p_model > p_market + threshold
    Short (buy No): p_model < p_market - threshold
    """
    signals: list[EdgeSignal] = []
    min_threshold = ladder_threshold

    for bucket in markets:
        if not (_EPS < bucket.market_price < 1 - _EPS):
            continue

        if bucket.temp_unit == "C":
            lookup_temp = round(bucket.temp_value * 9 / 5 + 32)
        else:
            lookup_temp = bucket.temp_value

        p_model = forecast.buckets.get(lookup_temp, 0.0)
        volume_24h = getattr(bucket, "volume_24h", 2000.0)
        slippage = estimate_slippage(volume_24h, bet_size=slippage_bet_size)
        edge = p_model - bucket.market_price

        from datetime import datetime
        now = datetime.now(UTC)
        age_hours = (now - forecast.updated_at).total_seconds() / 3600.0
        if age_hours < 1.0:
            forecast_weight = 1.3
        elif age_hours < 2.0:
            forecast_weight = 1.0
        elif age_hours < 4.0:
            forecast_weight = 0.8
        else:
            forecast_weight = 0.6

        # Long: model thinks event MORE likely than market
        if edge > min_threshold:
            ev = calculate_ev(p_model, bucket.market_price, fee_rate, slippage)
            if ev > 0:
                odds = (1.0 - bucket.market_price) / bucket.market_price
                signals.append(EdgeSignal(
                    city=bucket.city, date=bucket.date,
                    temp_value=bucket.temp_value, temp_unit=bucket.temp_unit,
                    token_id=bucket.token_id, p_model=p_model,
                    p_market=bucket.market_price, edge=edge, odds=odds,
                    ensemble_spread=forecast.ensemble_spread,
                    forecast_age_hours=round(age_hours, 2),
                    weight=forecast_weight, side="buy",
                ))

        # Short: model thinks event LESS likely than market
        elif edge < -min_threshold:
            ev_short = calculate_ev_short(p_model, bucket.market_price, fee_rate, slippage)
            if ev_short > 0:
                odds_no = bucket.market_price / (1.0 - bucket.market_price)
                signals.append(EdgeSignal(
                    city=bucket.city, date=bucket.date,
                    temp_value=bucket.temp_value, temp_unit=bucket.temp_unit,
                    token_id=bucket.token_id, p_model=p_model,
                    p_market=bucket.market_price, edge=-edge, odds=odds_no,
                    ensemble_spread=forecast.ensemble_spread,
                    forecast_age_hours=round(age_hours, 2),
                    weight=forecast_weight, side="sell",
                ))

    return signals
