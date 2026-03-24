from __future__ import annotations

from wedge.market.models import Position
from wedge.strategy.kelly import fractional_kelly
from wedge.strategy.models import EdgeSignal


def evaluate_ladder(
    signals: list[EdgeSignal],
    budget: float,
    edge_threshold: float = 0.05,
    kelly_fraction: float = 0.10,  # Reduced from 0.15
    max_bet: float = 50.0,  # Reduced from 100
    max_bet_pct: float = 0.03,  # Reduced from 0.05
    spread_baseline: float = 3.0,  # Ensemble spread baseline (°F) for Kelly damping
    fee_rate: float = 0.0,
) -> list[Position]:
    """Select ladder positions: center-region buckets with edge > threshold.

    Handles both long (buy Yes) and short (buy No) signals.
    """
    ladder_signals = [s for s in signals if s.edge > edge_threshold]
    if not ladder_signals:
        return []

    # Sort by edge descending to prioritize best opportunities
    ladder_signals.sort(key=lambda s: s.edge, reverse=True)

    positions: list[Position] = []
    remaining = budget

    for signal in ladder_signals:
        # For short signals, Kelly uses No probabilities
        if signal.side == "sell":
            kelly_p = 1.0 - signal.p_model
            kelly_price = 1.0 - signal.p_market
        else:
            kelly_p = signal.p_model
            kelly_price = signal.p_market

        result = fractional_kelly(
            p_model=kelly_p,
            market_price=kelly_price,
            bankroll=remaining,
            fraction=kelly_fraction * signal.weight,
            max_bet=max_bet,
            max_bet_pct=max_bet_pct,
            ensemble_spread=signal.ensemble_spread,
            spread_baseline=spread_baseline,
            fee_rate=fee_rate,
        )
        bet = result.bet_size
        if bet <= 0:
            continue
        if bet > remaining:
            break

        from wedge.market.models import MarketBucket

        positions.append(
            Position(
                bucket=MarketBucket(
                    token_id=signal.token_id,
                    city=signal.city,
                    date=signal.date,
                    temp_value=signal.temp_value,
                    temp_unit=signal.temp_unit,
                    market_price=signal.p_market,
                    implied_prob=signal.p_market,
                ),
                side=signal.side,
                size=bet,
                entry_price=signal.p_market if signal.side == "buy" else 1.0 - signal.p_market,
                strategy="ladder",
                p_model=signal.p_model,
                edge=signal.edge,
            )
        )
        remaining -= bet

    return positions
