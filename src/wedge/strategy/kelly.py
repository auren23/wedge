from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-6


@dataclass
class KellyResult:
    """Kelly calculation result with breakdown."""

    bet_size: float
    kelly_full: float  # Full Kelly fraction
    kelly_fractional: float  # Actual fraction used
    edge: float  # p_model - p_market
    ev: float  # Expected value
    reasoning: str = ""  # Explanation if bet was capped/reduced


def fractional_kelly(
    p_model: float,
    market_price: float,
    bankroll: float,
    fraction: float = 0.10,  # Reduced from 0.15 to 0.10 for more conservatism
    max_bet: float = 50.0,  # Reduced from 100 to 50
    max_bet_pct: float = 0.03,  # Reduced from 5% to 3%
    capital_lockup_days: int = 3,
    funding_rate: float = 0.08,  # Annual funding cost
    ensemble_spread: float = 0.0,  # Forecast ensemble spread in °F
    spread_baseline: float = 3.0,  # Baseline spread for damping normalization
    fee_rate: float = 0.0,  # Weather markets are fee-free by default
) -> KellyResult:
    """Calculate fractional Kelly bet size with binary option corrections.

    Uses Thorp's Binary Kelly formula with additional safety margins for:
    - Binary option risk (100% loss possible)
    - Fat-tail weather events
    - Capital lockup costs
    - Ensemble spread damping (high spread = high uncertainty = smaller bet)

    Args:
        p_model: Model probability
        market_price: Market price (0-1)
        bankroll: Available bankroll
        fraction: Kelly fraction (default 10% for binary options)
        max_bet: Maximum bet in USD (default $50)
        max_bet_pct: Maximum bet as % of bankroll (default 3%)
        capital_lockup_days: Days until settlement
        funding_rate: Annual funding cost rate
        ensemble_spread: Forecast ensemble spread in °F (0 = no damping)
        spread_baseline: Spread at which damping = 0.5 (default 3.0°F)

    Returns:
        KellyResult with bet size and breakdown
    """
    if bankroll <= 0:
        return KellyResult(
            bet_size=0.0,
            kelly_full=0.0,
            kelly_fractional=0.0,
            edge=0.0,
            ev=0.0,
            reasoning="bankroll <= 0",
        )

    if not (_EPS < market_price < 1 - _EPS):
        return KellyResult(
            bet_size=0.0,
            kelly_full=0.0,
            kelly_fractional=0.0,
            edge=0.0,
            ev=0.0,
            reasoning=f"invalid market_price: {market_price}",
        )

    if p_model <= market_price:
        return KellyResult(
            bet_size=0.0,
            kelly_full=0.0,
            kelly_fractional=0.0,
            edge=p_model - market_price,
            ev=0.0,
            reasoning="no edge (p_model <= p_market)",
        )

    # Binary Kelly formula (Thorp correction for 100% loss risk)
    win_prob = p_model
    lose_prob = 1.0 - p_model
    win_odds = (1.0 - market_price) / market_price
    loss_odds = 1.0  # Binary option: lose 100% of stake

    # Binary Kelly: f* = (p*odds - q) / odds
    # This is more conservative than standard Kelly for binary outcomes
    f_full = (win_prob * win_odds - lose_prob * loss_odds) / win_odds

    if f_full <= 0 or not math.isfinite(f_full):  # pragma: no cover
        return KellyResult(
            bet_size=0.0,
            kelly_full=f_full if math.isfinite(f_full) else 0.0,
            kelly_fractional=0.0,
            edge=p_model - market_price,
            ev=0.0,
            reasoning="negative or infinite Kelly fraction",
        )

    # Apply fractional Kelly with additional safety margin for fat tails
    # Weather events have fat tails - extreme events more likely than normal distribution
    fat_tail_discount = 0.8  # 20% additional safety margin

    # Ensemble spread damping: high forecast uncertainty -> smaller bet
    # damping = 1 / (1 + spread / baseline), range (0, 1]
    # spread=0 -> damping=1.0 (no effect)
    # spread=baseline -> damping=0.5
    # spread=2*baseline -> damping=0.33
    if ensemble_spread > 0 and spread_baseline > 0:
        spread_damping = 1.0 / (1.0 + ensemble_spread / spread_baseline)
    else:
        spread_damping = 1.0

    f_actual = f_full * fraction * fat_tail_discount * spread_damping

    # Calculate raw bet size
    bet = f_actual * bankroll

    # Apply capital lockup cost
    # Money locked for N days has opportunity cost
    daily_funding = funding_rate / 365.0
    lockup_cost = daily_funding * capital_lockup_days
    bet *= max(0.0, 1.0 - lockup_cost)  # Reduce bet by lockup cost, floor at 0

    # Apply hard caps
    cap = min(max_bet, bankroll * max_bet_pct)
    original_bet = bet
    if bet > cap:
        bet = cap

    # Build reasoning
    reasoning_parts = []
    if original_bet > cap:
        reasoning_parts.append(f"capped at {cap:.2f}")
    if lockup_cost > 0:
        reasoning_parts.append(f"lockup cost: {lockup_cost * 100:.2f}%")
    if ensemble_spread > 0:
        reasoning_parts.append(
            f"spread_damping: {spread_damping:.2f} "
            f"(spread={ensemble_spread:.1f}°F)"
        )
    reasoning_parts.append(f"edge: {(p_model - market_price) * 100:.1f}%")

    # Calculate EV for reporting
    ev = p_model * (1 - fee_rate) / market_price - 1.0

    if not math.isfinite(bet):  # pragma: no cover
        return KellyResult(
            bet_size=0.0,
            kelly_full=f_full,
            kelly_fractional=f_actual,
            edge=p_model - market_price,
            ev=ev,
            reasoning="infinite bet size",
        )

    return KellyResult(
        bet_size=max(0.0, bet),
        kelly_full=f_full,
        kelly_fractional=f_actual,
        edge=p_model - market_price,
        ev=ev,
        reasoning="; ".join(reasoning_parts) if reasoning_parts else "ok",
    )

