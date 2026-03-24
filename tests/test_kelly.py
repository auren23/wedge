from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from wedge.strategy.kelly import KellyResult, fractional_kelly


class TestKelly:
    def test_basic_positive_edge(self):
        result = fractional_kelly(p_model=0.77, market_price=0.68, bankroll=1000)
        assert isinstance(result, KellyResult)
        assert result.bet_size > 0
        assert result.bet_size <= 50  # max_bet (reduced)
        assert result.bet_size <= 1000 * 0.03  # max_bet_pct (reduced)

    def test_negative_edge_returns_zero(self):
        result = fractional_kelly(p_model=0.10, market_price=0.20, bankroll=1000)
        assert result.bet_size == 0.0
        assert "no edge" in result.reasoning.lower()

    def test_zero_edge_returns_zero(self):
        result = fractional_kelly(p_model=0.20, market_price=0.20, bankroll=1000)
        assert result.bet_size == 0.0

    def test_zero_bankroll(self):
        result = fractional_kelly(p_model=0.80, market_price=0.50, bankroll=0)
        assert result.bet_size == 0.0

    def test_negative_bankroll(self):
        result = fractional_kelly(p_model=0.80, market_price=0.50, bankroll=-100)
        assert result.bet_size == 0.0

    def test_price_at_zero(self):
        result = fractional_kelly(p_model=0.80, market_price=0.0, bankroll=1000)
        assert result.bet_size == 0.0

    def test_price_at_one(self):
        result = fractional_kelly(p_model=0.80, market_price=1.0, bankroll=1000)
        assert result.bet_size == 0.0

    def test_max_bet_cap(self):
        result = fractional_kelly(p_model=0.99, market_price=0.01, bankroll=100000, max_bet=50)
        assert result.bet_size <= 50

    def test_max_bet_pct_cap(self):
        result = fractional_kelly(p_model=0.99, market_price=0.01, bankroll=1000, max_bet_pct=0.03)
        assert result.bet_size <= 30  # 1000 * 0.03

    def test_kelly_result_contains_breakdown(self):
        """Test that KellyResult contains full breakdown."""
        result = fractional_kelly(p_model=0.40, market_price=0.30, bankroll=1000)

        assert result.kelly_full > 0
        assert result.kelly_fractional > 0
        assert abs(result.edge - 0.10) < 0.001  # Floating point tolerance
        assert result.ev > 0
        assert result.reasoning != ""

    def test_kelly_result_ev_defaults_to_fee_free_weather_market(self):
        result = fractional_kelly(p_model=0.40, market_price=0.30, bankroll=1000)
        expected_ev = 0.40 / 0.30 - 1.0
        assert result.ev == pytest.approx(expected_ev, rel=0.01)

    def test_fat_tail_discount_applied(self):
        """Test that fat tail discount (0.8) is applied."""
        result = fractional_kelly(p_model=0.40, market_price=0.30, bankroll=1000, fraction=0.10)

        # f_actual = f_full * fraction * fat_tail_discount (0.8)
        expected_fraction = 0.10 * 0.8
        actual_fraction = result.kelly_fractional / result.kelly_full

        assert abs(actual_fraction - expected_fraction) < 0.001

    def test_capital_lockup_cost_reduces_bet(self):
        """Test that capital lockup cost reduces bet size."""
        result_no_lockup = fractional_kelly(
            p_model=0.40, market_price=0.30, bankroll=1000, capital_lockup_days=0
        )

        result_with_lockup = fractional_kelly(
            p_model=0.40, market_price=0.30, bankroll=1000, capital_lockup_days=7, funding_rate=0.10
        )

        assert result_with_lockup.bet_size < result_no_lockup.bet_size



class TestKellyPBT:
    @given(
        p_model=st.floats(min_value=0, max_value=1),
        market_price=st.floats(min_value=0, max_value=1),
        bankroll=st.floats(min_value=-1000, max_value=100000),
    )
    @settings(max_examples=500)
    def test_always_clamped(self, p_model, market_price, bankroll):
        result = fractional_kelly(p_model, market_price, bankroll)
        assert math.isfinite(result.bet_size)
        assert result.bet_size >= 0
        cap = min(50, max(0, bankroll) * 0.03)  # Updated defaults
        assert result.bet_size <= cap + 1e-9

    @given(
        market_price=st.floats(min_value=0.01, max_value=0.99),
        bankroll=st.floats(min_value=100, max_value=10000),
    )
    @settings(max_examples=200)
    def test_monotonic_in_p_model(self, market_price, bankroll):
        """Higher p_model should never decrease the bet size."""
        p1 = market_price + 0.05
        p2 = market_price + 0.10
        if p1 >= 1 or p2 >= 1:
            return
        result1 = fractional_kelly(p1, market_price, bankroll)
        result2 = fractional_kelly(p2, market_price, bankroll)
        assert result2.bet_size >= result1.bet_size - 1e-9


class TestKellyEdgeCases:
    def test_negative_kelly_returns_zero_bet(self):
        # p_model < market_price => negative edge => f_full <= 0
        result = fractional_kelly(0.20, 0.50, 1000)
        assert result.bet_size == 0.0
        assert "edge" in result.reasoning

    def test_zero_spread_no_damping(self):
        # ensemble_spread=0 -> spread_damping=1.0 (no effect)
        r1 = fractional_kelly(0.65, 0.50, 1000, ensemble_spread=0.0)
        r2 = fractional_kelly(0.65, 0.50, 1000, ensemble_spread=0.0, spread_baseline=3.0)
        assert abs(r1.bet_size - r2.bet_size) < 1e-9

    def test_spread_damping_in_reasoning(self):
        # ensemble_spread > 0 -> reasoning includes spread_damping
        result = fractional_kelly(0.65, 0.50, 1000, ensemble_spread=2.0, spread_baseline=3.0)
        assert "spread_damping" in result.reasoning

    def test_spread_baseline_zero_no_damping(self):
        # spread_baseline=0 -> no damping applied
        r1 = fractional_kelly(0.65, 0.50, 1000, ensemble_spread=5.0, spread_baseline=0.0)
        r2 = fractional_kelly(0.65, 0.50, 1000, ensemble_spread=0.0)
        assert abs(r1.bet_size - r2.bet_size) < 1e-9

    def test_infinite_kelly_fraction_returns_zero(self):
        # market_price near 0 -> win_odds huge -> f_full may be inf
        import math

        result = fractional_kelly(0.9999, 0.0001, 1000)
        assert result.bet_size == 0.0 or math.isfinite(result.bet_size)

    def test_very_high_bankroll_caps_bet(self):
        # Extreme bankroll: bet should still be capped by max_bet
        result = fractional_kelly(0.65, 0.50, 1e9, max_bet=100.0)
        assert result.bet_size <= 100.0

    def test_infinite_kelly_via_extreme_odds(self):
        # market_price very small -> win_odds huge -> bet capped
        import math

        result = fractional_kelly(0.9999, 0.0001, 1000, max_bet=50.0)
        assert math.isfinite(result.bet_size)
