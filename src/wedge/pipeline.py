from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import httpx
import structlog

from wedge.config import CityConfig, Settings
from wedge.db import Database
from wedge.execution.models import OrderRequest
from wedge.log import get_logger
from wedge.market.models import MarketBucket
from wedge.market.polymarket import PolymarketClient, PublicPolymarketClient
from wedge.market.scanner import discover_weather_markets, rank_market_buckets, scan_weather_markets
from wedge.strategy.edge import _EPS, detect_edges
from wedge.strategy.ladder import evaluate_ladder
from wedge.strategy.portfolio import allocate
from wedge.weather.client import fetch_actual_temperature, fetch_ensemble
from wedge.weather.ensemble import parse_distribution

if TYPE_CHECKING:
    from wedge.execution.dry_run import DryRunExecutor
    from wedge.execution.live import LiveExecutor
else:
    from wedge.execution.dry_run import DryRunExecutor
    from wedge.execution.live import LiveExecutor

log = get_logger("pipeline")


async def run_pipeline(settings: Settings, db: Database) -> None:
    """Execute one full trading pipeline cycle across all cities."""
    run_id = uuid.uuid4().hex[:16]
    now = datetime.now(UTC)
    structlog.contextvars.bind_contextvars(run_id=run_id)

    await db.insert_run(run_id, now.isoformat())
    log.info("pipeline_start", mode=settings.mode, bankroll=settings.bankroll)

    # Restore balance from last snapshot (persists across pipeline runs)
    current_balance = await db.get_last_balance(default=settings.bankroll)
    # Set up executor and shared Polymarket client
    # For market data: use public client (no auth needed)
    # For trading: use authenticated client (requires credentials)
    poly_client: PolymarketClient | PublicPolymarketClient | None = None

    if settings.mode == "live":
        # Live mode requires authenticated client for trading
        if not (
            settings.polymarket_private_key
            and settings.polymarket_api_key
            and settings.polymarket_api_secret
        ):
            raise ValueError("Live mode requires Polymarket API credentials")
        poly_client = PolymarketClient(
            settings.polymarket_private_key,
            settings.polymarket_api_key,
            settings.polymarket_api_secret,
        )
        await poly_client.connect()
        executor = LiveExecutor(db, poly_client, current_balance, settings.max_bet)
    else:
        # Dry-run mode: use public client for market data (no credentials needed)
        poly_client = PublicPolymarketClient()
        executor = DryRunExecutor(db, current_balance, settings.max_bet)

    # Budget allocation based on current balance, not initial bankroll
    ladder_budget, _, _ = allocate(
        current_balance,
        settings.ladder_alloc,
    )

    total_orders = 0

    cities_processed = 0
    cities_failed = 0

    # Check exit conditions for open positions before new trades
    async with httpx.AsyncClient() as _exit_http:
        await check_exit_positions(
            settings,
            db,
            executor,
            http_client=_exit_http,
        )

    city_filter = {c.name: True for c in settings.cities}

    async with httpx.AsyncClient() as http_client:
        for city_cfg in settings.cities:
            try:
                # Skip cities with poor recent forecast performance
                if not city_filter.get(city_cfg.name, True):
                    log.info("city_skipped_performance", city=city_cfg.name)
                    continue

                # Compute target dates per city timezone (contract settlement is local)
                # Only trade 1-2 day forecasts for better accuracy
                city_tz = ZoneInfo(city_cfg.timezone)
                local_today = datetime.now(city_tz).date()

                # Scan markets for next 2 days only (tomorrow and day after)
                # Skip today (0 days) as it's too close to settlement
                for days_ahead in range(1, 3):
                    target_date = local_today + timedelta(days=days_ahead)

                    orders = await _process_city(
                        http_client=http_client,
                        settings=settings,
                        db=db,
                        executor=executor,
                        city_cfg=city_cfg,
                        target_date=target_date,
                        run_id=run_id,
                        ladder_budget=ladder_budget,
                        poly_client=poly_client,
                    )
                    total_orders += orders

                    # Update position prices for dry-run mode
                    if settings.mode == "dry_run" and poly_client:
                        markets = await scan_weather_markets(
                            poly_client,
                            city_cfg.name,
                            target_date,
                            min_volume=settings.market_min_volume,
                        )
                        await executor.update_position_prices(markets)

                cities_processed += 1

            except Exception as e:
                log.error("city_failed", city=city_cfg.name, error=str(e))
                cities_failed += 1
                # Continue with next city instead of failing entire pipeline

    # Determine status based on success rate
    if cities_failed == len(settings.cities):
        status = "failed"  # All cities failed
    elif cities_failed > 0:
        status = "partial"  # Some cities failed
    else:
        status = "completed"  # All cities succeeded
    await db.complete_run(run_id, datetime.now(UTC).isoformat(), status)

    # Calculate unrealized P&L for dry-run mode
    unrealized_pnl = 0.0
    if settings.mode == "dry_run":
        unrealized_pnl = await executor.get_unrealized_pnl()

    await db.insert_bankroll_snapshot(
        await executor.get_balance(), unrealized_pnl, datetime.now(UTC).isoformat()
    )
    log.info(
        "pipeline_complete",
        status=status,
        total_orders=total_orders,
        cities_processed=cities_processed,
        cities_failed=cities_failed,
        balance=await executor.get_balance(),
        unrealized_pnl=unrealized_pnl,
    )

    structlog.contextvars.unbind_contextvars("run_id")


async def _process_city(
    *,
    http_client: httpx.AsyncClient,
    settings: Settings,
    db: Database,
    executor: DryRunExecutor | LiveExecutor,
    city_cfg: CityConfig,
    target_date: date,
    run_id: str,
    ladder_budget: float,
    poly_client: PolymarketClient | None = None,
) -> int:
    """Process a single city. Returns number of orders placed."""
    log.info("processing_city", city=city_cfg.name, date=str(target_date))

    # 1. Fetch weather data
    raw = await fetch_ensemble(
        http_client,
        city_cfg,
        target_date,
        parallel=settings.enable_parallel_noaa_fetch,
        max_concurrency=settings.readiness_fetch_concurrency,
        error_rate_threshold=settings.readiness_error_rate_threshold,
    )
    if not raw:
        log.warning("no_weather_data", city=city_cfg.name)
        return 0

    # 2. Parse distribution
    forecast = parse_distribution(raw, city_cfg.name, target_date)
    if not forecast:
        log.warning("no_distribution", city=city_cfg.name)
        return 0

    # 3. Store forecasts
    await db.insert_forecasts_batch(
        run_id=run_id,
        city=city_cfg.name,
        date=target_date.isoformat(),
        buckets=forecast.buckets,
        created_at=datetime.now(UTC).isoformat(),
    )

    # 4. Scan market (prefer real market data when available)
    discovered_at = datetime.now(UTC).isoformat()
    candidate_markets: list[MarketBucket]
    if poly_client:
        accepted_markets, rejected_markets = await discover_weather_markets(
            poly_client,
            city_cfg.name,
            target_date,
            min_volume=settings.market_min_volume,
            min_open_interest=settings.market_min_open_interest,
            max_spread=settings.market_max_spread,
        )
        ranked_markets = rank_market_buckets(
            accepted_markets,
            watchlist_size=settings.market_watchlist_size,
            rejected_buckets=rejected_markets,
        )
        await db.replace_market_discoveries(
            run_id=run_id,
            city=city_cfg.name,
            target_date=target_date.isoformat(),
            buckets=[*ranked_markets, *rejected_markets],
            discovered_at=discovered_at,
        )
        candidate_markets = [m for m in ranked_markets if m.selected_for_watchlist]
    elif settings.mode == "dry_run":
        log.warning("no_polymarket_client_using_synthetic", city=city_cfg.name)
        markets = _generate_synthetic_markets(forecast, city_cfg.name, target_date)
        candidate_markets = markets
    else:
        candidate_markets = []

    if not candidate_markets:
        log.warning("no_markets", city=city_cfg.name)
        return 0

    # 5. Detect edges
    signals = detect_edges(
        forecast,
        candidate_markets,
        ladder_threshold=settings.ladder_edge,
        fee_rate=settings.fee_rate,
        slippage_bet_size=settings.slippage_bet_size,
    )
    if not signals:
        log.info("no_edges", city=city_cfg.name)
        return 0

    log.info("edges_found", city=city_cfg.name, count=len(signals))

    # 6. Generate ladder positions only
    ladder_positions = evaluate_ladder(
        signals,
        ladder_budget,
        edge_threshold=settings.ladder_edge,
        kelly_fraction=settings.kelly_fraction,
        max_bet=settings.max_bet,
        max_bet_pct=settings.max_bet_pct,
        spread_baseline=settings.spread_baseline_f,
        fee_rate=settings.fee_rate,
    )

    # 7. Execute orders
    orders = 0
    for pos in ladder_positions:
        if await db.has_open_position(
            pos.bucket.city, pos.bucket.date.isoformat(), pos.bucket.temp_value
        ):
            log.info(
                "position_exists_skipping",
                city=pos.bucket.city,
                date=str(pos.bucket.date),
                temp_value=pos.bucket.temp_value,
            )
            continue

        request = OrderRequest(
            run_id=run_id,
            token_id=pos.bucket.token_id,
            city=pos.bucket.city,
            date=pos.bucket.date,
            temp_value=pos.bucket.temp_value,
            temp_unit=pos.bucket.temp_unit,
            strategy=pos.strategy,
            limit_price=pos.entry_price,
            size=pos.size,
            p_model=pos.p_model,
            p_market=pos.entry_price,
            edge=pos.edge,
        )
        result = await executor.place_order(request)
        if result.success:
            orders += 1

    return orders


async def check_exit_positions(
    settings: Settings,
    db: Database,
    executor: DryRunExecutor | LiveExecutor,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> int:
    """Check all open positions and exit those where probability has turned against us.

    Exit rules (probability-based):
    - Stop-loss: p_model < entry_price * exit_loss_factor  (probability collapsed)
    - Trailing stop: profit > activation_pct AND p_model < peak * (1 - trailing_pct)
    - Take-profit: p_model >= entry_price but EV <= exit_min_ev  (edge is gone)

    Returns number of positions closed.
    """


    positions = await db.get_open_positions()
    if not positions:
        return 0

    closed = 0
    own_client = False
    if http_client is None:
        http_client = httpx.AsyncClient()
        own_client = True

    try:
        for pos in positions:
            city_name = pos["city"]
            date_str = pos["date"]
            temp_f = pos["temp_value"]
            entry_price = pos["entry_price"]
            size = pos["size"]

            # Skip if settlement is imminent (avoid noise near expiry)
            try:
                settle_date = date.fromisoformat(date_str)
                city_cfg_tz = next((c for c in settings.cities if c.name == city_name), None)
                if city_cfg_tz:
                    city_tz = ZoneInfo(city_cfg_tz.timezone)
                    now_local = datetime.now(city_tz)
                    settle_dt = datetime(
                        settle_date.year,
                        settle_date.month,
                        settle_date.day,
                        23,
                        59,
                        tzinfo=city_tz,
                    )
                    hours_to_settle = (settle_dt - now_local).total_seconds() / 3600
                    if hours_to_settle <= settings.exit_min_hours_to_settle:
                        log.info(
                            "exit_skip_near_settlement",
                            city=city_name,
                            date=date_str,
                            hours_remaining=round(hours_to_settle, 1),
                        )
                        continue
            except Exception:
                pass

            # Find city config
            city_cfg = next((c for c in settings.cities if c.name == city_name), None)
            if not city_cfg:
                continue

            # Re-fetch latest forecast for this city
            try:
                target_date = date.fromisoformat(date_str)
                raw = await fetch_ensemble(http_client, city_cfg, target_date)
                if not raw:
                    continue
                forecast = parse_distribution(raw, city_name, target_date)
                if not forecast:
                    continue
            except Exception as e:
                log.warning("exit_check_forecast_failed", city=city_name, error=str(e))
                continue

            # Get latest p_model for this bucket
            p_model = forecast.buckets.get(int(temp_f))
            if p_model is None:
                # Try nearest bucket
                nearest = min(forecast.buckets.keys(), key=lambda t: abs(t - temp_f), default=None)
                if nearest is None:
                    continue
                p_model = forecast.buckets[nearest]

            # Track peak p_model for trailing stop
            peak_p_model = pos.get("peak_p_model", p_model)
            if peak_p_model is None:
                peak_p_model = p_model
            if p_model > peak_p_model:
                peak_p_model = p_model
                await db.update_peak_p_model(city_name, date_str, temp_f, peak_p_model)

            # Exit check uses p_model directly (model's fair value)
            side = pos.get("side", "buy")  # default buy for legacy rows
            exit_reason: str | None = None

            if side == "sell":  # bought No — profit when p_model stays LOW
                no_entry = 1.0 - entry_price  # price paid for No contract
                no_p_model = 1.0 - p_model    # model's fair value for No
                # Stop-loss: model now agrees temperature WILL happen (No collapsed)
                if p_model > (1.0 - no_entry * settings.exit_loss_factor):
                    exit_reason = "stop_loss"
                elif settings.trailing_activation_pct > 0 and no_entry > 0:
                    # Trailing stop on No side: No price risen from peak, now falling
                    profit_pct = (no_p_model - no_entry) / no_entry
                    if profit_pct >= settings.trailing_activation_pct:
                        trail_line = peak_p_model * (1.0 - settings.trailing_pct)
                        if p_model > trail_line:
                            exit_reason = "trailing_stop"
                if exit_reason is None and no_p_model < no_entry:
                    # Edge gone: model now thinks No is fairly priced or worse
                    exit_reason = "take_profit"
            else:  # bought Yes
                profit_pct = (p_model - entry_price) / entry_price if entry_price > 0 else 0
                if p_model < entry_price * settings.exit_loss_factor:
                    exit_reason = "stop_loss"
                elif settings.trailing_activation_pct > 0 and entry_price > 0:
                    profit_pct = (p_model - entry_price) / entry_price
                    if profit_pct >= settings.trailing_activation_pct:
                        trail_line = peak_p_model * (1.0 - settings.trailing_pct)
                        if p_model < trail_line:
                            exit_reason = "trailing_stop"
                if exit_reason is None and p_model < entry_price:
                    exit_reason = "take_profit"
            if exit_reason is None:
                # Log trailing stop status if activated
                if settings.trailing_activation_pct > 0 and entry_price > 0:
                    profit_pct = (p_model - entry_price) / entry_price
                    if profit_pct >= settings.trailing_activation_pct:
                        trail_line = peak_p_model * (1.0 - settings.trailing_pct)
                        log.info(
                            "exit_check_trailing_active",
                            city=city_name,
                            temp_f=temp_f,
                            profit_pct=round(profit_pct, 3),
                            peak_p_model=round(peak_p_model, 4),
                            trail_line=round(trail_line, 4),
                            p_model=round(p_model, 4),
                        )
                    else:
                        log.info(
                            "exit_check_hold",
                            city=city_name,
                            temp_f=temp_f,
                            entry_price=entry_price,
                            p_model=round(p_model, 4),
                            ev=round(ev, 4),
                            profit_pct=round(profit_pct, 3),
                        )
                else:
                    log.info(
                        "exit_check_hold",
                        city=city_name,
                        temp_f=temp_f,
                        entry_price=entry_price,
                        p_model=round(p_model, 4),
                        ev=round(ev, 4),
                    )
                # Check partial tier exits (scale out of profitable positions)
                if settings.exit_tier_pcts and settings.exit_tier_portions:
                    remaining = pos.get("remaining_size", size) or size
                    if remaining > 0 and entry_price > 0:
                        existing_exits = await db.get_tier_exits(city_name, date_str, temp_f)
                        completed_tiers = {e["tier_index"] for e in existing_exits}
                        for tier_idx, (tier_pct, portion) in enumerate(
                            zip(settings.exit_tier_pcts, settings.exit_tier_portions)
                        ):
                            if tier_idx in completed_tiers:
                                continue
                            profit_pct = (p_model - entry_price) / entry_price
                            if profit_pct >= tier_pct:
                                # Partial exit: sell `portion` of remaining
                                shares_total = remaining / entry_price
                                shares_sold = shares_total * portion
                                exit_value = shares_sold * p_model
                                cost_sold = remaining * portion
                                tier_pnl = exit_value - cost_sold
                                new_remaining = remaining * (1.0 - portion)
                                # Credit balance
                                executor._balance += exit_value
                                await db.record_tier_exit(
                                    city=city_name,
                                    date_str=date_str,
                                    temp_f=temp_f,
                                    tier_index=tier_idx,
                                    exit_price=round(p_model, 4),
                                    shares_sold=round(shares_sold, 4),
                                    pnl=round(tier_pnl, 4),
                                    new_remaining_size=round(new_remaining, 4),
                                )
                                log.info(
                                    "tier_exit",
                                    city=city_name,
                                    temp_f=temp_f,
                                    tier_index=tier_idx,
                                    tier_pct=tier_pct,
                                    profit_pct=round(profit_pct, 3),
                                    portion=portion,
                                    shares_sold=round(shares_sold, 2),
                                    exit_price=round(p_model, 4),
                                    pnl=round(tier_pnl, 4),
                                    remaining=round(new_remaining, 4),
                                )
                                remaining = new_remaining
                continue

            # Exit at current p_model as proxy for fair exit price
            exit_price = round(p_model, 4)
            pnl = await executor.close_position(
                city=city_name,
                date_str=date_str,
                temp_f=temp_f,
                exit_price=exit_price,
                exit_reason=exit_reason,
                db=db,
            )
            closed += 1

            log.info(
                "position_exited",
                city=city_name,
                temp_f=temp_f,
                exit_reason=exit_reason,
                p_model=round(p_model, 4),
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=round(pnl, 4),
            )


    finally:
        if own_client:
            await http_client.aclose()

    if closed:
        log.info("exit_check_complete", closed=closed)
    return closed


def _generate_synthetic_markets(forecast, city: str, target_date: date) -> list[MarketBucket]:
    """Generate synthetic market buckets for dry-run testing.
    Simulates market inefficiency by adding noise to model probabilities.
    Uses realistic ±2% noise to match real Polymarket spreads.
    Seeded by city+date for reproducibility."""
    import random

    rng = random.Random(f"{city}_{target_date}")
    markets = []
    for temp_f, p_model in forecast.buckets.items():
        # Realistic noise: ±2% to match actual Polymarket spreads
        noise = rng.uniform(-0.02, 0.02)
        market_price = max(0.01, min(0.99, p_model + noise))
        markets.append(
            MarketBucket(
                token_id=f"syn_{city}_{target_date}_{temp_f}",
                city=city,
                date=target_date,
                temp_value=temp_f,
                temp_unit="F",  # Synthetic markets always use °F
                market_price=round(market_price, 2),
                implied_prob=round(market_price, 2),
            )
        )
    return markets


async def run_settlement(
    settings: Settings, db: Database
) -> int:
    """Settle all trades whose target date has passed.

    Fetches actual observed temperatures and updates forecasts + trades.
    Returns total number of trades settled.

    Includes retry logic for API failures (3 attempts with exponential backoff).
    """
    import asyncio

    unsettled = await db.get_unsettled_dates()
    if not unsettled:
        log.info("settlement_no_pending")
        return 0

    log.info("settlement_start", pending_pairs=len(unsettled))

    city_map = {c.name: c for c in settings.cities}
    total_settled = 0
    pending_retry: list[tuple[str, str]] = []  # (city, date) pairs for retry

    async with httpx.AsyncClient() as http_client:
        for city_name, trade_date in unsettled:
            city_cfg = city_map.get(city_name)
            if not city_cfg:
                log.warning("settlement_unknown_city", city=city_name)
                continue

            # Retry logic: 3 attempts with exponential backoff
            actual_temp = None
            for attempt in range(1, 4):
                try:
                    actual_temp = await fetch_actual_temperature(http_client, city_cfg, trade_date)
                    if actual_temp is not None:
                        break
                except Exception as e:
                    log.warning(
                        "fetch_temp_retry",
                        city=city_name,
                        date=trade_date,
                        attempt=attempt,
                        error=str(e),
                    )
                    if attempt < 3:
                        await asyncio.sleep(2**attempt)  # Exponential backoff

            if actual_temp is None:
                log.warning(
                    "settlement_no_actual",
                    city=city_name,
                    date=trade_date,
                    attempts=3,
                )
                pending_retry.append((city_name, trade_date))
                continue

            # Update forecast and settle trades
            await db.update_forecast_actual(city_name, trade_date, actual_temp)
            count = await db.settle_trades(
                city_name, trade_date, actual_temp, fee_rate=settings.fee_rate
            )
            total_settled += count
            log.info(
                "settlement_settled",
                city=city_name,
                date=trade_date,
                actual_temp=actual_temp,
                trades_settled=count,
            )

    # Log pending retries for manual intervention
    if pending_retry:
        log.warning(
            "settlement_pending_retry",
            count=len(pending_retry),
            pairs=pending_retry,
        )



    log.info("settlement_complete", total_settled=total_settled)

    return total_settled


async def run_single_scan(settings: Settings, city_name: str) -> None:
    """Run a single scan for one city (CLI scan command)."""
    from wedge.log import setup_logging

    setup_logging()

    city_cfg = next((c for c in settings.cities if c.name.lower() == city_name.lower()), None)
    if not city_cfg:
        log.error("city_not_found", city=city_name)
        return

    city_tz = ZoneInfo(city_cfg.timezone)
    target_date = (datetime.now(city_tz) + timedelta(days=3)).date()

    async with httpx.AsyncClient() as http_client:
        raw = await fetch_ensemble(http_client, city_cfg, target_date)
        if not raw:
            log.error("no_weather_data", city=city_name)
            return

        forecast = parse_distribution(raw, city_cfg.name, target_date)
        if not forecast:
            log.error("no_distribution", city=city_name)
            return

        log.info(
            "forecast_distribution",
            city=city_name,
            date=str(target_date),
            members=forecast.member_count,
            spread=f"{forecast.ensemble_spread:.1f}°F",
        )

        for temp_f in sorted(forecast.buckets):
            prob = forecast.buckets[temp_f]
            log.info("bucket", temp_f=temp_f, probability=f"{prob:.1%}")


async def run_market_exit_check(
    settings: Settings,
    db: Database,
    executor: DryRunExecutor | LiveExecutor,
    poly_client: PolymarketClient | PublicPolymarketClient | None = None,
) -> int:
    """Check exits using real market prices (polled every N minutes).

    This runs independently from the weather pipeline. It fetches current
    Polymarket prices for open positions and checks:
    - Stop-loss: market_price < entry_price * exit_loss_factor
    - Trailing stop: profit > activation, then market_price drops from peak
    - Tier exits: profit % hits tier thresholds

    Does NOT check take-profit (EV-based) since that needs p_model.
    Returns number of positions closed.
    """
    if poly_client is None:
        return 0

    positions = await db.get_open_positions()
    if not positions:
        return 0

    closed = 0
    async with httpx.AsyncClient() as http_client:
        for pos in positions:
            city_name = pos["city"]
            date_str = pos["date"]
            temp_f = pos["temp_value"]
            entry_price = pos["entry_price"]
            size = pos["size"]

            # Skip if settlement is imminent
            try:
                settle_date = date.fromisoformat(date_str)
                city_cfg = next((c for c in settings.cities if c.name == city_name), None)
                if city_cfg:
                    city_tz = ZoneInfo(city_cfg.timezone)
                    now_local = datetime.now(city_tz)
                    settle_dt = datetime(
                        settle_date.year, settle_date.month, settle_date.day,
                        23, 59, tzinfo=city_tz,
                    )
                    hours_to_settle = (settle_dt - now_local).total_seconds() / 3600
                    if hours_to_settle <= settings.exit_min_hours_to_settle:
                        continue
            except Exception:
                pass

            # Fetch current market price for this bucket
            try:
                city_cfg = next((c for c in settings.cities if c.name == city_name), None)
                if not city_cfg:
                    continue
                target_date = date.fromisoformat(date_str)
                markets = await scan_weather_markets(
                    poly_client,
                    city_name,
                    target_date,
                    min_volume=settings.market_min_volume,
                )
                # Find matching market bucket
                matching = [m for m in markets if m.temp_value == temp_f]
                if not matching:
                    continue
                market_price = matching[0].market_price
            except Exception as e:
                log.warning("exit_check_market_fetch_failed", city=city_name, error=str(e))
                continue

            if not (_EPS < market_price < 1 - _EPS):
                continue

            # Track peak market price for trailing stop
            peak_p_model = pos.get("peak_p_model", market_price)
            if peak_p_model is None:
                peak_p_model = market_price
            if market_price > peak_p_model:
                peak_p_model = market_price
                await db.update_peak_p_model(city_name, date_str, temp_f, peak_p_model)

            remaining = pos.get("remaining_size", size) or size
            side = pos.get("side", "buy")  # default buy for legacy rows
            exit_reason: str | None = None

            if side == "sell":  # bought No — profit when market_price (No price) stays HIGH
                no_entry = 1.0 - entry_price   # price paid for No
                no_market = 1.0 - market_price  # current No market price
                # 1. Stop-loss: No price collapsed (market now believes temp WILL happen)
                if no_market < no_entry * settings.exit_loss_factor:
                    exit_reason = "stop_loss"
                # 2. Trailing stop: No price risen then fallen from peak
                elif settings.trailing_activation_pct > 0 and no_entry > 0:
                    profit_pct = (no_market - no_entry) / no_entry
                    if profit_pct >= settings.trailing_activation_pct:
                        trail_line = (1.0 - peak_p_model) * (1.0 - settings.trailing_pct)
                        if no_market < trail_line:
                            exit_reason = "trailing_stop"
            else:  # bought Yes
                # 1. Stop-loss: market price collapsed
                if market_price < entry_price * settings.exit_loss_factor:
                    exit_reason = "stop_loss"
                # 2. Trailing stop: market price dropped from peak
                elif settings.trailing_activation_pct > 0 and entry_price > 0:
                    profit_pct = (market_price - entry_price) / entry_price
                    if profit_pct >= settings.trailing_activation_pct:
                        trail_line = peak_p_model * (1.0 - settings.trailing_pct)
                        if market_price < trail_line:
                            exit_reason = "trailing_stop"
            # 3. Tier exits (partial)
            if exit_reason is None and settings.exit_tier_pcts and settings.exit_tier_portions:
                if remaining > 0 and entry_price > 0:
                    existing_exits = await db.get_tier_exits(city_name, date_str, temp_f)
                    completed_tiers = {e["tier_index"] for e in existing_exits}
                    for tier_idx, (tier_pct, portion) in enumerate(
                        zip(settings.exit_tier_pcts, settings.exit_tier_portions)
                    ):
                        if tier_idx in completed_tiers:
                            continue
                        profit_pct = (market_price - entry_price) / entry_price
                        if profit_pct >= tier_pct:
                            shares_total = remaining / entry_price
                            shares_sold = shares_total * portion
                            exit_value = shares_sold * market_price
                            cost_sold = remaining * portion
                            tier_pnl = exit_value - cost_sold
                            new_remaining = remaining * (1.0 - portion)
                            executor._balance += exit_value
                            await db.record_tier_exit(
                                city=city_name, date_str=date_str, temp_f=temp_f,
                                tier_index=tier_idx, exit_price=round(market_price, 4),
                                shares_sold=round(shares_sold, 4), pnl=round(tier_pnl, 4),
                                new_remaining_size=round(new_remaining, 4),
                            )
                            log.info(
                                "market_tier_exit", city=city_name, temp_f=temp_f,
                                tier_index=tier_idx, market_price=round(market_price, 4),
                                profit_pct=round(profit_pct, 3), pnl=round(tier_pnl, 4),
                            )
                            remaining = new_remaining

            if exit_reason is None:
                continue

            # Full exit at market price
            exit_price = round(market_price, 4)
            pnl = await executor.close_position(
                city=city_name, date_str=date_str, temp_f=temp_f,
                exit_price=exit_price, exit_reason=exit_reason, db=db,
            )
            closed += 1
            log.info(
                "market_position_exited", city=city_name, temp_f=temp_f,
                exit_reason=exit_reason, market_price=round(market_price, 4),
                entry_price=entry_price, pnl=round(pnl, 4),
            )

    if closed:
        log.info("market_exit_check_complete", closed=closed)
    return closed
