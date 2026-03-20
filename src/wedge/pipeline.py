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
from wedge.market.scanner import scan_weather_markets
from wedge.strategy.edge import detect_edges
from wedge.strategy.ladder import evaluate_ladder
from wedge.strategy.arbitrage import detect_bucket_arbitrage
from wedge.strategy.performance import update_city_performance, get_city_filter, update_all_city_performance
from wedge.strategy.portfolio import allocate
from wedge.strategy.tail import evaluate_tail
from wedge.weather.client import fetch_actual_temperature, fetch_ensemble, fetch_ensemble_auto
from wedge.weather.ensemble import parse_distribution

if TYPE_CHECKING:
    from wedge.execution.dry_run import DryRunExecutor
    from wedge.execution.live import LiveExecutor
else:
    from wedge.execution.dry_run import DryRunExecutor
    from wedge.execution.live import LiveExecutor

log = get_logger("pipeline")


async def run_pipeline(
    settings: Settings, db: Database, *, notifier: object | None = None
) -> None:
    """Execute one full trading pipeline cycle across all cities."""
    run_id = uuid.uuid4().hex[:16]
    now = datetime.now(UTC)
    structlog.contextvars.bind_contextvars(run_id=run_id)

    await db.insert_run(run_id, now.isoformat())
    log.info("pipeline_start", mode=settings.mode, bankroll=settings.bankroll)

    # Check Brier score before trading (weekly window)
    brier = await db.get_brier_score(days=7)
    if brier is not None and brier > settings.brier_threshold:
        log.warning(
            "brier_threshold_exceeded",
            brier=f"{brier:.4f}",
            threshold=settings.brier_threshold,
            action="skipping_trading",
        )
        await db.complete_run(run_id, datetime.now(UTC).isoformat(), "paused_brier")
        if notifier and hasattr(notifier, "send"):
            await notifier.send(
                f"⚠️ Trading paused: Brier score {brier:.4f} exceeds threshold {settings.brier_threshold}"
            )
        structlog.contextvars.unbind_contextvars("run_id")
        return

    # Restore balance from last snapshot (persists across pipeline runs)
    current_balance = await db.get_last_balance(default=settings.bankroll)

    # Set up executor and shared Polymarket client
    # For market data: use public client (no auth needed)
    # For trading: use authenticated client (requires credentials)
    poly_client: PolymarketClient | PublicPolymarketClient | None = None

    if settings.mode == "live":
        # Live mode requires authenticated client for trading
        if not (settings.polymarket_private_key and settings.polymarket_api_key and settings.polymarket_api_secret):
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
    ladder_budget, tail_budget, _ = allocate(
        current_balance,
        settings.ladder_alloc,
        settings.tail_alloc,
    )

    total_orders = 0

    cities_processed = 0
    cities_failed = 0

    # Check exit conditions for open positions before new trades
    async with httpx.AsyncClient() as _exit_http:
        await check_exit_positions(settings, db, executor, http_client=_exit_http, notifier=notifier)

    # Pre-fetch city performance filter (batch)
    city_names = [c.name for c in settings.cities]
    city_filter = await get_city_filter(
        db,
        city_names,
        max_brier=settings.min_city_brier_score,
        window_days=30,
    )

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
                        tail_budget=tail_budget,
                        poly_client=poly_client,
                    )
                    total_orders += orders

                    # Update position prices for dry-run mode
                    if settings.mode == "dry_run" and poly_client:
                        markets = await scan_weather_markets(
                            poly_client, city_cfg.name, target_date
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

    # Send notification if notifier is available
    if notifier and hasattr(notifier, "send"):
        from wedge.monitoring.notify import format_pipeline_summary

        summary = format_pipeline_summary(
            mode=settings.mode,
            cities=[c.name for c in settings.cities],
            edges_found=total_orders,  # approximate
            orders_placed=total_orders,
            balance=await executor.get_balance(),
        )
        await notifier.send(summary)

        # Also send positions summary if there are open positions
        positions = await db.get_open_positions()
        if positions:
            from wedge.monitoring.notify import format_positions
            await notifier.send(format_positions(positions))

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
    tail_budget: float,
    poly_client: PolymarketClient | None = None,
) -> int:
    """Process a single city. Returns number of orders placed."""
    log.info("processing_city", city=city_cfg.name, date=str(target_date))

    # 1. Fetch weather data
    raw = await fetch_ensemble_auto(http_client, city_cfg, settings, target_date)
    if not raw:
        log.warning("no_weather_data", city=city_cfg.name)
        return 0

    # 2. Parse distribution
    forecast = parse_distribution(raw, city_cfg.name, target_date)
    if not forecast:
        log.warning("no_distribution", city=city_cfg.name)
        return 0

    # 3. Store forecasts
    for temp_f, prob in forecast.buckets.items():
        await db.insert_forecast(
            run_id=run_id,
            city=city_cfg.name,
            date=target_date.isoformat(),
            temp_f=temp_f,  # Forecast always uses °F
            p_model=prob,
            created_at=datetime.now(UTC).isoformat(),
        )

    # 4. Scan market (prefer real market data when available)
    if poly_client:
        markets = await scan_weather_markets(poly_client, city_cfg.name, target_date)
    elif settings.mode == "dry_run":
        # Fallback to synthetic for dry-run without API credentials
        log.warning("no_polymarket_client_using_synthetic", city=city_cfg.name)
        markets = _generate_synthetic_markets(forecast, city_cfg.name, target_date)
    else:
        # Live mode requires real market data
        markets = []

    if not markets:
        log.warning("no_markets", city=city_cfg.name)
        return 0

    # 5. Detect edges (with calibration, fees, slippage)
    signals = detect_edges(
        forecast,
        markets,
        ladder_threshold=settings.ladder_edge,
        tail_threshold=settings.tail_edge,
        fee_rate=settings.fee_rate,
        target_date=target_date,
    )
    # 5b. Cross-bucket arbitrage detection (structural opportunity, no model needed)
    arb_signal = detect_bucket_arbitrage(
        markets,
        min_buckets=2,
    )
    if arb_signal:
        log.info(
            "arbitrage_opportunity",
            city=city_cfg.name,
            date=str(target_date),
            price_sum=round(arb_signal.price_sum, 4),
            gap=round(arb_signal.gap, 4),
            bucket_count=arb_signal.bucket_count,
        )
        import json as _json
        acted_on = 0

        # Execute arbitrage: buy all buckets proportionally
        # Each bucket gets equal share of arbitrage budget
        arb_budget = min(ladder_budget, tail_budget, settings.max_bet * arb_signal.bucket_count)
        per_bucket_size = round(arb_budget / arb_signal.bucket_count, 2)
        arb_orders = 0
        uniform_p = 1.0 / arb_signal.bucket_count
        for bucket in arb_signal.buckets:
            if await db.has_open_position(bucket.city, target_date.isoformat(), bucket.temp_value):
                continue
            # Skip buckets with market price below minimum threshold (extreme improbable outcomes)
            if bucket.market_price < settings.arb_min_price:
                log.debug(
                    "arb_bucket_skipped_low_price",
                    city=bucket.city,
                    temp_value=bucket.temp_value,
                    market_price=bucket.market_price,
                    min_price=settings.arb_min_price,
                )
                continue
            # Use forecast probability for this bucket; fall back to uniform if not available
            p_model = forecast.buckets.get(bucket.temp_value, uniform_p)
            bucket_edge = max(0.0, p_model - bucket.market_price)
            arb_request = OrderRequest(
                run_id=run_id,
                token_id=bucket.token_id,
                city=bucket.city,
                date=bucket.date,
                temp_value=bucket.temp_value,
                temp_unit=bucket.temp_unit,
                strategy="arbitrage",
                limit_price=bucket.market_price,
                size=per_bucket_size,
                p_model=p_model,
                p_market=bucket.market_price,
                edge=bucket_edge,
            )
            arb_result = await executor.place_order(arb_request)
            if arb_result.success:
                arb_orders += 1
        if arb_orders > 0:
            acted_on = 1
            log.info("arbitrage_executed", city=city_cfg.name, orders=arb_orders, gap=round(arb_signal.gap, 4))

        await db.record_arbitrage(
            run_id=run_id,
            city=city_cfg.name,
            date=target_date.isoformat(),
            bucket_count=arb_signal.bucket_count,
            price_sum=arb_signal.price_sum,
            gap=arb_signal.gap,
            token_ids=_json.dumps(arb_signal.token_ids),
            acted_on=acted_on,
        )

    if not signals:
        log.info("no_edges", city=city_cfg.name)
        return 0

    log.info("edges_found", city=city_cfg.name, count=len(signals))

    # 6. Generate positions with risk controls
    ladder_positions = evaluate_ladder(
        signals, ladder_budget,
        edge_threshold=settings.ladder_edge,
        kelly_fraction=settings.kelly_fraction,
        max_bet=settings.max_bet,
        max_bet_pct=settings.max_bet_pct,
        spread_baseline=settings.spread_baseline_f,
    )
    tail_positions = evaluate_tail(
        signals, tail_budget,
        edge_threshold=settings.tail_edge,
        min_odds=settings.tail_odds,
        kelly_fraction=settings.kelly_fraction,
        max_bet=settings.max_bet,
        max_bet_pct=settings.max_bet_pct,
        max_correlated=settings.tail_max_correlated,
        daily_loss_limit=settings.daily_loss_limit,
        spread_baseline=settings.spread_baseline_f,
    )

    # 7. Execute orders
    orders = 0
    for pos in ladder_positions + tail_positions:
        # Skip if already have an open position for this market
        if await db.has_open_position(pos.bucket.city, pos.bucket.date.isoformat(), pos.bucket.temp_value):
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
    executor: "DryRunExecutor | LiveExecutor",
    *,
    http_client: httpx.AsyncClient | None = None,
    notifier: object | None = None,
) -> int:
    """Check all open positions and exit those where probability has turned against us.

    Exit rules (probability-based):
    - Stop-loss: p_model < entry_price * exit_loss_factor  (probability collapsed)
    - Take-profit: p_model >= entry_price but EV <= exit_min_ev  (edge is gone)

    Returns number of positions closed.
    """
    from wedge.monitoring.notify import format_exit_notification

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
            temp_f = pos["temp_f"]
            entry_price = pos["entry_price"]

            # Skip if settlement is imminent (avoid noise near expiry)
            try:
                settle_date = date.fromisoformat(date_str)
                city_cfg_tz = next((c for c in settings.cities if c.name == city_name), None)
                if city_cfg_tz:
                    city_tz = ZoneInfo(city_cfg_tz.timezone)
                    now_local = datetime.now(city_tz)
                    settle_dt = datetime(settle_date.year, settle_date.month, settle_date.day, 23, 59, tzinfo=city_tz)
                    hours_to_settle = (settle_dt - now_local).total_seconds() / 3600
                    if hours_to_settle <= settings.exit_min_hours_to_settle:
                        log.info("exit_skip_near_settlement", city=city_name, date=date_str, hours_remaining=round(hours_to_settle, 1))
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
                raw = await fetch_ensemble_auto(http_client, city_cfg, settings, target_date)
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

            # Determine current market price (use entry_price as fallback)
            market_price = entry_price

            # Compute current EV: p_model * (1/market_price - 1) - (1 - p_model)
            if market_price > 0:
                ev = p_model * (1.0 / market_price - 1.0) - (1.0 - p_model)
            else:
                ev = -1.0

            exit_reason: str | None = None

            if p_model < entry_price * settings.exit_loss_factor:
                # Probability collapsed — stop loss
                exit_reason = "stop_loss"
            elif p_model >= entry_price and ev <= settings.exit_min_ev:
                # We're ahead but edge is gone — take profit
                exit_reason = "take_profit"

            if exit_reason is None:
                log.info(
                    "exit_check_hold",
                    city=city_name,
                    temp_f=temp_f,
                    entry_price=entry_price,
                    p_model=round(p_model, 4),
                    ev=round(ev, 4),
                )
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

            if notifier and hasattr(notifier, "send"):
                msg = format_exit_notification(
                    city=city_name,
                    date=date_str,
                    temp_f=temp_f,
                    exit_reason=exit_reason,
                    pnl=pnl,
                    p_model=p_model,
                    entry_price=entry_price,
                )
                await notifier.send(msg)
    finally:
        if own_client:
            await http_client.aclose()

    if closed:
        log.info("exit_check_complete", closed=closed)
    return closed


def _generate_synthetic_markets(
    forecast, city: str, target_date: date
) -> list[MarketBucket]:
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
    settings: Settings, db: Database, *, notifier: object | None = None
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
                    actual_temp = await fetch_actual_temperature(
                        http_client, city_cfg, trade_date
                    )
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
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff

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

    if total_settled > 0 and notifier and hasattr(notifier, "send"):
        await notifier.send(
            f"[Settlement] Settled {total_settled} trade(s) across {len(unsettled) - len(pending_retry)} date(s) ({len(pending_retry)} pending retry)"
        )

    log.info("settlement_complete", total_settled=total_settled)

    # Update per-city forecast performance after settlement
    if total_settled > 0:
        settled_cities = {city for city, _ in unsettled if city not in [p[0] for p in pending_retry]}
        for city_name in settled_cities:
            try:
                await update_city_performance(db, city_name, window_days=30)
                log.debug("city_performance_updated", city=city_name)
            except Exception as exc:  # noqa: BLE001
                log.warning("city_performance_update_failed", city=city_name, error=str(exc))

    return total_settled


async def run_single_scan(settings: Settings, city_name: str) -> None:
    """Run a single scan for one city (CLI scan command)."""
    from wedge.log import setup_logging
    setup_logging()

    city_cfg = next(
        (c for c in settings.cities if c.name.lower() == city_name.lower()), None
    )
    if not city_cfg:
        log.error("city_not_found", city=city_name)
        return

    city_tz = ZoneInfo(city_cfg.timezone)
    target_date = (datetime.now(city_tz) + timedelta(days=3)).date()

    async with httpx.AsyncClient() as http_client:

        raw = await fetch_ensemble_auto(http_client, city_cfg, settings, target_date)
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
