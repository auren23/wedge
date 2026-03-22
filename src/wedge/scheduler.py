from __future__ import annotations

import asyncio
import signal
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from wedge.config import Settings
from wedge.db import Database
from wedge.log import get_logger
from wedge.pipeline import run_market_exit_check, run_pipeline, run_settlement
from wedge.weather.client import ReadinessProbeResult, probe_cycle_readiness

_UTC = ZoneInfo("UTC")

log = get_logger("scheduler")


def _cycle_key(*, run_date: date, cycle_hour: int) -> str:
    return f"gefs:{run_date.strftime('%Y%m%d')}:{cycle_hour:02d}"


async def _select_probe_target(settings: Settings) -> tuple[object, date] | None:
    if not settings.cities:
        return None
    city_cfg = settings.cities[0]
    city_tz = ZoneInfo(city_cfg.timezone)
    target_date = datetime.now(city_tz).date() + timedelta(days=1)
    return city_cfg, target_date


async def _maybe_probe_cycle(
    *,
    settings: Settings,
    db: Database,
 ) -> ReadinessProbeResult | None:
    if settings.readiness_mode == "off":
        return None
    probe_target = await _select_probe_target(settings)
    if probe_target is None:
        return None
    city_cfg, target_date = probe_target
    async with __import__("httpx").AsyncClient() as client:
        return await probe_cycle_readiness(client, city_cfg, target_date)


async def run_scheduler(settings: Settings) -> None:
    """Main entry: start APScheduler and run until SIGINT/SIGTERM."""
    db = Database(settings.db_path)
    await db.connect()

    scheduler = AsyncIOScheduler()
    _running_lock = asyncio.Lock()

    async def _guarded_pipeline() -> None:
        if _running_lock.locked():  # pragma: no cover — concurrency guard
            log.warning("pipeline_skipped_already_running")
            return
        async with _running_lock:
            try:
                brier = await db.get_brier_score(days=settings.scheduler_brier_days)
                if brier is not None and brier > settings.brier_threshold:
                    log.warning("circuit_breaker_active", brier_score=f"{brier:.3f}")
                    return

                probe_result = await _maybe_probe_cycle(settings=settings, db=db)

                if settings.readiness_mode == "active" and probe_result is not None:
                    if not probe_result.ready:
                        log.info(
                            "readiness_probe_not_ready",
                            reason=probe_result.reason,
                            attempts=probe_result.attempts,
                        )
                        return

                    cycle_key = _cycle_key(
                        run_date=probe_result.run_date,
                        cycle_hour=probe_result.cycle_hour,
                    )
                    claimed = await db.claim_cycle_marker(
                        cycle_key,
                        trigger_mode="active",
                        status="claimed",
                        run_id=None,
                        created_at=datetime.now(UTC).isoformat(),
                    )
                    if not claimed:
                        log.info("cycle_marker_exists_skipping", cycle_key=cycle_key)
                        return

                    try:
                        await run_pipeline(settings, db)
                        await db.update_cycle_marker_status(
                            cycle_key,
                            status="completed",
                            updated_at=datetime.now(UTC).isoformat(),
                        )
                    except Exception:
                        await db.update_cycle_marker_status(
                            cycle_key,
                            status="failed",
                            updated_at=datetime.now(UTC).isoformat(),
                        )
                        raise
                    return

                await run_pipeline(settings, db)
            except Exception as e:
                log.error("pipeline_error", error=str(e))

    async def _run_settlement() -> None:
        try:
            await run_settlement(settings, db)
        except Exception as e:
            log.error("settlement_error", error=str(e))

    # Daily settlement at 23:45 UTC (after all daily highs are recorded)
    scheduler.add_job(
        _run_settlement,
        trigger=CronTrigger(hour=23, minute=45, timezone=_UTC),
        coalesce=True,
        misfire_grace_time=3600,
        id="settlement_daily",
    )

    scheduler.add_job(
        _guarded_pipeline,
        trigger=IntervalTrigger(minutes=1),
        coalesce=True,
        misfire_grace_time=60,
        id="pipeline_interval",
    )

    # Market price exit check (every N seconds, independent of weather pipeline)
    async def _run_market_exit_check() -> None:
        try:
            from wedge.market.polymarket import PolymarketClient, PublicPolymarketClient
            from wedge.execution.dry_run import DryRunExecutor
            from wedge.execution.live import LiveExecutor

            if settings.mode == "live":
                if settings.polymarket_private_key and settings.polymarket_api_key:
                    poly_client = PolymarketClient(
                        settings.polymarket_private_key,
                        settings.polymarket_api_key,
                        settings.polymarket_api_secret,
                    )
                    await poly_client.connect()
                    balance = await db.get_last_balance(default=settings.bankroll)
                    executor = LiveExecutor(db, poly_client, balance, settings.max_bet)
                    await run_market_exit_check(settings, db, executor, poly_client)
                else:
                    log.warning("exit_check_no_credentials")
            else:
                poly_client = PublicPolymarketClient()
                balance = await db.get_last_balance(default=settings.bankroll)
                executor = DryRunExecutor(db, balance, settings.max_bet)
                await run_market_exit_check(settings, db, executor, poly_client)
        except Exception as e:
            log.error("market_exit_check_error", error=str(e))

    if settings.exit_poll_interval_seconds > 0:
        scheduler.add_job(
            _run_market_exit_check,
            trigger=IntervalTrigger(seconds=settings.exit_poll_interval_seconds),
            coalesce=True,
            misfire_grace_time=60,
            id="market_exit_check",
        )
    scheduler.start()
    log.info(
        "scheduler_started",
        mode=settings.mode,
        windows=settings.offsets_utc,
        bankroll=settings.bankroll,
    )

    # Run one immediate cycle
    await _guarded_pipeline()

    # Wait for shutdown signal
    stop_event = asyncio.Event()

    def _handle_signal() -> None:  # pragma: no cover — OS signal handler
        log.info("shutdown_signal_received")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await stop_event.wait()

    log.info("shutting_down")
    scheduler.shutdown(wait=True)
    await db.close()
    log.info("shutdown_complete")
