from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from wedge.config import Settings
from wedge.config_manager import app as config_app
from wedge.log import setup_logging

app = typer.Typer(name="wedge", help="Weather prediction market trading bot")
app.add_typer(config_app, name="config")


@app.command()
def run(
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Run in simulation mode"),
    bankroll: float = typer.Option(None, "--bankroll", "-b", help="Starting bankroll"),
    max_bet: float = typer.Option(None, "--max-bet", help="Max bet per trade"),
    kelly: float = typer.Option(None, "--kelly", help="Kelly fraction (0-1)"),
    ladder_edge: float = typer.Option(None, "--ladder-edge", help="Ladder edge threshold"),
) -> None:
    """Start the 7x24 trading bot."""
    overrides = {
        "mode": "dry_run" if dry_run else "live",
    }
    if bankroll is not None:
        overrides["bankroll"] = bankroll
    if max_bet is not None:
        overrides["max_bet"] = max_bet
    if kelly is not None:
        overrides["kelly_fraction"] = kelly
    if ladder_edge is not None:
        overrides["ladder_edge"] = ladder_edge

    settings = Settings.load(**overrides)
    from datetime import UTC, datetime

    _log_file = Path(settings.log_dir) / f"wedge-{datetime.now(UTC).strftime('%Y-%m-%d')}.log"
    setup_logging(log_file=_log_file)

    from wedge.scheduler import run_scheduler

    asyncio.run(run_scheduler(settings))


@app.command()
def scan(
    city: str = typer.Option("NYC", "--city", help="City to scan"),
) -> None:
    """Run a single scan for a city."""
    settings = Settings.load()
    setup_logging()

    from wedge.pipeline import run_single_scan

    asyncio.run(run_single_scan(settings, city))


@app.command()
def stats(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to show"),
) -> None:
    """Show P&L, Brier score, and trade statistics."""
    settings = Settings.load()
    setup_logging()

    from wedge.monitoring.metrics import show_stats

    asyncio.run(show_stats(settings, days))


@app.command()
def watchlist(
    city: str | None = typer.Option(None, "--city", help="Filter by city"),
    target_date: str | None = typer.Option(None, "--date", help="Filter by ISO date YYYY-MM-DD"),
    include_all: bool = typer.Option(
        False,
        "--all",
        help="Include non-watchlist market discoveries as well",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of text table"),
) -> None:
    """Show persisted market watchlist/discovery rows."""
    settings = Settings.load()
    setup_logging()

    from wedge.monitoring.watchlist import show_watchlist

    asyncio.run(
        show_watchlist(
            settings,
            city=city,
            target_date=target_date,
            include_all=include_all,
            as_json=as_json,
        )
    )


@app.command()
def backtest(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to backtest"),
) -> None:
    """Run backtest on historical settled trades."""
    from datetime import datetime, timedelta

    settings = Settings.load()
    setup_logging()

    from wedge.backtest import run_backtest

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)
    asyncio.run(run_backtest(settings, start_date, end_date))


if __name__ == "__main__":  # pragma: no cover
    app()
