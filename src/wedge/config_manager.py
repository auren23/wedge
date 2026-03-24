"""Configuration file management."""

from __future__ import annotations

from typing import Any

import typer

from wedge.config import get_config_dir, get_config_path, get_data_dir, load_config_file

app = typer.Typer(name="config", help="Manage configuration")


def _write_config(data: dict[str, Any]) -> None:
    """Write config data to TOML file."""
    import tomli_w

    config_path = get_config_path()
    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)


@app.command()
def init(force: bool = typer.Option(False, "--force", help="Overwrite existing config")) -> None:
    """Initialize config file with defaults."""
    config_path = get_config_path()

    if config_path.exists() and not force:
        typer.echo(f"Config already exists: {config_path}")
        typer.echo("Use --force to overwrite")
        raise typer.Exit(1)

    default_config = {
        "mode": "dry_run",
        "bankroll": 1000.0,
        "max_bet": 100.0,
        "kelly_fraction": 0.15,
        "ladder_edge": 0.08,
        "ladder_alloc": 0.90,
        "market_min_volume": 2000.0,
        "market_min_open_interest": 1000.0,
        "market_max_spread": 0.10,
        "market_watchlist_size": 5,
        "slippage_bet_size": 50.0,
        "fee_rate": 0.0,
        "exit_loss_factor": 0.75,
        "exit_min_ev": 0.01,
        "exit_min_hours_to_settle": 6,
        "brier_threshold": 0.25,
        "scheduler_brier_days": 30,
        "spread_baseline_f": 3.0,
        "readiness_mode": "off",
        "readiness_probe_start_offset_minutes": 180,
        "readiness_probe_fast_poll_seconds": 60,
        "readiness_probe_fast_until_minutes": 210,
        "readiness_probe_slow_poll_seconds": 120,
        "readiness_probe_timeout_minutes": 240,
        "readiness_probe_max_attempts": 180,
        "readiness_fetch_concurrency": 16,
        "readiness_error_rate_threshold": 0.05,
        "enable_parallel_noaa_fetch": False,
        "polymarket_private_key": "",
        "polymarket_api_key": "",
        "polymarket_api_secret": "",
    }

    _write_config(default_config)
    typer.echo(f"✓ Config initialized: {config_path}")


@app.command()
def set(key: str, value: str) -> None:
    """Set a config value."""
    config_data = load_config_file()

    # Type conversion
    if value.lower() in ("true", "false"):
        typed_value: Any = value.lower() == "true"
    elif value.replace(".", "", 1).isdigit():
        typed_value = float(value) if "." in value else int(value)
    else:
        typed_value = value

    config_data[key] = typed_value
    _write_config(config_data)
    typer.echo(f"✓ Set {key} = {typed_value}")


@app.command()
def get(key: str) -> None:
    """Get a config value."""
    config_data = load_config_file()

    if key not in config_data:
        typer.echo(f"Key not found: {key}", err=True)
        raise typer.Exit(1)

    typer.echo(config_data[key])


@app.command()
def show() -> None:
    """Show all config values."""
    config_data = load_config_file()

    if not config_data:
        typer.echo("No config file found. Run 'wedge config init' first.")
        raise typer.Exit(1)

    typer.echo(f"Config: {get_config_path()}\n")
    for key, value in sorted(config_data.items()):
        # Mask sensitive values
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            display_value = "***" if value else "(not set)"
        else:
            display_value = value
        typer.echo(f"{key:25} = {display_value}")


@app.command()
def path() -> None:
    """Show config and data paths."""
    typer.echo(f"Config dir:  {get_config_dir()}")
    typer.echo(f"Config file: {get_config_path()}")
    typer.echo(f"Data dir:    {get_data_dir()}")
