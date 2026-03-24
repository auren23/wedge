from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_config_dir() -> Path:
    """Get XDG config directory."""
    config_dir = Path.home() / ".config" / "wedge"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def get_data_dir() -> Path:
    """Get XDG data directory."""
    data_dir = Path.home() / ".local" / "share" / "wedge"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_cache_dir() -> Path:
    """Get XDG cache directory."""
    cache_dir = Path.home() / ".cache" / "wedge"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_config_path() -> Path:
    """Get config file path."""
    return get_config_dir() / "config.toml"


def load_config_file() -> dict[str, Any]:
    """Load config from TOML file if exists."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}

    with open(config_path, "rb") as f:
        return tomllib.load(f)


class CityConfig(BaseModel):
    name: str
    lat: float
    lon: float
    timezone: str = "UTC"
    station: str = ""  # ICAO airport code (e.g. KLGA)


# CRITICAL: Coordinates MUST match the airport weather stations
# Polymarket resolves on. Using city center coords causes 3-8°F error.
# High liquidity markets only (>$25K daily volume)
DEFAULT_CITIES = [
    CityConfig(name="Seoul", lat=37.4602, lon=126.4407, timezone="Asia/Seoul", station="RKSI"),
    CityConfig(name="London", lat=51.4700, lon=-0.4543, timezone="Europe/London", station="EGLL"),
    CityConfig(name="NYC", lat=40.7772, lon=-73.8726, timezone="America/New_York", station="KLGA"),
    CityConfig(
        name="Shanghai",
        lat=31.1434,
        lon=121.8052,
        timezone="Asia/Shanghai",
        station="ZSPD",
    ),
    CityConfig(
        name="Miami",
        lat=25.7959,
        lon=-80.2870,
        timezone="America/New_York",
        station="KMIA",
    ),
    CityConfig(
        name="Wellington",
        lat=-41.3272,
        lon=174.8050,
        timezone="Pacific/Auckland",
        station="NZWN",
    ),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    mode: str = "dry_run"
    bankroll: float = 1000.0
    max_bet: float = 200.0
    kelly_fraction: float = 0.20
    max_bet_pct: float = 0.10
    db_path: str = Field(default_factory=lambda: str(get_data_dir() / "wedge.db"))
    log_dir: str = Field(default_factory=lambda: str(get_data_dir() / "logs"))

    # Ladder strategy
    ladder_edge: float = 0.06
    ladder_alloc: float = 0.95
    market_min_volume: float = 2000.0  # Minimum 24h volume for market inclusion
    market_min_open_interest: float = 1000.0  # Minimum open interest for market inclusion
    market_max_spread: float = 0.10  # Optional max bid/ask spread; ignored when spread unavailable
    market_watchlist_size: int = 5  # Limit entry evaluation to top-K liquid buckets per scan
    slippage_bet_size: float = 50.0  # Reference bet size for EV slippage estimation

    # Fees and execution
    fee_rate: float = 0.0  # Weather markets are fee-free; only slippage/liquidity apply
    # Exit strategy (probability-based)
    exit_loss_factor: float = 0.75  # Exit when p_model drops to 75% of entry
    exit_min_ev: float = 0.01  # Exit when edge drops below 1%
    exit_min_hours_to_settle: int = 6  # Don't exit within 6h of settlement
    exit_poll_interval_seconds: int = 60  # Check market prices every 1min for exits

    # Trailing stop (let profits run, protect gains)
    trailing_activation_pct: float = 0.20  # Trail activates after 20% gain from entry
    trailing_pct: float = 0.20  # Exit if price drops 20% from peak (after activation)

    # Partial take-profit tiers (scale out of positions)
    # Example: exit 33% at +50%, another 33% at +100%, rest runs to trailing/settlement
    exit_tier_pcts: list[float] = Field(default_factory=lambda: [0.50, 1.0])
    exit_tier_portions: list[float] = Field(default_factory=lambda: [0.33, 0.33])

    brier_threshold: float = 0.25
    scheduler_brier_days: int = 30
    spread_baseline_f: float = 3.0
    # NOAA latency-path rollout (Phase 1+2)
    readiness_mode: str = "off"  # off | shadow | active
    readiness_probe_start_offset_minutes: int = 180  # cycle + 3h — start polling as early as possible
    readiness_probe_fast_poll_seconds: int = 60  # check every 60s once probing starts
    readiness_probe_fast_until_minutes: int = 210  # cycle + 3h30m — fast poll window
    readiness_probe_slow_poll_seconds: int = 120  # slow to 2min after fast window
    readiness_probe_timeout_minutes: int = 240  # cycle + 4h — hard timeout
    readiness_probe_max_attempts: int = 180
    readiness_fetch_concurrency: int = 16
    readiness_error_rate_threshold: float = 0.05
    enable_parallel_noaa_fetch: bool = False

    offsets_utc: list[str] = Field(
        default_factory=lambda: ["03:00", "09:00", "15:00", "21:00"]
        # Start probing at cycle+3h. Readiness probe polls until data is available.
    )

    cities: list[CityConfig] = Field(default_factory=lambda: list(DEFAULT_CITIES))

    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""

    @classmethod
    def load(cls, **overrides: Any) -> Settings:
        """Load settings from config file, env vars, and overrides.

        Priority: overrides > env vars > config file > defaults
        """
        config_data = load_config_file()
        # Merge config file with overrides
        merged = {**config_data, **overrides}
        return cls(**merged)
