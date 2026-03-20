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
    CityConfig(name="Shanghai", lat=31.1434, lon=121.8052, timezone="Asia/Shanghai", station="ZSSS"),
    CityConfig(name="Miami", lat=25.7959, lon=-80.2870, timezone="America/New_York", station="KMIA"),
    CityConfig(name="Wellington", lat=-41.3272, lon=174.8050, timezone="Pacific/Auckland", station="NZWN"),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WEDGE_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    mode: str = "dry_run"
    bankroll: float = 1000.0
    max_bet: float = 100.0
    kelly_fraction: float = 0.15
    max_bet_pct: float = 0.05
    db_path: str = Field(default_factory=lambda: str(get_data_dir() / "wedge.db"))
    log_dir: str = Field(default_factory=lambda: str(get_data_dir() / "logs"))

    # Ladder strategy
    ladder_edge: float = 0.08
    ladder_alloc: float = 0.70

    # Tail strategy
    tail_edge: float = 0.12
    tail_odds: float = 10.0
    tail_alloc: float = 0.20
    tail_max_correlated: int = 2  # Max positions per climate region
    daily_loss_limit: float = 200.0  # Stop trading if daily loss exceeds this

    # Fee and slippage configuration
    fee_rate: float = 0.02  # Polymarket 2% fee on winnings
    slippage_model: str = "volume_based"  # volume_based or fixed

    # Risk management
    arb_min_price: float = 0.05  # Skip arb buckets with market price below this threshold

    # Exit strategy (probability-based)
    exit_loss_factor: float = 0.5    # Stop-loss: exit when p_model < entry_price * this factor
    exit_min_ev: float = 0.0         # Take-profit: exit when EV drops to or below this value
    exit_min_hours_to_settle: int = 12  # Don't exit within this many hours of settlement
    brier_threshold: float = 0.25  # Pause trading if weekly Brier score exceeds this
    brier_decomposition: bool = True  # Track Brier reliability/resolution
    min_city_brier_score: float = 0.22  # Skip city if 30-day Brier score exceeds this
    min_city_samples: int = 5  # Min settled trades before applying city filter
    spread_baseline_f: float = 3.0  # Ensemble spread baseline (°F) for Kelly damping

    weather_source: str = "openmeteo"  # 'openmeteo' or 'noaa'

    offsets_utc: list[str] = Field(
        default_factory=lambda: ["03:45", "09:45", "15:45", "21:45"]
        # GFS model runs at 00/06/12/18 UTC; data available ~3.5h later.
        # Running at :45 past the hour catches fresh data before market makers reprice.
    )

    cities: list[CityConfig] = Field(default_factory=lambda: list(DEFAULT_CITIES))

    polymarket_private_key: str = ""
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""

    telegram_token: str = ""
    telegram_chat_id: str = ""

    @classmethod
    def load(cls, **overrides: Any) -> Settings:
        """Load settings from config file, env vars, and overrides.

        Priority: overrides > env vars > config file > defaults
        """
        config_data = load_config_file()
        # Merge config file with overrides
        merged = {**config_data, **overrides}
        return cls(**merged)
