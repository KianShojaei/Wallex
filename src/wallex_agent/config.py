"""Typed configuration: YAML (strategy/risk) + .env (secrets)."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


class ChallengeCfg(BaseModel):
    initial_capital_tmn: float = 5_000_000
    quote_asset: str = "TMN"
    start: datetime
    end: datetime
    liquidate_before_end_minutes: int = 45


class MarketsCfg(BaseModel):
    allowed: List[str]
    min_24h_quote_volume_tmn: float = 0
    max_spread_pct: float = 0.35


class DataCfg(BaseModel):
    resolution: str = "60"
    fallback_resolution: str = "60"
    lookback_bars: int = 300
    candle_cache_dir: str = "data/candles"


class StrategyCfg(BaseModel):
    name: str = "ai_gated_mean_reversion"
    rsi_period: int = 14
    rsi_oversold: float = 32
    rsi_exit: float = 55
    bb_period: int = 20
    bb_std: float = 2.0
    ema_fast: int = 20
    ema_slow: int = 50
    trend_ema: int = 100
    atr_period: int = 14
    volume_zscore_min: float = -1.0
    min_signal_score: float = 0.55
    ai_enabled: bool = True
    ai_min_confidence: float = 0.6
    regime_refresh_minutes: int = 60


class RiskCfg(BaseModel):
    max_position_pct: float = 0.20
    max_total_exposure_pct: float = 0.40
    max_open_positions: int = 2
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 1.6
    trailing_stop_pct: float = 0.8
    max_holding_minutes: int = 720
    daily_drawdown_halt_pct: float = 3.0
    total_drawdown_kill_pct: float = 6.0
    cooldown_after_loss_minutes: int = 90
    min_order_notional_tmn: float = 300_000
    fee_pct: float = 0.25
    slippage_pct: float = 0.05


class ExecutionCfg(BaseModel):
    loop_interval_seconds: int = 60
    order_type: Literal["LIMIT", "MARKET"] = "LIMIT"
    limit_offset_pct: float = 0.10
    order_timeout_seconds: int = 45
    state_dir: str = "state"
    decision_log_dir: str = "logs/decisions"
    ai_cache_dir: str = "logs/ai_cache"


class Settings(BaseModel):
    challenge: ChallengeCfg
    markets: MarketsCfg
    data: DataCfg
    strategy: StrategyCfg
    risk: RiskCfg
    execution: ExecutionCfg
    # secrets / env
    wallex_api_key: str = ""
    wallex_base_url: str = "https://api.wallex.ir"
    ai_base_url: str = "https://ai.boxapi.ir/ai/v1"
    ai_api_key: str = ""
    ai_model_fast: str = "openai/gpt-6-luna"
    ai_model_deep: str = "openai/gpt-6-sol"
    run_mode: Literal["shadow", "live", "backtest"] = "shadow"

    @property
    def challenge_end_utc(self) -> datetime:
        return self.challenge.end.astimezone(timezone.utc)

    @property
    def challenge_start_utc(self) -> datetime:
        return self.challenge.start.astimezone(timezone.utc)


def load_settings(config_path: str | os.PathLike | None = None, env_file: str = ".env") -> Settings:
    load_dotenv(env_file, override=False)
    path = Path(config_path or os.getenv("CONFIG_PATH", "config/settings.yaml"))
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = dict(raw)
    raw.update(
        wallex_api_key=os.getenv("WALLEX_API_KEY", ""),
        wallex_base_url=os.getenv("WALLEX_BASE_URL", "https://api.wallex.ir"),
        ai_base_url=os.getenv("AI_BASE_URL", "https://ai.boxapi.ir/ai/v1"),
        ai_api_key=os.getenv("AI_API_KEY", ""),
        ai_model_fast=os.getenv("AI_MODEL_FAST", "openai/gpt-6-luna"),
        ai_model_deep=os.getenv("AI_MODEL_DEEP", "openai/gpt-6-sol"),
        run_mode=os.getenv("RUN_MODE", "shadow"),
    )
    return Settings(**raw)
