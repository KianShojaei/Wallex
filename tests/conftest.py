import numpy as np
import pandas as pd
import pytest

from wallex_agent.config import load_settings


@pytest.fixture
def settings():
    return load_settings("config/settings.yaml", env_file="/nonexistent.env")


def synthetic_candles(n=600, seed=1, start_price=1_000_000.0, vol=0.004, mean_revert=0.05, freq_min=60):
    """Mean-reverting random walk with occasional shocks: realistic enough to trigger signals."""
    rng = np.random.default_rng(seed)
    px = [start_price]
    anchor = start_price
    for i in range(1, n):
        shock = rng.normal(0, vol)
        if rng.random() < 0.03:
            shock -= abs(rng.normal(0, 4 * vol))      # sudden dips
        drift = mean_revert * (anchor / px[-1] - 1)
        px.append(px[-1] * (1 + drift + shock))
        anchor *= 1 + rng.normal(0, vol / 6)
    c = np.array(px)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    v = np.abs(rng.normal(100, 30, n))
    idx = pd.date_range("2026-09-01", periods=n, freq=f"{freq_min}min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)


@pytest.fixture
def candles():
    return synthetic_candles()
