import numpy as np

from wallex_agent.data.features import compute_indicators, snapshot_from
from wallex_agent.data.market_data import udf_to_df
from wallex_agent.strategy.signals import evaluate_entry, evaluate_exit
from tests.conftest import synthetic_candles


def test_udf_parsing_and_empty():
    df = udf_to_df({"s": "ok", "t": [0, 3600], "o": [1, 2], "h": [2, 3], "l": [0.5, 1], "c": [1.5, 2.5], "v": [10, 20]})
    assert len(df) == 2 and df.index.tz is not None
    assert udf_to_df({"s": "no_data"}).empty


def test_indicators_have_expected_ranges(settings, candles):
    df = compute_indicators(candles, settings.strategy)
    valid = df.dropna()
    assert len(valid) > 400
    assert valid["rsi"].between(0, 100).all()
    assert (valid["atr"] > 0).all()
    assert valid["bb_pos"].between(-0.5, 1.5).all()


def test_signal_fires_on_oversold_and_is_deterministic(settings):
    df = compute_indicators(synthetic_candles(seed=3), settings.strategy)
    scores = []
    for i in range(150, len(df)):
        snap = snapshot_from(df.iloc[: i + 1], "BTCTMN", spread_pct=0.1)
        sig = evaluate_entry(snap, settings.strategy)
        scores.append(sig.score)
        if sig.is_candidate:
            assert snap.rsi <= settings.strategy.rsi_oversold or snap.bb_pos <= 0.15
            assert sig.reasons
    assert max(scores) > 0.5, "synthetic series should produce at least one strong candidate"
    # deterministic: same input -> same output
    snap = snapshot_from(df, "BTCTMN", spread_pct=0.1)
    assert evaluate_entry(snap, settings.strategy).to_dict() == evaluate_entry(snap, settings.strategy).to_dict()


def test_vetoes(settings):
    df = compute_indicators(synthetic_candles(seed=3), settings.strategy)
    snap = snapshot_from(df, "BTCTMN", spread_pct=5.0)
    from types import SimpleNamespace
    cfg = SimpleNamespace(**settings.strategy.model_dump(), max_spread_pct=0.35)
    sig = evaluate_entry(snap, cfg)
    assert any("spread" in v for v in sig.vetoes)
    snap.rsi, snap.bb_pos = 70.0, 0.9
    assert evaluate_exit(snap, settings.strategy) is not None
