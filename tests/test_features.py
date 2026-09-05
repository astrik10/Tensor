"""
test_features.py
PROVIDED.

Run with:
    python -m pytest tests/test_features.py -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import compute_features, make_label  # noqa: E402


def make_buffer(prices):
    return [
        {"timestamp": f"t{i}", "coin": "bitcoin", "price_usd": p}
        for i, p in enumerate(prices)
    ]


def test_rolling_means():
    prices = list(range(1, 21))  # 1..20
    buffer = make_buffer(prices)
    feats = compute_features(buffer)
    assert abs(feats["rolling_mean_5"] - sum(prices[-5:]) / 5) < 1e-6
    assert abs(feats["rolling_mean_20"] - sum(prices[-20:]) / 20) < 1e-6


def test_momentum():
    prices = list(range(1, 21))
    buffer = make_buffer(prices)
    feats = compute_features(buffer)
    assert abs(feats["momentum_5"] - (prices[-1] - prices[-6])) < 1e-6


def test_volatility_is_nonnegative():
    prices = [100, 102, 101, 105, 99, 98, 103, 107, 110, 108]
    buffer = make_buffer(prices)
    feats = compute_features(buffer)
    assert feats["rolling_vol_10"] >= 0


def test_handles_short_buffer():
    prices = [100, 101, 102]
    buffer = make_buffer(prices)
    feats = compute_features(buffer)
    assert "rolling_mean_5" in feats


def test_make_label_up():
    buffer = make_buffer([100, 101, 102])
    next_tick = {"timestamp": "t3", "coin": "bitcoin", "price_usd": 103}
    assert make_label(buffer, next_tick) == 1


def test_make_label_down():
    buffer = make_buffer([100, 101, 102])
    next_tick = {"timestamp": "t3", "coin": "bitcoin", "price_usd": 101}
    assert make_label(buffer, next_tick) == 0
