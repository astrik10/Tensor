"""
features.py

Feature engineering for the BTC tick-prediction pipeline. Computes a
rolling-window feature vector for the most recent tick in a buffer, and
derives the up/down label once the next tick is known.

A "tick" is a dict like: {"timestamp": "...", "coin": "bitcoin", "price_usd": 61234.5}
The buffer passed in is a list of the most recent N ticks (suggested N=20),
ordered oldest -> newest.
"""

from typing import Dict, List

import numpy as np

REQUIRED_WINDOWS = {
    "rolling_mean_5": 5,
    "rolling_mean_20": 20,
    "rolling_vol_10": 10,
}
MOMENTUM_LAG = 5


def _extract_prices(buffer: List[Dict]) -> List[float]:
    """Pulls price_usd out of each tick, raising a clear error if a tick is malformed."""
    prices = []
    for i, tick in enumerate(buffer):
        if "price_usd" not in tick:
            raise ValueError(f"Tick at index {i} is missing 'price_usd': {tick}")
        prices.append(float(tick["price_usd"]))
    return prices


def compute_features(buffer: List[Dict]) -> Dict[str, float]:
    """
    Compute features for the CURRENT tick (the last item in buffer) using the
    rolling window of ticks that precede it.

    Required features (minimum):
        - rolling_mean_5:  mean price of the last 5 ticks (or fewer, early on)
        - rolling_mean_20: mean price of the last 20 ticks (or fewer, early on)
        - momentum_5:      current price - price 5 ticks ago (falls back to the
                            earliest available tick if fewer than 6 ticks exist)
        - rolling_vol_10:  std. dev. of price over the last 10 ticks (or fewer)

    Args:
        buffer: list of tick dicts, oldest first, newest last.
                len(buffer) may be less than 20 early on — handled by using
                only as many ticks as are available for each window.

    Returns:
        A dict of feature_name -> float value for the most recent tick.

    Raises:
        ValueError: if buffer is empty or a tick is missing 'price_usd'.
    """
    if not buffer:
        raise ValueError("compute_features requires at least one tick in buffer")

    prices = _extract_prices(buffer)
    current_price = prices[-1]

    rolling_mean_5 = float(np.mean(prices[-REQUIRED_WINDOWS["rolling_mean_5"]:]))
    rolling_mean_20 = float(np.mean(prices[-REQUIRED_WINDOWS["rolling_mean_20"]:]))
    rolling_vol_10 = float(np.std(prices[-REQUIRED_WINDOWS["rolling_vol_10"]:]))

    # "5 ticks ago" needs index -6 (5 ticks before the current, last one).
    # If we don't have that much history yet, fall back to the earliest
    # tick we do have rather than raising — this only affects the first
    # few predictions right after startup.
    if len(prices) > MOMENTUM_LAG:
        price_lag_5 = prices[-(MOMENTUM_LAG + 1)]
    else:
        price_lag_5 = prices[0]
    momentum_5 = float(current_price - price_lag_5)

    return {
        "rolling_mean_5": rolling_mean_5,
        "rolling_mean_20": rolling_mean_20,
        "momentum_5": momentum_5,
        "rolling_vol_10": rolling_vol_10,
    }


def make_label(buffer: List[Dict], next_tick: Dict) -> int:
    """
    Compute the training label for the tick at buffer[-1]: did price go UP (1)
    or DOWN/SAME (0) on the next tick?

    Args:
        buffer: rolling buffer ending at the tick being labeled.
        next_tick: the tick that came right after buffer[-1].

    Returns:
        1 if next_tick price > buffer[-1] price, else 0.

    Raises:
        ValueError: if buffer is empty or either tick is missing 'price_usd'.
    """
    if not buffer:
        raise ValueError("make_label requires a non-empty buffer")
    if "price_usd" not in buffer[-1]:
        raise ValueError(f"buffer[-1] is missing 'price_usd': {buffer[-1]}")
    if "price_usd" not in next_tick:
        raise ValueError(f"next_tick is missing 'price_usd': {next_tick}")

    current_price = float(buffer[-1]["price_usd"])
    next_price = float(next_tick["price_usd"])

    return 1 if next_price > current_price else 0