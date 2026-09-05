"""
run_pipeline.py

Orchestration only; the pieces live in features.py, models.py, monitor.py.

This is the main entry point. It:
    1. Bootstraps from data/historical_ticks.csv, building a training set
       via features.compute_features() and features.make_label().
    2. Trains both models via models.train_models().
    3. Serves: polls data/live_ticks.csv (written by scraper.py) for new
       ticks, maintains a rolling buffer, computes features for each new
       tick, and gets predictions from both models.
    4. Once the next tick arrives, resolves the actual outcome for the
       previous prediction and logs the whole record via
       monitor.log_prediction().
    5. Prints a live monitoring line each tick: accuracy, tick count,
       latency, drift flag.
    6. Every SAVE_EVERY_N_TICKS ticks, and again on shutdown, saves model
       checkpoints and writes run_log.json.

Run with:
    python src/run_pipeline.py

Make sure scraper.py is running in a separate terminal if you want to test
against live data. For development, this bootstraps from
data/historical_ticks.csv first, then polls data/live_ticks.csv.
"""

import csv
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Allows `python src/run_pipeline.py` to work directly (not just
# `python -m src.run_pipeline`) even though the imports below are
# package-qualified — without this, running the file as a plain script
# raises "ModuleNotFoundError: No module named 'src'".
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import compute_features, make_label
from src.models import *
from src.monitor import (
    compute_drift_flag,
    compute_rolling_accuracy,
    log_prediction,
    write_run_log,
)

BUFFER_SIZE = 20
BOOTSTRAP_SOURCE = os.path.join("data", "historical_ticks.csv")
LIVE_SOURCE = os.path.join("data", "live_ticks.csv")

POLL_SECONDS = 15
SAVE_EVERY_N_TICKS = 50
FEATURE_ORDER = ["rolling_mean_5", "rolling_mean_20", "momentum_5", "rolling_vol_10"]


def load_ticks_csv(path):
    """Load a tick CSV into a list of dicts, bitcoin rows only. Provided helper."""
    ticks = []
    if not os.path.exists(path):
        return ticks
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticks.append(
                {
                    "timestamp": row["timestamp"],
                    "coin": row["coin"],
                    "price_usd": float(row["price_usd"]),
                }
            )
    return [t for t in ticks if t["coin"] == "bitcoin"]


def _feature_vector(feature_dict):
    """Converts a compute_features() dict into the fixed-order list the models expect."""
    return [feature_dict[name] for name in FEATURE_ORDER]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bootstrap():
    """
    Loads BOOTSTRAP_SOURCE, slides a window of size BUFFER_SIZE across it,
    computing features and labels for each valid position.

    Returns:
        (X, y, training_volatility) so run_pipeline() can train the models
        and set a baseline for the drift flag.
    """
    ticks = load_ticks_csv(BOOTSTRAP_SOURCE)
    if len(ticks) < BUFFER_SIZE + 1:
        raise RuntimeError(
            f"Need at least {BUFFER_SIZE + 1} historical bitcoin ticks to "
            f"bootstrap, found {len(ticks)} in {BOOTSTRAP_SOURCE}"
        )

    X, y, volatilities = [], [], []

    for end in range(BUFFER_SIZE, len(ticks)):
        window = ticks[end - BUFFER_SIZE:end]
        next_tick = ticks[end]

        feature_dict = compute_features(window)
        label = make_label(window, next_tick)

        X.append(_feature_vector(feature_dict))
        y.append(label)
        volatilities.append(feature_dict["rolling_vol_10"])

    training_volatility = statistics.mean(volatilities) if volatilities else 0.0
    return X, y, training_volatility


def _save_checkpoint(models, start_time, total_ticks_processed):
    model_versions = save_models(models)
    final_accuracy = {
        "logreg": compute_rolling_accuracy("logreg", window=10_000),
        "random_forest": compute_rolling_accuracy("random_forest", window=10_000),
        "lstm": compute_rolling_accuracy("lstm", window=10_000),  # Add this line
    }
    write_run_log({
        "start_time": start_time,
        "end_time": _now_iso(),
        "model_versions": model_versions,
        "total_ticks_processed": total_ticks_processed,
        "final_accuracy": final_accuracy,
    })
    print(f"[pipeline] checkpoint saved -> {model_versions}")

def run_pipeline():
    print("[pipeline] bootstrapping from historical data...")
    X, y, training_volatility = bootstrap()
    print(f"[pipeline] bootstrap set: {len(X)} examples, "
          f"training_volatility={training_volatility:.4f}")

    print("[pipeline] training models...")
    models = train_models(X, y)

    start_time = _now_iso()
    total_ticks_processed = 0
    pending = None  # the most recent prediction whose outcome isn't known yet

    # Seed the live buffer with the tail of the bootstrap history so we
    # don't need another BUFFER_SIZE live ticks before predicting again.
    bootstrap_ticks = load_ticks_csv(BOOTSTRAP_SOURCE)
    buffer = bootstrap_ticks[-BUFFER_SIZE:]
    seen_timestamps = {t["timestamp"] for t in bootstrap_ticks}

    try:
        while True:
            try:
                live_ticks = load_ticks_csv(LIVE_SOURCE)
            except (OSError, csv.Error) as e:
                # scraper.py may be mid-write, or briefly unavailable —
                # skip this poll instead of crashing the pipeline.
                print(f"[pipeline] could not read live ticks this cycle: {e}")
                time.sleep(POLL_SECONDS)
                continue

            new_ticks = [t for t in live_ticks if t["timestamp"] not in seen_timestamps]

            for tick in new_ticks:
                seen_timestamps.add(tick["timestamp"])

                # Resolve the previous pending prediction now that we know
                # what actually happened next.
                if pending is not None:
                    actual = make_label(pending["buffer_snapshot"], tick)
                    log_prediction(
                        timestamp=pending["timestamp"],
                        features=pending["features"],
                        predictions=pending["predictions"],
                        actual=actual,
                        latency_ms=pending["latency_ms"],
                    )
                    total_ticks_processed += 1

                buffer.append(tick)
                if len(buffer) > BUFFER_SIZE:
                    buffer = buffer[-BUFFER_SIZE:]

                if len(buffer) < BUFFER_SIZE:
                    pending = None
                    continue

                predict_start = time.perf_counter()
                feature_dict = compute_features(buffer)
                feature_vector = _feature_vector(feature_dict)
                predictions = predict(models, feature_vector)
                latency_ms = (time.perf_counter() - predict_start) * 1000

                pending = {
                    "timestamp": tick["timestamp"],
                    "buffer_snapshot": list(buffer),
                    "features": feature_dict,
                    "predictions": predictions,
                    "latency_ms": latency_ms,
                }

                acc_logreg = compute_rolling_accuracy("logreg")
                acc_rf = compute_rolling_accuracy("random_forest")
                acc_lstm = compute_rolling_accuracy("lstm")  # Add this line
                drift = compute_drift_flag(feature_dict["rolling_vol_10"], training_volatility)

                print(
                    f"[pipeline] tick={tick['timestamp']} price={tick['price_usd']:.2f} "
                    f"pred(log={predictions.get('logreg')}, rf={predictions.get('random_forest')}, lstm={predictions.get('lstm')}) "
                    f"acc(log={acc_logreg:.3f}, rf={acc_rf:.3f}, lstm={acc_lstm:.3f}) "
                    f"latency={latency_ms:.2f}ms drift={drift} "
                    f"ticks_processed={total_ticks_processed}"
                )
                if total_ticks_processed and total_ticks_processed % SAVE_EVERY_N_TICKS == 0:
                    _save_checkpoint(models, start_time, total_ticks_processed)

            time.sleep(POLL_SECONDS)

    except KeyboardInterrupt:
        print("\n[pipeline] stopping...")

    finally:
        if pending is not None:
            log_prediction(
                timestamp=pending["timestamp"],
                features=pending["features"],
                predictions=pending["predictions"],
                actual=None,
                latency_ms=pending["latency_ms"],
            )
        _save_checkpoint(models, start_time, total_ticks_processed)


if __name__ == "__main__":
    run_pipeline()