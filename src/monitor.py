"""
monitor.py

Logging and live-monitoring helpers: per-tick prediction logging, rolling
accuracy, latency tracking, and a simple volatility-drift flag.
"""

from typing import Dict, List, Optional
import json
import os

LOG_PATH = os.path.join("logs", "predictions.jsonl")
RUN_LOG_PATH = "run_log.json"


def log_prediction(
    timestamp: str,
    features: Dict[str, float],
    predictions: Dict[str, int],
    actual: Optional[int],
    latency_ms: float,
) -> None:
    """
    Append one line of JSON to logs/predictions.jsonl recording:
        timestamp, features, predictions (per model), actual (may be None
        until the next tick arrives), latency_ms.

    Creates the logs/ directory if it doesn't exist.
    """
    log_dir = os.path.dirname(LOG_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    record = {
        "timestamp": timestamp,
        "features": features,
        "predictions": predictions,
        "actual": actual,
        "latency_ms": latency_ms,
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _read_log_records() -> List[Dict]:
    """Reads predictions.jsonl into a list of dicts, skipping any corrupted
    line (e.g. from a crash mid-write) instead of failing the whole read."""
    if not os.path.exists(LOG_PATH):
        return []

    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def compute_rolling_accuracy(model_name: str, window: int = 50) -> float:
    """
    Read logs/predictions.jsonl and compute accuracy for `model_name` over
    the last `window` ticks that have a non-null `actual` value.

    Returns:
        accuracy as a float in [0, 1]. Returns 0.0 if there's no data yet.
    """
    records = _read_log_records()

    resolved = [
        r for r in records
        if r.get("actual") is not None and model_name in r.get("predictions", {})
    ]
    if not resolved:
        return 0.0

    recent = resolved[-window:]
    correct = sum(1 for r in recent if r["predictions"][model_name] == r["actual"])
    return correct / len(recent)


def compute_drift_flag(
    current_volatility: float, training_volatility: float, threshold: float = 1.5
) -> bool:
    """
    Return True if current_volatility is more than `threshold`x the
    volatility seen during training (a simple, cheap drift signal).
    """
    if training_volatility <= 0:
        # No meaningful baseline to compare against yet — don't false-flag.
        return False
    return current_volatility > threshold * training_volatility


def write_run_log(metadata: Dict) -> None:
    """
    Write/overwrite run_log.json with run metadata, e.g.:
        {
          "start_time": "...",
          "model_versions": {"logreg": "models/logreg_v1_....pkl", ...},
          "total_ticks_processed": 123,
          "final_accuracy": {"logreg": 0.55, "random_forest": 0.58}
        }
    """
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)