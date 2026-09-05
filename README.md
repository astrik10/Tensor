# CryptoStream

## Background

Financial and crypto markets are a classic real-world streaming use case:
new price ticks arrive continuously, and systems need to predict, log, and
monitor in real time. In this problem you'll build a small end-to-end
pipeline: ingest live BTC price ticks, engineer features from a rolling
window, predict short-term price direction with classical ML, and wrap it
all with basic MLOps practices (versioning, logging, drift monitoring).

## Task

Using a rolling window of recent BTC price ticks, predict whether the
**next tick's price will go UP or DOWN** relative to the current tick
(binary classification). Bitcoin only.

## Architecture

```
scraper.py (provided)
      |
      v
data/live_ticks.csv  --------\
                               \
data/historical_ticks.csv ---> ingestion buffer (rolling window, N=20)
 (provided, for bootstrap)            |
                                       v
                              feature engineering
                            (src/features.py — TODO)
                                       |
                                       v
                        [ Logistic Regression + Random Forest ]
                              (src/models.py — TODO)
                                       |
                                       v
                        monitor / log / drift flag
                            (src/monitor.py — TODO)
```

## What's provided

- **`scraper.py`** fully working. Polls the CoinGecko public API
  (`/api/v3/simple/price`) every 15 seconds for BTC (and ETH, for the
  optional stretch goal) and appends readings to `data/live_ticks.csv`.
  The scraping itself is intentionally trivial one GET request, no
  auth so it can't become a source of unfair advantage or wasted time.
  Run it in a separate terminal: `python scraper.py`.
- **`data/historical_ticks.csv`** 2 hours of sample BTC tick data
  (15-second intervals) so you can build and test your whole pipeline
  without waiting on live data to accumulate. **Note:** this sample file
  is synthetically generated (a realistic random-walk series, not a
  scrape of actual historical CoinGecko data) it exists purely so your
  bootstrap/training logic has something to chew on immediately. Swap in
  a real scrape ahead of time if you want authentic numbers for the
  final demo.
- **`requirements.txt`** all the libraries you should need.
- Stub files in `src/` with function signatures and docstrings.
- **`tests/test_features.py`** tests that currently fail; make them pass.

## Your job

Implement the following:

1. **`src/features.py`**
   - `compute_features(buffer)` rolling mean (5), rolling mean (20),
     5-tick momentum, rolling volatility (10, std. dev.)
   - `make_label(buffer, next_tick)` 1 if price went up on the next
     tick, else 0
2. **`src/models.py`**
   - `train_models(X, y)` train Logistic Regression + Random Forest
     (KNN optional, for stretch)
   - `predict(models, feature_vector)` get a 0/1 prediction from each
     model
   - `save_models(models)` save each model to disk with a versioned
     filename, e.g. `models/rf_v1_<timestamp>.pkl`
3. **`src/monitor.py`**
   - `log_prediction(...)` append a JSON line per tick with features,
     predictions, actual outcome (once known), and latency
   - `compute_rolling_accuracy(model_name, window)` accuracy over the
     last `window` resolved predictions
   - `compute_drift_flag(current_volatility, training_volatility, threshold)`
     flag if current volatility is > 1.5x the training-time volatility
   - `write_run_log(metadata)` write `run_log.json` with run metadata
4. **`src/run_pipeline.py`**
   - Wire the above into one orchestration loop: bootstrap → train →
     ingest live ticks → feature engineer → predict → log → monitor →
     periodically save models + write run log
   - A live-updating view of accuracy / tick count / latency / drift flag
     (a simple CLI print loop is fine; Streamlit is a nice-to-have)

## Required pipeline stages (recap)

1. Ingestion: rolling buffer of last N ticks (N=20 suggested)
2. Feature engineering: the 4 features above, computed per new tick
3. Modeling: bootstrap training set (from historical CSV or ~15-20 min
   of live data), train LogReg + Random Forest
4. Serving: predict on each new tick after bootstrap, log correctness
   once the real next tick arrives
5. Monitoring: running accuracy per model, tick count, per-prediction
   latency, drift flag
6. Versioning/logging: versioned model files + `run_log.json`

## Constraints

- Classical ML only no deep learning.
- Handle a briefly-unavailable scraper (one failed request) without
  crashing.

## Deliverables

- Working pipeline runnable via one command, e.g. `python src/run_pipeline.py`
- `RUN.md` (fill in `RUN.md.template`) with exact run instructions
- `run_log.json` from a run of ≥ 100 processed ticks (historical data counts
  toward this if your live run is short)
- Final accuracy of both models

## Stretch goals (optional)

- Add Ethereum as a second tracked coin
- Add a 3rd model (KNN or SVM) to the comparison
- Alert if running accuracy drops below a threshold
- Auto-retrain trigger when the drift flag stays active for N consecutive ticks

## Judging weights

| Criterion | Weight |
|---|---|
| Core Functionality | 40% |
| Creativity/ Extra Features | 20% |
| Code Quality & Repo Hygiene | 15% |
| UI/ UX | 15% |
| Presentation / Documentation | 10% |

## Getting started

```bash
pip install -r requirements.txt
# terminal 1 start the live scraper (optional while developing)
python scraper.py
# terminal 2 develop against the historical CSV first, then switch to live
python -m pytest tests/test_features.py -v
python src/run_pipeline.py
```
