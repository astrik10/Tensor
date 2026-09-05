# RUN.md — TENSOR-TITANS

Real-time Bitcoin tick prediction and MLOps monitoring pipeline (classical ML: Logistic Regression + Random Forest).

## Pipeline Overview

```
scraper.py (CoinGecko poll, 15s)
        │
        ▼
data/live_ticks.csv  ◄──────── data/historical_ticks.csv (bootstrap)
        │
        ▼
src/features.py   → compute_features() (rolling mean 5/20, momentum 5, rolling vol 10)
        │
        ▼
src/models.py     → train_models() / predict() / save_models()
        │
        ▼
src/monitor.py    → log_prediction() / compute_rolling_accuracy() / compute_drift_flag() / write_run_log()
        │
        ▼
models/*.pkl · run_log.json · logs/predictions.jsonl
        │
        ▼
dashboard.py (Streamlit, optional — reads the above, doesn't compute anything new)
```

`src/run_pipeline.py` is the orchestrator: it bootstraps a training set from `data/historical_ticks.csv`, trains both models once, then polls `data/live_ticks.csv` for new ticks, predicts on each, resolves the previous prediction once the next tick is known, and logs everything.

## Setup (Windows / PyCharm)

**Command Prompt:**
```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**PowerShell** (if you get an execution-policy error on activate):
```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**In PyCharm:** File → Settings → Project → Python Interpreter → Add Interpreter → Virtualenv Environment, pointing at the `venv` folder above. Then just use PyCharm's built-in Terminal tab (already activates the venv) for every command below.

Dependencies installed (`requirements.txt`):
```
requests>=2.31.0
pandas>=2.2.0
numpy>=1.26.0
scikit-learn>=1.4.0
streamlit>=1.29.0
```
No dataset preparation step is needed — `data/historical_ticks.csv` (synthetic bootstrap data) ships with the repo.

## How to Run

Run these in order, each in its own terminal, from the project root.

**1. (Optional, recommended) Verify the feature engineering tests pass:**
```bat
python -m pytest tests/test_features.py -v
```

**2. Start the scraper** — polls CoinGecko every 15s and appends to `data\live_ticks.csv`. Needed if you want the pipeline to keep predicting past the bootstrap phase:
```bat
python scraper.py
```

**3. Start the pipeline** — this is the actual entry point:
```bat
python src\run_pipeline.py
```
What happens automatically, with no extra commands needed:
- Bootstraps a training set from `data\historical_ticks.csv` (sliding 20-tick window → features + labels) and trains Logistic Regression + Random Forest.
- Polls `data\live_ticks.csv` every 15s, computes features per new tick, predicts with both models.
- Once the *next* tick lands, resolves the previous prediction's correctness and appends a record to `logs\predictions.jsonl`.
- Every 50 resolved ticks — and again on `Ctrl+C` shutdown — saves versioned model checkpoints to `models\` and overwrites `run_log.json`.

**4. (Optional) Launch the monitoring dashboard:**
```bat
streamlit run dashboard.py
```
Purely a viewer — it reads `data\live_ticks.csv`, `models\`, `run_log.json`, and `logs\predictions.jsonl`; it does not train or predict independently of what's already on disk (it does recompute one live prediction per refresh from the latest saved models for display purposes).

## Where to Find the Results

| Artifact | Path |
|---|---|
| Run log (metadata, model versions, final accuracy) | `run_log.json` |
| Per-tick prediction log | `logs\predictions.jsonl` |
| Saved model checkpoints (versioned) | `models\` (e.g. `logreg_v7_<timestamp>.pkl`) |
| Live tick data | `data\live_ticks.csv` |
| Bootstrap/training data | `data\historical_ticks.csv` |

## Final Accuracy

From the most recent `run_log.json` in this repo (run `2026-09-05T06:45:22Z` → `2026-09-05T06:45:46Z`, 110 ticks processed):

| Model | Accuracy | Ticks evaluated |
|---|---:|---:|
| Logistic Regression | 85.20% | 110 |
| Random Forest | 87.44% | 110 |

> Note: `write_run_log()` **overwrites** `run_log.json` on every checkpoint, so these numbers reflect one specific run, not a fixed benchmark. Before final submission, open `run_log.json` yourself and confirm `final_accuracy` / `total_ticks_processed` match what's shown here — if you've run the pipeline again since, re-paste the current values.

## Troubleshooting (Windows / PyCharm)

- **`ModuleNotFoundError: No module named 'src'`** — run commands from the project root, not from inside `src\`. `run_pipeline.py` already inserts the project root onto `sys.path`, but only if it's executed as `python src\run_pipeline.py` from the root. In PyCharm, check the run configuration's **Working directory** is set to the project root, not `src\`.
- **`429 Client Error: Too Many Requests` from `scraper.py`** — CoinGecko's free public API rate-limits by IP. This is more likely on shared/university/office Wi-Fi where other devices hit the same endpoint. `scraper.py` already retries 3 times and skips that poll rather than crashing — no action needed unless it happens on nearly every poll, in which case increase `POLL_SECONDS` in `scraper.py`.
- **`RuntimeError: Need at least 21 historical bitcoin ticks to bootstrap`** — `data\historical_ticks.csv` has fewer than `BUFFER_SIZE + 1` (21) bitcoin rows. Don't edit this file unless you're intentionally replacing the sample dataset.
- **Dashboard shows "No saved model artifacts found yet"** — `src\run_pipeline.py` hasn't been run yet (or hasn't reached its first checkpoint / shutdown save). Run it first.
- **PowerShell won't let you activate the venv** — run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal session first (see Setup above).
- **`scikit-learn`/`pandas` import errors after install** — confirm the PyCharm interpreter is actually pointed at the `venv` you created, not the system Python (Settings → Project → Python Interpreter).

## Known Issues / Future Improvements

- **Bitcoin only** — the scraper also collects Ethereum prices into `data\live_ticks.csv`, but the pipeline filters to `coin == "bitcoin"` only; the ETH stretch goal isn't implemented.
- **Only 2 of 3 possible models** — `src\models.py` implements Logistic Regression and Random Forest; the optional KNN stretch model isn't present.
- **`training_volatility` isn't persisted** — `run_pipeline.bootstrap()` computes it in memory for the drift check but never writes it to `run_log.json` or anywhere on disk. Any other tool that needs it (e.g. a dashboard) has to recompute it from `data\historical_ticks.csv` independently, which is wasted computation and a risk of drift-check inconsistency between processes.
- **No hyperparameter tuning** — both models use fixed settings (`LogisticRegression(max_iter=1000)`, `RandomForestClassifier(n_estimators=200)`); no cross-validation or search was done.
- **`logs\predictions.jsonl` is append-only across runs** — restarting `run_pipeline.py` doesn't separate or version prediction history by run, so `compute_rolling_accuracy()` can mix predictions from different trained model versions if the pipeline has been restarted multiple times.
- **Single evaluation metric** — only accuracy is tracked; precision/recall/F1 would give a fuller picture, especially if UP/DOWN classes are imbalanced.
- **No automated tests beyond features** — `tests/` only covers `src\features.py`; `models.py`, `monitor.py`, and `run_pipeline.py` have no unit tests.
- **Small, synthetic bootstrap dataset** — `data\historical_ticks.csv` is a generated random-walk series (per `README.md`), not real historical CoinGecko data, so bootstrap-phase accuracy may not reflect real market behavior.