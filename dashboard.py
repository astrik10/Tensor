"""
dashboard.py

Streamlit dashboard for TENSOR-TITANS — real-time Bitcoin tick prediction
and MLOps monitoring.

This dashboard is READ-ONLY with respect to the pipeline: it does not
retrain or duplicate run_pipeline.py's orchestration loop. It:
    - reads data/live_ticks.csv directly for the price chart / KPIs
    - loads the most recently saved model artifacts from models/ to
      surface a live "next tick" prediction
    - reads logs/predictions.jsonl and run_log.json for monitoring,
      accuracy, and history

It imports compute_features() from src/features.py and
compute_rolling_accuracy()/compute_drift_flag() from src/monitor.py so
that every number shown matches the actual pipeline logic exactly.

One known gap: run_pipeline.py computes `training_volatility` in memory
during bootstrap() but never persists it (not in run_log.json, not
anywhere on disk). To show a real Drift Status, this dashboard recomputes
it from data/historical_ticks.csv using the *same* sliding-window logic
as run_pipeline.bootstrap() (see compute_training_volatility() below).
It is not a fabricated number — it's the same computation, just re-run
here since the backend doesn't expose it.

Run with:
    streamlit run dashboard.py
"""

import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

BACKEND_AVAILABLE = True
BACKEND_IMPORT_ERROR = None
try:
    from src.features import compute_features
    from src.models import predict as backend_predict
    from src.monitor import compute_drift_flag, compute_rolling_accuracy
except ImportError as e:  # pragma: no cover - defensive, not expected in normal use
    BACKEND_AVAILABLE = False
    BACKEND_IMPORT_ERROR = str(e)

PLOTLY_AVAILABLE = True
try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    PLOTLY_AVAILABLE = False

# Constants — mirror run_pipeline.py exactly so features/predictions line up
BUFFER_SIZE = 20
FEATURE_ORDER = ["rolling_mean_5", "rolling_mean_20", "momentum_5", "rolling_vol_10"]
MODEL_NAMES = ["logreg", "random_forest", "lstm"]
MODEL_DISPLAY = {"logreg": "Logistic Regression", "random_forest": "Random Forest", "lstm": "LSTM (Astra)"}
LIVE_PATH = os.path.join("data", "live_ticks.csv")
HIST_PATH = os.path.join("data", "historical_ticks.csv")
LOG_PATH = os.path.join("logs", "predictions.jsonl")
RUN_LOG_PATH = "run_log.json"
MODELS_DIR = "models"

POLL_SECONDS_DEFAULT = 15  # matches scraper.py / run_pipeline.py cadence


# Data loading helpers — all defensive, none of them raise into the UI
@st.cache_data(ttl=10)
def load_live_data() -> pd.DataFrame:
    """Load data/live_ticks.csv, bitcoin rows only, safely."""
    empty = pd.DataFrame(columns=["timestamp", "coin", "price_usd", "timestamp_parsed"])
    if not os.path.exists(LIVE_PATH):
        return empty
    try:
        df = pd.read_csv(LIVE_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return empty
    if not {"timestamp", "coin", "price_usd"}.issubset(df.columns):
        return empty

    df = df[df["coin"] == "bitcoin"].copy()
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["price_usd", "timestamp_parsed"])
    df = df.sort_values("timestamp_parsed").reset_index(drop=True)
    return df


@st.cache_data(ttl=10)
def load_prediction_log() -> pd.DataFrame:
    """Load logs/predictions.jsonl (written by monitor.log_prediction) safely."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()
    records = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return pd.DataFrame()
    if not records:
        return pd.DataFrame()

    df = pd.json_normalize(records)
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp_parsed").reset_index(drop=True)
    return df


@st.cache_data(ttl=30)
def load_run_log():
    """Load run_log.json (written by monitor.write_run_log) safely."""
    if not os.path.exists(RUN_LOG_PATH):
        return None
    try:
        with open(RUN_LOG_PATH) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _latest_model_file(name: str):
    """Same versioning scheme as models.save_models()/_next_version() — pick the
    highest v<N> for this model name currently on disk."""
    pattern = os.path.join(MODELS_DIR, f"{name}_v*_*.pkl")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    version_re = re.compile(rf"^{re.escape(name)}_v(\d+)_")

    def version_of(path):
        m = version_re.match(os.path.basename(path))
        return int(m.group(1)) if m else -1

    return max(candidates, key=version_of)


@st.cache_resource(ttl=60)
def load_latest_models():
    """Load the most recently saved logreg/random_forest artifacts from models/."""
    models, paths, errors = {}, {}, {}
    for name in MODEL_NAMES:
        path = _latest_model_file(name)
        if path is None:
            errors[name] = "no saved model file found in models/ yet"
            continue
        try:
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
            paths[name] = path
        except (pickle.UnpicklingError, EOFError, OSError, ValueError) as e:
            errors[name] = str(e)
    return models, paths, errors


@st.cache_data(ttl=300)
def compute_training_volatility():
    """Recompute training_volatility from data/historical_ticks.csv using the
    exact same sliding-window approach as run_pipeline.bootstrap(). Returns
    None if unavailable — never fabricated."""
    if not BACKEND_AVAILABLE or not os.path.exists(HIST_PATH):
        return None
    try:
        hist_df = pd.read_csv(HIST_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if not {"coin", "price_usd"}.issubset(hist_df.columns):
        return None

    hist_df = hist_df[hist_df["coin"] == "bitcoin"].copy()
    hist_df["price_usd"] = pd.to_numeric(hist_df["price_usd"], errors="coerce")
    hist_df = hist_df.dropna(subset=["price_usd"])
    ticks = hist_df.to_dict("records")
    if len(ticks) < BUFFER_SIZE + 1:
        return None

    vols = []
    for end in range(BUFFER_SIZE, len(ticks)):
        window = ticks[end - BUFFER_SIZE:end]
        try:
            fd = compute_features(window)
            vols.append(fd["rolling_vol_10"])
        except (ValueError, KeyError):
            continue
    return float(np.mean(vols)) if vols else None


def get_latest_btc_data(df: pd.DataFrame):
    return None if df.empty else df.iloc[-1]


def calculate_price_change(df: pd.DataFrame, periods: int = 20):
    if df.empty or len(df) < 2:
        return None
    latest = float(df["price_usd"].iloc[-1])
    idx = max(0, len(df) - 1 - periods)
    baseline = float(df["price_usd"].iloc[idx])
    abs_change = latest - baseline
    pct_change = (abs_change / baseline * 100) if baseline else 0.0
    return {"abs": abs_change, "pct": pct_change}


def compute_current_prediction(live_df: pd.DataFrame, models: dict):
    """Runs the real feature pipeline + real loaded models on the latest
    BUFFER_SIZE live ticks. Returns None if there isn't enough data yet —
    never invents a prediction."""
    if not BACKEND_AVAILABLE or live_df.empty or len(live_df) < BUFFER_SIZE:
        return None

    buffer = live_df.tail(BUFFER_SIZE).to_dict("records")
    try:
        feature_dict = compute_features(buffer)
    except (ValueError, KeyError):
        return None

    feature_vector = [feature_dict[name] for name in FEATURE_ORDER]

    predictions, confidences = {}, {}
    if models:
        try:
            predictions = backend_predict(models, feature_vector)
        except ValueError:
            predictions = {}
        for name, model in models.items():
            pred_label = predictions.get(name)
            if pred_label is None or not hasattr(model, "predict_proba"):
                continue
            try:
                proba = model.predict_proba([feature_vector])[0]
                class_idx = list(model.classes_).index(pred_label)
                confidences[name] = float(proba[class_idx])
            except (ValueError, IndexError):
                continue

    return {
        "features": feature_dict,
        "predictions": predictions,
        "confidences": confidences,
        "price": float(buffer[-1]["price_usd"]),
        "timestamp": buffer[-1]["timestamp"],
    }


# Small formatting / display helpers

def _direction_label(v) -> str:
    if v is None:
        return "—"
    return "UP" if int(v) == 1 else "DOWN"


def _fmt_pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def _fmt_ms(v):
    return "—" if v is None else f"{v:.1f} ms"


def _line_chart(df: pd.DataFrame, x: str, y: str, title: str, y_label: str):
    if df.empty or x not in df.columns or y not in df.columns:
        st.info(f"Not enough data yet for '{title}'.")
        return
    plot_df = df[[x, y]].dropna()
    if plot_df.empty:
        st.info(f"Not enough data yet for '{title}'.")
        return
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df[x], y=plot_df[y], mode="lines",
                                  line=dict(width=2, color="#4fd1c5")))
        fig.update_layout(
            title=title, template="plotly_dark", height=320,
            margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title=y_label, xaxis_title=None,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(title)
        st.line_chart(plot_df.set_index(x)[y], height=280)


# Page setup / styling
st.set_page_config(
    page_title="TENSOR TITANS — BTC Prediction & MLOps",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.6rem; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 12px 14px;
    }
    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
    }
    .badge-green { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.4);}
    .badge-red   { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.4);}
    .badge-amber { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4);}
    .badge-gray  { background: rgba(148,163,184,0.15); color: #94a3b8; border: 1px solid rgba(148,163,184,0.4);}
    .pred-box {
        border-radius: 14px; padding: 22px 24px; text-align: center;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .pred-up   { background: rgba(52,211,153,0.10); }
    .pred-down { background: rgba(248,113,113,0.10); }
    .pred-flat { background: rgba(148,163,184,0.08); }
</style>
""", unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("### TENSOR TITANS")
    st.caption("BTC tick prediction & MLOps monitor")
    st.divider()

    auto_refresh = st.toggle("Auto-refresh", value=False)
    refresh_interval = st.slider("Refresh interval (sec)", 5, 60, POLL_SECONDS_DEFAULT, step=5)
    st.divider()

    recent_n = st.number_input("Recent ticks to display", min_value=5, max_value=200, value=15, step=5)
    rolling_window = st.number_input("Rolling accuracy window", min_value=5, max_value=500, value=50, step=5)
    primary_model = st.selectbox(
        "Primary model (prediction panel)",
        options=MODEL_NAMES,
        format_func=lambda m: MODEL_DISPLAY[m],
    )
    shown_models = st.multiselect(
        "Models in performance table",
        options=MODEL_NAMES,
        default=MODEL_NAMES,
        format_func=lambda m: MODEL_DISPLAY[m],
    )

    if not BACKEND_AVAILABLE:
        st.error(f"Backend import failed: {BACKEND_IMPORT_ERROR}")

# Load everything up front
live_df = load_live_data()
log_df = load_prediction_log()
run_log = load_run_log()
models, model_paths, model_errors = load_latest_models() if BACKEND_AVAILABLE else ({}, {}, {})
current = compute_current_prediction(live_df, models)
training_volatility = compute_training_volatility()


# 1. Header
def render_header():
    latest = get_latest_btc_data(live_df)
    if latest is None:
        status_badge = '<span class="badge badge-red">NO DATA</span>'
        last_update_str = "—"
    else:
        age = (pd.Timestamp.now(tz="UTC") - latest["timestamp_parsed"]).total_seconds()
        if age < 45:
            status_badge = '<span class="badge badge-green">● LIVE</span>'
        elif age < 180:
            status_badge = '<span class="badge badge-amber">● DELAYED</span>'
        else:
            status_badge = '<span class="badge badge-red">● STALE</span>'
        last_update_str = latest["timestamp_parsed"].strftime("%Y-%m-%d %H:%M:%S UTC")

    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown("## TENSOR TITANS")
        st.caption("Real-Time Bitcoin Tick Prediction & MLOps Monitoring · BTC/USD")
    with c2:
        st.markdown(f"{status_badge}", unsafe_allow_html=True)
        st.caption(f"Last update: {last_update_str}")
    st.divider()

# 2. KPI cards
def render_kpis():
    latest = get_latest_btc_data(live_df)
    change = calculate_price_change(live_df, periods=min(20, max(1, len(live_df) - 1)))
    rolling_acc = compute_rolling_accuracy(primary_model, window=int(rolling_window)) if BACKEND_AVAILABLE else None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("BTC Price", f"${latest['price_usd']:,.2f}" if latest is not None else "—")
    c2.metric(
        "Price Change",
        f"{change['pct']:+.2f}%" if change else "—",
        delta=f"{change['abs']:+.2f} USD" if change else None,
    )
    pred_val = current["predictions"].get(primary_model) if current else None
    c3.metric("Prediction", _direction_label(pred_val) if current else "—")
    conf_val = current["confidences"].get(primary_model) if current else None
    c4.metric("Confidence", _fmt_pct(conf_val) if current else "—")
    if rolling_acc is not None:
        c5.metric(f"Rolling Accuracy ({MODEL_DISPLAY[primary_model]})", f"{rolling_acc * 100:.1f}%")
    else:
        c5.metric("Rolling Accuracy", "—")
    st.divider()


# 3. Main price chart
def render_price_chart():
    st.subheader("Bitcoin Price")
    if live_df.empty:
        st.info("No data yet in data/live_ticks.csv — start scraper.py to begin collecting ticks.")
        return
    plot_df = live_df.tail(max(int(recent_n) * 4, 100))
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["timestamp_parsed"], y=plot_df["price_usd"],
            mode="lines", line=dict(width=2, color="#f0b90b"),
            fill="tozeroy", fillcolor="rgba(240,185,11,0.06)",
        ))
        fig.update_layout(
            template="plotly_dark", height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Time", yaxis_title="Price (USD)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
        )
        fig.update_yaxes(tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(plot_df.set_index("timestamp_parsed")["price_usd"], height=400)
    st.divider()


# ---------------------------------------------------------------------------
# 4. Prediction panel
# ---------------------------------------------------------------------------
def render_prediction_panel():
    st.subheader("Next Tick Prediction")
    if not BACKEND_AVAILABLE:
        st.warning("Backend not available — cannot compute predictions.")
        return
    if not models:
        st.info("No saved model artifacts found yet in models/. Run `python src/run_pipeline.py` first.")
        return
    if current is None:
        st.info(f"Need at least {BUFFER_SIZE} bitcoin ticks in data/live_ticks.csv to predict — "
                f"currently have {len(live_df)}.")
        return

    cols = st.columns(len(models))
    for col, name in zip(cols, models.keys()):
        pred = current["predictions"].get(name)
        conf = current["confidences"].get(name)
        css_class = "pred-up" if pred == 1 else ("pred-down" if pred == 0 else "pred-flat")
        with col:
            st.markdown(f"""
            <div class="pred-box {css_class}">
                <div style="font-size:0.85rem; opacity:0.75;">{MODEL_DISPLAY[name]}</div>
                <div style="font-size:2rem; font-weight:700; margin:6px 0;">{_direction_label(pred)}</div>
                <div style="font-size:0.85rem; opacity:0.75;">Confidence: {_fmt_pct(conf)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(
        f"Based on price ${current['price']:,.2f} at {current['timestamp']} "
        f"· {len(models)} model(s) loaded from models/"
    )
    st.divider()


# ---------------------------------------------------------------------------
# 5. Model performance
# ---------------------------------------------------------------------------
def render_model_performance():
    st.subheader("Model Performance")
    if not BACKEND_AVAILABLE:
        st.warning("Backend not available — cannot compute accuracy.")
        return

    final_acc = (run_log or {}).get("final_accuracy", {})
    rows = []
    for name in shown_models or MODEL_NAMES:
        rolling_acc = compute_rolling_accuracy(name, window=int(rolling_window))
        rows.append({
            "Model": MODEL_DISPLAY[name],
            f"Rolling Accuracy (last {int(rolling_window)})": f"{rolling_acc * 100:.1f}%" if rolling_acc else "No data yet",
            "Last Run Final Accuracy": f"{final_acc[name] * 100:.1f}%" if name in final_acc else "—",
            "Status": "Active" if name in models else "No saved artifact",
            "_rolling_raw": rolling_acc,
        })

    if not rows:
        st.info("No models selected.")
        return

    best = max(rows, key=lambda r: r["_rolling_raw"]) if any(r["_rolling_raw"] > 0 for r in rows) else None
    df_display = pd.DataFrame(rows).drop(columns=["_rolling_raw"])
    st.dataframe(df_display, hide_index=True, use_container_width=True)
    if best and best["_rolling_raw"] > 0:
        st.caption(f"🏆 Best performing model right now: **{best['Model']}**")
    st.divider()


# ---------------------------------------------------------------------------
# 6. MLOps monitoring
# ---------------------------------------------------------------------------
def render_monitoring():
    st.subheader("MLOps Monitoring")
    if not BACKEND_AVAILABLE:
        st.warning("Backend not available — cannot compute monitoring signals.")
        return

    latest_latency = None
    if not log_df.empty and "latency_ms" in log_df.columns:
        latest_latency = log_df["latency_ms"].dropna().iloc[-1] if not log_df["latency_ms"].dropna().empty else None

    current_vol = current["features"]["rolling_vol_10"] if current else None
    drift = None
    if current_vol is not None and training_volatility is not None:
        drift = compute_drift_flag(current_vol, training_volatility)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inference Latency", _fmt_ms(latest_latency))
    c2.metric("Current Volatility", f"{current_vol:,.4f}" if current_vol is not None else "—")
    c3.metric("Training Volatility", f"{training_volatility:,.4f}" if training_volatility is not None else "—")
    with c4:
        if drift is None:
            st.markdown('<span class="badge badge-gray">DRIFT: UNKNOWN</span>', unsafe_allow_html=True)
            st.caption("Need current + training volatility")
        elif drift:
            st.markdown('<span class="badge badge-red">⚠ DRIFT DETECTED</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-green">NORMAL</span>', unsafe_allow_html=True)

    st.markdown("##### Pipeline Signal")
    st.caption("Derived from file freshness / contents — not a live process check.")
    p1, p2, p3, p4 = st.columns(4)

    latest_tick = get_latest_btc_data(live_df)
    if latest_tick is not None:
        age = (pd.Timestamp.now(tz="UTC") - latest_tick["timestamp_parsed"]).total_seconds()
        ingest_ok = age < 60
    else:
        ingest_ok = False
    p1.markdown(f"**Data Ingestion**<br><span class='badge {'badge-green' if ingest_ok else 'badge-amber'}'>"
                f"{'Fresh' if ingest_ok else 'Stale/No data'}</span>", unsafe_allow_html=True)

    feat_ok = current is not None
    p2.markdown(f"**Feature Engineering**<br><span class='badge {'badge-green' if feat_ok else 'badge-amber'}'>"
                f"{'Ready' if feat_ok else 'Insufficient ticks'}</span>", unsafe_allow_html=True)

    model_ok = len(models) > 0
    p3.markdown(f"**Model Artifacts**<br><span class='badge {'badge-green' if model_ok else 'badge-red'}'>"
                f"{'Loaded' if model_ok else 'Missing'}</span>", unsafe_allow_html=True)

    log_ok = not log_df.empty
    p4.markdown(f"**Prediction Logging**<br><span class='badge {'badge-green' if log_ok else 'badge-gray'}'>"
                f"{'Active' if log_ok else 'No logs yet'}</span>", unsafe_allow_html=True)

    st.divider()


# ---------------------------------------------------------------------------
# 7. Recent predictions table
# ---------------------------------------------------------------------------
def render_recent_predictions():
    st.subheader("Recent Predictions")
    if log_df.empty:
        st.info("No entries in logs/predictions.jsonl yet — predictions are logged once "
                "run_pipeline.py resolves each tick's outcome.")
        st.divider()
        return

    table = log_df.tail(int(recent_n)).copy()

    price_lookup = live_df.set_index("timestamp")["price_usd"] if not live_df.empty else pd.Series(dtype=float)
    table["BTC Price"] = table["timestamp"].map(price_lookup)

    display_cols = {"timestamp": "Timestamp", "BTC Price": "BTC Price"}
    for name in MODEL_NAMES:
        pred_col = f"predictions.{name}"
        if pred_col in table.columns:
            table[f"{MODEL_DISPLAY[name]} Pred"] = table[pred_col].apply(_direction_label)
            if "actual" in table.columns:
                table[f"{MODEL_DISPLAY[name]} Correct"] = table.apply(
                    lambda r: "—" if pd.isna(r.get("actual"))
                    else ("✓" if r.get(pred_col) == r.get("actual") else "✗"),
                    axis=1,
                )
    if "actual" in table.columns:
        table["Actual"] = table["actual"].apply(lambda v: "Pending" if pd.isna(v) else _direction_label(v))
    if "latency_ms" in table.columns:
        table["Latency (ms)"] = table["latency_ms"].round(2)

    cols_to_show = ["timestamp", "BTC Price"] + \
        [c for c in table.columns if c.endswith("Pred") or c.endswith("Correct")] + \
        [c for c in ["Actual", "Latency (ms)"] if c in table.columns]
    cols_to_show = [c for c in cols_to_show if c in table.columns]

    st.dataframe(
        table[cols_to_show].rename(columns={"timestamp": "Timestamp"}).iloc[::-1],
        hide_index=True, use_container_width=True,
    )
    st.divider()

# 8. Volatility / accuracy / latency history charts
def render_history_charts():
    st.subheader("Monitoring History")
    if log_df.empty:
        st.info("No logged predictions yet — charts will populate once run_pipeline.py has been running.")
        st.divider()
        return

    tab1, tab2, tab3 = st.tabs(["Volatility", "Rolling Accuracy", "Latency"])

    with tab1:
        if "features.rolling_vol_10" in log_df.columns:
            _line_chart(log_df, "timestamp_parsed", "features.rolling_vol_10",
                        "Rolling Volatility (10-tick)", "Volatility")
        else:
            st.info("No volatility data logged yet.")

    with tab2:
        if "actual" in log_df.columns:
            resolved = log_df.dropna(subset=["actual"]).copy()
            for name in shown_models or MODEL_NAMES:
                pred_col = f"predictions.{name}"
                if pred_col not in resolved.columns or resolved.empty:
                    continue
                resolved[f"correct_{name}"] = (resolved[pred_col] == resolved["actual"]).astype(int)
                w = min(20, len(resolved))
                resolved[f"rolling_acc_{name}"] = resolved[f"correct_{name}"].rolling(w, min_periods=1).mean()
                _line_chart(resolved, "timestamp_parsed", f"rolling_acc_{name}",
                            f"{MODEL_DISPLAY[name]} — Rolling Accuracy (window={w})", "Accuracy")
        else:
            st.info("No resolved outcomes yet.")

    with tab3:
        if "latency_ms" in log_df.columns:
            _line_chart(log_df, "timestamp_parsed", "latency_ms", "Inference Latency", "ms")
        else:
            st.info("No latency data logged yet.")

    st.divider()



# 9. Data quality / system info

def render_system_info():
    st.subheader("System Information")
    latest = get_latest_btc_data(live_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Live BTC Ticks", len(live_df))
        st.caption(f"Source: `{LIVE_PATH}` (CoinGecko via scraper.py)")
    with c2:
        st.metric("Latest Tick Timestamp", latest["timestamp_parsed"].strftime("%H:%M:%S UTC") if latest is not None else "—")
        st.metric("Feature Count", len(FEATURE_ORDER))
    with c3:
        if run_log:
            st.metric("Ticks Processed (last run)", run_log.get("total_ticks_processed", "—"))
        else:
            st.metric("Ticks Processed (last run)", "—")

    st.markdown("##### Current Model Versions")
    if model_paths:
        for name, path in model_paths.items():
            st.caption(f"**{MODEL_DISPLAY[name]}**: `{os.path.basename(path)}`")
    else:
        st.caption("No saved model artifacts found in `models/`.")
    if model_errors:
        for name, err in model_errors.items():
            st.caption(f"⚠ {MODEL_DISPLAY.get(name, name)}: {err}")

    st.markdown("##### Last Pipeline Run")
    if run_log:
        rc1, rc2 = st.columns(2)
        rc1.caption(f"Start: {run_log.get('start_time', '—')}")
        rc2.caption(f"End: {run_log.get('end_time', '—')}")
    else:
        st.caption("No `run_log.json` found yet — run `python src/run_pipeline.py`.")


# Render — each section isolated so one failure doesn't take down the page

sections = [
    render_header, render_kpis, render_price_chart, render_prediction_panel,
    render_model_performance, render_monitoring, render_recent_predictions,
    render_history_charts, render_system_info,
]
for section in sections:
    try:
        section()
    except Exception as e:  # keep the rest of the dashboard alive
        st.error(f"Error rendering '{section.__name__}': {e}")

if not PLOTLY_AVAILABLE:
    st.caption("Tip: `pip install plotly` and add `plotly>=5.20.0` to requirements.txt "
               "for interactive charts with zoom/hover — native charts are used as a fallback.")

# Auto-refresh — plain-Streamlit polling, no extra dependency required
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()
