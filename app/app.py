# app.py - VIKASIT NIVESH
# Ready to paste into your Streamlit app folder.
# Features:
# - UI title "VIKASIT NIVESH"
# - Full company description display (no trimming)
# - Robust sentiment scorer that handles non-string inputs
# - TimeSeriesSplit OOF stacking, Ridge meta-learner, conformal residual intervals
# - Decision recommendations concise for beginners
# - Prediction logging skeleton

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
import csv
import os
import re
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

# -------------------------
# Page and app title
# -------------------------
APP_NAME = "VIKASIT NIVESH"
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.markdown("<div style='color:gray'>Advanced models under the hood — beginner-friendly outputs. Inputs are in the sidebar only.</div>", unsafe_allow_html=True)

# -------------------------
# Sidebar Inputs (only here)
# -------------------------
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (no suffix)", value="")
    fmp_key = st.text_input("FMP API key (optional)", type="password")
    horizons = st.multiselect("Horizon(s)", ["3 days","15 days","1 month","3 months","6 months","1 year"], default=["3 days"])
    run = st.button("Run " + APP_NAME)
    st.markdown("---")
    st.write("Notes:")
    st.write("- App will try ticker, ticker.NS, ticker.BO")
    st.write("- Provide FMP API key to enable better news & profile fetching (optional)")

# -------------------------
# Lexicons for sentiment
# -------------------------
_POS = set("""
good great positive outperform beat strong upgrade profits growth excellent gain upward bull benefit rally recovery improve robust
""".split())

_NEG = set("""
bad poor negative underperform loss downgrade weak decline fall downward bear risk scandal fraud halt concern problem loss
""".split())

# -------------------------
# Utility: Robust sentiment scorer
# -------------------------
def sentiment_score(text):
    """
    Robust sentiment scorer: accepts any input, converts to text, tokenizes safely,
    and returns (score:int 0-100, label:str). Falls back to neutral (50, "Neutral") on any error.
    """
    try:
        if text is None:
            return 50, "Neutral"
        # Convert to string (handles numpy.nan, numbers, dicts, etc.)
        txt = str(text)
        txt = txt.lower()
        tokens = re.findall(r"\b\w+\b", txt)
        if not tokens:
            return 50, "Neutral"
        pos = sum(1 for t in tokens if t in _POS)
        neg = sum(1 for t in tokens if t in _NEG)
        if (pos + neg) == 0:
            return 50, "Neutral"
        score = int(50 + 40 * (pos - neg) / (pos + neg))
        if score >= 65:
            label = "Positive"
        elif score <= 35:
            label = "Negative"
        else:
            label = "Neutral"
        score = max(0, min(100, score))
        return score, label
    except Exception:
        return 50, "Neutral"

# -------------------------
# Data fetch helpers (cached)
# -------------------------
@st.cache_data(ttl=600)
def fetch_history_try(ticker, period='5y'):
    t = ticker.strip().upper()
    candidates = [t, t + '.NS', t + '.BO']
    last_err = None
    for s in candidates:
        try:
            tk = yf.Ticker(s)
            hist = tk.history(period=period, interval='1d', auto_adjust=False)
            if hist is not None and not hist.empty:
                hist.index = pd.to_datetime(hist.index)
                return hist, s, None
        except Exception as e:
            last_err = str(e)
            continue
    return pd.DataFrame(), None, last_err

@st.cache_data(ttl=3600)
def fetch_profile_fmp(ticker, apikey):
    if not apikey:
        return {}
    trials = [ticker.strip().upper(), ticker.strip().upper() + ".NS"]
    for s in trials:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{s}?apikey={apikey}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and j:
                    return j[0]
                if isinstance(j, dict):
                    return j
        except Exception:
            continue
    return {}

@st.cache_data(ttl=3600)
def fetch_yf_info(ticker):
    t = ticker.strip().upper()
    for s in [t, t + ".NS", t + ".BO"]:
        try:
            tk = yf.Ticker(s)
            info = getattr(tk, "info", {}) or {}
            if info:
                return info, s
        except Exception:
            continue
    return {}, None

@st.cache_data(ttl=600)
def fetch_news_and_sentiment(ticker, apikey):
    trials = [ticker.strip().upper(), ticker.strip().upper() + ".NS"]
    headlines = []
    # Try FMP news if API key provided
    if apikey:
        for s in trials:
            try:
                url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={s}&limit=20&apikey={apikey}"
                r = requests.get(url, timeout=8)
                if r.status_code == 200:
                    j = r.json()
                    if isinstance(j, list):
                        for it in j:
                            t = (it.get("title") or "") + " " + (it.get("text") or "")
                            headlines.append(t)
            except Exception:
                continue
    # Fallback: yfinance news
    if not headlines:
        for s in trials:
            try:
                tk = yf.Ticker(s)
                ny = getattr(tk, "news", None)
                if ny:
                    for it in ny[:20]:
                        if isinstance(it, dict):
                            headlines.append((it.get("title", "") + " " + (it.get("publisher") or "")).strip())
                        else:
                            headlines.append(str(it))
            except Exception:
                continue
    text = " ".join([h for h in headlines if h])
    if text.strip():
        return sentiment_score(text), "news"
    # fallback: profile descriptions from FMP then yfinance
    prof = fetch_profile_fmp(ticker, apikey)
    if prof and isinstance(prof, dict):
        desc = prof.get("description") or prof.get("longBusinessSummary") or ""
        if desc:
            return sentiment_score(desc), "FMP_desc"
    yfinfo, used = fetch_yf_info(ticker)
    if yfinfo and isinstance(yfinfo, dict):
        desc = yfinfo.get("longBusinessSummary") or yfinfo.get("shortBusinessSummary") or ""
        if desc:
            return sentiment_score(desc), f"yfinance:{used}"
    return (50, "Neutral"), "none"

# -------------------------
# Indicators
# -------------------------
def compute_indicators(df):
    df = df.copy()
    close = df["Close"]
    high = df["High"] if "High" in df.columns else close
    low = df["Low"] if "Low" in df.columns else close
    df["ret1"] = close.pct_change()
    df["ma7"] = close.rolling(7).mean()
    df["ma21"] = close.rolling(21).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()
    df["ema26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema12"] - df["ema26"]
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df["rsi14"] = 100.0 - (100.0 / (1.0 + (up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9))))
    df["bb_mid"] = close.rolling(20).mean()
    df["bb_up"] = df["bb_mid"] + 2 * close.rolling(20).std()
    df["bb_low"] = df["bb_mid"] - 2 * close.rolling(20).std()
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    low_k = low.rolling(14).min()
    high_k = high.rolling(14).max()
    df["stoch_k"] = 100 * ((close - low_k) / (high_k - low_k + 1e-9))
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["vol30"] = df["ret1"].rolling(30).std().fillna(0)
    df["log_ret"] = np.log1p(df["ret1"].fillna(0))
    df["range"] = (high - low) / (close + 1e-9)
    df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
    return df

# -------------------------
# Features & targets
# -------------------------
def make_features_targets(df, horizon_days):
    df = df.copy()
    days = horizon_days
    df["target_raw"] = df["Close"].shift(-days) / df["Close"] - 1.0
    df["vol"] = df["ret1"].rolling(20).std().fillna(0)
    df["target_scaled"] = df["target_raw"] / (df["vol"] + 1e-9)
    df["lag_1"] = df["ret1"].shift(1)
    df["lag_2"] = df["ret1"].shift(2)
    df["lag_3"] = df["ret1"].shift(3)
    df["rolling_mean_7"] = df["ret1"].rolling(7).mean().shift(1)
    df["rolling_std_14"] = df["ret1"].rolling(14).std().shift(1)
    df["ma_diff"] = (df["ma7"] - df["ma21"]) / (df["ma21"] + 1e-9)
    features = [
        "lag_1","lag_2","lag_3","rolling_mean_7","rolling_std_14",
        "ma_diff","macd","rsi14","bb_up","bb_low","atr14","vol30","log_ret","range"
    ]
    df = df.dropna()
    if df.empty:
        return None, None, None
    X = df[features]
    y_raw = df["target_raw"]
    y_scaled = df["target_scaled"]
    return X, y_raw, y_scaled, df

# -------------------------
# Training & prediction (stacking + conformal)
# -------------------------
def train_and_predict_full(df, horizon_days):
    res = make_features_targets(df, horizon_days)
    if res is None:
        return 0.0, {"error": "not_enough_data"}
    X, y_raw, y_scaled, full = res
    if len(X) < 120:
        # fallback momentum
        try:
            recent_return = df["Close"].pct_change(horizon_days).iloc[-1]
            return float(recent_return), {"method":"momentum_fallback"}
        except Exception:
            return 0.0, {"error":"insufficient_data"}
    tscv = TimeSeriesSplit(n_splits=4)
    base_models = [
        ("rf", RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=1)),
        ("et", ExtraTreesRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=1)),
        ("hgb", HistGradientBoostingRegressor(max_iter=160))
    ]
    # OOF stacking
    oof_preds = np.zeros((len(X), len(base_models)))
    maes_oof = {name: [] for name,_ in base_models}
    for m_idx, (name, model) in enumerate(base_models):
        col = np.zeros(len(X))
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y_raw.iloc[train_idx]
            try:
                model.fit(X_tr, y_tr)
                col[val_idx] = model.predict(X_val)
                maes_oof[name].append(mean_absolute_error(y_raw.iloc[val_idx], col[val_idx]))
            except Exception:
                # try scaled target fallback
                try:
                    model.fit(X_tr, y_scaled.iloc[train_idx])
                    col[val_idx] = model.predict(X_val)
                except Exception:
                    col[val_idx] = np.nan
        oof_preds[:, m_idx] = col
    valid_rows = ~np.any(np.isnan(oof_preds), axis=1)
    if valid_rows.sum() < 50:
        meta = RidgeCV(alphas=[0.1,1.0,10.0])
        # use nan_to_num to avoid failures, but this is an imperfect fallback
        meta.fit(np.nan_to_num(oof_preds), y_raw)
    else:
        meta = RidgeCV(alphas=[0.1,1.0,10.0])
        meta.fit(oof_preds[valid_rows], y_raw.iloc[valid_rows])
    # Fit base models on full X and collect preds
    fitted = []
    model_info = {}
    for name, model in base_models:
        try:
            model.fit(X, y_raw)
            fitted.append((name, model))
            p = float(model.predict(X.iloc[[-1]])[0])
            model_info[name] = {"pred": p, "oof_mae": float(np.mean(maes_oof.get(name, [np.nan])))}
        except Exception as e:
            model_info[name] = {"error": str(e)}
            fitted.append((name, None))
    # Make predictions for last row
    last_X = X.iloc[[-1]]
    preds = []
    for name, model in fitted:
        if model is None:
            preds.append(np.nan)
            continue
        try:
            preds.append(float(model.predict(last_X)[0]))
        except Exception:
            preds.append(np.nan)
    preds_arr = np.array([p for p in preds if not (p is None or np.isnan(p))])
    if preds_arr.size == 0:
        return float(df["Close"].pct_change(horizon_days).iloc[-1]), {"method":"all_models_failed"}
    # meta pred
    try:
        meta_input = np.array([p if not (p is None or np.isnan(p)) else np.nan for p in preds]).reshape(1, -1)
        meta_input = np.nan_to_num(meta_input, nan=np.nanmedian(meta_input))
        meta_pred = float(meta.predict(meta_input)[0])
    except Exception:
        meta_pred = float(np.nanmedian(preds_arr))
    # weighted average by inverse OOF MAE
    maes = []
    for name,_ in fitted:
        info = model_info.get(name, {})
        maes.append(info.get("oof_mae") if info.get("oof_mae") is not None else np.nan)
    maes = np.array([m if m is not None and not np.isnan(m) else np.nanmedian(maes) for m in maes])
    inv = 1.0 / (maes + 1e-9)
    weights = inv / inv.sum()
    weighted = float(np.nansum(np.array([w*p for w,p in zip(weights, preds) if not (p is None or np.isnan(p))])))
    final_pred = 0.6 * meta_pred + 0.4 * weighted
    final_pred = float(np.clip(final_pred, -0.6, 2.0))
    # conformal residuals using last 20%
    cal_size = max(10, int(0.2 * len(X)))
    try:
        X_train_full = X.iloc[:-cal_size]; y_train_full = y_raw.iloc[:-cal_size]
        X_cal = X.iloc[-cal_size:]; y_cal = y_raw.iloc[-cal_size:]
        fin = RidgeCV(alphas=[0.1,1.0,10.0])
        fin.fit(X_train_full, y_train_full)
        pred_cal = fin.predict(X_cal)
        residuals = np.abs(y_cal - pred_cal)
        q90 = float(np.quantile(residuals, 0.90))
        q10 = float(np.quantile(residuals, 0.10))
    except Exception:
        q90 = None; q10 = None
    # quantile attempt
    q_low = None; q_high = None
    try:
        hq_low = HistGradientBoostingRegressor(loss="quantile", quantile=0.10, max_iter=120)
        hq_high = HistGradientBoostingRegressor(loss="quantile", quantile=0.90, max_iter=120)
        hq_low.fit(X, y_raw); hq_high.fit(X, y_raw)
        q_low = float(hq_low.predict(last_X)[0]); q_high = float(hq_high.predict(last_X)[0])
    except Exception:
        q_low = None; q_high = None
    # final intervals in returns
    low_ret = None; high_ret = None
    if q90 is not None:
        low_ret = final_pred - q90; high_ret = final_pred + q90
    if q_low is not None and q_high is not None:
        if low_ret is None:
            low_ret = q_low; high_ret = q_high
        else:
            low_ret = min(low_ret, q_low); high_ret = max(high_ret, q_high)
    diag = {"models": model_info, "oof_sample": oof_preds[-5:].tolist(), "final_raw_return": final_pred, "quantiles": (low_ret, high_ret)}
    return final_pred, diag

# -------------------------
# Fundamentals scoring
# -------------------------
def fundamentals_score(profile, yf_info=None):
    def safe(x):
        try: return float(x)
        except: return None
    roe = None; debt = None; pe = None; rev_growth = None
    if isinstance(profile, dict):
        roe = safe(profile.get("returnOnEquity") or profile.get("roe") or profile.get("returnOnEquityTTM"))
        debt = safe(profile.get("debtToEquity") or profile.get("debtToEquityTTM"))
        pe = safe(profile.get("priceEarningsRatio") or profile.get("pe") or profile.get("trailingPE"))
        rev_growth = safe(profile.get("revenueGrowth") or profile.get("revenueGrowthTTM"))
    if (roe is None or debt is None or pe is None) and yf_info:
        try:
            roe = roe or safe(yf_info.get("returnOnEquity") or yf_info.get("roe"))
            debt = debt or safe(yf_info.get("debtToEquity") or yf_info.get("debtToEquityTTM"))
            pe = pe or safe(yf_info.get("trailingPE") or yf_info.get("forwardPE"))
            rev_growth = rev_growth or safe(yf_info.get("revenueGrowth"))
        except Exception:
            pass
    def score_positive(x, good=0.06, great=0.18):
        if x is None: return None
        x = float(x)
        if x >= great: return 100
        if x <= good: return int(50 + 50 * (x - good) / max(1e-9, (great - good)))
        return int(50 + 50 * (x - good) / max(1e-9, (great - good)))
    def score_inverse(x, bad=2.0, good=0.5):
        if x is None: return None
        x = float(x)
        if x <= good: return 100
        if x >= bad: return 0
        return int(100 * (bad - x) / max(1e-9, (bad - good)))
    scores = {}
    scores["revenue_growth"] = score_positive(rev_growth, 0.03, 0.15)
    scores["roe"] = score_positive(roe, 0.06, 0.18)
    scores["debt_equity"] = score_inverse(debt, 2.0, 0.5)
    if pe is None:
        scores["pe"] = None
    else:
        if pe <= 0:
            scores["pe"] = 50
        else:
            scores["pe"] = int(max(0, min(100, int(100 * (1 - (pe / 30.0))))))
    weights = {"revenue_growth":0.25, "roe":0.35, "debt_equity":0.25, "pe":0.15}
    total = 0.0; wsum = 0.0
    for k, w in weights.items():
        v = scores.get(k)
        if v is None: continue
        total += v * w; wsum += w
    if wsum == 0:
        return 50, {k:(scores.get(k) if scores.get(k) is not None else "NA") for k in scores}
    final = int(round(total / wsum))
    final = max(0, min(100, final))
    return final, {k:(scores.get(k) if scores.get(k) is not None else "NA") for k in scores}

# -------------------------
# Confidence calibration & logging
# -------------------------
LOG_CSV = "predictions_log.csv"

def log_prediction_row(row):
    header = ["timestamp","ticker","horizon","current_price","pred_price","pred_return","confidence","fund_score","sentiment","source"]
    write_header = not os.path.exists(LOG_CSV)
    try:
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
            writer.writerow([row.get(h) for h in header])
    except Exception:
        # logging failure shouldn't crash app
        pass

def calibrate_confidence(raw_score):
    # Simple piecewise remapping to be user-friendly
    if raw_score <= 30: return int(20 + raw_score * 0.5)
    if raw_score <= 60: return int(40 + (raw_score - 30) * 1.0)
    return int(70 + (raw_score - 60) * 0.66)

def compute_confidence(diag, recent_volatility):
    try:
        if not diag or not isinstance(diag, dict):
            return 40
        preds = []
        for v in (diag.get("models") or {}).values():
            if isinstance(v, dict) and "pred" in v:
                try:
                    preds.append(float(v["pred"]))
                except Exception:
                    continue
        if not preds:
            return 40
        arr = np.array(preds)
        mean_abs = max(1e-9, np.mean(np.abs(arr)))
        cv = np.std(arr) / mean_abs if mean_abs > 0 else np.inf
        base = 60.0
        if cv < 0.03: base += 12
        elif cv < 0.07: base += 6
        elif cv < 0.15: base += 2
        elif cv < 0.25: base -= 6
        else: base -= 12
        maes = []
        for v in (diag.get("models") or {}).values():
            if isinstance(v, dict) and "oof_mae" in v and v["oof_mae"] is not None:
                try:
                    maes.append(float(v["oof_mae"]))
                except Exception:
                    pass
        if maes:
            avg_mae = float(np.mean(maes))
            pen = min(18.0, avg_mae * 45.0)
            base -= pen
        if recent_volatility is not None and recent_volatility > 0:
            vol_pen = min(15.0, recent_volatility * 150.0)
            base -= vol_pen * 0.5
        floor = 20
        raw = int(max(floor, min(95, round(base))))
        return calibrate_confidence(raw)
    except Exception:
        return 40

# -------------------------
# Decision rule helper
# -------------------------
def decision_from_outputs(current_price, pred_price, confidence, fund_score, q_low_ret, q_high_ret, vol_buffer):
    implied = (pred_price / current_price - 1) * 100
    verdict = "Hold"
    # robust PI check: ensure q_low_ret is a numeric value
    try:
        if q_low_ret is not None and isinstance(q_low_ret, (int, float)):
            if (q_low_ret * current_price) > current_price and confidence >= 50 and fund_score >= 55:
                verdict = "Buy"
    except Exception:
        pass
    if verdict != "Buy":
        if implied > 8 and confidence >= 50 and fund_score >= 50:
            verdict = "Buy"
        elif implied > 5 and confidence >= 45 and fund_score >= 55:
            verdict = "Buy"
        elif implied < -3 and confidence >= 45:
            verdict = "Avoid"
        elif confidence < 35:
            verdict = "Defer"
        else:
            verdict = "Hold"
    buy_around = current_price * (1 - vol_buffer)
    stop_loss = buy_around * (1 - 0.8 * vol_buffer)
    return verdict, buy_around, stop_loss, implied

# -------------------------
# Horizon map
# -------------------------
horizon_map = {"3 days":3, "15 days":15, "1 month":22, "3 months":66, "6 months":132, "1 year":260}

# -------------------------
# Main run
# -------------------------
if run:
    if not ticker:
        st.error("Enter ticker in the sidebar and click Run.")
        st.stop()
    with st.spinner("Running VIKASIT NIVESH..."):
        hist, used_symbol, err = fetch_history_try(ticker, period="5y")
        if hist.empty:
            st.error(f"Failed to fetch historical data for '{ticker}'. Error: {err}")
            st.stop()
        hist = compute_indicators(hist)
        current_price = float(hist["Close"].iloc[-1])
        outputs = {}
        diagnostics = {"price_source": used_symbol or "yfinance", "model_info": {}}
        for h in horizons:
            days = horizon_map.get(h, 22)
            pred_ret, diag = train_and_predict_full(hist, days)
            pred_price = current_price * (1 + pred_ret)
            outputs[h] = {"pred_price": float(round(pred_price, 4)), "pred_return": float(pred_ret)}
            diagnostics["model_info"][h] = diag
        # profile & sentiment
        profile = fetch_profile_fmp(ticker, fmp_key)
        yf_info, yf_src = fetch_yf_info(ticker)
        sent_res, sent_src = fetch_news_and_sentiment(ticker, fmp_key)
        if isinstance(sent_res, tuple):
            sent_score, sent_label = sent_res
        else:
            sent_score, sent_label = (sent_res if isinstance(sent_res, tuple) else (50, "Neutral"))
        fund_score, fund_parts = fundamentals_score(profile, yf_info)
        recent_vol = float(hist["vol30"].iloc[-1]) if "vol30" in hist.columns else None
        momentum = int(50 + 50 * np.tanh(hist["Close"].pct_change(7).iloc[-1] * 10))
        # UI outputs
        st.markdown("<div style='padding:10px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04)'><h2>Predictions</h2><div style='color:gray'>Outputs only — inputs in sidebar</div></div>", unsafe_allow_html=True)
        left, right = st.columns([2,3])
        with left:
            st.markdown("<div style='padding:12px;border-radius:8px;background:linear-gradient(180deg,#fff,#fbfcff);'><div style='color:gray'>Current price</div><div style='font-weight:800;font-size:22px'>₹{:.4f}</div></div>".format(current_price), unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            for h in horizons:
                diag = diagnostics["model_info"].get(h, {})
                conf = compute_confidence(diag, recent_vol)
                predp = outputs[h]["pred_price"]
                implied = (predp / current_price - 1) * 100
                arrow = "▲" if implied > 0 else ("▼" if implied < 0 else "—")
                color = "green" if implied > 0 else ("red" if implied < 0 else "gray")
                if conf >= 70:
                    badge_style = "background:rgba(16,185,129,0.12);color:#059669;padding:6px 10px;border-radius:999px;font-weight:700;"
                elif conf >= 50:
                    badge_style = "background:rgba(245,158,11,0.10);color:#b45309;padding:6px 10px;border-radius:999px;font-weight:700;"
                else:
                    badge_style = "background:rgba(239,68,68,0.12);color:#b91c1c;padding:6px 10px;border-radius:999px;font-weight:700;"
                q_low, q_high = (None, None)
                if diag and isinstance(diag, dict):
                    q_low, q_high = diag.get("quantiles", (None, None))
                qtext = ""
                if q_low is not None and q_high is not None:
                    low_p = current_price * (1 + q_low)
                    high_p = current_price * (1 + q_high)
                    qtext = "<div style='margin-top:6px;padding:8px;border-radius:6px;background:#f8fafc;border:1px solid rgba(2,6,23,0.03);'>Expected range (approx): ₹{:.2f} — ₹{:.2f}</div>".format(low_p, high_p)
                html = ("<div style='padding:10px;border-radius:8px;margin-bottom:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);'>"
                        "<div style='display:flex;justify-content:space-between;align-items:center;'>"
                        "<div><strong>{}</strong><div style='font-size:18px;font-weight:700;margin-top:6px;'>₹{:.4f}</div><div style='color:gray'>Predicted price</div></div>"
                        "<div style='text-align:right;'><div style='{}'>Confidence {}%</div></div></div>"
                        "<div style='margin-top:6px;color:{};font-weight:700'>{} {:.2f}%</div>{}</div>").format(h, predp, badge_style, conf, color, arrow, implied, qtext)
                st.markdown(html, unsafe_allow_html=True)
        with right:
            st.markdown(("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);display:flex;justify-content:space-between;align-items:center;'>"
                         "<div><strong>Fundamentals score</strong><div style='font-size:20px;font-weight:700;margin-top:6px'>{}/100</div><div style='color:gray'>ROE, Rev growth, D/E, PE</div></div>"
                         "<div style='text-align:right;'><div style='color:gray'>Momentum: {}/100</div></div></div>").format(fund_score, momentum), unsafe_allow_html=True)
            st.markdown(("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'>"
                         "<strong>Sentiment</strong><div style='font-size:16px;font-weight:700;margin-top:6px'>{} ({}/100)</div><div style='color:gray'>Source: {}</div></div>").format(sent_label, sent_score, sent_src or "N/A"), unsafe_allow_html=True)

            # Company description - full display (no trimming)
            st.markdown("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'><strong>Company description</strong><div style='color:gray;margin-top:6px'>Fetched from APIs (full text)</div>", unsafe_allow_html=True)
            desc = None
            if isinstance(profile, dict):
                desc = profile.get("description") or profile.get("longBusinessSummary") or None
            if not desc and isinstance(yf_info, dict):
                desc = yf_info.get("longBusinessSummary") or yf_info.get("shortBusinessSummary") or None
            if desc:
                # display full description
                try:
                    st.write(desc)
                except Exception:
                    # fallback if extremely large: show first 4000 characters
                    st.write(str(desc)[:4000])
            else:
                st.write("Not available via API.")
            st.markdown("</div>", unsafe_allow_html=True)

            # Recommendation block
            st.markdown("<div style='padding:12px;border-radius:8px;background:#fff;border:1px solid rgba(2,6,23,0.04);margin-top:12px;'><strong>Recommendation</strong>", unsafe_allow_html=True)
            recs = []
            for h in horizons:
                predp = outputs[h]["pred_price"]
                pred_ret = outputs[h]["pred_return"]
                diag = diagnostics["model_info"].get(h, {})
                conf = compute_confidence(diag, recent_vol)
                try:
                    atr = float(hist["atr14"].iloc[-1]) if "atr14" in hist.columns else 0.0
                except Exception:
                    atr = 0.0
                vol_buffer = min(0.15, max(0.01, atr / max(1e-3, current_price)))
                buy_around = current_price * (1 - vol_buffer)
                stop_loss = buy_around * (1 - vol_buffer * 0.8)
                q_low, q_high = diag.get("quantiles", (None, None)) if isinstance(diag, dict) else (None, None)
                verdict, b_a, s_l, implied = decision_from_outputs(current_price, predp, conf, fund_score, q_low, q_high, vol_buffer)
                reasons = []
                if fund_score >= 60:
                    reasons.append("Fundamentals strong.")
                else:
                    reasons.append("Fundamentals modest.")
                if sent_label == "Positive":
                    reasons.append("Positive sentiment.")
                elif sent_label == "Negative":
                    reasons.append("Negative sentiment.")
                if conf >= 65:
                    reasons.append("Models agree.")
                else:
                    reasons.append("Lower model confidence.")
                # concise descriptive paragraphs (beginner friendly)
                para = ("Horizon {h}: {verdict}. Target ₹{target:.2f} ({implied:.2f}%).\n"
                        "Buy-around: ₹{b:.2f} — Stop-loss: ₹{s:.2f}.\n"
                        "Reason: {reason} Confidence: {c}%.").format(
                    h=h, verdict=verdict, target=predp, implied=implied, b=b_a, s=s_l,
                    reason=" ".join(reasons), c=conf
                )
                recs.append(para)
                # logging
                try:
                    log_row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                               "ticker": ticker.upper(),
                               "horizon": h,
                               "current_price": current_price,
                               "pred_price": predp,
                               "pred_return": pred_ret,
                               "confidence": conf,
                               "fund_score": fund_score,
                               "sentiment": sent_score,
                               "source": diagnostics.get("price_source")}
                    log_prediction_row(log_row)
                except Exception:
                    pass
            # Aggregate verdict
            agg_conf = int(np.mean([compute_confidence(diagnostics["model_info"].get(h, {}), recent_vol) for h in horizons])) if horizons else 50
            agg_pred = np.mean([outputs[h]["pred_price"] for h in horizons]) if horizons else current_price
            agg_verdict = "Hold"
            if agg_pred > current_price * 1.05 and fund_score > 55 and agg_conf >= 55:
                agg_verdict = "Buy"
            if agg_pred < current_price * 0.98 and agg_conf < 50:
                agg_verdict = "Avoid"
            st.write("**Overall:** {} — Average target across horizons ₹{:.2f}. Overall confidence {}%.".format(agg_verdict, agg_pred, agg_conf))
            for r in recs:
                st.write(r)
            with st.expander("Model diagnostics (advanced)"):
                st.write("Price source:", diagnostics.get("price_source"))
                for h in horizons:
                    diag = diagnostics["model_info"].get(h, {})
                    st.write("--- {} ---".format(h))
                    st.write("Models summary:")
                    models = diag.get("models", {})
                    for k, v in models.items():
                        st.write(f"{k}: {v}")
                    q = diag.get("quantiles")
                    if q and q[0] is not None:
                        low_p = current_price * (1 + q[0]); high_p = current_price * (1 + q[1])
                        st.write("Approx range: ₹{:.2f} — ₹{:.2f}".format(low_p, high_p))
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Enter ticker and optional FMP API key in the sidebar and click Run.")

# End of app.py
