
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests, math
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
)
from sklearn.linear_model import RidgeCV, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="VIKASIT NIVESH", layout="wide")
css = """<style>
body { font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; background: #f7fafc; color: #0f172a;}
.card{padding:14px;border-radius:12px;background:#fff;border:1px solid rgba(2,6,23,0.04);box-shadow:0 6px 18px rgba(2,6,23,0.03);margin-bottom:10px;}
.pred{padding:10px;border-radius:8px;background:linear-gradient(180deg,#fff,#fbfcff);margin-bottom:8px;}
.small{color:#6b7280;font-size:13px;}
.badge-green{background:rgba(16,185,129,0.12);color:#059669;padding:6px 10px;border-radius:999px;font-weight:700;}
.badge-amber{background:rgba(245,158,11,0.10);color:#b45309;padding:6px 10px;border-radius:999px;font-weight:700;}
.badge-red{background:rgba(239,68,68,0.12);color:#b91c1c;padding:6px 10px;border-radius:999px;font-weight:700;}
.qbox{padding:8px;border-radius:8px;background:#f0fdf4;border:1px solid rgba(16,185,129,0.08);margin-top:6px;}
</style>"""
st.markdown(css, unsafe_allow_html=True)

# App title
st.title("VIKASIT NIVESH")
st.markdown('<div class="small">Beginner-friendly Indian stock prediction — not financial advice.</div>', unsafe_allow_html=True)

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker (no suffix)", value="")
    fmp_key = st.text_input("FMP API key (optional)", type="password")
    horizons = st.multiselect("Horizon(s)", ["3 days","15 days","1 month","3 months","6 months","1 year"], default=["15 days"])
    run = st.button("Run NIVESH")
    st.markdown("---")
    st.caption("Models: RandomForest + HistGradientBoosting + ExtraTrees + Stacking + BayesianRidge. Inputs only in sidebar.")

@st.cache_data(ttl=120)
def fetch_history(ticker, period='5y'):
    t = ticker.strip().upper()
    candidates = [t, t + '.NS', t + '.BO']
    last_exc = None
    for s in candidates:
        try:
            tk = yf.Ticker(s)
            hist = tk.history(period=period, interval='1d', auto_adjust=False)
            if hist is not None and not hist.empty:
                hist.index = pd.to_datetime(hist.index)
                return hist, s, None
        except Exception as e:
            last_exc = e
            continue
    return pd.DataFrame(), None, str(last_exc)

@st.cache_data(ttl=3600)
def fetch_profile_fmp(ticker, apikey):
    if not apikey:
        return {}
    trials = [ticker.strip().upper(), ticker.strip().upper()+'.NS', ticker.strip().upper()+'.BO']
    for s in trials:
        try:
            url = f"https://financialmodelingprep.com/api/v3/profile/{s}?apikey={apikey}"
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                j = r.json()
                if isinstance(j, list) and j:
                    return j[0]
                if isinstance(j, dict) and j:
                    return j
        except Exception:
            continue
    return {}

@st.cache_data(ttl=3600)
def fetch_yf_info(ticker):
    t = ticker.strip().upper()
    for s in [t, t+'.NS', t+'.BO']:
        try:
            tk = yf.Ticker(s)
            info = getattr(tk, 'info', {}) or {}
            if info:
                return info, s
        except Exception:
            continue
    return {}, None

def compute_indicators(df):
    df = df.copy()
    close = df['Close']
    high = df['High'] if 'High' in df.columns else close
    low = df['Low'] if 'Low' in df.columns else close
    vol = df['Volume'] if 'Volume' in df.columns else pd.Series(0, index=df.index)
    df['ret1'] = close.pct_change()
    df['ma7'] = close.rolling(7).mean()
    df['ma21'] = close.rolling(21).mean()
    df['ema12'] = close.ewm(span=12, adjust=False).mean()
    df['ema26'] = close.ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    df['rsi14'] = 100.0 - (100.0 / (1.0 + (up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9))))
    df['bb_mid'] = close.rolling(20).mean()
    df['bb_up'] = df['bb_mid'] + 2 * close.rolling(20).std()
    df['bb_low'] = df['bb_mid'] - 2 * close.rolling(20).std()
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    low_k = low.rolling(14).min()
    high_k = high.rolling(14).max()
    df['stoch_k'] = 100 * ((close - low_k) / (high_k - low_k + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['willr'] = -100 * ((high_k - close) / (high_k - low_k + 1e-9))
    df['obv'] = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    df['vol30'] = df['ret1'].rolling(30).std().fillna(0)
    df['skew'] = df['ret1'].rolling(30).skew().fillna(0)
    df['kurt'] = df['ret1'].rolling(30).kurt().fillna(0)
    # extra features
    df['log_ret'] = np.log1p(df['ret1'].fillna(0))
    df['range'] = (high - low) / (close + 1e-9)
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    return df

def make_features_targets(df, horizon):
    df = df.copy()
    df['target'] = df['Close'].shift(-horizon) / df['Close'] - 1.0
    df['lag_1'] = df['ret1'].shift(1)
    df['lag_2'] = df['ret1'].shift(2)
    df['lag_3'] = df['ret1'].shift(3)
    df['rolling_mean_7'] = df['ret1'].rolling(7).mean().shift(1)
    df['rolling_std_14'] = df['ret1'].rolling(14).std().shift(1)
    df['ma_diff'] = (df['ma7'] - df['ma21']) / (df['ma21'] + 1e-9)
    features = ['lag_1','lag_2','lag_3','rolling_mean_7','rolling_std_14','ma_diff','macd','rsi14','bb_up','bb_low','atr14','obv','vol30','skew','kurt','log_ret','range']
    df = df.dropna()
    if df.empty:
        return None, None, None
    X = df[features]; y = df['target']
    return X, y, df

def train_and_predict_ensemble(df, horizon, current_price):
    X, y, full = make_features_targets(df, horizon)
    if X is None or len(X) < 120:
        try:
            recent_return = df['Close'].pct_change(horizon).iloc[-1]
            return float(recent_return), {"method":"momentum_fallback"}
        except Exception:
            return 0.0, {"method":"momentum_fallback_error"}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model_results = {}
    preds = []
    maes = []

    # RandomForest
    try:
        rf = RandomForestRegressor(n_estimators=140, random_state=42, max_depth=10, n_jobs=1)
        rf.fit(X_train_s, y_train)
        p_test = rf.predict(X_test_s)
        mae_rf = mean_absolute_error(y_test, p_test)
        p_last = rf.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["RandomForest"] = {"pred": float(p_last), "mae": float(mae_rf)}
        preds.append(float(p_last)); maes.append(float(mae_rf))
    except Exception as e:
        model_results["RandomForest"] = {"error": str(e)}

    # HistGradientBoosting
    try:
        hgb = HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06)
        hgb.fit(X_train_s, y_train)
        p_test = hgb.predict(X_test_s)
        mae_h = mean_absolute_error(y_test, p_test)
        p_last = hgb.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["HistGradientBoosting"] = {"pred": float(p_last), "mae": float(mae_h)}
        preds.append(float(p_last)); maes.append(float(mae_h))
    except Exception as e:
        model_results["HistGradientBoosting"] = {"error": str(e)}

    # ExtraTrees
    try:
        et = ExtraTreesRegressor(n_estimators=140, random_state=42, max_depth=12, n_jobs=1)
        et.fit(X_train_s, y_train)
        p_test = et.predict(X_test_s)
        mae_et = mean_absolute_error(y_test, p_test)
        p_last = et.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["ExtraTrees"] = {"pred": float(p_last), "mae": float(mae_et)}
        preds.append(float(p_last)); maes.append(float(mae_et))
    except Exception as e:
        model_results["ExtraTrees"] = {"error": str(e)}

    # RidgeCV
    try:
        rc = RidgeCV(alphas=[0.1,1.0,10.0], cv=3)
        rc.fit(X_train_s, y_train)
        p_test = rc.predict(X_test_s)
        mae_rc = mean_absolute_error(y_test, p_test)
        p_last = rc.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["RidgeCV"] = {"pred": float(p_last), "mae": float(mae_rc)}
        preds.append(float(p_last)); maes.append(float(mae_rc))
    except Exception as e:
        model_results["RidgeCV"] = {"error": str(e)}

    # BayesianRidge - robust linear model
    try:
        br = BayesianRidge()
        br.fit(X_train_s, y_train)
        p_test = br.predict(X_test_s)
        mae_br = mean_absolute_error(y_test, p_test)
        p_last = br.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["BayesianRidge"] = {"pred": float(p_last), "mae": float(mae_br)}
        preds.append(float(p_last)); maes.append(float(mae_br))
    except Exception as e:
        model_results["BayesianRidge"] = {"error": str(e)}

    # Stacking (meta-learner)
    try:
        estimators = [
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=100, max_depth=12, random_state=42)),
            ('hgb', HistGradientBoostingRegressor(max_iter=120, learning_rate=0.06))
        ]
        stack = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=3, n_jobs=1, passthrough=False)
        stack.fit(X_train_s, y_train)
        p_test = stack.predict(X_test_s)
        mae_stack = mean_absolute_error(y_test, p_test)
        p_last = stack.predict(scaler.transform(X.iloc[[-1]]))[0]
        model_results["Stacking"] = {"pred": float(p_last), "mae": float(mae_stack)}
        preds.append(float(p_last)); maes.append(float(mae_stack))
    except Exception as e:
        model_results["Stacking"] = {"error": str(e)}

    # RecentMean baseline
    try:
        lookback = min(len(df)-1, max(10, horizon))
        recent_mean_return = df['Close'].pct_change().dropna().tail(lookback).mean()
        baseline_pred_return = float(recent_mean_return * horizon)
        model_results["RecentMean"] = {"pred": baseline_pred_return, "mae": None}
        preds.append(float(baseline_pred_return))
    except Exception as e:
        model_results["RecentMean"] = {"error": str(e)}

    if not preds:
        recent_return = df['Close'].pct_change(horizon).iloc[-1]
        return float(recent_return), {"method":"fallback_all_failed", "details": model_results}

    # Clean predictions
    cleaned_preds = []
    for p in preds:
        try:
            if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
                continue
            if abs(p) > 5:
                continue
            cleaned_preds.append(float(p))
        except Exception:
            continue
    if not cleaned_preds:
        recent_return = df['Close'].pct_change(horizon).iloc[-1]
        return float(recent_return), {"method":"no_valid_model_preds", "details": model_results}

    # Weighted aggregation by 1/mae
    weight_map = {}
    total_w = 0.0
    for k,v in model_results.items():
        if isinstance(v,dict) and 'mae' in v and v['mae'] and v['mae']>0:
            w = 1.0/(v['mae']+1e-9)
            weight_map[k]=w; total_w+=w
    final_return = None
    if total_w>0 and len(weight_map)>=1:
        s=0.0; sw=0.0
        for k,w in weight_map.items():
            p = model_results.get(k,{}).get('pred', None)
            if p is None: continue
            s += p*w; sw += w
        if sw>0: final_return = s/sw
    if final_return is None:
        final_return = float(np.median(cleaned_preds))

    # Clip extreme returns
    final_return = float(np.clip(final_return, -0.30, 2.0))

    # Bootstrap / quantiles for uncertainty (10th, 90th)
    try:
        arr = np.array(cleaned_preds)
        lower_q = float(np.percentile(arr, 10))
        upper_q = float(np.percentile(arr, 90))
    except Exception:
        lower_q = None; upper_q = None

    # Permutation importance on stacking model (best effort)
    feat_importance = None
    try:
        if 'Stacking' in model_results and 'stack' in locals():
            res = permutation_importance(stack, X_test_s, y_test, n_repeats=6, random_state=42, n_jobs=1)
            importances = list(zip(X.columns, res.importances_mean))
            importances_sorted = sorted(importances, key=lambda x: -abs(x[1]))[:8]
            feat_importance = importances_sorted
    except Exception:
        feat_importance = None

    diagnostics = {"models": model_results, "pred_list": cleaned_preds, "final_return_raw": final_return, "aggregate_mae": float(np.mean([m for m in maes])) if maes else None, "quantiles": (lower_q, upper_q), "feature_importance": feat_importance}
    return final_return, diagnostics

def fundamentals_score(profile, yf_info=None):
    if not profile and not yf_info:
        return 50, {}
    def safe(x):
        try:
            return float(x)
        except Exception:
            return None
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
        x=float(x)
        if x>=great: return 100
        if x<=good: return int(50 + 50*(x-good)/max(1e-9,(great-good)))
        return int(50 + 50*(x-good)/max(1e-9,(great-good)))
    def score_inverse(x, bad=2.0, good=0.5):
        if x is None: return None
        x=float(x)
        if x<=good: return 100
        if x>=bad: return 0
        return int(100*(bad-x)/max(1e-9,(bad-good)))
    scores={}
    scores["revenue_growth"] = score_positive(rev_growth, 0.03, 0.15)
    scores["roe"] = score_positive(roe, 0.06, 0.18)
    scores["debt_equity"] = score_inverse(debt, 2.0, 0.5)
    if pe is None:
        scores["pe"] = None
    else:
        if pe<=0: scores["pe"]=50
        else: scores["pe"] = int(max(0,min(100,int(100*(1-(pe/30.0))))))
    weights = {"revenue_growth":0.25,"roe":0.35,"debt_equity":0.25,"pe":0.15}
    total=0.0; wsum=0.0
    for k,w in weights.items():
        v = scores.get(k)
        if v is None: continue
        total += v*w; wsum += w
    if wsum==0:
        return 50, {k:(scores.get(k) if scores.get(k) is not None else 'NA') for k in scores}
    final = int(round(total/wsum))
    final = max(0, min(100, final))
    return final, {k:(scores.get(k) if scores.get(k) is not None else 'NA') for k in scores}

def compute_confidence(diag, recent_volatility):
    try:
        if not diag or not isinstance(diag, dict):
            return 40
        preds = diag.get("pred_list") or []
        if not preds:
            return 40
        arr = np.array(preds)
        mean_abs = max(1e-9, np.mean(np.abs(arr)))
        cv = np.std(arr) / mean_abs if mean_abs > 0 else np.inf
        base = 72.0
        if cv < 0.03:
            base += 14
        elif cv < 0.07:
            base += 8
        elif cv < 0.15:
            base += 2
        elif cv < 0.25:
            base -= 6
        else:
            base -= 16
        maes = []
        for v in (diag.get("models") or {}).values():
            if isinstance(v, dict) and "mae" in v and v["mae"] is not None:
                try:
                    maes.append(float(v["mae"]))
                except:
                    pass
        if maes:
            avg_mae = float(np.mean(maes))
            pen = min(18.0, avg_mae * 45.0)
            base -= pen
        if recent_volatility is not None and recent_volatility > 0:
            vol_pen = min(15.0, recent_volatility * 150.0)
            base -= vol_pen * 0.5
        floor = 35
        return int(max(floor, min(95, round(base))))
    except Exception:
        return 40

# sentiment lexicon already defined earlier; reuse it
def sentiment_score(text):
    if not text or not isinstance(text, str): return 50, "Neutral"
    txt = text.lower()
    tokens = txt.split()
    pos = sum(1 for t in tokens if t in _POS)
    neg = sum(1 for t in tokens if t in _NEG)
    score = 50
    if pos+neg > 0:
        score = int(50 + 40 * (pos - neg) / (pos + neg))
    label = "Neutral"
    if score >= 65: label = "Positive"
    elif score <= 35: label = "Negative"
    return score, label

horizon_map = {"3 days":3, "15 days":15, "1 month":22, "3 months":66, "6 months":132, "1 year":260}

if run:
    if not ticker:
        st.error("Enter ticker in the sidebar")
    else:
        with st.spinner("Running NIVESH..."):
            hist, used_symbol, err = fetch_history(ticker, period="5y")
            if hist.empty:
                st.error(f"Failed to fetch historical data for '{ticker}'. Try a different ticker or check network.")
                st.stop()
            hist = compute_indicators(hist)
            current_price = float(hist['Close'].iloc[-1])
            outputs = {}
            diagnostics = {"price_source": used_symbol or "yfinance", "model_info": {}}
            for h in horizons:
                days = horizon_map.get(h, 22)
                pred_ret, diag = train_and_predict_ensemble(hist, days, current_price)
                pred_price = current_price * (1 + pred_ret)
                outputs[h] = {"pred_price": float(round(pred_price,4)), "pred_return": float(pred_ret)}
                diagnostics["model_info"][h] = diag
            profile = fetch_profile_fmp(ticker, fmp_key)
            yf_info, yf_src = fetch_yf_info(ticker)
            desc = None; desc_src = None
            if isinstance(profile, dict):
                desc = profile.get("description") or profile.get("longBusinessSummary") or None
                if desc: desc_src = "FMP"
            if not desc and isinstance(yf_info, dict):
                desc = yf_info.get("longBusinessSummary") or yf_info.get("shortBusinessSummary") or None
                if desc: desc_src = f"yfinance:{yf_src}"
            fund_score, fund_parts = fundamentals_score(profile, yf_info)
            recent_vol = float(hist['vol30'].iloc[-1]) if 'vol30' in hist.columns else None
            mom_score = int(50 + 50 * np.tanh(hist['Close'].pct_change(7).iloc[-1] * 10))
            sent_score, sent_label = sentiment_score(desc or "")
        # Output UI
        st.markdown("<div class='card'><h2>NIVESH — Predictions</h2><div class='small'>Main outputs only — inputs are in the sidebar</div></div>", unsafe_allow_html=True)
        left, right = st.columns([2,3])
        with left:
            st.markdown(f"<div class='card'><div class='small'>Current price</div><div style='font-weight:800;font-size:22px'>₹{current_price:.4f}</div></div>", unsafe_allow_html=True)
            st.markdown("<br/>", unsafe_allow_html=True)
            for h in horizons:
                diag = diagnostics['model_info'].get(h, {})
                conf = compute_confidence(diag, recent_vol)
                predp = outputs[h]['pred_price']
                implied = (predp / current_price - 1) * 100
                arrow = "▲" if implied > 0 else ("▼" if implied < 0 else "—")
                color = "green" if implied > 0 else ("red" if implied < 0 else "gray")
                badge_class = "badge-green" if conf >= 80 else ("badge-amber" if conf >= 60 else "badge-red")
                # quantiles
                q_low, q_high = (None, None)
                if diag and isinstance(diag, dict):
                    q_low, q_high = diag.get("quantiles", (None, None))
                qtext = ""
                if q_low is not None and q_high is not None:
                    low_p = current_price * (1 + q_low); high_p = current_price * (1 + q_high)
                    qtext = f"<div class='qbox'>Expected range (10%-90%): ₹{low_p:.2f} — ₹{high_p:.2f}</div>"
                st.markdown(f"<div class='pred'><div style='display:flex;justify-content:space-between;align-items:center;'><div><strong>{h}</strong><div style='font-size:18px;font-weight:700;margin-top:6px;'>₹{predp:.4f}</div><div class='small'>Predicted price</div></div><div style='text-align:right;'><div class='{badge_class}'>Confidence {conf}%</div></div></div><div style='margin-top:6px;color:{color};font-weight:700'>{arrow} {implied:.2f}%</div>{qtext}</div>", unsafe_allow_html=True)
        with right:
            st.markdown(f"<div class='card'><div style='display:flex;justify-content:space-between;align-items:center;'><div><strong>Fundamentals score</strong><div style='font-size:20px;font-weight:700;margin-top:6px'>{fund_score}/100</div><div class='small'>ROE, Rev growth, D/E, PE</div></div><div style='text-align:right;'><div class='small'>Momentum: {mom_score}/100</div></div></div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card' style='margin-top:12px;'><strong>Sentiment</strong><div style='font-size:16px;font-weight:700;margin-top:6px'>{sent_label} ({sent_score}/100)</div><div class='small'>Source: {desc_src or 'N/A'}</div></div>", unsafe_allow_html=True)
            st.markdown("<div class='card' style='margin-top:12px;'><strong>Company description</strong><div class='small' style='margin-top:6px'>Fetched from APIs (trimmed)</div>", unsafe_allow_html=True)
            if desc:
                # display full description without trimming
                try:
                    st.write(desc)
                except Exception:
                    # fallback to shorter write if very large
                    st.write(str(desc)[:4000])
            else:
                st.write("Not available via API.")
            st.markdown("</div>", unsafe_allow_html=True)
            # Recommendation block - concise but descriptive
            st.markdown("<div class='card' style='margin-top:12px;'><strong>Recommendation</strong>", unsafe_allow_html=True)
            recs = []
            for h in horizons:
                predp = outputs[h]['pred_price']; pred_ret = outputs[h]['pred_return']
                implied = (predp / current_price -1)*100
                diag = diagnostics['model_info'].get(h, {}); conf = compute_confidence(diag, recent_vol)
                try:
                    atr = float(hist['atr14'].iloc[-1]) if 'atr14' in hist.columns else 0.0
                except Exception:
                    atr = 0.0
                vol_buffer = min(0.15, max(0.01, atr / max(1e-3, current_price)))
                buy_around = current_price * (1 - vol_buffer)
                stop_loss = buy_around * (1 - vol_buffer*0.8)
                target = predp
                # concise beginner-friendly reasons
                reasons = []
                if fund_score >= 60:
                    reasons.append("Fundamentals strong.")
                else:
                    reasons.append("Fundamentals modest.")
                if sent_label == "Positive":
                    reasons.append("Positive sentiment.")
                elif sent_label == "Negative":
                    reasons.append("Negative sentiment.")
                if conf >= 70:
                    reasons.append("Models agree — higher reliability.")
                else:
                    reasons.append("Lower model confidence.")
                # verdict
                verdict = "Hold"
                if implied > 5 and fund_score > 55 and conf >= 55:
                    verdict = "Buy"
                if implied < -3 and conf < 50:
                    verdict = "Avoid"
                # shorter recommendation paragraph (concise but clear)
                para = f"Horizon {h}: {verdict}. Target ₹{target:.2f} ({implied:.2f}%).\nBuy-around: ₹{buy_around:.2f} — Stop-loss: ₹{stop_loss:.2f}.\nReasons: {' '.join(reasons)}\nConfidence: {conf}%."
                recs.append(para)
            # overall summary
            agg_conf = int(np.mean([compute_confidence(diagnostics['model_info'].get(h, {}), recent_vol) for h in horizons])) if horizons else 50
            agg_pred = np.mean([outputs[h]['pred_price'] for h in horizons]) if horizons else current_price
            agg_verdict = "Hold"
            if agg_pred > current_price * 1.05 and fund_score > 55 and agg_conf >= 60:
                agg_verdict = "Buy"
            if agg_pred < current_price * 0.98 and agg_conf < 50:
                agg_verdict = "Avoid"
            st.write(f"**Overall:** {agg_verdict} — Average target across horizons ₹{agg_pred:.2f}. Overall confidence {agg_conf}%.")
            for r in recs:
                st.write(r)

            # Advanced diagnostics in an expander
            with st.expander("Model diagnostics (advanced)"):
                st.write("Price source:", diagnostics.get("price_source"))
                for h in horizons:
                    diag = diagnostics["model_info"].get(h, {})
                    st.write(f"--- {h} ---")
                    st.write("Models summary:")
                    models = diag.get("models", {})
                    for k,v in models.items():
                        if isinstance(v, dict) and "pred" in v:
                            st.write(f"{k}: pred={v.get('pred'):.6f}, mae={v.get('mae')}")
                        else:
                            st.write(f"{k}: {v}")
                    q = diag.get("quantiles")
                    if q and q[0] is not None:
                        low_p = current_price * (1 + q[0]); high_p = current_price * (1 + q[1])
                        st.write(f"10-90% range: ₹{low_p:.2f} — ₹{high_p:.2f}")
                    fi = diag.get("feature_importance")
                    if fi:
                        st.write("Top features (permutation importance):")
                        for fname,imp in fi:
                            st.write(f"{fname}: {imp:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Enter ticker and optional FMP API key in the sidebar and click Run NIVESH.")
