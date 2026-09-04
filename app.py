import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone

# =============================================================================
# APP
# =============================================================================
st.set_page_config(page_title="CoinDCX Pattern Learning Scanner", page_icon="🧠", layout="wide")
st.title("🧠 CoinDCX Historical Pattern Learning Scanner")
st.caption("Automatically fetches CoinDCX Futures history, learns from similar market behavior across multiple historical pools, and explains what happened next.")

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"

MEME_WORDS = {
    "DOGE","SHIB","PEPE","BONK","FLOKI","WIF","BOME","MEME","BRETT","MOG",
    "TURBO","MEW","NEIRO","BABYDOGE","1000SHIB","1000PEPE","1000BONK","1000FLOKI",
    "1000LUNC","PONKE","MYRO","SLERF","LADYS","DEGEN","MOTHER","MAGA","TRUMP"
}

# =============================================================================
# COINDCX DATA
# =============================================================================
@st.cache_data(ttl=30, show_spinner=False)
def active_instruments(margin="USDT"):
    r = requests.get(
        f"{API}/exchange/v1/derivatives/futures/data/active_instruments",
        params=[("margin_currency_short_name[]", margin)], timeout=20
    )
    r.raise_for_status()
    x = r.json()
    if not isinstance(x, list):
        raise RuntimeError(f"Unexpected instruments response: {x}")
    return x

@st.cache_data(ttl=5, show_spinner=False)
def futures_prices():
    r = requests.get(f"{PUBLIC}/market_data/v3/current_prices/futures/rt", timeout=20)
    r.raise_for_status()
    x = r.json()
    return x.get("prices", {}) if isinstance(x, dict) else {}

@st.cache_data(ttl=60, show_spinner=False)
def candles(pair, resolution, start_ts, end_ts):
    params = {"pair": pair, "from": int(start_ts), "to": int(end_ts), "resolution": resolution, "pcode": "f"}
    r = requests.get(f"{PUBLIC}/market_data/candlesticks", params=params, timeout=30)
    r.raise_for_status()
    x = r.json()
    rows = x.get("data", []) if isinstance(x, dict) else x
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected candle response for {pair}: {x}")
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in d:
            raise RuntimeError(f"{pair} candle response missing {c}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d["time"], unit="ms", errors="coerce", utc=True)
    return d.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=60, show_spinner=False)
def get_tf(pair, tf, days):
    now = int(time.time())
    if tf == "1W":
        # CoinDCX daily data is aggregated to weekly so weekly availability does not
        # depend on a separate weekly API resolution.
        d = candles(pair, "1D", now - int(days * 86400), now)
        return resample_weekly(d)
    resolution = {"1m":"1", "5m":"5", "15m":"15", "1H":"60", "4H":"240", "1D":"1D"}[tf]
    return candles(pair, resolution, now - int(days * 86400), now)

# =============================================================================
# INDICATORS
# =============================================================================
def indicators(d):
    x = d.copy()
    if x.empty:
        return x
    for n in [20, 50, 100, 200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    e12 = x.close.ewm(span=12, adjust=False).mean()
    e26 = x.close.ewm(span=26, adjust=False).mean()
    x["macd"] = e12 - e26
    x["macd_signal"] = x.macd.ewm(span=9, adjust=False).mean()
    tr = pd.concat([x.high-x.low, (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["atr_pct"] = x.atr / x.close.replace(0, np.nan) * 100
    x["volma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume / x.volma.replace(0, np.nan)
    x["bbmid"] = x.close.rolling(20).mean()
    x["bbstd"] = x.close.rolling(20).std()
    x["bbup"] = x.bbmid + 2*x.bbstd
    x["bblow"] = x.bbmid - 2*x.bbstd
    up = x.high.diff()
    dn = -x.low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=x.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=x.index)
    atr = x.atr.replace(0, np.nan)
    x["pdi"] = 100 * plus.ewm(alpha=1/14, adjust=False).mean() / atr
    x["mdi"] = 100 * minus.ewm(alpha=1/14, adjust=False).mean() / atr
    dx = 100 * (x.pdi-x.mdi).abs() / (x.pdi+x.mdi).replace(0, np.nan)
    x["adx"] = dx.ewm(alpha=1/14, adjust=False).mean()
    return x

def resample_weekly(d):
    if d is None or d.empty:
        return pd.DataFrame()
    x = d.copy().set_index("time")
    out = x.resample("W-SUN", label="right", closed="right").agg({
        "open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"
    }).dropna().reset_index()
    return out

def completed(d):
    # Drop the latest candle because it may still be forming.
    if d is None or len(d) < 2:
        return d.copy() if isinstance(d, pd.DataFrame) else pd.DataFrame()
    return d.iloc[:-1].copy().reset_index(drop=True)

def structure(d):
    if d is None or len(d) < 40:
        return "Mixed"
    recent = d.tail(12)
    prior = d.iloc[-36:-12]
    if recent.high.max() > prior.high.max() and recent.low.min() > prior.low.min():
        return "Bullish"
    if recent.high.max() < prior.high.max() and recent.low.min() < prior.low.min():
        return "Bearish"
    return "Mixed"

# =============================================================================
# SIMPLE HELPERS
# =============================================================================
def fmt(v):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):,.8f}".rstrip("0").rstrip(".")
    except Exception:
        return "—"

def normalize(s):
    q = str(s).strip().upper().replace("/","").replace("-","").replace("_","")
    for quote in ("USDT","INR","USDC"):
        if q.endswith(quote) and len(q) > len(quote):
            return q[:-len(quote)]
    return q

def coin_matches(pair, symbol, requested, quote):
    req = normalize(requested)
    names = [str(pair).upper().replace("-","").replace("_",""), str(symbol).upper().replace("-","").replace("_","")]
    for n in names:
        variants = [n]
        if n.startswith("B"): variants.append(n[1:])
        if n.startswith("I"): variants.append(n[1:])
        for v in variants:
            if v == req or v == req+quote or v.startswith(req+quote): return True
    return False

def current_price(p):
    for key in ("ls","lp","last_price","price","mp","mark_price"):
        try:
            v = float(p.get(key,0) or 0)
            if v > 0: return v
        except Exception:
            pass
    return 0.0

def safe(v, default=np.nan):
    try:
        return float(v) if pd.notna(v) else default
    except Exception:
        return default

# =============================================================================
# MULTI-TIMEFRAME EMA ENGINE
# =============================================================================
TF_CONFIG = [("1m",2),("5m",3),("15m",5),("1H",14),("4H",60),("1D",260),("1W",1100)]

def ema_alignment(tf_data):
    rows = {}
    bullish = bearish = total = 0
    for tf, d in tf_data.items():
        x = indicators(completed(d))
        if x.empty:
            rows[tf] = {"state":"NO DATA","count":0}
            continue
        last = x.iloc[-1]
        checks = []
        for n in (20,50,100):
            val = safe(last.get(f"ema{n}"))
            close = safe(last.get("close"))
            checks.append(1 if pd.notna(val) and pd.notna(close) and close > val else -1 if pd.notna(val) and pd.notna(close) else 0)
        pos = sum(v == 1 for v in checks)
        neg = sum(v == -1 for v in checks)
        state = "BULLISH" if pos == 3 else "BEARISH" if neg == 3 else "MIXED"
        rows[tf] = {"state":state,"count":pos if state=="BULLISH" else neg if state=="BEARISH" else max(pos,neg)}
        if state == "BULLISH": bullish += 1
        elif state == "BEARISH": bearish += 1
        total += 1
    return rows, bullish, bearish, total

# =============================================================================
# HISTORICAL EVENT / PATTERN ENGINE
# =============================================================================

def rolling_features(x, i):
    """
    Build a normalized description of the market at candle i.

    The old engine concentrated heavily on absolute returns. This version also
    learns the *shape* of the setup: momentum acceleration, EMA extension,
    EMA stack, volume behavior, volatility, RSI, candle rejection and trend
    structure. That makes BTC, large caps and tiny meme coins more comparable.
    """
    if i < 50 or i >= len(x):
        return None

    row = x.iloc[i]
    close = safe(row.close)
    if not np.isfinite(close) or close <= 0:
        return None

    def ret(n):
        if i < n:
            return np.nan
        prev = safe(x.iloc[i-n].close)
        return (close / prev - 1) * 100 if prev > 0 else np.nan

    look24 = x.iloc[max(0, i-24):i+1]
    look12 = x.iloc[max(0, i-12):i+1]
    look6 = x.iloc[max(0, i-6):i+1]

    hi24 = safe(look24.high.max())
    lo24 = safe(look24.low.min())
    hi12 = safe(look12.high.max())
    lo12 = safe(look12.low.min())

    ema20 = safe(row.ema20)
    ema50 = safe(row.ema50)
    ema100 = safe(row.ema100)
    atr_pct = safe(row.atr_pct)

    range24 = ((hi24 - lo24) / close * 100) if hi24 > 0 else np.nan
    range12 = ((hi12 - lo12) / close * 100) if hi12 > 0 else np.nan

    body_pct = abs(safe(row.close) - safe(row.open)) / close * 100
    upper_wick_pct = max(
        0,
        (safe(row.high) - max(safe(row.open), safe(row.close))) / close * 100
    )
    lower_wick_pct = max(
        0,
        (min(safe(row.open), safe(row.close)) - safe(row.low)) / close * 100
    )

    stack = (
        1 if np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema100)
        and ema20 > ema50 > ema100
        else -1 if np.isfinite(ema20) and np.isfinite(ema50) and np.isfinite(ema100)
        and ema20 < ema50 < ema100
        else 0
    )

    ema20_dist = ((close / ema20) - 1) * 100 if ema20 > 0 else np.nan
    ema50_dist = ((close / ema50) - 1) * 100 if ema50 > 0 else np.nan
    ema100_dist = ((close / ema100) - 1) * 100 if ema100 > 0 else np.nan

    vol_ratio = safe(row.vol_ratio)
    prior_vol = safe(x.iloc[max(0, i-5):i].volume.mean()) if i > 5 else np.nan
    vol_accel = safe(row.volume) / prior_vol if prior_vol > 0 else np.nan

    # Momentum acceleration: is the move getting faster or slower?
    r4 = ret(4)
    r12 = ret(12)
    r24 = ret(24)
    acceleration = r4 - (r12 / 3 if np.isfinite(r12) else 0)

    # Location inside the recent range. Near 1 = pressing highs, near 0 = lows.
    range_position = (
        (close - lo24) / (hi24 - lo24)
        if hi24 > lo24 else 0.5
    )

    return {
        "ret4": r4,
        "ret12": r12,
        "ret24": r24,
        "range_pct": range24,
        "range12_pct": range12,
        "range_position": range_position,
        "body_pct": body_pct,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "rsi": safe(row.rsi),
        "adx": safe(row.adx),
        "vol_ratio": vol_ratio,
        "vol_accel": vol_accel,
        "atr_pct": atr_pct,
        "acceleration": acceleration,
        "ema_stack": stack,
        "ema20_dist": ema20_dist,
        "ema50_dist": ema50_dist,
        "ema100_dist": ema100_dist,
        "above20": 1 if close > ema20 else 0,
        "above50": 1 if close > ema50 else 0,
        "above100": 1 if close > ema100 else 0,
        "structure": structure(x.iloc[:i+1]),
    }


def feature_vector(f):
    if f is None:
        return None

    # Robust scales. These are deliberately broad so a setup does not need
    # to be numerically identical to qualify as historically similar.
    vals = [
        safe(f["ret4"]),
        safe(f["ret12"]),
        safe(f["ret24"]),
        safe(f["range_pct"]),
        safe(f["range12_pct"]),
        safe(f["range_position"]),
        safe(f["body_pct"]),
        safe(f["upper_wick_pct"]),
        safe(f["lower_wick_pct"]),
        safe(f["rsi"]),
        safe(f["adx"]),
        safe(f["vol_ratio"]),
        safe(f["vol_accel"]),
        safe(f["atr_pct"]),
        safe(f["acceleration"]),
        safe(f["ema_stack"]),
        safe(f["ema20_dist"]),
        safe(f["ema50_dist"]),
        safe(f["ema100_dist"]),
        safe(f["above20"]),
        safe(f["above50"]),
        safe(f["above100"]),
    ]

    return np.array(
        [0 if not np.isfinite(v) else v for v in vals],
        dtype=float
    )


def scaled_distance(a, b):
    # Feature-specific scales + weights.
    scales = np.array([
        12, 22, 40, 25, 20, 0.50,
        6, 5, 5, 20, 20, 3, 2.5, 10, 15,
        1, 12, 18, 25, 1, 1, 1
    ], dtype=float)

    weights = np.array([
        1.5, 1.4, 1.2, 0.8, 0.7, 0.6,
        0.5, 0.6, 0.6, 1.4, 0.8, 1.2, 0.9, 0.8, 1.2,
        1.1, 1.0, 0.8, 0.6, 0.8, 0.8, 0.8
    ], dtype=float)

    z = ((a - b) / scales) ** 2
    return float(np.sqrt(np.sum(z * weights) / np.sum(weights)))



def event_outcome(x, i, horizon, direction="UP"):
    if i + 1 >= len(x):
        return None

    future = x.iloc[i+1:min(len(x), i+1+horizon)]
    if future.empty:
        return None

    entry = safe(x.iloc[i].close)
    if entry <= 0:
        return None

    closes = future.close.astype(float)
    highs = future.high.astype(float)
    lows = future.low.astype(float)

    end_ret = (safe(closes.iloc[-1]) / entry - 1) * 100
    best = (safe(highs.max()) / entry - 1) * 100
    worst = (safe(lows.min()) / entry - 1) * 100

    if direction == "DOWN":
        if worst <= -10 and best < 12:
            label = "CONTINUED"
        elif best >= 20:
            label = "REVERSED / BOUNCED"
        else:
            label = "SIDEWAYS / PULLBACK"
    else:
        if best >= 10 and worst > -12:
            label = "CONTINUED"
        elif worst <= -20:
            label = "DUMPED"
        else:
            label = "SIDEWAYS / PULLBACK"

    # Path metrics. These tell us whether the coin continued first and reversed
    # later, which an endpoint-only model cannot see.
    if direction == "UP":
        mfe_idx = int(np.argmax(highs.values))
        mae_idx = int(np.argmin(lows.values))
        favorable = best
        adverse = worst
    else:
        # For a DOWN event, favorable movement is negative and adverse movement
        # is positive.
        favorable = -worst
        adverse = best
        mfe_idx = int(np.argmin(lows.values))
        mae_idx = int(np.argmax(highs.values))

    path = {
        "favorable": favorable,
        "adverse": adverse,
        "favorable_bar": mfe_idx + 1,
        "adverse_bar": mae_idx + 1,
        "end": end_ret,
    }

    return {
        "end": end_ret,
        "best": best,
        "worst": worst,
        "label": label,
        "path": path,
    }


def multi_horizon_outcomes(x, i, direction="UP"):
    """
    Measure the historical path at 4H, 8H, 12H and 24H.

    Because the learning timeframe is 4H, these are 1, 2, 3 and 6
    completed 4H candles after the historical setup.
    """
    result = {}

    for name, bars in [
        ("4H", 1),
        ("8H", 2),
        ("12H", 3),
        ("24H", 6),
    ]:
        result[name] = event_outcome(x, i, bars, direction)

    # Full 24H path is used for the primary historical classification.
    result["24H_path"] = result["24H"]

    if result["24H"]:
        p = result["24H"]["path"]

        # Detect the important "another leg then reversal" pattern.
        if direction == "UP":
            another_leg = p["favorable"] >= 10
            ended_lower = p["end"] <= 0
            significant_reversal = (
                p["favorable"] >= 15 and
                p["end"] <= p["favorable"] - 15
            )
        else:
            another_leg = p["favorable"] >= 10
            ended_lower = p["end"] >= 0
            significant_reversal = (
                p["favorable"] >= 15 and
                p["end"] >= -p["favorable"] + 15
            )

        if significant_reversal:
            result["path_type"] = "SECOND LEG THEN REVERSAL"
        elif another_leg and not ended_lower:
            result["path_type"] = "CONTINUATION"
        elif p["adverse"] >= 15:
            result["path_type"] = "EARLY REJECTION"
        else:
            result["path_type"] = "CHOP / MIXED"
    else:
        result["path_type"] = "UNKNOWN"

    return result


def find_pump_events(d, horizon=6, min_pump=15):
    """Find completed pump setups without using future candles to define them."""
    if d is None or len(d) < 100:
        return []

    x = indicators(completed(d))
    events = []
    last_event = -999

    for i in range(50, len(x)-horizon-1):
        f = rolling_features(x, i)
        if not f:
            continue

        # The event itself is already visible at candle i.
        if safe(f["ret24"]) >= min_pump:
            # Avoid collecting every candle of one long pump as independent examples.
            if i - last_event < 12:
                continue

            outcomes = multi_horizon_outcomes(x, i, "UP")
            if outcomes["24H_path"]:
                events.append({
                    "i": i,
                    "time": x.iloc[i].time,
                    "features": f,
                    "vector": feature_vector(f),
                    "outcome": outcomes,
                })
                last_event = i

    return events


def find_breakout_events(d, mode="ATH", horizon=6):
    if d is None or len(d) < 100:
        return []

    x = indicators(completed(d))
    events = []
    last = -999

    for i in range(50, len(x)-horizon-1):
        hist = x.iloc[:i]
        price = safe(x.iloc[i].close)
        if price <= 0:
            continue

        extreme = safe(hist.high.max() if mode == "ATH" else hist.low.min())
        if extreme <= 0:
            continue

        event = price > extreme if mode == "ATH" else price < extreme

        if event and i - last >= 8:
            f = rolling_features(x, i)
            outcomes = multi_horizon_outcomes(
                x, i, "UP" if mode == "ATH" else "DOWN"
            )
            if f and outcomes["24H_path"]:
                events.append({
                    "i": i,
                    "time": x.iloc[i].time,
                    "features": f,
                    "vector": feature_vector(f),
                    "outcome": outcomes,
                })
                last = i

    return events


def current_pattern(d):
    x = indicators(completed(d))
    if len(x) < 55:
        return None
    return rolling_features(x, len(x)-1)


def classify_current_event(d):
    x = completed(d)
    if len(x) < 40:
        return "NORMAL"

    ind = indicators(x)
    last = ind.iloc[-1]
    hist = ind.iloc[:-1]

    price = safe(last.close)
    prior_ath = safe(hist.high.max())
    prior_atl = safe(hist.low.min())

    if prior_ath > 0 and price > prior_ath:
        return "ATH BREAKOUT"
    if prior_atl > 0 and price < prior_atl:
        return "ATL BREAKDOWN"

    ret24 = (
        (price / safe(x.iloc[-24].close) - 1) * 100
        if len(x) > 24 else 0
    )

    if ret24 >= 15:
        return "HOT / PUMP"
    if ret24 <= -15:
        return "FAST DUMP"

    return "NORMAL"



def behavior_bucket(f):
    """Convert extreme numerical moves into comparable behavioral regimes."""
    if not f:
        return "UNKNOWN"

    r24 = safe(f.get("ret24"))
    r12 = safe(f.get("ret12"))
    rsi = safe(f.get("rsi"))
    ema = safe(f.get("ema20_dist"))
    vol = safe(f.get("vol_ratio"))

    if np.isfinite(r24):
        abs_move = abs(r24)
    else:
        abs_move = 0

    if abs_move >= 200:
        move_regime = "EXTREME_200"
    elif abs_move >= 100:
        move_regime = "EXTREME_100"
    elif abs_move >= 60:
        move_regime = "EXTREME_60"
    elif abs_move >= 30:
        move_regime = "STRONG_30"
    elif abs_move >= 15:
        move_regime = "STRONG_15"
    else:
        move_regime = "NORMAL"

    if np.isfinite(rsi):
        momentum_regime = (
            "OVERHEATED" if rsi >= 85
            else "HOT" if rsi >= 70
            else "WEAK" if rsi <= 35
            else "NORMAL"
        )
    else:
        momentum_regime = "UNKNOWN"

    if np.isfinite(ema):
        extension_regime = (
            "VERY_EXTENDED" if abs(ema) >= 50
            else "EXTENDED" if abs(ema) >= 25
            else "MODERATE"
        )
    else:
        extension_regime = "UNKNOWN"

    if np.isfinite(vol):
        volume_regime = (
            "SURGE" if vol >= 2.5
            else "ELEVATED" if vol >= 1.3
            else "NORMAL"
        )
    else:
        volume_regime = "UNKNOWN"

    direction = "UP" if np.isfinite(r24) and r24 >= 0 else "DOWN"

    return "|".join([
        direction,
        move_regime,
        momentum_regime,
        extension_regime,
        volume_regime,
    ])


def similarity_components(target_features, event_features):
    """Return interpretable similarity dimensions for the UI."""
    keys = [
        ("Momentum", "ret24", 80),
        ("Recent momentum", "ret4", 30),
        ("RSI", "rsi", 20),
        ("Volume", "vol_ratio", 3),
        ("EMA extension", "ema20_dist", 25),
        ("Volatility", "atr_pct", 15),
        ("Acceleration", "acceleration", 25),
    ]

    result = {}
    for name, key, scale in keys:
        a = safe(target_features.get(key))
        b = safe(event_features.get(key))
        if np.isfinite(a) and np.isfinite(b):
            diff = abs(a - b)

            # For very large price moves, absolute percentage difference is
            # less useful than regime similarity. Compress the return feature.
            if key in {"ret24", "ret4", "acceleration"}:
                aa = np.sign(a) * np.log1p(abs(a))
                bb = np.sign(b) * np.log1p(abs(b))
                diff = abs(aa - bb) * scale / np.log1p(scale)

            result[name] = max(0, 100 * (1 - diff / scale))
        else:
            result[name] = 0

    return result


def adaptive_similarity(target_features, event_features, event_type=None):
    """
    Behavioral similarity score.

    Exact numerical similarity is useful for normal markets. For extreme
    movers, regime/shape similarity gets more weight so a +180% historical
    move can still teach us about a +250% current move.
    """
    tv = feature_vector(target_features)
    ev = feature_vector(event_features)

    if tv is None or ev is None:
        return 0.0, {}

    raw_dist = scaled_distance(tv, ev)
    numerical = max(0, 100 * (1 - raw_dist))

    t_bucket = behavior_bucket(target_features)
    e_bucket = behavior_bucket(event_features)

    t_parts = t_bucket.split("|")
    e_parts = e_bucket.split("|")

    matches = sum(a == b for a, b in zip(t_parts, e_parts))
    regime_score = matches / max(1, len(t_parts)) * 100

    components = similarity_components(target_features, event_features)
    shape_score = float(np.mean(list(components.values()))) if components else 0

    extreme_target = any(
        tag in t_bucket for tag in ("EXTREME_60", "EXTREME_100", "EXTREME_200")
    )

    if extreme_target:
        # Behavioral regime dominates for extreme events.
        score = (
            0.35 * numerical +
            0.40 * regime_score +
            0.25 * shape_score
        )
    else:
        score = (
            0.55 * numerical +
            0.20 * regime_score +
            0.25 * shape_score
        )

    return float(max(0, min(100, score))), {
        "numerical": numerical,
        "regime": regime_score,
        "shape": shape_score,
        "target_bucket": t_bucket,
        "event_bucket": e_bucket,
    }


def similar_events(target_features, event_pool, max_matches=50, min_similarity=None):
    if target_features is None:
        return []

    target_bucket = behavior_bucket(target_features)
    extreme_target = any(
        tag in target_bucket for tag in ("EXTREME_60", "EXTREME_100", "EXTREME_200")
    )

    if min_similarity is None:
        # Adaptive threshold: extreme patterns get a wider neighborhood.
        min_similarity = 38 if extreme_target else 50

    scored = []

    for e in event_pool:
        ev_features = e.get("features")
        if not ev_features:
            continue

        similarity, meta = adaptive_similarity(
            target_features,
            ev_features,
            e.get("event_type")
        )

        if similarity >= min_similarity:
            e2 = dict(e)
            e2["similarity_components"] = similarity_components(
                target_features, ev_features
            )
            e2["similarity_meta"] = meta
            scored.append((similarity, e2))

    scored.sort(key=lambda z: z[0], reverse=True)
    return scored[:max_matches]


def _weighted_mean(values, weights):
    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(vals) & np.isfinite(w)
    if not mask.any():
        return np.nan

    return float(np.average(vals[mask], weights=w[mask]))


def _weighted_percent(values, weights):
    if not values:
        return 0.0

    vals = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(w)

    if not mask.any():
        return 0.0

    return float(np.average(vals[mask], weights=w[mask]))



def outcome_summary(matches):
    if not matches:
        return None

    weights = np.array([
        max(1.0, sim / 10)
        for sim, _ in matches
    ])

    usable = [
        (idx, sim, e)
        for idx, (sim, e) in enumerate(matches)
        if e.get("outcome") and e["outcome"].get("24H_path")
    ]

    if not usable:
        return None

    labels = [
        e["outcome"]["24H_path"]["label"]
        for _, _, e in usable
    ]

    total = len(labels)

    counts = {
        k: labels.count(k)
        for k in [
            "CONTINUED",
            "SIDEWAYS / PULLBACK",
            "DUMPED",
            "REVERSED / BOUNCED",
        ]
    }

    def horizon_stats(h):
        ends, bests, worsts, local_weights = [], [], [], []

        for idx, sim, e in usable:
            o = e["outcome"].get(h)
            if not o:
                continue

            ends.append(o["end"])
            bests.append(o["best"])
            worsts.append(o["worst"])
            local_weights.append(weights[idx])

        return {
            "end": _weighted_mean(ends, local_weights),
            "best": _weighted_mean(bests, local_weights),
            "worst": _weighted_mean(worsts, local_weights),
        }

    path_types = [
        e["outcome"].get("path_type", "UNKNOWN")
        for _, _, e in usable
    ]

    path_counts = {
        "SECOND LEG THEN REVERSAL": path_types.count("SECOND LEG THEN REVERSAL"),
        "CONTINUATION": path_types.count("CONTINUATION"),
        "EARLY REJECTION": path_types.count("EARLY REJECTION"),
        "CHOP / MIXED": path_types.count("CHOP / MIXED"),
    }

    # Estimate when the strongest adverse move occurred.
    reversal_bars = []
    for _, _, e in usable:
        p = e["outcome"]["24H_path"].get("path", {})
        favorable = safe(p.get("favorable"))
        adverse = safe(p.get("adverse"))

        if favorable >= 15 and adverse >= 15:
            reversal_bars.append(safe(p.get("adverse_bar")))

    reversal_timing = (
        float(np.mean(reversal_bars)) * 4
        if reversal_bars else np.nan
    )

    return {
        "total": total,
        "counts": counts,
        "continue_pct": counts["CONTINUED"] / total * 100,
        "dump_pct": counts["DUMPED"] / total * 100,
        "reverse_pct": counts["REVERSED / BOUNCED"] / total * 100,
        "side_pct": counts["SIDEWAYS / PULLBACK"] / total * 100,

        "path_counts": path_counts,
        "second_leg_pct": path_counts["SECOND LEG THEN REVERSAL"] / total * 100,
        "path_continuation_pct": path_counts["CONTINUATION"] / total * 100,
        "early_rejection_pct": path_counts["EARLY REJECTION"] / total * 100,
        "chop_pct": path_counts["CHOP / MIXED"] / total * 100,
        "reversal_timing_hours": reversal_timing,

        "4H": horizon_stats("4H"),
        "8H": horizon_stats("8H"),
        "12H": horizon_stats("12H"),
        "24H": horizon_stats("24H"),

        "avg_end": horizon_stats("24H")["end"],
        "avg_best": horizon_stats("24H")["best"],
        "avg_worst": horizon_stats("24H")["worst"],
        "median_similarity": float(np.median([sim for _, sim, _ in usable])),
        "max_similarity": float(max(sim for _, sim, _ in usable)),
        "min_similarity": float(min(sim for _, sim, _ in usable)),
    }


def historical_edge(summary):
    if not summary or summary["total"] < 8:
        return 0

    return float(
        summary["continue_pct"] -
        summary["dump_pct"] +
        0.20 * (
            summary["reverse_pct"] -
            summary["side_pct"]
        )
    )


def evidence_grade(summary):
    if not summary:
        return "INSUFFICIENT"

    n = summary["total"]
    median = summary["median_similarity"]

    if n >= 25 and median >= 60:
        return "STRONG"
    if n >= 15 and median >= 55:
        return "GOOD"
    if n >= 8 and median >= 50:
        return "MODERATE"
    return "LIMITED"



def risk_profile(summary, current):
    """
    Separate trend direction from exhaustion/reversal risk.

    This prevents a weak sample from being converted directly into a strong
    LONG/SHORT-style conclusion.
    """
    if not summary:
        return {
            "trend": "UNKNOWN",
            "exhaustion": "UNKNOWN",
            "immediate": "UNKNOWN",
            "reversal_24h": "UNKNOWN",
            "volatility": "UNKNOWN",
        }

    target = current.get("target") or {}
    extreme = event_is_extreme(target)
    rsi = safe(target.get("rsi"))
    ema = safe(target.get("ema20_dist"))
    accel = safe(target.get("acceleration"))

    if current.get("event") in {"ATH BREAKOUT", "HOT / PUMP"}:
        trend = (
            "BULLISH" if current.get("bull", 0) >= current.get("bear", 0)
            else "MIXED"
        )
    elif current.get("event") in {"ATL BREAKDOWN", "FAST DUMP"}:
        trend = (
            "BEARISH" if current.get("bear", 0) >= current.get("bull", 0)
            else "MIXED"
        )
    else:
        trend = "MIXED"

    exhaustion_score = 0

    if extreme:
        exhaustion_score += 2
    if np.isfinite(rsi) and rsi >= 85:
        exhaustion_score += 2
    elif np.isfinite(rsi) and rsi >= 75:
        exhaustion_score += 1

    if np.isfinite(ema) and abs(ema) >= 50:
        exhaustion_score += 2
    elif np.isfinite(ema) and abs(ema) >= 25:
        exhaustion_score += 1

    if np.isfinite(accel) and accel < -3 and trend == "BULLISH":
        exhaustion_score += 2

    exhaustion = (
        "VERY HIGH" if exhaustion_score >= 6
        else "HIGH" if exhaustion_score >= 4
        else "MODERATE" if exhaustion_score >= 2
        else "LOW"
    )

    # Immediate direction should remain unknown unless the short horizon has
    # a meaningful historical edge.
    h4 = summary["4H"]["end"]
    h12 = summary["12H"]["end"]
    h24 = summary["24H"]["end"]

    if np.isfinite(h4) and abs(h4) >= 5:
        immediate = "BULLISH" if h4 > 0 else "BEARISH"
    else:
        immediate = "UNKNOWN"

    if summary["second_leg_pct"] >= 35:
        reversal_24h = "HIGH"
    elif summary["dump_pct"] >= 55 or summary["reverse_pct"] >= 55:
        reversal_24h = "HIGH"
    elif abs(h24) >= 8:
        reversal_24h = "ELEVATED"
    else:
        reversal_24h = "UNKNOWN"

    worst = abs(summary["avg_worst"]) if np.isfinite(summary["avg_worst"]) else 0
    best = abs(summary["avg_best"]) if np.isfinite(summary["avg_best"]) else 0

    volatility = (
        "VERY HIGH" if max(best, worst) >= 30
        else "HIGH" if max(best, worst) >= 20
        else "MODERATE"
    )

    return {
        "trend": trend,
        "exhaustion": exhaustion,
        "immediate": immediate,
        "reversal_24h": reversal_24h,
        "volatility": volatility,
    }


def simple_path_conclusion(summary, current):
    if not summary or summary["total"] < 8:
        return (
            "There is not enough historical evidence to describe the likely path. "
            "The scanner will not force a directional prediction."
        )

    risk = risk_profile(summary, current)

    if summary["second_leg_pct"] >= 35:
        return (
            f"In {summary['second_leg_pct']:.0f}% of the comparable cases, the coin "
            "made another meaningful move in the original direction before a "
            "significant reversal. This means the danger may not be an immediate "
            "dump; the larger historical risk is a delayed reversal after another leg."
        )

    if risk["exhaustion"] in {"HIGH", "VERY HIGH"} and summary["dump_pct"] >= 45:
        return (
            "The trend can remain bullish while the setup becomes increasingly "
            "dangerous. Similar cases frequently experienced a large pullback, so "
            "the historical evidence supports caution rather than chasing the move."
        )

    if summary["continue_pct"] >= 55:
        return (
            "Similar setups more often continued than failed. The historical path "
            "still contained pullbacks, so continuation should not be interpreted "
            "as a guarantee."
        )

    return (
        "The historical paths are mixed. There is no strong enough directional "
        "edge to treat the setup as a reliable long or short signal."
    )


def human_result(summary, current):
    if not summary or summary["total"] < 8:
        return (
            "🟡 NOT ENOUGH EVIDENCE",
            "I found too few similar historical situations. "
            "The tool should not pretend it knows what happens next."
        )

    # A strong label requires both enough observations and reasonable match quality.
    # A nine-case sample with 39% median similarity is therefore not promoted to
    # HIGH DUMP RISK simply because most examples happened to fall.
    if (
        summary["total"] < 12
        or summary["median_similarity"] < 48
    ):
        return (
            "🟠 LIMITED HISTORICAL EDGE",
            f"I found {summary['total']} comparable cases, but the historical "
            f"match quality is limited (median similarity "
            f"{summary['median_similarity']:.0f}%). The results can still show "
            "risk and historical behavior, but they are not strong enough for a "
            "high-confidence directional call."
        )

    c = summary["continue_pct"]
    d = summary["dump_pct"]
    r = summary["reverse_pct"]
    side = summary["side_pct"]
    edge = historical_edge(summary)

    down_event = current["event"] in {"ATL BREAKDOWN", "FAST DUMP"}

    if down_event:
        if c >= 58 and c - r >= 18 and edge < -5:
            title = "🔴 LIKELY CONTINUATION DOWN"
            text = (
                f"I found {summary['total']} similar historical breakdowns. "
                f"About {c:.0f}% kept falling, while {r:.0f}% bounced strongly. "
                "Historically this type of downside setup has favored continuation."
            )
        elif r >= 55 and r - c >= 15:
            title = "🟢 HIGH BOUNCE RISK"
            text = (
                f"I found {summary['total']} similar historical breakdowns. "
                f"About {r:.0f}% bounced strongly versus {c:.0f}% that continued down. "
                "Historically this type of fall has often produced a reversal."
            )
        else:
            title = "🟡 MIXED / WAIT"
            text = (
                f"I found {summary['total']} similar breakdowns, but outcomes are mixed: "
                f"{c:.0f}% continued down, {r:.0f}% bounced and "
                f"{side:.0f}% were sideways/pullback cases."
            )
    else:
        if c >= 58 and c - d >= 18 and edge > 5:
            title = "🟢 LIKELY CONTINUATION"
            text = (
                f"I found {summary['total']} similar historical setups. "
                f"About {c:.0f}% continued and {d:.0f}% dumped. "
                "Historically the pattern favors continuation, although it is not guaranteed."
            )
        elif d >= 55 and d - c >= 15:
            title = "🔴 HIGH DUMP RISK"
            text = (
                f"I found {summary['total']} similar historical setups. "
                f"About {d:.0f}% dumped and {c:.0f}% continued. "
                "Historically this type of setup has often weakened after the initial move."
            )
        else:
            title = "🟡 MIXED / WAIT"
            text = (
                f"I found {summary['total']} similar historical setups, but the outcomes "
                f"are mixed: {c:.0f}% continued, {side:.0f}% pulled back/sideways and "
                f"{d:.0f}% dumped. There is not a strong historical edge."
            )

    return title, text


def confirmation_text(current):
    last4 = current["4h"]
    last1 = current["1d"]

    checks = []

    if safe(last4.close) > safe(last4.ema20):
        checks.append("4H price is above EMA20")
    else:
        checks.append("4H price is below EMA20")

    if safe(last4.macd) > safe(last4.macd_signal):
        checks.append("4H momentum is improving")
    else:
        checks.append("4H momentum is weak")

    if safe(last4.vol_ratio) >= 1:
        checks.append("4H volume is above its average")
    else:
        checks.append("4H volume is below its average")

    if safe(last1.close) > safe(last1.ema20):
        checks.append("1D price is above EMA20")
    else:
        checks.append("1D price is below EMA20")

    return checks


def setup_description(f):
    if not f:
        return []

    out = []

    r24 = safe(f.get("ret24"))
    r4 = safe(f.get("ret4"))
    rsi = safe(f.get("rsi"))
    vol = safe(f.get("vol_ratio"))
    ema = safe(f.get("ema20_dist"))
    accel = safe(f.get("acceleration"))

    bucket = behavior_bucket(f)

    if bucket != "UNKNOWN":
        out.append(f"behavior regime: {bucket.replace('|', ' • ')}")

    if np.isfinite(r24):
        out.append(f"24-bar momentum: {r24:+.1f}%")
    if np.isfinite(r4):
        out.append(f"recent 4-bar momentum: {r4:+.1f}%")
    if np.isfinite(rsi):
        out.append(f"RSI: {rsi:.1f}")
    if np.isfinite(vol):
        out.append(f"volume: {vol:.1f}x average")
    if np.isfinite(ema):
        out.append(f"price vs EMA20: {ema:+.1f}%")
    if np.isfinite(accel):
        out.append(
            "momentum accelerating" if accel > 3
            else "momentum decelerating" if accel < -3
            else "momentum stable"
        )

    return out


def market_cap_bucket(pair, symbol):
    """
    We cannot reliably infer market cap from the public futures feed.
    Use contract naming only as a coarse peer grouping and label it honestly.
    """
    s = f"{pair} {symbol}".upper()

    if any(x in s for x in ["1000", "10000"]):
        return "MULTIPLIER_STYLE"

    if any(w in s for w in MEME_WORDS):
        return "MEME"

    return "GENERAL"


def peer_group(pair, symbol):
    return market_cap_bucket(pair, symbol)


def event_is_extreme(f):
    if not f:
        return False

    r24 = safe(f.get("ret24"))
    r12 = safe(f.get("ret12"))
    rsi = safe(f.get("rsi"))
    ema = safe(f.get("ema20_dist"))

    return (
        (np.isfinite(r24) and abs(r24) >= 60)
        or (np.isfinite(r12) and abs(r12) >= 80)
        or (np.isfinite(rsi) and rsi >= 85)
        or (np.isfinite(ema) and abs(ema) >= 50)
    )


def event_profile_score(target_features, event):
    """
    Secondary score used to diversify the historical sample.

    A single coin with a long uninterrupted pump should not dominate the
    result. We therefore reward different contracts and different dates while
    still ranking primarily by pattern similarity.
    """
    if not event:
        return 0

    score = 0

    if event_is_extreme(target_features) == event_is_extreme(event.get("features")):
        score += 20

    if target_features.get("ema_stack") == event.get("features", {}).get("ema_stack"):
        score += 10

    if target_features.get("structure") == event.get("features", {}).get("structure"):
        score += 10

    return score


def diversified_matches(matches, max_matches=50, per_coin=4):
    """Limit repeated examples from one contract so the model learns broadly."""
    selected = []
    counts = {}

    for sim, e in matches:
        coin = e.get("pair", "UNKNOWN")
        if counts.get(coin, 0) >= per_coin:
            continue

        selected.append((sim, e))
        counts[coin] = counts.get(coin, 0) + 1

        if len(selected) >= max_matches:
            break

    return selected

# =============================================================================
# CURRENT COIN PROFILE
# =============================================================================
def analyze_current_coin(pair, price_info):
    tf_data={}
    # Enough history for EMA100 and meaningful structure, without requesting huge 1m history.
    days={"1m":3,"5m":8,"15m":20,"1H":90,"4H":240,"1D":700,"1W":1100}
    for tf, d in days.items():
        tf_data[tf]=get_tf(pair,tf,d)
    ema_rows,bull,bear,total=ema_alignment(tf_data)
    d4=indicators(completed(tf_data["4H"]))
    d1=indicators(completed(tf_data["1D"]))
    d15=indicators(completed(tf_data["15m"]))
    if d4.empty or d1.empty:
        raise RuntimeError("Not enough 4H/1D history")
    current=current_price(price_info)
    last4=d4.iloc[-1]; last1=d1.iloc[-1]
    event=classify_current_event(tf_data["4H"])
    target=current_pattern(tf_data["4H"])
    return {
        "tf_data":tf_data,"ema_rows":ema_rows,"bull":bull,"bear":bear,"total":total,
        "current":current,"event":event,"target":target,
        "4h":last4,"1d":last1,"15m":d15.iloc[-1] if not d15.empty else None,
        "structure4":structure(d4),"structure1":structure(d1),"structure15":structure(d15) if not d15.empty else "Mixed"
    }

# =============================================================================
# MARKET-WIDE LEARNING POOL
# =============================================================================
def universe_rows(margin, meme_only, max_coins):
    active=active_instruments(margin); prices=futures_prices(); rows=[]
    for pair in active:
        p=prices.get(pair)
        if not p: continue
        symbol=str(p.get("mkt",pair)).upper()
        if meme_only and not any(w in symbol or w in pair.upper() for w in MEME_WORDS):
            continue
        pc=safe(p.get("pc",0),0); cur=current_price(p)
        if cur<=0: continue
        rows.append((pair,p,symbol,pc,cur))
    # Hot movers are the most useful first-pass learning universe.
    rows.sort(key=lambda z:abs(z[3]),reverse=True)
    return rows[:max_coins]

@st.cache_data(ttl=900, show_spinner=False)
def build_learning_pool(pairs_signature, margin, max_coins, event_mode):
    """
    Build a broader learning universe.

    The target event is still the primary filter, but historical examples are
    tagged by behavioral regime and peer group. This allows the matcher to
    learn from:
      1. normal event matches,
      2. extreme-mover behavior,
      3. meme/general peer behavior,
      4. different contracts/dates.

    We intentionally keep event construction free of future leakage.
    """
    pool = []
    failures = []

    for pair, symbol in pairs_signature:
        try:
            d = get_tf(pair, "4H", 240)
            if len(d) < 100:
                continue

            if event_mode == "PUMP":
                events = find_pump_events(d, horizon=6, min_pump=15)
                event_type = "PUMP"
            elif event_mode == "ATH":
                events = find_breakout_events(d, "ATH", horizon=6)
                event_type = "ATH"
            else:
                events = find_breakout_events(d, "ATL", horizon=6)
                event_type = "ATL"

            group = peer_group(pair, symbol)

            for e in events:
                e = dict(e)
                e["pair"] = pair
                e["symbol"] = symbol
                e["peer_group"] = group
                e["event_type"] = event_type
                e["behavior_bucket"] = behavior_bucket(e["features"])
                pool.append(e)

        except Exception as exc:
            failures.append(
                f"{pair}: {type(exc).__name__}: {exc}"
            )

    return pool, failures


def build_same_coin_pool(pair, event_mode):
    """Learn from the target contract's own historical behavior."""
    try:
        d = get_tf(pair, "4H", 700)

        if len(d) < 150:
            return []

        if event_mode == "PUMP":
            events = find_pump_events(d, horizon=6, min_pump=15)
        elif event_mode == "ATH":
            events = find_breakout_events(d, "ATH", horizon=6)
        else:
            events = find_breakout_events(d, "ATL", horizon=6)

        for e in events:
            e["pair"] = pair
            e["event_type"] = event_mode

        return events

    except Exception:
        return []


def build_extreme_pool(pool):
    """Extract the historical extreme-mover subset."""
    return [
        e for e in pool
        if event_is_extreme(e.get("features"))
    ]


def merge_learning_pools(*pools):
    """Deduplicate examples by contract + historical timestamp."""
    merged = []
    seen = set()

    for pool in pools:
        for e in pool:
            key = (
                e.get("pair"),
                str(e.get("time")),
                e.get("event_type"),
            )
            if key in seen:
                continue

            seen.add(key)
            merged.append(e)

    return merged

# =============================================================================
# SIMPLE PREDICTION LANGUAGE
# =============================================================================

def risk_profile(summary, current):
    """
    Separate trend direction from exhaustion/reversal risk.

    This prevents a weak sample from being converted directly into a strong
    LONG/SHORT-style conclusion.
    """
    if not summary:
        return {
            "trend": "UNKNOWN",
            "exhaustion": "UNKNOWN",
            "immediate": "UNKNOWN",
            "reversal_24h": "UNKNOWN",
            "volatility": "UNKNOWN",
        }

    target = current.get("target") or {}
    extreme = event_is_extreme(target)
    rsi = safe(target.get("rsi"))
    ema = safe(target.get("ema20_dist"))
    accel = safe(target.get("acceleration"))

    if current.get("event") in {"ATH BREAKOUT", "HOT / PUMP"}:
        trend = (
            "BULLISH" if current.get("bull", 0) >= current.get("bear", 0)
            else "MIXED"
        )
    elif current.get("event") in {"ATL BREAKDOWN", "FAST DUMP"}:
        trend = (
            "BEARISH" if current.get("bear", 0) >= current.get("bull", 0)
            else "MIXED"
        )
    else:
        trend = "MIXED"

    exhaustion_score = 0

    if extreme:
        exhaustion_score += 2
    if np.isfinite(rsi) and rsi >= 85:
        exhaustion_score += 2
    elif np.isfinite(rsi) and rsi >= 75:
        exhaustion_score += 1

    if np.isfinite(ema) and abs(ema) >= 50:
        exhaustion_score += 2
    elif np.isfinite(ema) and abs(ema) >= 25:
        exhaustion_score += 1

    if np.isfinite(accel) and accel < -3 and trend == "BULLISH":
        exhaustion_score += 2

    exhaustion = (
        "VERY HIGH" if exhaustion_score >= 6
        else "HIGH" if exhaustion_score >= 4
        else "MODERATE" if exhaustion_score >= 2
        else "LOW"
    )

    # Immediate direction should remain unknown unless the short horizon has
    # a meaningful historical edge.
    h4 = summary["4H"]["end"]
    h12 = summary["12H"]["end"]
    h24 = summary["24H"]["end"]

    if np.isfinite(h4) and abs(h4) >= 5:
        immediate = "BULLISH" if h4 > 0 else "BEARISH"
    else:
        immediate = "UNKNOWN"

    if summary["second_leg_pct"] >= 35:
        reversal_24h = "HIGH"
    elif summary["dump_pct"] >= 55 or summary["reverse_pct"] >= 55:
        reversal_24h = "HIGH"
    elif abs(h24) >= 8:
        reversal_24h = "ELEVATED"
    else:
        reversal_24h = "UNKNOWN"

    worst = abs(summary["avg_worst"]) if np.isfinite(summary["avg_worst"]) else 0
    best = abs(summary["avg_best"]) if np.isfinite(summary["avg_best"]) else 0

    volatility = (
        "VERY HIGH" if max(best, worst) >= 30
        else "HIGH" if max(best, worst) >= 20
        else "MODERATE"
    )

    return {
        "trend": trend,
        "exhaustion": exhaustion,
        "immediate": immediate,
        "reversal_24h": reversal_24h,
        "volatility": volatility,
    }


def simple_path_conclusion(summary, current):
    if not summary or summary["total"] < 8:
        return (
            "There is not enough historical evidence to describe the likely path. "
            "The scanner will not force a directional prediction."
        )

    risk = risk_profile(summary, current)

    if summary["second_leg_pct"] >= 35:
        return (
            f"In {summary['second_leg_pct']:.0f}% of the comparable cases, the coin "
            "made another meaningful move in the original direction before a "
            "significant reversal. This means the danger may not be an immediate "
            "dump; the larger historical risk is a delayed reversal after another leg."
        )

    if risk["exhaustion"] in {"HIGH", "VERY HIGH"} and summary["dump_pct"] >= 45:
        return (
            "The trend can remain bullish while the setup becomes increasingly "
            "dangerous. Similar cases frequently experienced a large pullback, so "
            "the historical evidence supports caution rather than chasing the move."
        )

    if summary["continue_pct"] >= 55:
        return (
            "Similar setups more often continued than failed. The historical path "
            "still contained pullbacks, so continuation should not be interpreted "
            "as a guarantee."
        )

    return (
        "The historical paths are mixed. There is no strong enough directional "
        "edge to treat the setup as a reliable long or short signal."
    )


def human_result(summary, current):
    if not summary or summary["total"]<8:
        return "🟡 NOT ENOUGH EVIDENCE", "I found too few similar historical situations. The tool should not pretend it knows what happens next."
    c=summary["continue_pct"]; d=summary["dump_pct"]; r=summary.get("reverse_pct",0); side=summary["side_pct"]
    down_event=current["event"] == "ATL BREAKDOWN" or current["event"] == "FAST DUMP"
    if down_event:
        if c>=60 and c-r>=20:
            title="🔴 LIKELY CONTINUATION DOWN"
            text=f"I found {summary['total']} similar breakdowns. About {c:.0f}% kept falling, while {r:.0f}% bounced strongly. Historically the downside move has usually continued."
        elif r>=60 and r-c>=20:
            title="🟢 HIGH BOUNCE RISK"
            text=f"I found {summary['total']} similar breakdowns. About {r:.0f}% bounced strongly, while {c:.0f}% kept falling. Historically this type of fall has often produced a reversal."
        else:
            title="🟡 MIXED / WAIT"
            text=f"I found {summary['total']} similar breakdowns, but the outcomes are mixed: {c:.0f}% continued down, {r:.0f}% bounced and {side:.0f}% were sideways/pullback cases."
    else:
        if c>=60 and c-d>=20:
            title="🟢 LIKELY CONTINUATION"
            text=f"I found {summary['total']} similar historical setups. About {c:.0f}% continued and {d:.0f}% dumped. That means the historical pattern favors continuation, although it is not guaranteed."
        elif d>=60 and d-c>=20:
            title="🔴 HIGH DUMP RISK"
            text=f"I found {summary['total']} similar historical setups. About {d:.0f}% dumped and {c:.0f}% continued. Historically this type of setup has usually weakened after the move."
        else:
            title="🟡 MIXED / WAIT"
            text=f"I found {summary['total']} similar historical setups, but the outcomes are mixed: {c:.0f}% continued, {side:.0f}% pulled back/sideways and {d:.0f}% dumped. There is not a strong historical edge."
    return title,text

def confirmation_text(current):
    last4=current["4h"]; last1=current["1d"]
    p=current["current"]
    checks=[]
    if safe(last4.close)>safe(last4.ema20): checks.append("4H price is above EMA20")
    else: checks.append("4H price is below EMA20")
    if safe(last4.macd)>safe(last4.macd_signal): checks.append("4H momentum is improving")
    else: checks.append("4H momentum is weak")
    if safe(last4.vol_ratio)>=1: checks.append("4H volume is above its average")
    else: checks.append("4H volume is below its average")
    if safe(last1.close)>safe(last1.ema20): checks.append("1D price is above EMA20")
    else: checks.append("1D price is below EMA20")
    return checks

# =============================================================================

# =============================================================================
# V5: CONTINUATION vs REVERSAL ENGINE
# =============================================================================
# v5 keeps the historical-learning engine, but adds a critical distinction:
# an extreme pump can remain bullish for a while before it actually reverses.
# We therefore combine historical path behavior with CURRENT short-term
# confirmation instead of treating overbought/extended as an immediate short.


def v5_event_outcome(x, i, horizon, direction="UP"):
    if i + 1 >= len(x):
        return None
    future = x.iloc[i+1:min(len(x), i+1+horizon)]
    if future.empty:
        return None
    entry = safe(x.iloc[i].close)
    if not np.isfinite(entry) or entry <= 0:
        return None

    closes = future.close.astype(float)
    highs = future.high.astype(float)
    lows = future.low.astype(float)
    end_ret = (safe(closes.iloc[-1]) / entry - 1) * 100
    best = (safe(highs.max()) / entry - 1) * 100
    worst = (safe(lows.min()) / entry - 1) * 100

    if direction == "DOWN":
        if worst <= -10 and best < 12:
            label = "CONTINUED"
        elif best >= 20:
            label = "REVERSED / BOUNCED"
        else:
            label = "SIDEWAYS / PULLBACK"
        favorable = -worst
        adverse = best
        favorable_bar = int(np.argmin(lows.values)) + 1
        adverse_bar = int(np.argmax(highs.values)) + 1
    else:
        if best >= 10 and worst > -15:
            label = "CONTINUED"
        elif worst <= -20:
            label = "DUMPED"
        else:
            label = "SIDEWAYS / PULLBACK"
        favorable = best
        adverse = abs(worst)
        favorable_bar = int(np.argmax(highs.values)) + 1
        adverse_bar = int(np.argmin(lows.values)) + 1

    # Track the sequence, not just the final candle.
    first_end = []
    for n in (1, min(2, horizon), min(3, horizon)):
        if n <= len(closes):
            first_end.append((n, (safe(closes.iloc[n-1]) / entry - 1) * 100))

    return {
        "end": end_ret,
        "best": best,
        "worst": worst,
        "label": label,
        "path": {
            "favorable": favorable,
            "adverse": adverse,
            "favorable_bar": favorable_bar,
            "adverse_bar": adverse_bar,
            "end": end_ret,
            "first_end": first_end,
        },
    }


def v5_multi_horizon_outcomes(x, i, direction="UP"):
    result = {}
    for name, bars in [("4H",1),("8H",2),("12H",3),("24H",6)]:
        result[name] = v5_event_outcome(x, i, bars, direction)

    result["24H_path"] = result["24H"]
    p = result["24H"]["path"] if result.get("24H") else None
    if not p:
        result["path_type"] = "UNKNOWN"
        return result

    if direction == "UP":
        # A second leg means the coin first moved materially in the original
        # direction. A delayed reversal means that the later move erased a
        # meaningful part of that gain. This is the behavior we want to learn.
        early_gain = max([v for _, v in p.get("first_end", [])] + [0])
        delayed_reversal = (
            p["favorable"] >= 15 and
            p["favorable_bar"] <= 5 and
            p["end"] <= p["favorable"] - 12
        )
        second_leg = p["favorable"] >= 10
        early_rejection = p["adverse"] >= 12 and early_gain < 8
        if delayed_reversal:
            result["path_type"] = "SECOND LEG THEN REVERSAL"
        elif second_leg and p["end"] >= 5:
            result["path_type"] = "CLEAN CONTINUATION"
        elif early_rejection:
            result["path_type"] = "EARLY REJECTION"
        elif second_leg:
            result["path_type"] = "SECOND LEG / MIXED"
        else:
            result["path_type"] = "CHOP / MIXED"
    else:
        early_drop = min([v for _, v in p.get("first_end", [])] + [0])
        delayed_bounce = (
            p["favorable"] >= 15 and
            p["favorable_bar"] <= 5 and
            p["end"] >= -p["favorable"] + 12
        )
        second_leg = p["favorable"] >= 10
        early_rejection = p["adverse"] >= 12 and early_drop > -8
        if delayed_bounce:
            result["path_type"] = "SECOND LEG THEN BOUNCE"
        elif second_leg and p["end"] <= -5:
            result["path_type"] = "CLEAN CONTINUATION"
        elif early_rejection:
            result["path_type"] = "EARLY REJECTION"
        elif second_leg:
            result["path_type"] = "SECOND LEG / MIXED"
        else:
            result["path_type"] = "CHOP / MIXED"
    return result

# Historical event creation calls the global function at runtime, so these
# replacements automatically make the learning pool use the v5 path logic.
event_outcome = v5_event_outcome
multi_horizon_outcomes = v5_multi_horizon_outcomes


def v5_outcome_summary(matches):
    if not matches:
        return None
    usable = [
        (idx, sim, e) for idx, (sim,e) in enumerate(matches)
        if e.get("outcome") and e["outcome"].get("24H_path")
    ]
    if not usable:
        return None

    weights = np.array([max(1.0, sim/10.0) for _,sim,_ in usable])
    labels=[e["outcome"]["24H_path"]["label"] for _,_,e in usable]
    path_types=[e["outcome"].get("path_type","UNKNOWN") for _,_,e in usable]
    total=len(usable)

    def pct(label):
        return labels.count(label)/total*100
    def pathpct(label):
        return path_types.count(label)/total*100
    def hstats(h):
        ends=[]; bests=[]; worsts=[]; ws=[]
        for j,(idx,sim,e) in enumerate(usable):
            o=e["outcome"].get(h)
            if not o: continue
            ends.append(o["end"]); bests.append(o["best"]); worsts.append(o["worst"]); ws.append(weights[j])
        return {"end":_weighted_mean(ends,ws),"best":_weighted_mean(bests,ws),"worst":_weighted_mean(worsts,ws)}

    reversal_times=[]; second_leg_gain=[]
    for _,_,e in usable:
        p=e["outcome"]["24H_path"].get("path",{})
        if e["outcome"].get("path_type")=="SECOND LEG THEN REVERSAL":
            if np.isfinite(safe(p.get("adverse_bar"))): reversal_times.append(safe(p.get("adverse_bar"))*4)
            if np.isfinite(safe(p.get("favorable"))): second_leg_gain.append(safe(p.get("favorable")))

    h4=hstats("4H"); h8=hstats("8H"); h12=hstats("12H"); h24=hstats("24H")
    return {
        "total":total,
        "continue_pct":pct("CONTINUED"),
        "dump_pct":pct("DUMPED"),
        "reverse_pct":pct("REVERSED / BOUNCED"),
        "side_pct":pct("SIDEWAYS / PULLBACK"),
        "second_leg_pct":pathpct("SECOND LEG THEN REVERSAL"),
        "clean_continuation_pct":pathpct("CLEAN CONTINUATION"),
        "early_rejection_pct":pathpct("EARLY REJECTION"),
        "second_leg_mixed_pct":pathpct("SECOND LEG / MIXED"),
        "chop_pct":pathpct("CHOP / MIXED"),
        "reversal_timing_hours":float(np.median(reversal_times)) if reversal_times else np.nan,
        "second_leg_gain":float(np.median(second_leg_gain)) if second_leg_gain else np.nan,
        "4H":h4,"8H":h8,"12H":h12,"24H":h24,
        "avg_end":h24["end"],"avg_best":h24["best"],"avg_worst":h24["worst"],
        "median_similarity":float(np.median([sim for _,sim,_ in usable])),
        "max_similarity":float(max(sim for _,sim,_ in usable)),
        "min_similarity":float(min(sim for _,sim,_ in usable)),
    }

outcome_summary = v5_outcome_summary


def short_term_state(current):
    d=current.get("tf_data",{}).get("15m")
    if d is None or len(d)<30:
        return {"state":"NO DATA","score":0,"reversal_confirmed":False,"reasons":[]}
    x=indicators(completed(d))
    if len(x)<25:
        return {"state":"NO DATA","score":0,"reversal_confirmed":False,"reasons":[]}
    r=x.iloc[-1]
    close=safe(r.close); ema20=safe(r.ema20); ema50=safe(r.ema50); ema100=safe(r.ema100)
    adx=safe(r.adx); macd=safe(r.macd); sig=safe(r.macd_signal); vol=safe(r.vol_ratio); rsi=safe(r.rsi)
    slope20=((safe(x.iloc[-1].ema20)/safe(x.iloc[-5].ema20))-1)*100 if safe(x.iloc[-5].ema20)>0 else np.nan
    recent=x.tail(8); prior=x.iloc[-16:-8]
    hh=recent.high.max()>prior.high.max() if not prior.empty else False
    hl=recent.low.min()>prior.low.min() if not prior.empty else False
    recent_high=safe(x.tail(24).high.max())
    pullback=(close/recent_high-1)*100 if recent_high>0 else np.nan

    score=0; reasons=[]
    if close>ema20: score+=20; reasons.append("15m price is above EMA20")
    if ema20>ema50: score+=15; reasons.append("15m EMA20 is above EMA50")
    if ema50>ema100: score+=10; reasons.append("15m EMA50 is above EMA100")
    if np.isfinite(adx) and adx>=25: score+=15; reasons.append(f"ADX is strong ({adx:.0f})")
    if np.isfinite(macd) and np.isfinite(sig) and macd>sig: score+=10; reasons.append("MACD is bullish")
    if np.isfinite(slope20) and slope20>0.15: score+=10; reasons.append("EMA20 is still rising")
    if hh and hl: score+=15; reasons.append("recent candles are making higher highs/higher lows")
    if np.isfinite(vol) and vol>=1: score+=5; reasons.append("volume is supporting the move")

    # Do not call an overbought coin a reversal until the short-term structure
    # actually breaks. This is deliberately conservative.
    below20=close<ema20
    below50=close<ema50
    slope_down=np.isfinite(slope20) and slope20<-0.10
    lower_structure=(not hh) and (recent.low.min()<prior.low.min() if not prior.empty else False)
    four_h=current.get("4h")
    four_h_break=(safe(four_h.close)<safe(four_h.ema20)) if four_h is not None else False
    reversal_confirmed = (below20 and slope_down and lower_structure) or (below50 and four_h_break)

    if reversal_confirmed:
        state="REVERSAL CONFIRMED"
    elif score>=70:
        state="STRONG CONTINUATION"
    elif score>=50:
        state="BULLISH / CONTINUATION"
    else:
        state="WEAK / WAIT"

    return {
        "state":state,"score":score,"reversal_confirmed":reversal_confirmed,
        "close":close,"ema20":ema20,"ema50":ema50,"ema100":ema100,
        "adx":adx,"macd":macd,"signal":sig,"volume":vol,"rsi":rsi,
        "ema20_slope":slope20,"pullback":pullback,"higher_highs":hh,"higher_lows":hl,
        "reasons":reasons,
        "break_ema20":below20,"break_ema50":below50,"lower_structure":lower_structure,
    }


def v5_decision(summary,current):
    st15=short_term_state(current)
    event=current.get("event","")
    down=event in {"ATL BREAKDOWN","FAST DUMP"}
    extreme=event_is_extreme(current.get("target"))
    enough=bool(summary and summary.get("total",0)>=8)
    quality=(summary.get("median_similarity",0)>=48) if summary else False

    if down:
        if st15["reversal_confirmed"]:
            return "🔴 DOWN MOVE CONFIRMED", "The short-term structure is still bearish and has not shown a strong reversal."
        return "🟡 DOWN TREND / WAIT", "The coin is weak, but the short-term structure is not strong enough to claim the next move with confidence."

    # Most important v5 rule: extreme + bullish structure + no break = continuation
    # mode, even if RSI/extension is very high.
    if extreme and not st15["reversal_confirmed"] and st15["score"]>=65:
        if summary and summary.get("second_leg_pct",0)>=30:
            return "🚀 CONTINUATION MODE — DELAYED REVERSAL RISK", (
                "The coin is extremely extended, but its short-term trend is still intact. "
                f"Historically, {summary['second_leg_pct']:.0f}% of comparable cases made another leg before a major reversal."
            )
        return "🚀 CONTINUATION MODE — REVERSAL NOT CONFIRMED", (
            "The coin is very extended, but the short-term bullish structure is still intact. "
            "High RSI alone is not treated as a reversal signal."
        )

    if st15["reversal_confirmed"]:
        return "🔴 REVERSAL CONFIRMED", (
            "The short-term structure has actually broken: price/EMA structure and price action now support a reversal."
        )

    if enough and quality and summary.get("continue_pct",0)>=58 and summary.get("continue_pct",0)-summary.get("dump_pct",0)>=15:
        return "🟢 HISTORICAL CONTINUATION BIAS", "Similar historical setups favored continuation and the current short-term structure has not confirmed a reversal."

    return "🟡 BULLISH BUT WAIT FOR CONFIRMATION", "The trend may continue, but the evidence is not strong enough to call the next move."


def v5_simple_language(summary,current,decision_title):
    s15=short_term_state(current)
    target=current.get("target") or {}
    rsi=safe(target.get("rsi")); ema=safe(target.get("ema20_dist"))
    lines=[]
    if decision_title.startswith("🚀"):
        lines.append("The important point: this coin is still pumping because the trend has not broken.")
        lines.append("Being overbought or far above EMA20 does NOT automatically mean the next candle must dump.")
        if summary and summary.get("second_leg_pct",0)>=30:
            lines.append(f"Historical matches show a delayed pattern in about {summary['second_leg_pct']:.0f}% of cases: another leg up first, reversal later.")
        lines.append("Watch for a real 15m structure break before treating this as a reversal.")
    elif decision_title.startswith("🔴"):
        lines.append("This is different from simply being overbought: the short-term structure has actually started breaking.")
        lines.append("A reversal signal becomes more meaningful when price loses EMA20/EMA50 and starts making lower highs/lows.")
    else:
        lines.append("The trend is not enough by itself to predict the next candle. Wait for confirmation rather than guessing.")
    if np.isfinite(rsi): lines.append(f"Current RSI: {rsi:.1f}.")
    if np.isfinite(ema): lines.append(f"Price vs EMA20: {ema:+.1f}%.")
    lines.append(f"15m state: {s15['state']} ({s15['score']}/100).")
    return lines


def confirmation_text_v5(current):
    s=short_term_state(current); out=[]
    out.extend(s.get("reasons",[])[:6])
    if s["reversal_confirmed"]:
        out.append("⚠️ Reversal confirmation is active on the short-term structure.")
    else:
        out.append("🟢 No confirmed short-term reversal yet.")
        out.append("⚠️ If price breaks 15m EMA20/EMA50 and starts making lower highs/lows, reassess the pump.")
    return out

confirmation_text=confirmation_text_v5

# =============================================================================
# V5 UI
# =============================================================================
st.title("🧠 CoinDCX Historical Pattern Learning Scanner V5")
st.caption("Learns from historical CoinDCX Futures behavior and separates active continuation from a confirmed reversal. Analysis only — no orders.")

margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Use meme-focused learning universe",value=False)
peer_limit=st.slider("Historical comparison universe",20,150,100,10,help="More contracts provide more historical examples but require more CoinDCX API calls.")
st.info("V5 rule: an extreme pump is NOT treated as an immediate short. The scanner checks whether the short-term trend is still intact and whether a real reversal has been confirmed.")

st.divider()
st.header("🔎 Analyze a Coin")
coin=st.text_input("Coin / Futures pair",placeholder="USELESS, DOGE, PEPE, B-DOGE_USDT")

if st.button("🧠 Analyze Coin & Learn From CoinDCX",type="primary"):
    try:
        with st.spinner("Fetching CoinDCX history and studying continuation vs reversal..."):
            prices=futures_prices(); req=normalize(coin); found=[]
            for q in [margin]+[x for x in ("USDT","INR") if x!=margin]:
                for pair in active_instruments(q):
                    p=prices.get(pair)
                    if not p: continue
                    symbol=str(p.get("mkt",pair)).upper()
                    if coin_matches(pair,symbol,req,q): found.append((pair,p,symbol,q))
            if not found:
                st.error(f"No active CoinDCX Futures contract found for '{coin}'."); st.stop()
            found.sort(key=lambda z:(0 if z[3]==margin else 1,len(z[0])))
            pair,p,symbol,_=found[0]
            current=analyze_current_coin(pair,p)
            mode="ATH" if current["event"]=="ATH BREAKOUT" else "ATL" if current["event"]=="ATL BREAKDOWN" else "PUMP"
            universe=universe_rows(margin,meme_only,peer_limit)
            pairs_sig=tuple((z[0],z[2]) for z in universe if z[0]!=pair)
            pool,failures=build_learning_pool(pairs_sig,margin,peer_limit,mode)
            same_coin_pool=build_same_coin_pool(pair,mode)
            extreme_pool=build_extreme_pool(pool)
            combined_pool=merge_learning_pools(pool,same_coin_pool,extreme_pool)
            raw_matches=similar_events(current["target"],combined_pool,max_matches=100)
            matches=diversified_matches(raw_matches,max_matches=60,per_coin=4)
            summary=outcome_summary(matches)
            decision_title,decision_text=v5_decision(summary,current)

            st.subheader(f"{symbol} — V5 Simple Prediction")
            st.write(f"**Event:** {current['event']}")
            a,b,c,d=st.columns(4)
            a.metric("Current",fmt(current["current"]))
            b.metric("24h",f"{safe(p.get('pc',0),0):+.2f}%")
            b4=current["4h"]
            c.metric("4H RSI",f"{safe(b4.rsi):.1f}" if pd.notna(b4.rsi) else "—")
            d.metric("4H Volume",f"{safe(b4.vol_ratio):.1f}x" if pd.notna(b4.vol_ratio) else "—")

            if decision_title.startswith("🚀") or decision_title.startswith("🟢"):
                st.success(decision_title)
            elif decision_title.startswith("🔴"):
                st.error(decision_title)
            else:
                st.warning(decision_title)
            st.markdown(f"### {decision_title}")
            st.write(decision_text)

            s15=short_term_state(current)
            st.markdown("### 📱 What is happening RIGHT NOW? (15-minute)")
            q1,q2,q3,q4,q5=st.columns(5)
            q1.metric("15m state",s15["state"])
            q2.metric("Trend score",f"{s15['score']}/100")
            q3.metric("ADX",f"{s15['adx']:.1f}" if np.isfinite(s15['adx']) else "—")
            q4.metric("EMA20 slope",f"{s15['ema20_slope']:+.2f}%" if np.isfinite(s15['ema20_slope']) else "—")
            q5.metric("Pullback from 24-bar high",f"{s15['pullback']:+.1f}%" if np.isfinite(s15['pullback']) else "—")

            st.markdown("### 🧭 Trend vs. reversal")
            t1,t2,t3,t4=st.columns(4)
            t1.metric("Current trend", "BULLISH" if current["bull"]>=current["bear"] else "MIXED")
            t2.metric("Momentum", "EXTREME" if event_is_extreme(current["target"]) else "NORMAL")
            t3.metric("Reversal confirmed", "YES" if s15["reversal_confirmed"] else "NO")
            t4.metric("15m structure",current.get("structure15","Mixed"))

            if summary:
                st.markdown("### 📚 What happened to similar coins AFTER the setup?")
                h1,h2,h3,h4=st.columns(4)
                for col,label,key in [(h1,"4H later","4H"),(h2,"8H later","8H"),(h3,"12H later","12H"),(h4,"24H later","24H")]:
                    val=summary[key]["end"]
                    col.metric(label,f"{val:+.1f}%" if np.isfinite(val) else "—")

                a,b,c,d=st.columns(4)
                a.metric("Continued",f"{summary['continue_pct']:.0f}%")
                b.metric("Dumped",f"{summary['dump_pct']:.0f}%")
                c.metric("Sideways",f"{summary['side_pct']:.0f}%")
                d.metric("Strong bounce",f"{summary['reverse_pct']:.0f}%")

                st.markdown("### 🛣️ The sequence the engine learned")
                p1,p2,p3,p4=st.columns(4)
                p1.metric("Another leg → reversal",f"{summary['second_leg_pct']:.0f}%")
                p2.metric("Clean continuation",f"{summary['clean_continuation_pct']:.0f}%")
                p3.metric("Early rejection",f"{summary['early_rejection_pct']:.0f}%")
                p4.metric("Typical reversal time",f"{summary['reversal_timing_hours']:.0f}H" if np.isfinite(summary['reversal_timing_hours']) else "—")

                st.write(f"**Sample:** {summary['total']} historical cases | median similarity {summary['median_similarity']:.0f}% | strongest {summary['max_similarity']:.0f}% | evidence: **{evidence_grade(summary)}**")
                if np.isfinite(summary.get("second_leg_gain",np.nan)):
                    st.write(f"**When the second-leg pattern occurred, the typical maximum move before the reversal was about +{summary['second_leg_gain']:.0f}%.**")

            st.markdown("### 📌 Simple explanation")
            for line in v5_simple_language(summary,current,decision_title):
                st.write("• "+line)

            st.markdown("### 👀 What should be watched now?")
            for line in confirmation_text(current):
                st.write("• "+line)

            st.markdown("### 📊 7-Timeframe EMA picture")
            ema_table=[]
            for tf in ["1m","5m","15m","1H","4H","1D","1W"]:
                r=current["ema_rows"].get(tf,{})
                ema_table.append({"Timeframe":tf,"EMA20/50/100":r.get("state","NO DATA"),"Alignment":f"{r.get('count',0)}/3"})
            st.dataframe(pd.DataFrame(ema_table),use_container_width=True,hide_index=True)
            st.write(f"**Full EMA alignment:** {current['bull']*3}/21 bullish conditions | {current['bear']*3}/21 bearish conditions.")

            if matches:
                st.markdown("### 🔎 Closest historical examples")
                rows=[]
                for sim,e in matches[:20]:
                    f=e["features"]; o=e["outcome"]
                    rows.append({
                        "Similarity":f"{sim:.0f}%","Coin":e.get("pair","—"),"Date":str(e.get("time","—"))[:16],
                        "Regime":e.get("behavior_bucket","—"),"24-bar move":f"{safe(f.get('ret24')):+.1f}%",
                        "RSI":f"{safe(f.get('rsi')):.0f}","Volume":f"{safe(f.get('vol_ratio')):.1f}x",
                        "4H":f"{o['4H']['end']:+.1f}%" if o.get("4H") else "—",
                        "12H":f"{o['12H']['end']:+.1f}%" if o.get("12H") else "—",
                        "24H":f"{o['24H']['end']:+.1f}%" if o.get("24H") else "—",
                        "Path":o.get("path_type","—"),"Best":f"{o['24H']['best']:+.1f}%" if o.get("24H") else "—",
                        "Worst":f"{o['24H']['worst']:+.1f}%" if o.get("24H") else "—",
                    })
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

            with st.expander("Advanced details"):
                st.write(f"**4H structure:** {current['structure4']} | **1D:** {current['structure1']} | **15m:** {current['structure15']}")
                st.write(f"**15m:** ADX {s15['adx']:.1f} | MACD {'Bullish' if s15['macd']>s15['signal'] else 'Bearish'} | EMA20 slope {s15['ema20_slope']:+.2f}%" if np.isfinite(s15['adx']) else "15m indicators unavailable")
                st.write(f"**Learning pool:** {len(pool)} events from {len(pairs_sig)} comparison contracts + {len(same_coin_pool)} same-coin events + {len(extreme_pool)} extreme events.")
                st.write("V5 learns both the historical outcome and the sequence: continuation first, delayed reversal, early rejection or mixed behavior. Current 15m structure is used to decide whether a reversal is actually confirmed.")
                if failures: st.code("\n".join(failures[:50]))

            st.session_state["last_analysis"]={"symbol":symbol,"pair":pair,"current":current,"summary":summary,"matches":matches}
    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")

# =============================================================================
# CURRENT HOT / ATH / ATL DISCOVERY
# =============================================================================
st.divider()
st.header("🔥 Hot / ATH / ATL Discovery")
st.caption("Find current movers first. Copy a coin into Analyze a Coin for the deeper V5 study.")
if st.button("🔍 Scan Current Hot / ATH / ATL Coins"):
    try:
        with st.spinner("Reading current CoinDCX Futures prices..."):
            prices=futures_prices(); rows=[]; active=active_instruments(margin)
            for pair in active:
                p=prices.get(pair)
                if not p: continue
                symbol=str(p.get("mkt",pair)).upper(); cur=current_price(p); pc=safe(p.get("pc",0),0)
                if cur<=0: continue
                if meme_only and not any(w in symbol or w in pair.upper() for w in MEME_WORDS): continue
                rows.append((pair,p,symbol,pc,cur))
            rows.sort(key=lambda z:abs(z[3]),reverse=True)
            out=[]; failures=[]
            for pair,p,symbol,pc,cur in rows[:min(peer_limit,80)]:
                try:
                    d=get_tf(pair,"1D",700); dc=completed(d)
                    if len(dc)<30: continue
                    prior_ath=safe(dc.iloc[:-1].high.max()); prior_atl=safe(dc.iloc[:-1].low.min())
                    ath_dist=(cur/prior_ath-1)*100 if prior_ath>0 else np.nan
                    atl_dist=(cur/prior_atl-1)*100 if prior_atl>0 else np.nan
                    ind=indicators(dc); last=ind.iloc[-1]; tag=None
                    if ath_dist>0: tag="🔥 ATH BREAKOUT"
                    elif ath_dist>=-5: tag="🟢 NEAR ATH"
                    elif atl_dist<0: tag="🩸 ATL BREAKDOWN"
                    elif atl_dist<=5: tag="🟠 NEAR ATL"
                    elif pc>=15: tag="🚀 HOT"
                    elif pc<=-15: tag="🔻 FAST DUMP"
                    if tag:
                        out.append({"Coin":symbol,"Price":fmt(cur),"24h":f"{pc:+.2f}%","ATH distance":f"{ath_dist:+.2f}%","ATL distance":f"{atl_dist:+.2f}%","RSI":f"{safe(last.rsi):.1f}" if pd.notna(last.rsi) else "—","Volume":f"{safe(last.vol_ratio):.1f}x" if pd.notna(last.vol_ratio) else "—","Event":tag})
                except Exception as exc:
                    failures.append(f"{symbol}: {type(exc).__name__}: {exc}")
            if out: st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)
            else: st.warning("No current hot/ATH/ATL candidates were found in the scanned universe.")
            if failures:
                with st.expander("Scan diagnostics"): st.code("\n".join(failures[:50]))
    except Exception as e:
        st.error(f"Discovery scan failed: {type(e).__name__}: {e}")

st.divider()
st.caption("Analysis only. No orders, balances, API keys or withdrawals are used. Historical behavior is evidence, not a guarantee or financial advice.")
