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
    """Discover the complete active CoinDCX Futures universe robustly.

    CoinDCX responses have appeared in several shapes (plain list, nested data,
    keyed dictionaries and price-feed objects).  This function normalizes all
    of them.  It never converts an API failure into a fake empty market.
    """
    url = f"{API}/exchange/v1/derivatives/futures/data/active_instruments"
    errors = []

    def flatten_records(obj):
        out = []
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    out.append(item)
                elif isinstance(item, str):
                    out.append({"pair": item, "symbol": item})
        elif isinstance(obj, dict):
            # Normal documented/nested response containers.
            for key in ("data", "instruments", "active_instruments", "result", "markets", "items"):
                if key in obj:
                    out.extend(flatten_records(obj[key]))
            # Also support keyed dictionaries such as {"B-BTC_USDT": {...}}.
            for key, value in obj.items():
                if isinstance(value, dict):
                    rec = dict(value)
                    if not any(rec.get(k) for k in ("pair", "symbol", "market", "instrument", "coindcx_name")):
                        if isinstance(key, str) and ("_USDT" in key.upper() or "USDT" in key.upper()):
                            rec["pair"] = key
                    if any(rec.get(k) for k in ("pair", "symbol", "market", "instrument", "coindcx_name")):
                        out.append(rec)
        return out

    attempts = [
        {"margin_currency_short_name[]": margin},
        {"margin_currency_short_name": margin},
        {"margin_currency_short_name[]": [margin]},
        {},
    ]
    for params in attempts:
        try:
            r = requests.get(url, params=params, timeout=25)
            r.raise_for_status()
            payload = r.json()
            rows = flatten_records(payload)
            if rows:
                # Keep the legacy V5 contract: callers expect a list of pair strings.
                # V6/V6.2 also accepts strings via v61_instrument_pair().
                pairs = []
                seen_pairs = set()
                for rec in rows:
                    if isinstance(rec, str):
                        pair = rec.strip()
                    elif isinstance(rec, dict):
                        pair = next((rec.get(k) for k in ("pair", "symbol", "market", "instrument", "coindcx_name", "id") if isinstance(rec.get(k), str) and rec.get(k).strip()), None)
                    else:
                        pair = None
                    if pair and pair not in seen_pairs:
                        seen_pairs.add(pair)
                        pairs.append(pair)
                if pairs:
                    return pairs
            errors.append(f"empty response params={params}; payload_type={type(payload).__name__}")
        except Exception as exc:
            errors.append(f"instrument endpoint {type(exc).__name__}: {exc}")

    # Robust fallback: the public real-time Futures feed itself is a live market
    # universe. This is especially useful if the active_instruments schema changes.
    try:
        raw = requests.get(f"{PUBLIC}/market_data/v3/current_prices/futures/rt", timeout=25)
        raw.raise_for_status()
        payload = raw.json()
        feed = payload.get("prices", payload) if isinstance(payload, dict) else payload
        derived = []
        if isinstance(feed, dict):
            iterator = feed.items()
        elif isinstance(feed, list):
            iterator = []
            for item in feed:
                if isinstance(item, dict):
                    key = item.get("pair") or item.get("symbol") or item.get("mkt") or item.get("market")
                    if key:
                        iterator.append((key, item))
        else:
            iterator = []
        seen = set()
        for key, value in iterator:
            pair = None
            if isinstance(value, dict):
                pair = value.get("pair") or value.get("symbol") or value.get("mkt") or value.get("market") or key
            else:
                pair = key
            if isinstance(pair, str):
                pair = pair.strip()
                up = pair.upper()
                if pair and ("USDT" in up or margin.upper() in up) and pair not in seen:
                    seen.add(pair)
                    derived.append({"pair": pair, "symbol": pair, "margin_currency_short_name": margin})
        if derived:
            return [x["pair"] for x in derived if isinstance(x, dict) and x.get("pair")]
        errors.append(f"price-feed fallback returned no {margin} Futures pairs; payload_type={type(feed).__name__}")
    except Exception as exc:
        errors.append(f"price-feed fallback {type(exc).__name__}: {exc}")

    raise RuntimeError("CoinDCX Futures universe discovery failed. " + " | ".join(errors[-5:]))


@st.cache_data(ttl=5, show_spinner=False)
def futures_prices():
    """Return current Futures prices normalized to {pair: price-record}."""
    r = requests.get(f"{PUBLIC}/market_data/v3/current_prices/futures/rt", timeout=25)
    r.raise_for_status()
    payload = r.json()
    feed = payload.get("prices", payload) if isinstance(payload, dict) else payload
    out = {}

    def add(pair, value):
        if not isinstance(pair, str) or not pair.strip():
            return
        pair = pair.strip()
        if isinstance(value, dict):
            rec = dict(value)
            rec.setdefault("pair", pair)
            # Some variants expose the last price under different names.
            if not any(k in rec for k in ("price", "last_price", "last", "close")):
                for k in ("p", "lp", "mark_price", "mp"):
                    if k in rec:
                        rec["price"] = rec[k]
                        break
            out[pair] = rec
        else:
            num = v6_num(value)
            if np.isfinite(num):
                out[pair] = {"pair": pair, "price": num}

    if isinstance(feed, dict):
        for key, value in feed.items():
            pair = None
            if isinstance(value, dict):
                pair = value.get("pair") or value.get("symbol") or value.get("mkt") or value.get("market") or key
            else:
                pair = key
            add(pair, value)
    elif isinstance(feed, list):
        for item in feed:
            if isinstance(item, dict):
                pair = item.get("pair") or item.get("symbol") or item.get("mkt") or item.get("market")
                if pair:
                    add(pair, item)
    if not out:
        raise RuntimeError(f"CoinDCX Futures price feed returned no usable prices (payload_type={type(feed).__name__})")
    return out

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
            pair_key = e.get("pair")
            if isinstance(pair_key, dict):
                pair_key = pair_key.get("pair") or pair_key.get("symbol") or str(pair_key)
            event_key = e.get("event_type")
            if isinstance(event_key, dict):
                event_key = event_key.get("event_type") or event_key.get("type") or str(event_key)
            key = (
                str(pair_key),
                str(e.get("time")),
                str(event_key),
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
        return {"state":"NO DATA","score":0,"reversal_confirmed":False,
                "reversal_score":0,"reversal_stage":"UNKNOWN","reasons":[],"reversal_reasons":[]}
    x=indicators(completed(d))
    if len(x)<25:
        return {"state":"NO DATA","score":0,"reversal_confirmed":False,
                "reversal_score":0,"reversal_stage":"UNKNOWN","reasons":[],"reversal_reasons":[]}

    r=x.iloc[-1]
    close=safe(r.close); ema20=safe(r.ema20); ema50=safe(r.ema50); ema100=safe(r.ema100)
    adx=safe(r.adx); macd=safe(r.macd); sig=safe(r.macd_signal); vol=safe(r.vol_ratio); rsi=safe(r.rsi)
    slope20=((safe(x.iloc[-1].ema20)/safe(x.iloc[-5].ema20))-1)*100 if safe(x.iloc[-5].ema20)>0 else np.nan

    recent=x.tail(8); prior=x.iloc[-16:-8]
    hh=recent.high.max()>prior.high.max() if not prior.empty else False
    hl=recent.low.min()>prior.low.min() if not prior.empty else False
    lower_high=(recent.high.max()<prior.high.max()) if not prior.empty else False
    lower_low=(recent.low.min()<prior.low.min()) if not prior.empty else False
    recent_high=safe(x.tail(24).high.max())
    pullback=(close/recent_high-1)*100 if recent_high>0 else np.nan

    # -------------------------------------------------------------------------
    # CONTINUATION SCORE (0-100)
    # -------------------------------------------------------------------------
    score=0; reasons=[]
    if close>ema20: score+=20; reasons.append("15m price is above EMA20")
    if ema20>ema50: score+=15; reasons.append("15m EMA20 is above EMA50")
    if ema50>ema100: score+=10; reasons.append("15m EMA50 is above EMA100")
    if np.isfinite(adx) and adx>=25: score+=15; reasons.append(f"ADX is strong ({adx:.0f})")
    if np.isfinite(macd) and np.isfinite(sig) and macd>sig: score+=10; reasons.append("MACD is bullish")
    if np.isfinite(slope20) and slope20>0.15: score+=10; reasons.append("EMA20 is still rising")
    if hh and hl: score+=15; reasons.append("recent candles are making higher highs/higher lows")
    if np.isfinite(vol) and vol>=1: score+=5; reasons.append("volume is supporting the move")

    # -------------------------------------------------------------------------
    # REVERSAL SCORE (0-100)
    # A single EMA break is NOT enough. We require several independent pieces
    # of evidence before using the strong "REVERSAL CONFIRMED" label.
    # -------------------------------------------------------------------------
    below20=np.isfinite(close) and np.isfinite(ema20) and close<ema20
    below50=np.isfinite(close) and np.isfinite(ema50) and close<ema50
    ema_bear_stack=np.isfinite(ema20) and np.isfinite(ema50) and ema20<ema50
    slope_down=np.isfinite(slope20) and slope20<-0.10
    macd_bearish=np.isfinite(macd) and np.isfinite(sig) and macd<sig
    volume_confirmation=np.isfinite(vol) and vol>=1.30
    four_h=current.get("4h")
    four_h_below20=(safe(four_h.close)<safe(four_h.ema20)) if four_h is not None else False

    reversal_score=0; reversal_reasons=[]
    if below20:
        reversal_score+=15; reversal_reasons.append("15m price is below EMA20")
    if below50:
        reversal_score+=20; reversal_reasons.append("15m price is below EMA50")
    if ema_bear_stack:
        reversal_score+=10; reversal_reasons.append("15m EMA20 is below EMA50")
    if slope_down:
        reversal_score+=10; reversal_reasons.append(f"EMA20 slope is falling ({slope20:+.2f}%)")
    if lower_high:
        reversal_score+=15; reversal_reasons.append("15m has formed a lower high")
    if lower_low:
        reversal_score+=15; reversal_reasons.append("15m has formed a lower low")
    if macd_bearish:
        reversal_score+=5; reversal_reasons.append("MACD is bearish")
    if volume_confirmation:
        reversal_score+=5; reversal_reasons.append(f"volume confirms weakness ({vol:.1f}x average)")
    if four_h_below20:
        reversal_score+=5; reversal_reasons.append("4H price is below EMA20")

    # Strong confirmation requires a genuine price-structure break plus
    # supporting momentum/EMA evidence.  This prevents a single 4H EMA break
    # from overriding a still-bullish 15m market.
    structure_break = lower_high and lower_low
    momentum_confirmation = (slope_down and (below50 or ema_bear_stack)) or (macd_bearish and below20)
    reversal_confirmed = (
        reversal_score>=65
        and structure_break
        and momentum_confirmation
    )

    if reversal_confirmed:
        reversal_stage="CONFIRMED"
        state="REVERSAL CONFIRMED"
    elif reversal_score>=45:
        reversal_stage="DEVELOPING"
        state="REVERSAL DEVELOPING"
    elif score>=70:
        reversal_stage="NONE"
        state="STRONG CONTINUATION"
    elif score>=50:
        reversal_stage="NONE"
        state="BULLISH / CONTINUATION"
    else:
        reversal_stage="WATCH"
        state="WEAK / WAIT"

    return {
        "state":state,"score":score,"reversal_confirmed":reversal_confirmed,
        "reversal_score":reversal_score,"reversal_stage":reversal_stage,
        "close":close,"ema20":ema20,"ema50":ema50,"ema100":ema100,
        "adx":adx,"macd":macd,"signal":sig,"volume":vol,"rsi":rsi,
        "ema20_slope":slope20,"pullback":pullback,"higher_highs":hh,"higher_lows":hl,
        "lower_high":lower_high,"lower_low":lower_low,
        "reasons":reasons,"reversal_reasons":reversal_reasons,
        "break_ema20":below20,"break_ema50":below50,
        "ema_bear_stack":ema_bear_stack,"four_h_below20":four_h_below20,
    }


def v5_decision(summary,current):
    st15=short_term_state(current)
    event=current.get("event","")
    down=event in {"ATL BREAKDOWN","FAST DUMP"}
    extreme=event_is_extreme(current.get("target"))
    enough=bool(summary and summary.get("total",0)>=8)
    quality=(summary.get("median_similarity",0)>=48) if summary else False

    # A confirmed reversal always requires the dedicated multi-confirmation gate.
    if st15["reversal_confirmed"]:
        if down:
            return "🔴 DOWN MOVE CONFIRMED", "The short-term bearish structure is confirmed by multiple independent signals and has not shown a strong reversal."
        return "🔴 REVERSAL CONFIRMED", "The short-term structure has broken with multiple confirmations: lower-high/lower-low price action plus bearish EMA/momentum evidence."

    # Reversal is developing, but not yet strong enough to call confirmed.
    if st15["reversal_stage"]=="DEVELOPING":
        return "🟠 REVERSAL DEVELOPING", (
            f"The short-term structure is weakening (reversal score {st15['reversal_score']}/100), "
            "but the confirmation threshold has not been reached. Avoid treating an EMA break alone as a confirmed reversal."
        )

    if down:
        return "🟡 DOWN TREND / WAIT", "The coin is weak, but the short-term structure is not strong enough to claim the next move with confidence."

    # Most important V5 rule: extreme + intact short-term structure =
    # continuation mode, even when the coin is extended.
    if extreme and st15["reversal_score"]<45 and st15["score"]>=65:
        if summary and summary.get("second_leg_pct",0)>=30:
            return "🚀 CONTINUATION MODE — DELAYED REVERSAL RISK", (
                "The coin is extremely extended, but its short-term trend is still intact. "
                f"Historically, {summary['second_leg_pct']:.0f}% of comparable cases made another leg before a major reversal."
            )
        return "🚀 CONTINUATION MODE — REVERSAL NOT CONFIRMED", (
            "The coin is very extended, but the short-term bullish structure is still intact. "
            "High RSI or an EMA displacement alone is not treated as a reversal signal."
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
        lines.append("The important point: this coin is still pumping because the short-term trend has not broken enough to confirm a reversal.")
        lines.append("Being overbought or far from EMA20 does NOT automatically mean the next move must dump.")
        if summary and summary.get("second_leg_pct",0)>=30:
            lines.append(f"Historical matches show a delayed pattern in about {summary['second_leg_pct']:.0f}% of cases: another leg first, reversal later.")
        lines.append("Watch for lower highs + lower lows and bearish momentum before treating this as a confirmed reversal.")
    elif decision_title.startswith("🔴"):
        lines.append("This is different from simply being overbought: multiple short-term reversal conditions have now aligned.")
        lines.append("The strongest confirmation is a lower high + lower low together with bearish EMA/momentum evidence.")
    elif decision_title.startswith("🟠"):
        lines.append("The short-term trend is weakening, but the reversal is still developing rather than fully confirmed.")
        lines.append("A single EMA break is not enough; wait for lower-high/lower-low structure and supporting momentum.")
    else:
        lines.append("The trend is not enough by itself to predict the next candle. Wait for confirmation rather than guessing.")
    if np.isfinite(rsi): lines.append(f"Current RSI: {rsi:.1f}.")
    if np.isfinite(ema): lines.append(f"Price vs EMA20: {ema:+.1f}%.")
    lines.append(f"15m state: {s15['state']} ({s15['score']}/100).")
    lines.append(f"Reversal score: {s15['reversal_score']}/100 ({s15['reversal_stage']}).")
    return lines


def confirmation_text_v5(current):
    s=short_term_state(current); out=[]
    out.extend(s.get("reasons",[])[:4])
    out.extend(s.get("reversal_reasons",[])[:6])
    if s["reversal_confirmed"]:
        out.append("🔴 Reversal confirmation is active: the multi-confirmation gate has been passed.")
    elif s["reversal_stage"]=="DEVELOPING":
        out.append("🟠 Reversal is developing, but confirmation is not complete yet.")
        out.append("⚠️ Wait for lower high + lower low and bearish momentum/EMA confirmation.")
    else:
        out.append("🟢 No confirmed short-term reversal yet.")
        out.append("⚠️ Reassess if price breaks EMA20/EMA50 and begins making lower highs/lower lows.")
    return out

confirmation_text=confirmation_text_v5

confirmation_text=confirmation_text_v5

# =============================================================================
# V6.1 MARKET-WIDE SIGNAL ENGINE — RANGE + BREAKOUT + PUMP/DUMP
# =============================================================================
# Purpose: turn the existing V5/V6 analysis into a "tell me when to trade"
# scanner.  It keeps the original V5 code intact and adds a market-wide layer.
# It is deliberately analysis/paper-only; it does not place live orders.

V61_VERSION = "6.3-STRUCTURE-SIGNAL"
V61_DEFAULTS = {
    "lookback_15m": 160,
    "lookback_1h": 120,
    "lookback_4h": 100,
    "range_min_touches": 2,
    "range_max_width_pct": 8.0,
    "entry_zone_pct": 0.45,
    "min_rr": 2.0,
    "min_score": 78,
    "max_scan_workers": 6,
}


def v61_instrument_pair(x):
    if isinstance(x, str) and x.strip():
        return x.strip()
    if not isinstance(x, dict):
        return None
    for k in ("pair", "symbol", "market", "instrument", "coindcx_name", "id"):
        v = x.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def v61_symbol(x, pair):
    if isinstance(x, dict):
        for k in ("symbol", "pair", "display_name", "market"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return pair


def v61_price_for_pair(prices, pair):
    if not isinstance(prices, dict):
        return np.nan
    target = str(pair).upper()
    for k in (pair, pair.upper(), pair.lower()):
        if k in prices:
            v = prices[k]
            if isinstance(v, dict):
                for kk in ("price", "last_price", "last", "close", "p", "lp", "mark_price", "mp"):
                    if kk in v:
                        q = v6_num(v[kk])
                        if np.isfinite(q): return q
            else:
                q = v6_num(v)
                if np.isfinite(q): return q
    for k, v in prices.items():
        if isinstance(v, dict):
            ident = str(v.get("pair") or v.get("symbol") or v.get("mkt") or v.get("market") or k).upper()
            if ident == target:
                for kk in ("price", "last_price", "last", "close", "p", "lp", "mark_price", "mp"):
                    if kk in v:
                        q = v6_num(v[kk])
                        if np.isfinite(q): return q
    return np.nan


def v61_fetch_candidate(pair):
    """Fetch only the timeframes needed by the fast market-wide signal engine."""
    try:
        d15 = get_tf(pair, "15m", 3)
        d1h = get_tf(pair, "1H", 7)
        d4h = get_tf(pair, "4H", 25)
        if any(d is None or d.empty for d in (d15, d1h, d4h)):
            return None
        return pair, d15, d1h, d4h
    except Exception:
        return None


def v61_atr(x):
    if x is None or len(x) < 20:
        return np.nan
    z = indicators(completed(x))
    return v6_num(z.iloc[-1].get("atr")) if not z.empty else np.nan


def v61_cluster_levels(d, lookback=120, tolerance_pct=0.45):
    """Cluster swing highs/lows into practical zones rather than exact prices."""
    if d is None or d.empty:
        return [], []
    x = completed(d).tail(lookback).reset_index(drop=True)
    if len(x) < 30:
        return [], []
    highs = x.high.astype(float).to_numpy()
    lows = x.low.astype(float).to_numpy()
    close = x.close.astype(float).to_numpy()
    last = float(close[-1])
    if not np.isfinite(last) or last <= 0:
        return [], []
    # Local extrema, then cluster nearby extrema.
    hi_pts, lo_pts = [], []
    for i in range(2, len(x)-2):
        if highs[i] >= highs[i-1] and highs[i] >= highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
            hi_pts.append(float(highs[i]))
        if lows[i] <= lows[i-1] and lows[i] <= lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
            lo_pts.append(float(lows[i]))

    def cluster(points):
        if not points:
            return []
        pts = sorted(points)
        clusters = []
        for p in pts:
            if not clusters:
                clusters.append([p])
                continue
            center = float(np.mean(clusters[-1]))
            if abs(p-center)/center*100 <= tolerance_pct:
                clusters[-1].append(p)
            else:
                clusters.append([p])
        out = []
        for c in clusters:
            center = float(np.mean(c))
            touches = len(c)
            out.append({"level": center, "touches": touches, "strength": min(1.0, 0.35 + 0.18*touches)})
        return out

    return cluster(hi_pts), cluster(lo_pts)


def v61_nearest_levels(d15, d1h, d4h, price):
    """Build support/resistance from 15m, 1H and 4H swing clusters."""
    levels_hi, levels_lo = [], []
    for d, weight in ((d15, 1.0), (d1h, 1.35), (d4h, 1.7)):
        hi, lo = v61_cluster_levels(d)
        for q in hi:
            q = dict(q); q["strength"] *= weight; levels_hi.append(q)
        for q in lo:
            q = dict(q); q["strength"] *= weight; levels_lo.append(q)

    def merge(levels):
        if not levels:
            return []
        levels = sorted(levels, key=lambda z: z["level"])
        merged = []
        for z in levels:
            if not merged or abs(z["level"]-merged[-1]["level"])/merged[-1]["level"]*100 > 0.55:
                merged.append({"level": z["level"], "strength": z["strength"], "touches": z["touches"]})
            else:
                old = merged[-1]
                w1, w2 = old["strength"], z["strength"]
                old["level"] = (old["level"]*w1 + z["level"]*w2)/(w1+w2)
                old["strength"] += z["strength"]
                old["touches"] += z["touches"]
        return merged

    hi, lo = merge(levels_hi), merge(levels_lo)
    supports = sorted([z for z in lo if z["level"] < price], key=lambda z: price-z["level"])
    resistances = sorted([z for z in hi if z["level"] > price], key=lambda z: z["level"]-price)
    return supports, resistances


def v61_range_state(d15, d1h, d4h, price):
    """Detect a range and return its practical support/resistance zones."""
    if any(d is None or d.empty for d in (d15, d1h, d4h)) or not np.isfinite(price) or price <= 0:
        return {"is_range": False, "score": 0}
    x15 = indicators(completed(d15)).tail(100)
    x1h = indicators(completed(d1h)).tail(80)
    x4h = indicators(completed(d4h)).tail(60)
    if min(len(x15), len(x1h), len(x4h)) < 30:
        return {"is_range": False, "score": 0}

    # Trend strength: ranges are strongest when ADX is modest and EMA spread is small.
    adx = v6_num(x15.iloc[-1].get("adx"), 0)
    ema20, ema50 = v6_num(x15.iloc[-1].get("ema20")), v6_num(x15.iloc[-1].get("ema50"))
    ema_spread = abs(ema20-ema50)/price*100 if np.isfinite(ema20) and np.isfinite(ema50) else 99

    supports, resistances = v61_nearest_levels(d15, d1h, d4h, price)
    s = supports[0] if supports else None
    r = resistances[0] if resistances else None
    if not s or not r or r["level"] <= s["level"]:
        return {"is_range": False, "score": 0, "support": s, "resistance": r}

    width_pct = (r["level"]-s["level"])/price*100
    position = (price-s["level"])/(r["level"]-s["level"])
    # Repeated boundary tests on the 1H chart are strong evidence of a tradable range.
    h = x1h.high.to_numpy(dtype=float)
    l = x1h.low.to_numpy(dtype=float)
    touch_r = int(np.sum(np.abs(h-r["level"])/r["level"]*100 <= 0.65))
    touch_s = int(np.sum(np.abs(l-s["level"])/s["level"]*100 <= 0.65))

    score = 0
    if width_pct <= V61_DEFAULTS["range_max_width_pct"]: score += 25
    if adx < 25: score += 20
    elif adx < 30: score += 10
    if ema_spread < 1.2: score += 15
    elif ema_spread < 2.0: score += 8
    if touch_r >= 2: score += 15
    if touch_s >= 2: score += 15
    if 0.12 <= position <= 0.88: score += 10

    return {
        "is_range": score >= 55 and width_pct <= V61_DEFAULTS["range_max_width_pct"] and touch_r >= 2 and touch_s >= 2,
        "score": int(min(100, score)),
        "support": s,
        "resistance": r,
        "width_pct": width_pct,
        "position": position,
        "touch_r": touch_r,
        "touch_s": touch_s,
        "adx": adx,
        "ema_spread": ema_spread,
    }


def v61_regime(d1h, d4h):
    x1 = indicators(completed(d1h))
    x4 = indicators(completed(d4h))
    if x1.empty or x4.empty:
        return "UNKNOWN", 0
    a, b = x1.iloc[-1], x4.iloc[-1]
    bull = sum([
        v6_num(a.close) > v6_num(a.ema20),
        v6_num(a.ema20) > v6_num(a.ema50),
        v6_num(a.macd) > v6_num(a.macd_signal),
        v6_num(b.close) > v6_num(b.ema20),
        v6_num(b.ema20) > v6_num(b.ema50),
    ])
    bear = sum([
        v6_num(a.close) < v6_num(a.ema20),
        v6_num(a.ema20) < v6_num(a.ema50),
        v6_num(a.macd) < v6_num(a.macd_signal),
        v6_num(b.close) < v6_num(b.ema20),
        v6_num(b.ema20) < v6_num(b.ema50),
    ])
    if bull >= 4: return "BULL TREND", bull*20
    if bear >= 4: return "BEAR TREND", bear*20
    return "MIXED", 50


def v61_momentum(d15):
    x = indicators(completed(d15))
    if len(x) < 30:
        return {}
    a = x.iloc[-1]
    prev20 = x.iloc[-21].close if len(x) >= 21 else np.nan
    close = v6_num(a.close)
    return {
        "close": close,
        "rsi": v6_num(a.rsi),
        "macd": v6_num(a.macd),
        "macd_signal": v6_num(a.macd_signal),
        "atr": v6_num(a.atr),
        "atr_pct": v6_num(a.atr_pct),
        "vol_ratio": v6_num(a.vol_ratio, 1),
        "ema20": v6_num(a.ema20),
        "ema50": v6_num(a.ema50),
        "return_5h_pct": v6_pct(close, prev20),
        "close_open": v6_pct(close, x.iloc[-1].open),
    }


def v61_breakout_status(d15, support, resistance):
    x = completed(d15).tail(12)
    if x.empty or not support or not resistance:
        return "NONE"
    close = float(x.close.iloc[-1]); prev = float(x.close.iloc[-2]) if len(x) > 1 else close
    r = resistance["level"]; s = support["level"]
    atr = v61_atr(d15)
    buf = max(atr*0.35 if np.isfinite(atr) else 0, close*0.0015)
    if prev <= r and close > r + buf: return "BREAKOUT_UP"
    if prev >= s and close < s - buf: return "BREAKDOWN_DOWN"
    return "NONE"


def v61_trade_from_setup(pair, side, price, support, resistance, atr, score, reason):
    """Return entry/SL/TP levels.  Entry is a zone; trade is valid only after trigger."""
    if not np.isfinite(price) or price <= 0 or not np.isfinite(atr) or atr <= 0:
        return None
    s = support["level"] if support else np.nan
    r = resistance["level"] if resistance else np.nan
    if side == "LONG":
        if not np.isfinite(s): return None
        entry = s * 1.0015
        stop = min(s - 0.85*atr, entry - 1.15*atr)
        target1 = r * 0.997 if np.isfinite(r) else entry + 2*atr
        risk = entry-stop
        if risk <= 0: return None
        target2 = entry + max(2.8*risk, (target1-entry)*1.55)
        rr1 = (target1-entry)/risk
        rr2 = (target2-entry)/risk
    else:
        if not np.isfinite(r): return None
        entry = r * 0.9985
        stop = max(r + 0.85*atr, entry + 1.15*atr)
        target1 = s * 1.003 if np.isfinite(s) else entry - 2*atr
        risk = stop-entry
        if risk <= 0: return None
        target2 = entry - max(2.8*risk, (entry-target1)*1.55)
        rr1 = (entry-target1)/risk
        rr2 = (entry-target2)/risk
    if rr1 < V61_DEFAULTS["min_rr"]:
        return None
    return {
        "pair": pair, "side": side, "entry": entry, "stop": stop,
        "tp1": target1, "tp2": target2, "rr1": rr1, "rr2": rr2,
        "score": score, "reason": reason,
    }


def v61_analyze_candidate(pair, symbol, price, d15, d1h, d4h):
    m = v61_momentum(d15)
    if not m or not np.isfinite(price):
        return None
    regime, regime_score = v61_regime(d1h, d4h)
    supports, resistances = v61_nearest_levels(d15, d1h, d4h, price)
    support = supports[0] if supports else None
    resistance = resistances[0] if resistances else None
    rng = v61_range_state(d15, d1h, d4h, price)
    brk = v61_breakout_status(d15, support, resistance)
    structure = v63_structure_engine(d15, d1h, price)
    ema_cross = v63_ema_cross_context({"15m": d15, "4H": d4h})
    atr = m.get("atr", np.nan)
    candidates = []

    # ---------------- TRUE HH/HL / LH/LL STRUCTURE ----------------
    # Only strong two-sided swing sequences become actionable structure alerts.
    if structure.get("side") in ("LONG", "SHORT") and structure.get("score", 0) >= 55:
        st = v63_structure_trade(pair, symbol, price, structure, structure["side"])
        if st:
            st.update({"regime": regime, "rsi": m.get("rsi", np.nan), "vol_ratio": m.get("vol_ratio", np.nan),
                       "structure_state": structure.get("state")})
            candidates.append(st)

    # ---------------- RANGE LONG ----------------
    if rng.get("is_range") and support:
        dist_s = abs(price-support["level"])/price*100
        near_s = dist_s <= max(V61_DEFAULTS["entry_zone_pct"], m.get("atr_pct", 0)*1.25)
        rejection = m.get("rsi", 50) < 48 and m.get("macd", 0) >= m.get("macd_signal", 0)
        score = 55 + int(rng["score"]*0.25)
        if near_s: score += 15
        if rejection: score += 10
        if m.get("vol_ratio", 1) >= 1.25: score += 5
        if regime == "BEAR TREND": score -= 15
        score = max(0, min(100, score))
        if score >= V61_DEFAULTS["min_score"]:
            t = v61_trade_from_setup(pair, "LONG", price, support, resistance, atr, score, "RANGE SUPPORT REJECTION")
            if t:
                t.update({"symbol": symbol, "type": "RANGE LONG", "status": "LONG NOW" if near_s and rejection else "LONG SETUP",
                          "support": support["level"], "resistance": resistance["level"] if resistance else np.nan,
                          "range_score": rng["score"], "regime": regime, "rsi": m["rsi"], "vol_ratio": m["vol_ratio"]})
                candidates.append(t)

    # ---------------- RANGE SHORT ----------------
    if rng.get("is_range") and resistance:
        dist_r = abs(resistance["level"]-price)/price*100
        near_r = dist_r <= max(V61_DEFAULTS["entry_zone_pct"], m.get("atr_pct", 0)*1.25)
        rejection = m.get("rsi", 50) > 52 and m.get("macd", 0) <= m.get("macd_signal", 0)
        score = 55 + int(rng["score"]*0.25)
        if near_r: score += 15
        if rejection: score += 10
        if m.get("vol_ratio", 1) >= 1.25: score += 5
        if regime == "BULL TREND": score -= 15
        score = max(0, min(100, score))
        if score >= V61_DEFAULTS["min_score"]:
            t = v61_trade_from_setup(pair, "SHORT", price, support, resistance, atr, score, "RANGE RESISTANCE REJECTION")
            if t:
                t.update({"symbol": symbol, "type": "RANGE SHORT", "status": "SHORT NOW" if near_r and rejection else "SHORT SETUP",
                          "support": support["level"] if support else np.nan, "resistance": resistance["level"],
                          "range_score": rng["score"], "regime": regime, "rsi": m["rsi"], "vol_ratio": m["vol_ratio"]})
                candidates.append(t)

    # ---------------- BREAKOUT LONG ----------------
    if brk == "BREAKOUT_UP":
        score = 72
        if m.get("vol_ratio", 1) >= 1.5: score += 10
        if m.get("rsi", 50) < 78: score += 7
        if regime == "BULL TREND": score += 8
        if regime == "BEAR TREND": score -= 20
        if score >= V61_DEFAULTS["min_score"]:
            # For breakout, use old resistance as entry and the next ATR-based stop.
            fake_support = {"level": resistance["level"] - max(atr, price*0.006)} if resistance else None
            t = v61_trade_from_setup(pair, "LONG", price, fake_support, {"level": price+2.8*atr}, atr, score, "15M BREAKOUT + CONFIRMATION")
            if t:
                t.update({"symbol": symbol, "type": "BREAKOUT LONG", "status": "LONG NOW",
                          "support": fake_support["level"], "resistance": resistance["level"] if resistance else np.nan,
                          "range_score": rng.get("score", 0), "regime": regime, "rsi": m["rsi"], "vol_ratio": m["vol_ratio"]})
                candidates.append(t)

    # ---------------- BREAKDOWN SHORT ----------------
    if brk == "BREAKDOWN_DOWN":
        score = 72
        if m.get("vol_ratio", 1) >= 1.5: score += 10
        if m.get("rsi", 50) > 22: score += 7
        if regime == "BEAR TREND": score += 8
        if regime == "BULL TREND": score -= 20
        if score >= V61_DEFAULTS["min_score"]:
            fake_res = {"level": support["level"] + max(atr, price*0.006)} if support else None
            t = v61_trade_from_setup(pair, "SHORT", price, {"level": price-2.8*atr}, fake_res, atr, score, "15M BREAKDOWN + CONFIRMATION")
            if t:
                t.update({"symbol": symbol, "type": "BREAKDOWN SHORT", "status": "SHORT NOW",
                          "support": support["level"] if support else np.nan, "resistance": fake_res["level"],
                          "range_score": rng.get("score", 0), "regime": regime, "rsi": m["rsi"], "vol_ratio": m["vol_ratio"]})
                candidates.append(t)

    # ---------------- EMA20 / EMA100 CROSS CONFIRMATION ----------------
    # A fresh bearish cross is treated as a downside confirmation, not as a
    # standalone short trigger.  A fresh bullish cross is the mirror image.
    for t in candidates:
        side=t.get("side")
        cross_score=ema_cross["bear_score"] if side == "SHORT" else ema_cross["bull_score"] if side == "LONG" else 0
        if cross_score:
            bonus=min(25, int(round(cross_score*0.35)))
            t["score"]=min(100, int(t.get("score",0))+bonus)
            matched=[]
            c15=ema_cross["15m"]; c4=ema_cross["4H"]
            if side == "SHORT":
                if c15.get("fresh_bearish"): matched.append("15m EMA20/100 bearish cross")
                if c4.get("fresh_bearish"): matched.append("4H EMA20/100 bearish cross")
            else:
                if c15.get("fresh_bullish"): matched.append("15m EMA20/100 bullish cross")
                if c4.get("fresh_bullish"): matched.append("4H EMA20/100 bullish cross")
            if matched:
                t["reason"]=str(t.get("reason", ""))+" | "+", ".join(matched)
        t["ema_cross"] = ema_cross

    # Dedicated EMA-cross watch: useful when the crossover has just happened
    # but the full entry trigger is not confirmed yet.
    ret = m.get("return_5h_pct", 0)
    vol = m.get("vol_ratio", 1)
    rsi = m.get("rsi", 50)
    pump_score = 0
    dump_score = 0
    if ret >= 3: pump_score += 25
    if ret >= 6: pump_score += 15
    if vol >= 1.5: pump_score += 20
    if vol >= 2.5: pump_score += 10
    if rsi >= 60: pump_score += 10
    if m.get("macd", 0) > m.get("macd_signal", 0): pump_score += 10
    if regime == "BULL TREND": pump_score += 10
    if ret <= -3: dump_score += 25
    if ret <= -6: dump_score += 15
    if vol >= 1.5: dump_score += 20
    if vol >= 2.5: dump_score += 10
    if rsi <= 40: dump_score += 10
    if m.get("macd", 0) < m.get("macd_signal", 0): dump_score += 10
    if regime == "BEAR TREND": dump_score += 10

    watches = []
    if pump_score >= 55:
        watches.append({"pair":pair,"symbol":symbol,"watch":"PUMP WATCH","score":min(100,pump_score),"price":price,
                        "return_5h_pct":ret,"vol_ratio":vol,"rsi":rsi,"regime":regime})
    if dump_score >= 55:
        watches.append({"pair":pair,"symbol":symbol,"watch":"DUMP WATCH","score":min(100,dump_score),"price":price,
                        "return_5h_pct":ret,"vol_ratio":vol,"rsi":rsi,"regime":regime})

    if ema_cross["bear_score"] >= 28:
        watches.append({"pair":pair,"symbol":symbol,"watch":"EMA20/100 BEAR CROSS",
                        "score":ema_cross["bear_score"],"price":price,"return_5h_pct":ret,
                        "vol_ratio":vol,"rsi":rsi,"regime":regime,
                        "ema15":ema_cross["15m"]["state"],"ema4h":ema_cross["4H"]["state"]})
    if ema_cross["bull_score"] >= 28:
        watches.append({"pair":pair,"symbol":symbol,"watch":"EMA20/100 BULL CROSS",
                        "score":ema_cross["bull_score"],"price":price,"return_5h_pct":ret,
                        "vol_ratio":vol,"rsi":rsi,"regime":regime,
                        "ema15":ema_cross["15m"]["state"],"ema4h":ema_cross["4H"]["state"]})

    # V7 early-transition radar is computed while the candles are already in memory.
    # This avoids another market-wide candle fetch when the V7 UI runs.
    try:
        early_structure = v71_early_structure_transition(d15, d1h)
    except Exception:
        early_structure = {"state":"UNAVAILABLE", "side":None, "score":0, "trigger":"", "sequence":"",
                           "prior_state":"", "first_break_price":np.nan, "pullback_price":np.nan,
                           "breakout_price":np.nan, "age_bars":None}

    return {"candidates": candidates, "watches": watches, "range": rng, "regime": regime,
            "momentum": m, "support": support, "resistance": resistance, "structure": structure,
            "ema_cross": ema_cross, "early_structure": early_structure}


def v61_scan_all(progress=None, max_workers=6):
    """Scan every active USDT Futures contract.

    The active-instrument list is the source of truth.  The live price feed is
    only a preferred price source; a contract is NOT discarded when its price
    key does not match the instrument key.  In that case the latest completed
    15m candle supplies the current price.  This prevents a valid 500+ contract
    universe from collapsing to zero because of a feed-key naming difference.
    """
    instruments = active_instruments("USDT")
    try:
        prices = futures_prices()
    except Exception:
        prices = {}

    items = []
    seen = set()
    for inst in instruments:
        pair = v61_instrument_pair(inst)
        if not pair:
            continue
        canonical = str(pair).strip().upper()
        if canonical in seen:
            continue
        seen.add(canonical)
        live_price = v61_price_for_pair(prices, pair)
        items.append((pair, v61_symbol(inst, pair), live_price))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(v61_fetch_candidate, p[0]): p for p in items}
        for fut in as_completed(futs):
            base = futs[fut]
            fetched = fut.result()
            done += 1
            if progress:
                progress(done, len(items))
            if not fetched:
                continue
            pair, d15, d1h, d4h = fetched
            try:
                # Prefer the live price, but ALWAYS fall back to the latest
                # completed 15m close when the real-time feed key differs.
                price = base[2]
                if not np.isfinite(price) or price <= 0:
                    c = completed(d15)
                    if c is not None and not c.empty:
                        price = v6_num(c.iloc[-1].close)
                if not np.isfinite(price) or price <= 0:
                    continue
                a = v61_analyze_candidate(pair, base[1], price, d15, d1h, d4h)
                if a:
                    a["price"] = price
                    a["price_source"] = "LIVE_FEED" if np.isfinite(base[2]) and base[2] > 0 else "15M_CANDLE_FALLBACK"
                    results.append(a)
            except Exception:
                continue
    return results, len(items)


def v61_fmt_price(v):
    v = v6_num(v)
    if not np.isfinite(v): return "—"
    if abs(v) >= 1000: return f"{v:,.2f}"
    if abs(v) >= 1: return f"{v:,.4f}"
    if abs(v) >= .01: return f"{v:,.6f}"
    return f"{v:,.8f}"


def v61_signal_card(t):
    side = t.get("side", "")
    emoji = "🟢" if side == "LONG" else "🔴"
    st.write(f"### {emoji} {t.get('status','SIGNAL')} — {t.get('symbol', t.get('pair'))}")
    st.write(f"**Type:** {t.get('type','')}  |  **Score:** {t.get('score',0)}/100  |  **Regime:** {t.get('regime','')}  ")
    st.write(f"**Entry:** `{v61_fmt_price(t.get('entry'))}`  |  **SL:** `{v61_fmt_price(t.get('stop'))}`  |  **TP1:** `{v61_fmt_price(t.get('tp1'))}`  |  **TP2:** `{v61_fmt_price(t.get('tp2'))}`")
    st.write(f"**R:R:** 1:{t.get('rr1',0):.2f} / 1:{t.get('rr2',0):.2f}  |  **Reason:** {t.get('reason','')}")




# =============================================================================
# V5 UI
# =============================================================================
st.title("🧠 CoinDCX Historical Pattern Learning Scanner V5")
st.caption("Learns from historical CoinDCX Futures behavior and separates active continuation from a confirmed reversal. Analysis only — no orders.")

margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Use meme-focused learning universe",value=False)
peer_limit=st.slider("Historical comparison universe",20,150,100,10,help="More contracts provide more historical examples but require more CoinDCX API calls.")
st.info("V5 rule: an extreme pump is NOT treated as an immediate short. Reversal now requires multiple confirmations; EMA weakness alone moves the setup to REVERSAL DEVELOPING, not REVERSAL CONFIRMED.")

st.divider()
st.header("🔎 Analyze a Coin")
coin=st.text_input("Coin / Futures pair",placeholder="USELESS, DOGE, PEPE, B-DOGE_USDT")

if st.button("🧠 Analyze Coin & Learn From CoinDCX",type="primary"):
    try:
        with st.spinner("Fetching CoinDCX history and studying continuation vs reversal..."):
            prices=futures_prices(); req=normalize(coin); found=[]

            # CoinDCX can expose the same Futures contract under slightly
            # different casing/key fields in the active-instruments endpoint
            # and the real-time price feed.  Resolve the pair through the
            # normalized helper instead of doing a fragile exact dictionary
            # lookup.  This keeps the legacy V5 analyzer compatible with the
            # V6/V6.2 market-discovery layer.
            def price_record_for_pair(price_map, target_pair):
                if not isinstance(price_map, dict):
                    return None
                target = str(target_pair).strip().upper()
                for key in (target_pair, str(target_pair).upper(), str(target_pair).lower()):
                    rec = price_map.get(key)
                    if isinstance(rec, dict):
                        return rec
                for key, rec in price_map.items():
                    if not isinstance(rec, dict):
                        continue
                    ident = str(rec.get("pair") or rec.get("symbol") or rec.get("mkt") or rec.get("market") or key).strip().upper()
                    if ident == target:
                        return rec
                return None

            for q in [margin]+[x for x in ("USDT","INR") if x!=margin]:
                for raw_pair in active_instruments(q):
                    pair = v61_instrument_pair(raw_pair)
                    if not pair:
                        continue

                    # Match against the active Futures universe first. The live
                    # price feed may use a different key/field spelling, so it
                    # must not decide whether the contract exists.
                    active_symbol = pair
                    if isinstance(raw_pair, dict):
                        active_symbol = str(raw_pair.get("symbol") or raw_pair.get("display_name") or raw_pair.get("market") or raw_pair.get("pair") or pair)
                    if not coin_matches(pair, active_symbol, req, q):
                        continue

                    p = price_record_for_pair(prices, pair)

                    # Fuzzy price-feed lookup for B-LSK_USDT / LSK_USDT / LSKUSDT.
                    if p is None:
                        target = str(pair).upper().replace("-", "").replace("_", "")
                        for key, rec in prices.items():
                            ident = str(key).upper().replace("-", "").replace("_", "")
                            if ident == target or (ident.startswith("B") and ident[1:] == target):
                                p = rec if isinstance(rec, dict) else {"pair": key, "price": rec}
                                break

                    # Last-resort price from the latest completed 15m candle.
                    if p is None:
                        try:
                            probe = completed(get_tf(pair, "15m", 2))
                            if not probe.empty:
                                last_close = float(probe.iloc[-1]["close"])
                                p = {"pair": pair, "price": last_close, "ls": last_close}
                        except Exception:
                            p = None
                    if p is None:
                        continue

                    symbol = str(p.get("mkt") or p.get("symbol") or p.get("pair") or active_symbol or pair).upper()
                    found.append((pair, p, symbol, q))
            if not found:
                st.error(f"No active CoinDCX Futures contract found for '{coin}'. Active Futures were discovered, but the requested coin did not match. Try the exact pair shown by CoinDCX, e.g. B-LSK_USDT."); st.stop()
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
            q1,q2,q3,q4,q5,q6=st.columns(6)
            q1.metric("15m state",s15["state"])
            q2.metric("Trend score",f"{s15['score']}/100")
            q3.metric("Reversal score",f"{s15['reversal_score']}/100")
            q4.metric("ADX",f"{s15['adx']:.1f}" if np.isfinite(s15['adx']) else "—")
            q5.metric("EMA20 slope",f"{s15['ema20_slope']:+.2f}%" if np.isfinite(s15['ema20_slope']) else "—")
            q6.metric("Pullback from 24-bar high",f"{s15['pullback']:+.1f}%" if np.isfinite(s15['pullback']) else "—")

            st.markdown("### 🧭 Trend vs. reversal")
            t1,t2,t3,t4=st.columns(4)
            t1.metric("Current trend", "BULLISH" if current["bull"]>=current["bear"] else "MIXED")
            t2.metric("Momentum", "EXTREME" if event_is_extreme(current["target"]) else "NORMAL")
            t3.metric("Reversal stage",s15["reversal_stage"])
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
                st.write(f"**15m:** ADX {s15['adx']:.1f} | MACD {'Bullish' if s15['macd']>s15['signal'] else 'Bearish'} | EMA20 slope {s15['ema20_slope']:+.2f}% | Trend score {s15['score']}/100 | Reversal score {s15['reversal_score']}/100" if np.isfinite(s15['adx']) else "15m indicators unavailable")
                st.write(f"**Learning pool:** {len(pool)} events from {len(pairs_sig)} comparison contracts + {len(same_coin_pool)} same-coin events + {len(extreme_pool)} extreme events.")
                st.write("V5 learns both the historical outcome and the sequence: continuation first, delayed reversal, early rejection or mixed behavior. Current 15m structure is used to decide whether a reversal is actually confirmed.")
                if failures: st.code("\n".join(failures[:50]))

            st.session_state["last_analysis"]={"symbol":symbol,"pair":pair,"current":current,"summary":summary,"matches":matches}
    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")

# =============================================================================
# CURRENT HOT / ATH / ATL DISCOVERY — V5.2
# =============================================================================
st.divider()
st.header("🔥 Hot / ATH / ATL Discovery")
st.caption("Scans the live CoinDCX Futures universe and identifies hot movers, historical-high breakouts, near-ATH coins, ATL breakdowns and near-ATL coins.")

def discovery_event(pair, symbol, price_info, margin):
    """Return a robust current discovery record for one futures contract."""
    cur = current_price(price_info)
    pc = safe(price_info.get("pc", 0), 0)
    if cur <= 0:
        return None

    # Use completed 4H candles for the primary discovery window. This catches
    # intraday ATH/ATL events that a 1D-only scan can miss.
    d4 = get_tf(pair, "4H", 260)
    dc4 = completed(d4)
    if dc4 is None or len(dc4) < 60:
        return None

    # Use the longest practical daily history as a second, broader reference.
    d1 = get_tf(pair, "1D", 700)
    dc1 = completed(d1)

    hist4 = dc4
    hist1 = dc1 if dc1 is not None and len(dc1) >= 30 else dc4

    ath4 = safe(hist4.high.max())
    atl4 = safe(hist4.low.min())
    ath1 = safe(hist1.high.max())
    atl1 = safe(hist1.low.min())

    # Treat the broader daily record as the primary ATH/ATL reference, while
    # retaining the 4H reference so recent intraday extremes are visible.
    ath = max(v for v in (ath4, ath1) if np.isfinite(v) and v > 0)
    atl = min(v for v in (atl4, atl1) if np.isfinite(v) and v > 0)
    ath_dist = (cur / ath - 1) * 100 if ath > 0 else np.nan
    atl_dist = (cur / atl - 1) * 100 if atl > 0 else np.nan

    # 4H momentum gives a more useful "hot" signal than relying only on the
    # exchange 24h field when that field is missing or stale.
    ret24_4h = np.nan
    if len(dc4) >= 7:
        base = safe(dc4.iloc[-7].close)
        if base > 0:
            ret24_4h = (cur / base - 1) * 100
    momentum24 = pc if np.isfinite(pc) else ret24_4h

    ind4 = indicators(dc4)
    last = ind4.iloc[-1]

    # Priority: true historical breakout/breakdown > near extreme > hot/dump.
    # A 4H intraday breakout is included when live price has crossed the
    # historical daily/4H extreme.
    if cur > ath:
        tag = "🔥 ATH BREAKOUT"
    elif ath_dist >= -3:
        tag = "🟢 NEAR ATH"
    elif cur < atl:
        tag = "🩸 ATL BREAKDOWN"
    elif atl_dist <= 3:
        tag = "🟠 NEAR ATL"
    elif np.isfinite(momentum24) and momentum24 >= 15:
        tag = "🚀 HOT"
    elif np.isfinite(momentum24) and momentum24 <= -15:
        tag = "🔻 FAST DUMP"
    else:
        return None

    return {
        "Coin": symbol,
        "Pair": pair,
        "Price": fmt(cur),
        "24h": f"{momentum24:+.2f}%" if np.isfinite(momentum24) else "—",
        "ATH distance": f"{ath_dist:+.2f}%" if np.isfinite(ath_dist) else "—",
        "ATL distance": f"{atl_dist:+.2f}%" if np.isfinite(atl_dist) else "—",
        "RSI": f"{safe(last.rsi):.1f}" if pd.notna(last.rsi) else "—",
        "Volume": f"{safe(last.vol_ratio):.1f}x" if pd.notna(last.vol_ratio) else "—",
        "Event": tag,
    }

if st.button("🔍 Scan Current Hot / ATH / ATL Coins"):
    try:
        with st.spinner("Scanning live CoinDCX Futures + historical extremes..."):
            prices = futures_prices()
            active = active_instruments(margin)
            candidates = []
            failures = []

            for pair in active:
                try:
                    # Do not require an exact dictionary-key match. CoinDCX can
                    # expose the same Futures contract under B-XXX_USDT,
                    # XXX_USDT, or another normalized key. Resolve the price
                    # independently and build the legacy price_info structure.
                    p = None
                    if isinstance(prices, dict):
                        raw = prices.get(pair) or prices.get(str(pair).upper()) or prices.get(str(pair).lower())
                        if isinstance(raw, dict):
                            p = dict(raw)
                        elif raw is not None:
                            p = {"price": raw, "pair": pair}

                    resolved_price = v61_price_for_pair(prices, pair)
                    if not np.isfinite(resolved_price) or resolved_price <= 0:
                        # Last-resort historical close. This keeps discovery
                        # useful when the real-time feed omits one symbol.
                        try:
                            d15_probe = get_tf(pair, "15m", 2)
                            dc_probe = completed(d15_probe)
                            if dc_probe is not None and not dc_probe.empty:
                                resolved_price = safe(dc_probe.iloc[-1].close)
                        except Exception:
                            pass
                    if not np.isfinite(resolved_price) or resolved_price <= 0:
                        continue

                    if p is None:
                        p = {"pair": pair, "price": resolved_price, "lp": resolved_price}
                    else:
                        p.setdefault("pair", pair)
                        p.setdefault("price", resolved_price)
                        p.setdefault("lp", resolved_price)

                    symbol = str(p.get("mkt") or p.get("symbol") or p.get("pair") or pair).upper()
                    if meme_only and not any(w in symbol or w in pair.upper() for w in MEME_WORDS):
                        continue

                    rec = discovery_event(pair, symbol, p, margin)
                    if rec:
                        candidates.append(rec)
                except Exception as exc:
                    failures.append(f"{pair}: {type(exc).__name__}: {exc}")

            # Put actionable extremes first, then strongest movers.
            rank = {
                "🔥 ATH BREAKOUT": 0, "🩸 ATL BREAKDOWN": 0,
                "🟢 NEAR ATH": 1, "🟠 NEAR ATL": 1,
                "🚀 HOT": 2, "🔻 FAST DUMP": 2,
            }
            candidates.sort(key=lambda r: (rank.get(r["Event"], 9), -abs(safe(r["24h"].replace("%", ""), 0))))
            out = candidates[:min(max(peer_limit, 50), 150)]

            if out:
                st.success(f"Found {len(candidates)} current discovery candidates across {len(active)} active {margin} Futures contracts.")
                st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
            else:
                st.warning(f"No current hot/ATH/ATL candidates were found after evaluating {len(active)} active {margin} Futures contracts. This means the market did not meet the configured discovery thresholds, or some history was unavailable.")

            if failures:
                with st.expander(f"Scan diagnostics ({len(failures)} contracts with errors)"):
                    st.code("\n".join(failures[:100]))
    except Exception as e:
        st.error(f"Discovery scan failed: {type(e).__name__}: {e}")

st.divider()
st.caption("Analysis only. No orders, balances, API keys or withdrawals are used. Historical behavior is evidence, not a guarantee or financial advice.")
# =============================================================================
# V6 PROFESSIONAL INTRADAY FUTURES AGENT — ADD-ON
# =============================================================================
# V6 keeps the complete V5 historical-learning engine above and adds a separate
# intraday decision/risk/paper-trading layer. It intentionally does NOT invent
# private CoinDCX order endpoints. Live execution should be connected only after
# the strategy has been backtested and paper-traded successfully.

import json
import os
from pathlib import Path

V6_VERSION = "6.0-INTRADAY"
PAPER_FILE = Path("paper_trades.json")

# ----------------------------- V6 CONFIG -------------------------------------
V6_DEFAULTS = {
    "risk_per_trade_pct": 0.50,
    "max_daily_loss_pct": 2.0,
    "max_open_positions": 3,
    "min_setup_score": 72,
    "min_rr": 2.0,
    "atr_stop_mult": 1.20,
    "max_leverage": 5,
}


def v6_num(v, default=np.nan):
    try:
        z = float(v)
        return z if np.isfinite(z) else default
    except Exception:
        return default


def v6_pct(a, b):
    a, b = v6_num(a), v6_num(b)
    return (a / b - 1.0) * 100 if np.isfinite(a) and np.isfinite(b) and b else np.nan


def v6_last_closed(tf_data, tf):
    d = tf_data.get(tf)
    if d is None or d.empty:
        return None
    x = indicators(completed(d))
    return x.iloc[-1] if not x.empty else None


def v6_swing_levels(d, lookback=80):
    """Recent support/resistance levels from completed candles."""
    if d is None or d.empty:
        return {}
    x = completed(d).tail(lookback)
    if len(x) < 10:
        return {}
    highs = x.high.astype(float)
    lows = x.low.astype(float)
    return {
        "recent_high": float(highs.max()),
        "recent_low": float(lows.min()),
        "last_close": float(x.close.iloc[-1]),
        "last_high": float(x.high.iloc[-1]),
        "last_low": float(x.low.iloc[-1]),
    }


def v6_pivot_levels(d):
    """Previous completed daily OHLC, useful as intraday reference levels."""
    if d is None or len(d) < 3:
        return {}
    x = completed(d)
    if len(x) < 2:
        return {}
    p = x.iloc[-2]
    h, l, c = v6_num(p.high), v6_num(p.low), v6_num(p.close)
    if not all(np.isfinite(v) for v in (h, l, c)):
        return {}
    pivot = (h + l + c) / 3.0
    return {
        "prev_day_high": h,
        "prev_day_low": l,
        "prev_day_close": c,
        "pivot": pivot,
        "r1": 2 * pivot - l,
        "s1": 2 * pivot - h,
        "r2": pivot + (h - l),
        "s2": pivot - (h - l),
    }


def v6_market_regime(tf_data):
    """Determine broad regime before allowing an intraday setup."""
    rows = {}
    bull = bear = 0
    for tf in ("1D", "4H", "1H"):
        r = v6_last_closed(tf_data, tf)
        if r is None:
            rows[tf] = "NO DATA"
            continue
        close = v6_num(r.close)
        e20, e50, e100 = v6_num(r.ema20), v6_num(r.ema50), v6_num(r.ema100)
        macd, sig = v6_num(r.macd), v6_num(r.macd_signal)
        if close > e20 > e50 and e50 > e100 and macd > sig:
            rows[tf] = "BULLISH"
            bull += 1
        elif close < e20 < e50 and e50 < e100 and macd < sig:
            rows[tf] = "BEARISH"
            bear += 1
        else:
            rows[tf] = "MIXED"
    if bull >= 2 and bear == 0:
        regime = "BULL TREND"
    elif bear >= 2 and bull == 0:
        regime = "BEAR TREND"
    elif bull >= 1 and bear == 0:
        regime = "BULLISH / MIXED"
    elif bear >= 1 and bull == 0:
        regime = "BEARISH / MIXED"
    else:
        regime = "RANGE / MIXED"
    return {"regime": regime, "rows": rows, "bull": bull, "bear": bear}


def v6_relative_strength(tf_data):
    """Coin-only momentum quality; BTC/ETH cross-asset feed can be added later."""
    out = {}
    for tf in ("5m", "15m", "1H", "4H"):
        d = completed(tf_data.get(tf, pd.DataFrame()))
        if d is None or len(d) < 5:
            out[tf] = np.nan
            continue
        n = {"5m": 12, "15m": 8, "1H": 6, "4H": 6}[tf]
        if len(d) <= n:
            out[tf] = np.nan
        else:
            out[tf] = v6_pct(d.close.iloc[-1], d.close.iloc[-1-n])
    return out


def v6_volume_quality(d):
    if d is None or len(d) < 25:
        return {"ratio": np.nan, "state": "NO DATA"}
    x = indicators(completed(d))
    if x.empty:
        return {"ratio": np.nan, "state": "NO DATA"}
    r = v6_num(x.iloc[-1].vol_ratio)
    state = "SURGE" if r >= 2 else "STRONG" if r >= 1.3 else "NORMAL" if r >= 0.8 else "THIN"
    return {"ratio": r, "state": state}


def v6_structure_signal(d):
    """More intraday-specific structure classification."""
    if d is None or len(d) < 30:
        return "UNKNOWN"
    x = completed(d).tail(24)
    a = x.iloc[:12]
    b = x.iloc[12:]
    ah, al = a.high.max(), a.low.min()
    bh, bl = b.high.max(), b.low.min()
    if bh > ah and bl > al:
        return "HH/HL"
    if bh < ah and bl < al:
        return "LH/LL"
    return "RANGE"


def v63_structure_engine(d15, d1h, price):
    """True swing-sequence detector for HH/HL and LH/LL structures.

    Unlike the older two-half comparison, this uses confirmed pivot swings and
    requires the latest two swing highs/lows to progress in the same direction.
    It is intentionally conservative: structure is a setup condition, not an
    automatic market order.
    """
    def pivots(d, left=2, right=2, lookback=140):
        if d is None or d.empty:
            return [], []
        x = completed(d).tail(lookback).reset_index(drop=True)
        if len(x) < left + right + 8:
            return [], []
        highs, lows = [], []
        h = pd.to_numeric(x["high"], errors="coerce").to_numpy(float)
        l = pd.to_numeric(x["low"], errors="coerce").to_numpy(float)
        for i in range(left, len(x)-right):
            if np.isfinite(h[i]) and h[i] >= np.nanmax(h[i-left:i+right+1]) and h[i] > h[i-1] and h[i] >= h[i+1]:
                highs.append({"idx": i, "price": float(h[i])})
            if np.isfinite(l[i]) and l[i] <= np.nanmin(l[i-left:i+right+1]) and l[i] < l[i-1] and l[i] <= l[i+1]:
                lows.append({"idx": i, "price": float(l[i])})
        return highs, lows

    h15, l15 = pivots(d15)
    h1, l1 = pivots(d1h, left=2, right=2, lookback=100)
    if len(h15) < 2 or len(l15) < 2 or not np.isfinite(price) or price <= 0:
        return {"state":"INSUFFICIENT STRUCTURE", "score":0, "side":None,
                "hh":False,"hl":False,"lh":False,"ll":False}

    ph0, ph1 = h15[-2], h15[-1]
    pl0, pl1 = l15[-2], l15[-1]
    atr = v61_atr(d15)
    atr_pct = (atr/price*100) if np.isfinite(atr) and price > 0 else 0.0
    min_move = max(0.10, atr_pct * 0.18)
    high_change = (ph1["price"]-ph0["price"])/ph0["price"]*100
    low_change = (pl1["price"]-pl0["price"])/pl0["price"]*100
    hh = high_change >= min_move
    hl = low_change >= min_move
    lh = high_change <= -min_move
    ll = low_change <= -min_move

    # Higher timeframe alignment strengthens, but does not create, the 15m structure.
    h1hh=h1hl=h1lh=h1ll=False
    if len(h1) >= 2 and len(l1) >= 2:
        h1hh = h1[-1]["price"] > h1[-2]["price"]
        h1hl = l1[-1]["price"] > l1[-2]["price"]
        h1lh = h1[-1]["price"] < h1[-2]["price"]
        h1ll = l1[-1]["price"] < l1[-2]["price"]

    bullish = hh and hl
    bearish = lh and ll
    bull_score = 0
    bear_score = 0
    if bullish:
        bull_score = 55
        if h1hh and h1hl: bull_score += 25
        elif h1hh or h1hl: bull_score += 12
        if price >= ph1["price"]: bull_score += 10
        elif price >= pl1["price"]: bull_score += 5
        if high_change >= max(min_move*2, 0.30): bull_score += 5
        if low_change >= max(min_move*2, 0.30): bull_score += 5
    elif bearish:
        bear_score = 55
        if h1lh and h1ll: bear_score += 25
        elif h1lh or h1ll: bear_score += 12
        if price <= pl1["price"]: bear_score += 10
        elif price <= ph1["price"]: bear_score += 5
        if abs(high_change) >= max(min_move*2, 0.30): bear_score += 5
        if abs(low_change) >= max(min_move*2, 0.30): bear_score += 5

    if bullish and bull_score >= 75:
        state, side, score = "CONFIRMED HH/HL", "LONG", min(100,bull_score)
    elif bullish:
        state, side, score = "DEVELOPING HH/HL", "LONG", min(100,bull_score)
    elif bearish and bear_score >= 75:
        state, side, score = "CONFIRMED LH/LL", "SHORT", min(100,bear_score)
    elif bearish:
        state, side, score = "DEVELOPING LH/LL", "SHORT", min(100,bear_score)
    else:
        # One-sided progression is useful as an early warning, but not a trade signal.
        if hh or hl:
            state, side, score = "EARLY BULLISH STRUCTURE", "LONG", 40
        elif lh or ll:
            state, side, score = "EARLY BEARISH STRUCTURE", "SHORT", 40
        else:
            state, side, score = "MIXED STRUCTURE", None, 0

    return {
        "state": state, "side": side, "score": int(score),
        "hh": bool(hh), "hl": bool(hl), "lh": bool(lh), "ll": bool(ll),
        "h1_hh": bool(h1hh), "h1_hl": bool(h1hl), "h1_lh": bool(h1lh), "h1_ll": bool(h1ll),
        "last_high": ph1["price"], "previous_high": ph0["price"],
        "last_low": pl1["price"], "previous_low": pl0["price"],
        "high_change_pct": high_change, "low_change_pct": low_change,
        "atr": atr, "min_move_pct": min_move,
    }



def v63_ema20100_cross(d, recent_bars=6):
    """Detect fresh EMA20/EMA100 crossovers on completed candles.

    A bearish cross is fast EMA moving from >= slow EMA to < slow EMA.
    A bullish cross is the mirror image.  The function also reports whether
    the cross is still fresh and the current EMA spread.  A crossover alone
    is NOT a trade signal; structure, momentum and risk filters must agree.
    """
    if d is None or d.empty or len(d) < 110:
        return {"state":"NO DATA", "bearish":False, "bullish":False,
                "fresh_bearish":False, "fresh_bullish":False, "age_bars":None,
                "spread_pct":np.nan}
    x = indicators(completed(d)).dropna(subset=["ema20","ema100"]).reset_index(drop=True)
    if len(x) < 3:
        return {"state":"NO DATA", "bearish":False, "bullish":False,
                "fresh_bearish":False, "fresh_bullish":False, "age_bars":None,
                "spread_pct":np.nan}
    fast=x["ema20"].astype(float).to_numpy()
    slow=x["ema100"].astype(float).to_numpy()
    diff=fast-slow
    current_bear=bool(diff[-1] < 0)
    current_bull=bool(diff[-1] > 0)
    fresh_bear=fresh_bull=False
    age=None
    look=min(int(recent_bars), len(diff)-1)
    for j in range(1, look+1):
        prev=diff[-j-1]; cur=diff[-j]
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        if prev >= 0 and cur < 0 and not fresh_bear:
            fresh_bear=True; age=j-1
        if prev <= 0 and cur > 0 and not fresh_bull:
            fresh_bull=True; age=j-1
    price=v6_num(x.iloc[-1].close)
    spread=(diff[-1]/price*100) if np.isfinite(price) and price>0 else np.nan
    if fresh_bear:
        state="FRESH BEARISH EMA20/100 CROSS"
    elif fresh_bull:
        state="FRESH BULLISH EMA20/100 CROSS"
    elif current_bear:
        state="EMA20 BELOW EMA100"
    elif current_bull:
        state="EMA20 ABOVE EMA100"
    else:
        state="EMA20/100 FLAT"
    return {"state":state, "bearish":current_bear, "bullish":current_bull,
            "fresh_bearish":fresh_bear, "fresh_bullish":fresh_bull,
            "age_bars":age, "spread_pct":spread}


def v63_ema_cross_context(tf_data):
    """Combine the 15m and 4H EMA20/100 states, weighting 4H more heavily."""
    c15=v63_ema20100_cross(tf_data.get("15m"), recent_bars=6)
    c4=v63_ema20100_cross(tf_data.get("4H"), recent_bars=4)
    bear_points=0; bull_points=0; reasons=[]
    if c15.get("fresh_bearish"): bear_points += 18; reasons.append("fresh 15m EMA20 crossed below EMA100")
    elif c15.get("bearish"): bear_points += 5
    if c4.get("fresh_bearish"): bear_points += 28; reasons.append("fresh 4H EMA20 crossed below EMA100")
    elif c4.get("bearish"): bear_points += 8
    if c15.get("fresh_bullish"): bull_points += 18; reasons.append("fresh 15m EMA20 crossed above EMA100")
    elif c15.get("bullish"): bull_points += 5
    if c4.get("fresh_bullish"): bull_points += 28; reasons.append("fresh 4H EMA20 crossed above EMA100")
    elif c4.get("bullish"): bull_points += 8
    if c15.get("fresh_bearish") and c4.get("fresh_bearish"):
        bear_points += 10; reasons.append("15m + 4H bearish EMA20/100 alignment")
    if c15.get("fresh_bullish") and c4.get("fresh_bullish"):
        bull_points += 10; reasons.append("15m + 4H bullish EMA20/100 alignment")
    return {"15m":c15,"4H":c4,"bear_score":min(100,bear_points),
            "bull_score":min(100,bull_points),"reasons":reasons}

def v63_structure_trade(pair, symbol, price, structure, direction):
    """Create a structure-based alert with breakout/pullback trigger levels."""
    atr = v6_num(structure.get("atr"))
    if not np.isfinite(atr) or atr <= 0 or price <= 0:
        return None
    if direction == "LONG":
        hl = v6_num(structure.get("last_low"))
        hh = v6_num(structure.get("last_high"))
        trigger = hh + 0.12*atr
        stop = hl - 0.25*atr
        risk = trigger-stop
        if risk <= 0: return None
        tp1, tp2 = trigger + 2*risk, trigger + 3*risk
        near_pullback = abs(price-hl) <= max(0.75*atr, price*0.004)
        triggered = price >= trigger
        status = "LONG NOW" if triggered else ("LONG PULLBACK ZONE" if near_pullback else "WAIT FOR HH BREAK")
        return {"pair":pair,"symbol":symbol,"side":"LONG","direction":"LONG","type":"HH/HL STRUCTURE",
                "status":status,"score":structure["score"],"entry":trigger,"stop":stop,"tp1":tp1,"tp2":tp2,
                "rr1":2.0,"rr2":3.0,"support":hl,"resistance":hh,
                "reason":f"15m {structure['state']} | HH +{structure['high_change_pct']:.2f}% | HL +{structure['low_change_pct']:.2f}%"}
    lh = v6_num(structure.get("last_high")); ll = v6_num(structure.get("last_low"))
    trigger = ll - 0.12*atr
    stop = lh + 0.25*atr
    risk = stop-trigger
    if risk <= 0: return None
    tp1, tp2 = trigger - 2*risk, trigger - 3*risk
    near_pullback = abs(price-lh) <= max(0.75*atr, price*0.004)
    triggered = price <= trigger
    status = "SHORT NOW" if triggered else ("SHORT PULLBACK ZONE" if near_pullback else "WAIT FOR LL BREAK")
    return {"pair":pair,"symbol":symbol,"side":"SHORT","direction":"SHORT","type":"LH/LL STRUCTURE",
            "status":status,"score":structure["score"],"entry":trigger,"stop":stop,"tp1":tp1,"tp2":tp2,
            "rr1":2.0,"rr2":3.0,"support":ll,"resistance":lh,
            "reason":f"15m {structure['state']} | LH {structure['high_change_pct']:.2f}% | LL {structure['low_change_pct']:.2f}%"}


def v6_squeeze_breakout(d):
    """Detect compression followed by a range/volume expansion."""
    if d is None or len(d) < 35:
        return {"state": "UNKNOWN", "score": 0}
    x = indicators(completed(d))
    if len(x) < 30:
        return {"state": "UNKNOWN", "score": 0}
    now = x.iloc[-1]
    prior = x.iloc[-8:-1]
    width_now = v6_num(now.bbup - now.bblow)
    width_prior = v6_num((prior.bbup - prior.bblow).median())
    vol = v6_num(now.vol_ratio)
    rng = v6_num(now.high - now.low)
    atr = v6_num(now.atr)
    score = 0
    if np.isfinite(width_now) and np.isfinite(width_prior) and width_prior > 0 and width_now > width_prior * 1.15:
        score += 35
    if np.isfinite(vol) and vol >= 1.5:
        score += 35
    if np.isfinite(rng) and np.isfinite(atr) and atr > 0 and rng >= atr * 1.2:
        score += 30
    return {"state": "EXPANSION" if score >= 60 else "NO CLEAR EXPANSION", "score": score}


def v6_trade_levels(tf_data, direction, price, cfg):
    """ATR/structure-based entry, invalidation and multi-target levels."""
    d = completed(tf_data.get("15m", pd.DataFrame()))
    x = indicators(d) if d is not None and not d.empty else pd.DataFrame()
    if x.empty:
        return None
    r = x.iloc[-1]
    atr = v6_num(r.atr)
    if not np.isfinite(atr) or atr <= 0 or price <= 0:
        return None
    swing = v6_swing_levels(d, 32)
    pad = atr * float(cfg["atr_stop_mult"])
    if direction == "LONG":
        structural = min(v6_num(swing.get("recent_low"), price - pad), price - pad)
        stop = structural - 0.10 * atr
        risk = price - stop
        if risk <= 0:
            return None
        tp1, tp2, tp3 = price + 1.5*risk, price + 2.5*risk, price + 4.0*risk
    else:
        structural = max(v6_num(swing.get("recent_high"), price + pad), price + pad)
        stop = structural + 0.10 * atr
        risk = stop - price
        if risk <= 0:
            return None
        tp1, tp2, tp3 = price - 1.5*risk, price - 2.5*risk, price - 4.0*risk
    return {
        "entry": price, "stop": stop, "risk_per_unit": risk,
        "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "rr_tp1": 1.5, "rr_tp2": 2.5, "rr_tp3": 4.0,
        "atr": atr,
    }


def v6_setup_engine(current, direction, cfg):
    """Professional-style deterministic LONG/SHORT scoring; V5 remains a vote."""
    tf = current.get("tf_data", {})
    r5 = v6_last_closed(tf, "5m")
    r15 = v6_last_closed(tf, "15m")
    r1 = v6_last_closed(tf, "1H")
    r4 = v6_last_closed(tf, "4H")
    if any(r is None for r in (r15, r1, r4)):
        return None

    price = v6_num(current.get("current"))
    if price <= 0:
        return None
    regime = v6_market_regime(tf)
    s15 = short_term_state(current)
    struct15 = v6_structure_signal(tf.get("15m"))
    volq = v6_volume_quality(tf.get("15m"))
    expansion = v6_squeeze_breakout(tf.get("15m"))
    rs = v6_relative_strength(tf)
    piv = v6_pivot_levels(tf.get("1D"))

    score = 0
    reasons = []
    blockers = []

    # Directional higher-timeframe agreement: 25 points.
    if direction == "LONG":
        if regime["regime"].startswith("BULL"):
            score += 15; reasons.append("higher-timeframe regime supports LONG")
        if v6_num(r4.close) > v6_num(r4.ema20) and v6_num(r1.close) > v6_num(r1.ema20):
            score += 10; reasons.append("1H and 4H are above EMA20")
        if struct15 == "HH/HL":
            score += 15; reasons.append("15m structure is HH/HL")
        if v6_num(r15.macd) > v6_num(r15.macd_signal):
            score += 8; reasons.append("15m MACD bullish")
        if v6_num(r15.ema20) > v6_num(r15.ema50):
            score += 8; reasons.append("15m EMA20 > EMA50")
        if v6_num(r15.rsi) >= 50:
            score += 5; reasons.append("15m RSI has bullish momentum")
        if volq["ratio"] >= 1.2:
            score += 8; reasons.append("15m volume confirms participation")
        if expansion["score"] >= 60:
            score += 6; reasons.append("range/volume expansion detected")
        if s15.get("reversal_confirmed"):
            blockers.append("15m reversal is confirmed")
        if regime["bear"] >= 2:
            blockers.append("higher-timeframe regime is bearish")
    else:
        if regime["regime"].startswith("BEAR"):
            score += 15; reasons.append("higher-timeframe regime supports SHORT")
        if v6_num(r4.close) < v6_num(r4.ema20) and v6_num(r1.close) < v6_num(r1.ema20):
            score += 10; reasons.append("1H and 4H are below EMA20")
        if struct15 == "LH/LL":
            score += 15; reasons.append("15m structure is LH/LL")
        if v6_num(r15.macd) < v6_num(r15.macd_signal):
            score += 8; reasons.append("15m MACD bearish")
        if v6_num(r15.ema20) < v6_num(r15.ema50):
            score += 8; reasons.append("15m EMA20 < EMA50")
        if v6_num(r15.rsi) <= 50:
            score += 5; reasons.append("15m RSI has bearish momentum")
        if volq["ratio"] >= 1.2:
            score += 8; reasons.append("15m volume confirms participation")
        if expansion["score"] >= 60:
            score += 6; reasons.append("range/volume expansion detected")
        if s15.get("reversal_confirmed"):
            score += 12; reasons.append("V5 short-term reversal confirmation supports SHORT")
        if regime["bull"] >= 2:
            blockers.append("higher-timeframe regime is bullish")

    # V5 historical evidence: 15 points maximum.
    last_summary = st.session_state.get("last_analysis", {}).get("summary")
    if last_summary:
        c = v6_num(last_summary.get("continue_pct"), 0)
        d = v6_num(last_summary.get("dump_pct"), 0)
        if direction == "LONG" and c >= 58 and c - d >= 15:
            score += 15; reasons.append(f"V5 historical continuation edge ({c:.0f}% continue)")
        elif direction == "SHORT" and c >= 58 and c - last_summary.get("reverse_pct", 0) >= 15:
            score += 15; reasons.append(f"V5 historical downside continuation evidence ({c:.0f}%)")
        elif direction == "SHORT" and last_summary.get("reverse_pct", 0) >= 55:
            blockers.append("V5 history shows elevated bounce risk")

    # Avoid chasing extremely extended candles.
    rsi15 = v6_num(r15.rsi)
    dist20 = v6_pct(r15.close, r15.ema20)
    if direction == "LONG" and np.isfinite(rsi15) and rsi15 >= 82:
        blockers.append("LONG is too extended (15m RSI >= 82)")
    if direction == "SHORT" and np.isfinite(rsi15) and rsi15 <= 18:
        blockers.append("SHORT is too extended (15m RSI <= 18)")
    if direction == "LONG" and np.isfinite(dist20) and dist20 >= 15:
        blockers.append("LONG is stretched far above EMA20")
    if direction == "SHORT" and np.isfinite(dist20) and dist20 <= -15:
        blockers.append("SHORT is stretched far below EMA20")

    levels = v6_trade_levels(tf, direction, price, cfg)
    if not levels:
        return None
    valid = score >= int(cfg["min_setup_score"]) and not blockers and levels["rr_tp2"] >= float(cfg["min_rr"])
    return {
        "direction": direction, "score": int(min(score, 100)), "valid": bool(valid),
        "regime": regime["regime"], "regime_rows": regime["rows"],
        "structure15": struct15, "volume": volq, "expansion": expansion,
        "relative_strength": rs, "pivot_levels": piv,
        "entry": levels["entry"], "stop": levels["stop"],
        "tp1": levels["tp1"], "tp2": levels["tp2"], "tp3": levels["tp3"],
        "risk_per_unit": levels["risk_per_unit"], "atr": levels["atr"],
        "reasons": reasons, "blockers": blockers,
    }


def v6_position_size(balance, entry, stop, risk_pct, leverage=5):
    """Risk-based position sizing; leverage caps exposure but does not define risk."""
    balance = v6_num(balance, 0)
    entry, stop = v6_num(entry), v6_num(stop)
    if balance <= 0 or entry <= 0 or stop <= 0 or entry == stop:
        return 0.0
    risk_cash = balance * float(risk_pct) / 100.0
    per_unit = abs(entry - stop)
    qty = risk_cash / per_unit
    max_notional_qty = (balance * float(leverage)) / entry
    return max(0.0, min(qty, max_notional_qty))


def v6_load_paper():
    try:
        if PAPER_FILE.exists():
            data = json.loads(PAPER_FILE.read_text())
            return data if isinstance(data, list) else []
    except Exception:
        pass
    return []


def v6_save_paper(rows):
    try:
        PAPER_FILE.write_text(json.dumps(rows, indent=2, default=str))
    except Exception:
        pass


def v6_paper_open(setup, balance, risk_pct, leverage, pair=None, symbol=None, source="MANUAL"):
    """Open a paper position without requiring the V5 last_analysis selection.

    The optional pair/symbol/source fields let the autonomous scanner open trades
    directly from the market-wide scan while preserving the original manual flow.
    """
    rows = v6_load_paper()
    pair = pair or setup.get("pair") or st.session_state.get("last_analysis", {}).get("pair", "")
    symbol = symbol or setup.get("symbol") or pair
    if not pair:
        return None
    # Never stack duplicate exposure on the same contract while an earlier paper
    # position is still open.
    if any(r.get("status") == "OPEN" and str(r.get("pair")) == str(pair) for r in rows):
        return None
    qty = v6_position_size(balance, setup["entry"], setup["stop"], risk_pct, leverage)
    if qty <= 0:
        return None
    now = datetime.now(timezone.utc)
    trade = {
        "id": now.strftime("%Y%m%d%H%M%S%f"),
        "time": now.isoformat(),
        "pair": pair,
        "symbol": symbol,
        "direction": setup["direction"],
        "score": setup["score"],
        "entry": setup["entry"], "stop": setup["stop"],
        "tp1": setup["tp1"], "tp2": setup["tp2"], "tp3": setup["tp3"],
        "qty": qty, "status": "OPEN", "pnl": 0.0,
        "last_price": setup["entry"],
        "tp1_hit": False, "tp2_hit": False,
        "source": source,
        "reason": " | ".join(setup.get("reasons", [])[:5]),
    }
    rows.append(trade)
    v6_save_paper(rows)
    return trade


def v6_paper_update(pair, price):
    """Update one open paper position and record TP milestones/exit state."""
    rows = v6_load_paper()
    changed = False
    for t in rows:
        if t.get("status") != "OPEN" or str(t.get("pair")) != str(pair):
            continue
        p = v6_num(price)
        entry = v6_num(t.get("entry")); qty = v6_num(t.get("qty"), 0)
        stop = v6_num(t.get("stop")); tp1 = v6_num(t.get("tp1")); tp2 = v6_num(t.get("tp2")); tp3 = v6_num(t.get("tp3"))
        direction = t.get("direction")
        if not np.isfinite(p) or p <= 0:
            continue
        t["last_price"] = p
        t["unrealized_pnl"] = ((p-entry) * qty if direction == "LONG" else (entry-p) * qty)
        if direction == "LONG":
            t["tp1_hit"] = bool(t.get("tp1_hit")) or p >= tp1
            t["tp2_hit"] = bool(t.get("tp2_hit")) or p >= tp2
            hit_stop = p <= stop
            hit_tp3 = p >= tp3
        else:
            t["tp1_hit"] = bool(t.get("tp1_hit")) or p <= tp1
            t["tp2_hit"] = bool(t.get("tp2_hit")) or p <= tp2
            hit_stop = p >= stop
            hit_tp3 = p <= tp3
        if hit_stop or hit_tp3:
            raw = (p-entry) * qty if direction == "LONG" else (entry-p) * qty
            t["exit"] = p; t["pnl"] = raw
            t["status"] = "STOP" if hit_stop else "TP3"
            t["closed_at"] = datetime.now(timezone.utc).isoformat()
        changed = True
    if changed:
        v6_save_paper(rows)
    return rows


def v6_daily_guard(balance, starting_balance, rows, cfg):
    """Kill switch based on today's closed paper PnL and the configured daily loss limit."""
    if starting_balance <= 0:
        return True, "No valid starting balance"
    today = datetime.now(timezone.utc).date().isoformat()
    closed = [r for r in rows if r.get("status") != "OPEN" and str(r.get("closed_at", "")).startswith(today)]
    pnl = sum(v6_num(r.get("pnl"), 0) for r in closed)
    loss_pct = max(0.0, -pnl / starting_balance * 100)
    if loss_pct >= float(cfg["max_daily_loss_pct"]):
        return False, f"DAILY KILL SWITCH: closed loss {loss_pct:.2f}% >= {cfg['max_daily_loss_pct']:.2f}%"
    return True, f"Daily risk OK: closed PnL {pnl:+.2f}"


# =============================================================================
# V6 AUTONOMOUS PAPER-TRADING LOOP
# =============================================================================
def v6_resolve_live_price(prices, pair):
    """Resolve a Futures price despite CoinDCX feed-key formatting differences."""
    if not pair:
        return None
    keys = [str(pair), str(pair).upper(), str(pair).lower()]
    for k in keys:
        if k in prices:
            return current_price(prices[k])
    target = str(pair).upper().replace("-", "").replace("_", "")
    for k, rec in (prices or {}).items():
        kk = str(k).upper().replace("-", "").replace("_", "")
        if kk == target:
            return current_price(rec)
    return None


def v6_run_market_scan(scan_limit, cfg):
    """Run the same V6 deterministic scan used by the manual button."""
    prices = futures_prices()
    active = active_instruments(margin)
    universe = []
    for pair in active:
        p = prices.get(pair) or prices.get(str(pair).upper()) or prices.get(str(pair).lower())
        if not p:
            # Some CoinDCX feeds use a differently formatted contract key.
            target = str(pair).upper().replace("-", "").replace("_", "")
            p = next((rec for k, rec in (prices or {}).items()
                      if str(k).upper().replace("-", "").replace("_", "") == target), None)
        if not p:
            continue
        symbol = str(p.get("mkt", pair)).upper()
        if meme_only and not any(w in symbol or w in str(pair).upper() for w in MEME_WORDS):
            continue
        cur = current_price(p)
        pc = v6_num(p.get("pc"), 0)
        if cur > 0:
            universe.append((pair, symbol, p, cur, abs(pc)))
    universe.sort(key=lambda z: z[4], reverse=True)
    universe = universe[:int(scan_limit)]

    records, failures = [], []
    for pair, symbol, p, cur, _ in universe:
        try:
            tf_data = {tf: get_tf(pair, tf, days) for tf, days in {
                "5m": 5, "15m": 12, "1H": 30, "4H": 120, "1D": 180
            }.items()}
            dummy = {"tf_data": tf_data, "current": cur, "pair": pair, "symbol": symbol}
            for direction in ("LONG", "SHORT"):
                setup = v6_setup_engine(dummy, direction, cfg)
                if not setup:
                    continue
                setup = dict(setup)
                setup["pair"] = pair
                setup["symbol"] = symbol
                records.append({
                    "Coin": symbol, "Pair": pair, "Direction": setup["direction"],
                    "Score": setup["score"],
                    "Valid": "✅ TRADE CANDIDATE" if setup["valid"] else "WAIT",
                    "Regime": setup["regime"], "15m Structure": setup["structure15"],
                    "Volume": f"{setup['volume']['ratio']:.1f}x" if np.isfinite(setup['volume']['ratio']) else "—",
                    "Entry": fmt(setup["entry"]), "Stop": fmt(setup["stop"]),
                    "TP1": fmt(setup["tp1"]), "TP2": fmt(setup["tp2"]), "TP3": fmt(setup["tp3"]),
                    "Blockers": "; ".join(setup["blockers"]) if setup["blockers"] else "—",
                    "Reasons": " | ".join(setup["reasons"][:4]), "setup": setup,
                })
        except Exception as exc:
            failures.append(f"{pair}: {type(exc).__name__}: {exc}")
    records.sort(key=lambda r: (r["Valid"] != "✅ TRADE CANDIDATE", -r["Score"]))
    return records, failures, len(universe)


def v6_autonomous_paper_cycle(balance, risk_pct, leverage, cfg, scan_limit, max_new_trades=1):
    """Scan, manage existing paper positions, and automatically open new valid setups."""
    # First mark existing positions using the freshest public Futures prices.
    prices = futures_prices()
    rows = v6_load_paper()
    for t in rows:
        if t.get("status") != "OPEN":
            continue
        live = v6_resolve_live_price(prices, t.get("pair"))
        if live is not None:
            v6_paper_update(t.get("pair"), live)
    rows = v6_load_paper()

    records, failures, scanned = v6_run_market_scan(scan_limit, cfg)
    valid = [r for r in records if r["Valid"] == "✅ TRADE CANDIDATE"]
    # Only one position per contract; if both directions qualify, take the stronger one.
    best_by_pair = {}
    for r in valid:
        pair = r["Pair"]
        if pair not in best_by_pair or r["Score"] > best_by_pair[pair]["Score"]:
            best_by_pair[pair] = r
    candidates = sorted(best_by_pair.values(), key=lambda r: r["Score"], reverse=True)

    opened, skipped = [], []
    open_pairs = {str(r.get("pair")) for r in rows if r.get("status") == "OPEN"}
    open_count = len(open_pairs)
    guard_ok, guard_msg = v6_daily_guard(balance, balance, rows, cfg)
    if not guard_ok:
        return records, failures, scanned, opened, [guard_msg], guard_msg

    for r in candidates:
        if len(opened) >= int(max_new_trades):
            skipped.append(f"{r['Coin']}: per-cycle auto-trade limit reached")
            break
        if open_count >= int(cfg["max_open_positions"]):
            skipped.append(f"{r['Coin']}: max open positions reached")
            break
        pair = r["Pair"]
        if str(pair) in open_pairs:
            skipped.append(f"{r['Coin']}: position already open")
            continue
        trade = v6_paper_open(r["setup"], balance, risk_pct, leverage,
                              pair=pair, symbol=r["Coin"], source="AUTONOMOUS")
        if trade:
            opened.append(trade)
            open_pairs.add(str(pair))
            open_count += 1

    msg = f"Auto cycle: scanned {scanned} contracts | {len(valid)} valid setups | opened {len(opened)} paper trade(s)"
    return records, failures, scanned, opened, skipped, msg


# =============================================================================
# V6 UI — runs after the original V5 UI, so the original functionality remains.
# =============================================================================
st.divider()
st.header("🤖 V6 Professional Intraday Futures Agent")
st.caption("V5 historical learning + multi-timeframe regime + LONG/SHORT scoring + ATR risk engine + paper execution.")

with st.expander("⚙️ V6 Risk Controls", expanded=False):
    vc1, vc2, vc3, vc4 = st.columns(4)
    v6_balance = vc1.number_input("Paper balance", min_value=100.0, value=1000.0, step=100.0)
    v6_risk = vc2.number_input("Risk / trade %", min_value=0.05, max_value=5.0, value=V6_DEFAULTS["risk_per_trade_pct"], step=0.05)
    v6_daily = vc3.number_input("Max daily loss %", min_value=0.25, max_value=20.0, value=V6_DEFAULTS["max_daily_loss_pct"], step=0.25)
    v6_lev = vc4.number_input("Max leverage", min_value=1, max_value=20, value=V6_DEFAULTS["max_leverage"], step=1)
    v6_cfg = dict(V6_DEFAULTS)
    v6_cfg.update({"risk_per_trade_pct": v6_risk, "max_daily_loss_pct": v6_daily, "max_leverage": v6_lev})
    st.warning("V6 is PAPER-TRADING ONLY. No live order is submitted by this add-on.")

v6_col1, v6_col2 = st.columns([1, 1])
with v6_col1:
    v6_scan_limit = st.slider("Intraday scan contracts", 10, 60, 25, 5)
with v6_col2:
    v6_min_score = st.slider("Minimum setup score", 60, 90, V6_DEFAULTS["min_setup_score"], 1)
v6_cfg["min_setup_score"] = v6_min_score

a1, a2, a3 = st.columns(3)
with a1:
    v6_auto_paper = st.checkbox("🤖 Autonomous Paper Trading", value=False, key="v6_auto_paper")
with a2:
    v6_auto_minutes = st.selectbox("Auto scan interval", [1, 5, 10, 15], index=1, key="v6_auto_minutes")
with a3:
    v6_auto_limit = st.slider("Max auto trades / cycle", 1, 3, 1, key="v6_auto_limit")

if v6_auto_paper:
    st.warning("AUTONOMOUS PAPER MODE: the agent will scan the V6 universe, open qualifying paper trades automatically, and update open positions. No live CoinDCX order is sent.")

if st.button("🚦 Run Professional LONG / SHORT Scan", type="primary"):
    try:
        with st.spinner("Scanning CoinDCX Futures with V6 intraday filters..."):
            prices = futures_prices()
            active = active_instruments(margin)
            universe = []
            for pair in active:
                p = prices.get(pair) or prices.get(str(pair).upper()) or prices.get(str(pair).lower())
                if not p:
                    continue
                symbol = str(p.get("mkt", pair)).upper()
                if meme_only and not any(w in symbol or w in pair.upper() for w in MEME_WORDS):
                    continue
                cur = current_price(p)
                pc = v6_num(p.get("pc"), 0)
                if cur > 0:
                    universe.append((pair, symbol, p, cur, abs(pc)))
            universe.sort(key=lambda z: z[4], reverse=True)
            universe = universe[:v6_scan_limit]

            records = []
            failures = []
            for pair, symbol, p, cur, _ in universe:
                try:
                    # Reuse the same public candle functions already used by V5.
                    tf_data = {tf: get_tf(pair, tf, days) for tf, days in {
                        "5m": 5, "15m": 12, "1H": 30, "4H": 120, "1D": 180
                    }.items()}
                    dummy = {"tf_data": tf_data, "current": cur}
                    long_setup = v6_setup_engine(dummy, "LONG", v6_cfg)
                    short_setup = v6_setup_engine(dummy, "SHORT", v6_cfg)
                    for setup in (long_setup, short_setup):
                        if not setup:
                            continue
                        records.append({
                            "Coin": symbol, "Pair": pair,
                            "Direction": setup["direction"],
                            "Score": setup["score"],
                            "Valid": "✅ TRADE CANDIDATE" if setup["valid"] else "WAIT",
                            "Regime": setup["regime"],
                            "15m Structure": setup["structure15"],
                            "Volume": f"{setup['volume']['ratio']:.1f}x" if np.isfinite(setup['volume']['ratio']) else "—",
                            "Entry": fmt(setup["entry"]),
                            "Stop": fmt(setup["stop"]),
                            "TP1": fmt(setup["tp1"]),
                            "TP2": fmt(setup["tp2"]),
                            "TP3": fmt(setup["tp3"]),
                            "Blockers": "; ".join(setup["blockers"]) if setup["blockers"] else "—",
                            "Reasons": " | ".join(setup["reasons"][:4]),
                            "setup": setup,
                        })
                except Exception as exc:
                    failures.append(f"{pair}: {type(exc).__name__}: {exc}")

            records.sort(key=lambda r: (r["Valid"] != "✅ TRADE CANDIDATE", -r["Score"]))
            st.session_state["v6_scan"] = records
            st.session_state["v6_scan_failures"] = failures

    except Exception as exc:
        st.error(f"V6 scan failed: {type(exc).__name__}: {exc}")

# The fragment reruns independently, so autonomous paper trading does not require
# repeatedly clicking the manual scan button. Streamlit >=1.37 supports fragments.
if v6_auto_paper:
    if hasattr(st, "fragment"):
        @st.fragment(run_every=f"{int(v6_auto_minutes)}m")
        def _v6_autonomous_runner():
            with st.status("🤖 Autonomous paper cycle running…", expanded=False):
                try:
                    auto_cfg = dict(v6_cfg)
                    recs, fails, scanned, opened, skipped, msg = v6_autonomous_paper_cycle(
                        v6_balance, v6_risk, v6_lev, auto_cfg, v6_scan_limit, v6_auto_limit
                    )
                    st.session_state["v6_scan"] = recs
                    st.session_state["v6_scan_failures"] = fails
                    st.session_state["v6_auto_last"] = datetime.now(timezone.utc).isoformat()
                    st.session_state["v6_auto_message"] = msg
                    if opened:
                        st.success("AUTO-OPENED: " + ", ".join(
                            f"{t.get('symbol', t.get('pair'))} {t.get('direction')} {t.get('score')}/100" for t in opened
                        ))
                    else:
                        st.info(msg)
                except Exception as exc:
                    st.error(f"Autonomous paper cycle failed: {type(exc).__name__}: {exc}")
        _v6_autonomous_runner()
    else:
        st.error("Autonomous mode requires a recent Streamlit version with st.fragment. Manual paper trading remains available.")

v6_records = st.session_state.get("v6_scan", [])
if v6_records:
    st.subheader("📋 Ranked Intraday Opportunities")
    display_cols = ["Coin","Pair","Direction","Score","Valid","Regime","15m Structure","Volume","Entry","Stop","TP1","TP2","TP3","Blockers"]
    st.dataframe(pd.DataFrame([{k:r[k] for k in display_cols} for r in v6_records]), use_container_width=True, hide_index=True)

    valid = [r for r in v6_records if r["Valid"] == "✅ TRADE CANDIDATE"]
    if valid:
        st.success(f"{len(valid)} setup(s) passed the V6 deterministic gate. Review the trade card before paper execution.")
        pick = st.selectbox("Select setup", range(len(valid)), format_func=lambda i: f"{valid[i]['Coin']} — {valid[i]['Direction']} — {valid[i]['Score']}/100")
        chosen = valid[pick]
        setup = chosen["setup"]
        st.markdown("### 🎯 Professional Trade Card")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Direction", setup["direction"])
        c2.metric("Score", f"{setup['score']}/100")
        c3.metric("Entry", fmt(setup["entry"]))
        c4.metric("Stop", fmt(setup["stop"]))
        c5.metric("TP2", fmt(setup["tp2"]))
        q1,q2,q3,q4 = st.columns(4)
        q1.metric("Risk / unit", fmt(setup["risk_per_unit"]))
        q2.metric("TP1 R", "1.5R")
        q3.metric("TP2 R", "2.5R")
        q4.metric("TP3 R", "4.0R")
        qty = v6_position_size(v6_balance, setup["entry"], setup["stop"], v6_risk, v6_lev)
        st.write(f"**Paper quantity:** {qty:.6f} | **Cash risk:** ~{v6_balance*v6_risk/100:.2f} | **Max leverage:** {v6_lev}x")
        st.write("**Why:** " + "; ".join(setup["reasons"]))
        if setup["blockers"]:
            st.error("BLOCKED: " + "; ".join(setup["blockers"]))
        else:
            if st.button("🧪 Open PAPER Trade", type="secondary"):
                rows = v6_load_paper()
                open_count = sum(r.get("status") == "OPEN" for r in rows)
                if open_count >= int(v6_cfg["max_open_positions"]):
                    st.error(f"Max open positions reached ({v6_cfg['max_open_positions']}).")
                else:
                    ok, guard = v6_daily_guard(v6_balance, v6_balance, rows, v6_cfg)
                    if not ok:
                        st.error(guard)
                    else:
                        trade = v6_paper_open(setup, v6_balance, v6_risk, v6_lev, pair=chosen.get("Pair"), symbol=chosen.get("Coin"), source="MANUAL")
                        if trade:
                            st.success(f"Paper trade opened: {trade['direction']} {trade['pair']} qty {trade['qty']:.6f}")

if st.session_state.get("v6_scan_failures"):
    with st.expander(f"V6 scan diagnostics ({len(st.session_state['v6_scan_failures'])})"):
        st.code("\n".join(st.session_state["v6_scan_failures"][:100]))

st.markdown("### 🧪 Paper Position Manager")
if v6_auto_paper:
    st.caption(f"🤖 Autonomous mode ON | Interval: {v6_auto_minutes} min | Last cycle: {st.session_state.get('v6_auto_last', 'waiting for first cycle')}")
paper_rows = v6_load_paper()
if paper_rows:
    pactive = [r for r in paper_rows if r.get("status") == "OPEN"]
    st.write(f"Open paper positions: **{len(pactive)}**")
    st.dataframe(pd.DataFrame(paper_rows), use_container_width=True, hide_index=True)
    if st.button("🔄 Refresh / Update Current Paper Prices"):
        prices = futures_prices()
        for t in paper_rows:
            if t.get("status") != "OPEN":
                continue
            live = v6_resolve_live_price(prices, t.get("pair"))
            if live is not None:
                v6_paper_update(t.get("pair"), live)
        st.rerun()
else:
    st.info("No paper trades yet. Run the V6 scan first.")

st.caption(f"V{V6_VERSION}: existing V5 engine retained above; V6 adds deterministic intraday analysis, risk sizing and paper trade management. Live execution is intentionally disabled until exchange order integration is explicitly implemented and verified.")

# =============================================================================
# V6.2 AUTONOMOUS HISTORICAL PATTERN LEARNING ENGINE
# =============================================================================
# This layer turns the scanner into a market-wide pattern learner.
# It does NOT train an LLM. It stores historical CoinDCX Futures candles,
# converts recurring market states into feature vectors, records what happened
# next, and uses similar historical states to improve the current signal score.
# The learner is deliberately local (SQLite) and does not place orders.

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

V62_VERSION = "6.2-LEARNING"
V62_DB = str(Path(__file__).with_name("coindcx_pattern_learning.db")) if "Path" in globals() else "coindcx_pattern_learning.db"
V62_LOCK = threading.Lock()
V62_FEATURES = [
    "ret_1h", "ret_4h", "ret_12h", "ret_24h",
    "rsi", "macd_atr", "ema20_gap", "ema50_gap", "ema200_gap",
    "atr_pct", "vol_ratio", "adx", "bb_pos", "range_pos",
    "body_pct", "upper_wick", "lower_wick", "trend_score"
]


def v62_db():
    con = sqlite3.connect(V62_DB, timeout=30, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS pattern_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pair TEXT NOT NULL,
            ts TEXT NOT NULL,
            side_bias TEXT NOT NULL,
            ret_1h REAL, ret_4h REAL, ret_12h REAL, ret_24h REAL,
            rsi REAL, macd_atr REAL, ema20_gap REAL, ema50_gap REAL, ema200_gap REAL,
            atr_pct REAL, vol_ratio REAL, adx REAL, bb_pos REAL, range_pos REAL,
            body_pct REAL, upper_wick REAL, lower_wick REAL, trend_score REAL,
            long_max_4h REAL, long_min_4h REAL, short_max_4h REAL, short_min_4h REAL,
            long_win INTEGER, short_win INTEGER,
            UNIQUE(pair, ts)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_pattern_ts ON pattern_samples(ts)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pattern_pair_ts ON pattern_samples(pair, ts)")
    con.commit()
    return con


def v62_pct_raw(a, b):
    try:
        a, b = float(a), float(b)
        if not np.isfinite(a) or not np.isfinite(b) or b == 0:
            return np.nan
        return (a / b - 1.0) * 100.0
    except Exception:
        return np.nan


def v62_resample_ohlcv(d, rule):
    if d is None or d.empty:
        return pd.DataFrame()
    x = d.copy().set_index("time")
    y = x.resample(rule).agg({
        "open":"first", "high":"max", "low":"min", "close":"last", "volume":"sum"
    }).dropna().reset_index()
    return y


def v62_feature_row(x, i):
    """Build a normalized market-state vector from a completed 15m candle."""
    if x is None or len(x) < 220 or i < 205 or i >= len(x):
        return None
    z = x.iloc[:i+1]
    a = indicators(z).iloc[-1]
    close = v6_num(a.close)
    atr = v6_num(a.atr)
    if not np.isfinite(close) or close <= 0 or not np.isfinite(atr) or atr <= 0:
        return None

    def ret(n):
        if len(z) <= n:
            return np.nan
        return v62_pct_raw(close, z.iloc[-n-1].close)

    hi = float(z.high.tail(96).max())
    lo = float(z.low.tail(96).min())
    bb_up = v6_num(a.bbup)
    bb_lo = v6_num(a.bblow)
    bb_pos = (close-bb_lo)/(bb_up-bb_lo) if np.isfinite(bb_up) and np.isfinite(bb_lo) and bb_up > bb_lo else 0.5
    range_pos = (close-lo)/(hi-lo) if hi > lo else 0.5
    op = v6_num(a.open)
    high = v6_num(a.high)
    low = v6_num(a.low)
    body_pct = abs(close-op)/close*100 if np.isfinite(op) else 0
    upper_wick = max(0, high-max(op, close))/close*100 if np.isfinite(high) and np.isfinite(op) else 0
    lower_wick = max(0, min(op, close)-low)/close*100 if np.isfinite(low) and np.isfinite(op) else 0

    bull_flags = [
        close > v6_num(a.ema20),
        v6_num(a.ema20) > v6_num(a.ema50),
        v6_num(a.ema50) > v6_num(a.ema200),
        v6_num(a.macd) > v6_num(a.macd_signal),
        v6_num(a.pdi) > v6_num(a.mdi),
    ]
    trend_score = (sum(bull_flags) - (len(bull_flags)-sum(bull_flags))) / len(bull_flags)
    side_bias = "LONG" if trend_score >= 0.2 else ("SHORT" if trend_score <= -0.2 else "NEUTRAL")

    vals = {
        "ret_1h": ret(4), "ret_4h": ret(16), "ret_12h": ret(48), "ret_24h": ret(96),
        "rsi": (v6_num(a.rsi, 50)-50)/50,
        "macd_atr": v6_num(a.macd)/atr,
        "ema20_gap": (close-v6_num(a.ema20))/close*100,
        "ema50_gap": (close-v6_num(a.ema50))/close*100,
        "ema200_gap": (close-v6_num(a.ema200))/close*100,
        "atr_pct": v6_num(a.atr_pct, 0),
        "vol_ratio": min(v6_num(a.vol_ratio, 1), 8),
        "adx": min(v6_num(a.adx, 0), 80),
        "bb_pos": float(np.clip(bb_pos, -1, 2)),
        "range_pos": float(np.clip(range_pos, -1, 2)),
        "body_pct": min(body_pct, 20),
        "upper_wick": min(upper_wick, 20),
        "lower_wick": min(lower_wick, 20),
        "trend_score": trend_score,
    }
    if any(not np.isfinite(v) for v in vals.values()):
        return None
    return vals, side_bias


def v62_label_sample(x, i, feat, atr):
    """Label what happened after the pattern using ATR-normalized movement.
    A trade is considered a historical winner when price moved at least 1.8 ATR
    in the expected direction before making a 1.0 ATR adverse move over 4h.
    This is a learning label, not a guarantee for future trades.
    """
    if i + 16 >= len(x) or not np.isfinite(atr) or atr <= 0:
        return None
    entry = float(x.iloc[i].close)
    future = x.iloc[i+1:i+17]
    if future.empty:
        return None
    max_up = (float(future.high.max())/entry-1)*100
    min_down = (float(future.low.min())/entry-1)*100
    atr_pct = atr/entry*100
    long_win = int(max_up >= 1.8*atr_pct and min_down > -1.0*atr_pct)
    short_win = int(min_down <= -1.8*atr_pct and max_up < 1.0*atr_pct)
    return {
        "long_max_4h": max_up, "long_min_4h": min_down,
        "short_max_4h": max_up, "short_min_4h": min_down,
        "long_win": long_win, "short_win": short_win,
    }


def v62_train_one(pair, days=45, sample_every=4):
    """Download historical 15m candles for one active Futures contract and learn.
    sample_every=4 means one training observation per hour, reducing duplicate states.
    """
    try:
        d = get_tf(pair, "15m", days)
        d = completed(d)
        if d is None or len(d) < 260:
            return pair, 0, 0, "insufficient history"
        x = d.reset_index(drop=True)
        con = v62_db()
        inserted = 0
        skipped = 0
        # Recalculate indicators once for labeling efficiency.
        ind = indicators(x)
        with V62_LOCK:
            for i in range(205, len(x)-16, sample_every):
                feat_bias = v62_feature_row(x, i)
                if feat_bias is None:
                    skipped += 1
                    continue
                feat, side_bias = feat_bias
                atr = v6_num(ind.iloc[i].atr)
                label = v62_label_sample(x, i, feat, atr)
                if label is None:
                    skipped += 1
                    continue
                ts = pd.to_datetime(x.iloc[i].time, utc=True).isoformat()
                cols = [
                    pair, ts, side_bias,
                    *[feat[k] for k in V62_FEATURES],
                    label["long_max_4h"], label["long_min_4h"],
                    label["short_max_4h"], label["short_min_4h"],
                    label["long_win"], label["short_win"]
                ]
                con.execute("""
                    INSERT OR REPLACE INTO pattern_samples
                    (pair,ts,side_bias,ret_1h,ret_4h,ret_12h,ret_24h,rsi,macd_atr,
                     ema20_gap,ema50_gap,ema200_gap,atr_pct,vol_ratio,adx,bb_pos,range_pos,
                     body_pct,upper_wick,lower_wick,trend_score,long_max_4h,long_min_4h,
                     short_max_4h,short_min_4h,long_win,short_win)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, cols)
                inserted += 1
            con.commit()
            count = con.execute("SELECT COUNT(*) FROM pattern_samples WHERE pair=?", (pair,)).fetchone()[0]
        con.close()
        return pair, inserted, int(count), "ok"
    except Exception as e:
        return pair, 0, 0, str(e)[:160]


def v62_train_all(progress=None, days=45, workers=4):
    instruments = active_instruments("USDT")
    pairs = []
    seen = set()
    for inst in instruments:
        p = v61_instrument_pair(inst)
        if p and p not in seen:
            seen.add(p); pairs.append(p)
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(v62_train_one, p, days, 4): p for p in pairs}
        for fut in as_completed(futs):
            res = fut.result(); results.append(res); done += 1
            if progress:
                progress(done, len(pairs), res)
    return results, len(pairs)


def v62_db_stats():
    try:
        con = v62_db()
        row = con.execute("SELECT COUNT(*), COUNT(DISTINCT pair), MIN(ts), MAX(ts) FROM pattern_samples").fetchone()
        con.close()
        return {"samples": row[0] or 0, "coins": row[1] or 0, "first": row[2], "last": row[3]}
    except Exception:
        return {"samples":0,"coins":0,"first":None,"last":None}


def v62_nearest_patterns(current_features, side, pair=None, limit=2500):
    """Find historically similar states across ALL trained CoinDCX Futures.
    Same-coin observations receive a modest preference, but cross-coin patterns
    are included so the learner can recognize market-wide setups.
    """
    try:
        con = v62_db()
        cols = ",".join(V62_FEATURES) + ",pair,side_bias,long_win,short_win,long_max_4h,long_min_4h,short_max_4h,short_min_4h"
        df = pd.read_sql_query(f"SELECT {cols} FROM pattern_samples ORDER BY id DESC LIMIT ?", con, params=(limit,))
        con.close()
        if df.empty:
            return pd.DataFrame()
        arr = df[V62_FEATURES].to_numpy(dtype=float)
        cur = np.array([current_features[k] for k in V62_FEATURES], dtype=float)
        mu = np.nanmedian(arr, axis=0)
        mad = np.nanmedian(np.abs(arr-mu), axis=0)
        scale = np.where(mad > 1e-8, mad*1.4826, np.nanstd(arr, axis=0))
        scale = np.where(scale > 1e-8, scale, 1.0)
        dist = np.sqrt(np.nanmean(((arr-cur)/scale)**2, axis=1))
        df["distance"] = dist
        df["same_pair"] = (df["pair"].astype(str) == str(pair)).astype(int)
        # Side-specific historical outcome.
        df["win"] = df["long_win"] if side == "LONG" else df["short_win"]
        df = df.sort_values(["distance","same_pair"], ascending=[True,False]).head(100)
        return df
    except Exception:
        return pd.DataFrame()


def v62_learning_score(pair, d15):
    """Return a learned probability/evidence score for the current pattern."""
    feat_bias = v62_feature_row(completed(d15).reset_index(drop=True), len(completed(d15))-1)
    if feat_bias is None:
        return {"long_prob":50,"short_prob":50,"long_n":0,"short_n":0,"evidence":"NONE"}
    feat, _ = feat_bias
    out = {}
    for side in ("LONG","SHORT"):
        near = v62_nearest_patterns(feat, side, pair=pair, limit=3000)
        if near.empty:
            out[side.lower()+"_prob"] = 50
            out[side.lower()+"_n"] = 0
            continue
        # Distance-weighted historical hit rate.
        w = 1/(0.25 + near.distance.to_numpy(dtype=float))
        same = near.same_pair.to_numpy(dtype=float)
        w *= (1 + 0.15*same)
        win = near.win.to_numpy(dtype=float)
        prob = float(np.sum(w*win)/np.sum(w)*100)
        out[side.lower()+"_prob"] = round(float(np.clip(prob,0,100)),1)
        out[side.lower()+"_n"] = int(len(near))
    evidence_n = max(out.get("long_n",0), out.get("short_n",0))
    out["evidence"] = "STRONG" if evidence_n >= 50 else ("MEDIUM" if evidence_n >= 15 else ("WEAK" if evidence_n > 0 else "NONE"))
    return out


def v62_enhance_scan(scan):
    """Add learned probabilities to current V6.1 candidates without changing V5."""
    stats = v62_db_stats()
    if stats["samples"] < 100:
        return scan
    enhanced = []
    for a in scan:
        try:
            pair = a.get("pair") or next((t.get("pair") for t in a.get("candidates",[])), None)
            d15 = a.get("d15")
            if d15 is None:
                # Current V6.1 objects do not retain candles; fetch one short history.
                d15 = get_tf(pair, "15m", 3) if pair else pd.DataFrame()
            learn = v62_learning_score(pair, d15) if pair else {}
            a = dict(a)
            a["learning"] = learn
            new_candidates = []
            for t in a.get("candidates", []):
                t = dict(t)
                side = t.get("side","").lower()
                lp = learn.get(side+"_prob",50)
                # Blend deterministic score with historical pattern probability.
                old = float(t.get("score",0))
                t["raw_score"] = old
                t["learned_probability"] = lp
                t["score"] = round(0.70*old + 0.30*lp, 1)
                t["learning_evidence"] = learn.get("evidence","NONE")
                # Historical disagreement downgrades a setup; strong agreement upgrades it.
                if lp < 40:
                    t["status"] = "WAIT — HISTORY CONFLICT"
                elif lp >= 65 and t.get("status") in ("LONG NOW","SHORT NOW"):
                    t["status"] = t.get("status") + " + LEARNED CONFIRMATION"
                new_candidates.append(t)
            a["candidates"] = new_candidates
            enhanced.append(a)
        except Exception:
            enhanced.append(a)
    return enhanced


# ----------------------------- MARKET SIGNAL UI ------------------------------
st.divider()
st.header("🎯 V6.1 — Market-Wide Long / Short Signals")
st.caption("Scans all active CoinDCX USDT Futures using the existing public market-data API. Existing V5 remains above. No live orders are placed.")

with st.expander("How the signal works", expanded=False):
    st.markdown("""
**Your screen should answer one question: where is the trade?**

- 🟢 **LONG NOW** = price has reached/confirmed a qualifying long trigger.
- 🔴 **SHORT NOW** = price has reached/confirmed a qualifying short trigger.
- 🟢/🔴 **SETUP** = level is identified, but the trigger is not confirmed yet.
- 🟦 **RANGE** = repeated support/resistance behaviour; buy support / short resistance only with confirmation.
- 🚀 **PUMP WATCH** and 🔻 **DUMP WATCH** are momentum warnings, **not automatic trade signals**.
- A strong breakout/breakdown invalidates the range logic rather than blindly fading it.
""")

c1, c2, c3 = st.columns(3)
with c1:
    v61_min_score = st.slider("Minimum signal score", 70, 95, 78, 1, key="v61_min_score")
with c2:
    v61_workers = st.slider("Concurrent API workers", 2, 10, 6, 1, key="v61_workers")
with c3:
    v61_auto = st.checkbox("Auto-refresh after scan", value=False, key="v61_auto")

if st.button("🔎 SCAN ALL COINDCX FUTURES", type="primary", key="v61_scan"):
    V61_DEFAULTS["min_score"] = v61_min_score
    bar = st.progress(0, text="Starting whole-market scan…")
    def _progress(done, total):
        pct = int(done/max(total,1)*100)
        bar.progress(pct, text=f"Scanning Futures: {done}/{total}")
    with st.spinner("Fetching multi-timeframe data and calculating signals…"):
        scan, total = v61_scan_all(_progress, max_workers=v61_workers)
    bar.progress(100, text=f"Scan complete: {total} active contracts discovered")
    # If the historical learner has been populated, blend its evidence into
    # the current market scan. The deterministic V5/V6.1 logic remains intact.
    if v62_db_stats()["samples"] >= 100:
        with st.spinner("Comparing current setups with learned historical patterns…"):
            scan = v62_enhance_scan(scan)
    st.session_state["v61_scan_results"] = scan
    st.session_state["v61_scan_total"] = total
    st.session_state["v61_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

scan = st.session_state.get("v61_scan_results", [])
if scan:
    trades = []
    watches = []
    ranges = []
    for a in scan:
        for t in a.get("candidates", []):
            if t.get("score",0) >= v61_min_score:
                trades.append(t)
        watches.extend(a.get("watches", []))
        r = a.get("range", {})
        if r.get("is_range"):
            ranges.append({"pair":a.get("candidates", [{}])[0].get("symbol") if a.get("candidates") else "", "price":a.get("price"), **r})

    longs = sorted([x for x in trades if x.get("side")=="LONG"], key=lambda x:x.get("score",0), reverse=True)
    shorts = sorted([x for x in trades if x.get("side")=="SHORT"], key=lambda x:x.get("score",0), reverse=True)
    pumps = sorted([x for x in watches if x.get("watch")=="PUMP WATCH"], key=lambda x:x.get("score",0), reverse=True)
    dumps = sorted([x for x in watches if x.get("watch")=="DUMP WATCH"], key=lambda x:x.get("score",0), reverse=True)
    ema_bears = sorted([x for x in watches if x.get("watch")=="EMA20/100 BEAR CROSS"], key=lambda x:x.get("score",0), reverse=True)
    ema_bulls = sorted([x for x in watches if x.get("watch")=="EMA20/100 BULL CROSS"], key=lambda x:x.get("score",0), reverse=True)

    st.caption(f"Last scan: {st.session_state.get('v61_scan_time','—')} | Active contracts: {st.session_state.get('v61_scan_total','—')}")
    a,b,c,d = st.columns(4)
    a.metric("LONG signals", len(longs))
    b.metric("SHORT signals", len(shorts))
    c.metric("Pump watch", len(pumps))
    d.metric("Dump watch", len(dumps))

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["🟢 LONG", "🔴 SHORT", "🚀 PUMP", "🔻 DUMP", "🟦 RANGES", "📈 HH/HL • LH/LL", "📉 EMA20/100 CROSS"])
    with tab1:
        if longs:
            for t in longs[:10]: v61_signal_card(t)
        else: st.info("No qualifying LONG signal. No trade is the correct result.")
    with tab2:
        if shorts:
            for t in shorts[:10]: v61_signal_card(t)
        else: st.info("No qualifying SHORT signal. No trade is the correct result.")
    with tab3:
        if pumps:
            st.dataframe(pd.DataFrame(pumps[:15])[['symbol','score','price','return_5h_pct','vol_ratio','rsi','regime']], use_container_width=True, hide_index=True)
        else: st.info("No unusual pump behaviour detected.")
    with tab4:
        if dumps:
            st.dataframe(pd.DataFrame(dumps[:15])[['symbol','score','price','return_5h_pct','vol_ratio','rsi','regime']], use_container_width=True, hide_index=True)
        else: st.info("No unusual dump behaviour detected.")
    with tab5:
        # Reconstruct range rows from analysis objects for a compact view.
        range_rows=[]
        for a in scan:
            r=a.get("range",{})
            if r.get("is_range") and a.get("support") and a.get("resistance"):
                range_rows.append({
                    "symbol": next((x.get("symbol") for x in a.get("candidates",[]) if x.get("symbol")), ""),
                    "price":a.get("price"), "support":a["support"]["level"], "resistance":a["resistance"]["level"],
                    "range_score":r.get("score"), "width_pct":r.get("width_pct"), "support_touches":r.get("touch_s"),
                    "resistance_touches":r.get("touch_r"), "regime":a.get("regime")})
        if range_rows:
            st.dataframe(pd.DataFrame(range_rows).sort_values("range_score", ascending=False).head(20), use_container_width=True, hide_index=True)
        else: st.info("No clean ranges detected.")
    with tab6:
        structure_rows=[]
        for a in scan:
            z=a.get("structure", {})
            if z.get("side") in ("LONG", "SHORT") and z.get("score",0) >= 40:
                structure_rows.append({
                    "symbol": next((x.get("symbol") for x in a.get("candidates",[]) if x.get("symbol")), ""),
                    "structure": z.get("state"), "side": z.get("side"), "score": z.get("score"),
                    "HH %": z.get("high_change_pct"), "HL/LL %": z.get("low_change_pct"),
                    "last swing high": z.get("last_high"), "last swing low": z.get("last_low"),
                    "1H aligned": (z.get("h1_hh") and z.get("h1_hl")) if z.get("side")=="LONG" else (z.get("h1_lh") and z.get("h1_ll")),
                    "price": a.get("price")
                })
        if structure_rows:
            df_struct=pd.DataFrame(structure_rows).sort_values(["score","1H aligned"], ascending=[False,False])
            st.dataframe(df_struct.head(30), use_container_width=True, hide_index=True)
            st.caption("HH/HL = higher highs + higher lows. LH/LL = lower highs + lower lows. These are structure alerts; the entry trigger is shown in the LONG/SHORT cards.")
        else: st.info("No strong HH/HL or LH/LL sequence detected right now.")
    with tab7:
        st.caption("Fresh EMA20/EMA100 crosses are confirmation signals. A bearish cross can support SHORT/DUMP analysis; it is not, by itself, an entry trigger.")
        if ema_bears or ema_bulls:
            rows=[]
            for x in ema_bears + ema_bulls:
                rows.append({"symbol":x.get("symbol"),"signal":x.get("watch"),"score":x.get("score"),
                             "15m":x.get("ema15"),"4H":x.get("ema4h"),"price":x.get("price"),"regime":x.get("regime")})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No fresh EMA20/EMA100 crossover detected in the scanned market.")
else:
    st.info("Click **SCAN ALL COINDCX FUTURES**. The scanner will do the market-wide analysis for you and show only actionable candidates.")

st.caption(f"V{V61_VERSION} + V{V62_VERSION}: V5 retained + market-wide signals + historical pattern learning. Signals are analytical, not guarantees. Live order execution remains disabled.")

# ------------------------- LEARNING CONTROL PANEL ----------------------------
st.divider()
st.header("🧠 V6.2 — Autonomous Historical Pattern Learning")
st.caption("The agent learns from completed CoinDCX Futures 15-minute chart states across the whole active market. It records what happened after similar patterns and uses that evidence in future signals.")

ls1, ls2, ls3 = st.columns(3)
with ls1:
    v62_days = st.selectbox("Training history", [15, 30, 45, 60], index=2, key="v62_days")
with ls2:
    v62_workers = st.slider("Training API workers", 2, 8, 4, 1, key="v62_workers")
with ls3:
    v62_stats = v62_db_stats()
    st.metric("Learned samples", f"{v62_stats['samples']:,}")

st.write(f"**Coins learned:** {v62_stats['coins']:,}  |  **History:** {v62_stats['first'] or '—'} → {v62_stats['last'] or '—'}")

if st.button("🧠 TRAIN / REFRESH ALL COINDCX CHART PATTERNS", type="secondary", key="v62_train"):
    bar2 = st.progress(0, text="Starting historical learning…")
    errors = []
    def _learn_progress(done, total, result):
        pct = int(done/max(total,1)*100)
        status = f"Learning {done}/{total}: {result[0]} (+{result[1]} samples)"
        bar2.progress(pct, text=status)
        if result[3] != "ok":
            errors.append(result)
    try:
        with st.spinner(f"Reading {v62_days} days of 15m charts for every active Futures contract…"):
            results, learned_total = v62_train_all(_learn_progress, days=v62_days, workers=v62_workers)
    except Exception as exc:
        bar2.progress(100, text="Learning stopped — CoinDCX market discovery failed")
        st.error(f"Historical learning could not start: {type(exc).__name__}: {exc}")
        results, learned_total = [], 0
    final_stats = v62_db_stats()
    bar2.progress(100, text=f"Learning complete: {final_stats['coins']:,} coins / {final_stats['samples']:,} samples")
    st.session_state["v62_train_stats"] = final_stats
    if learned_total == 0:
        st.error("Training found 0 active Futures contracts. CoinDCX instrument discovery failed; no learning was performed.")
    elif errors:
        st.warning(f"{len(errors)} contracts could not be learned. The scanner will continue using the contracts that succeeded.")
    else:
        st.success(f"Training processed {learned_total:,} active Futures contracts successfully.")

if v62_db_stats()["samples"] >= 100:
    st.info("🧠 Learning is active. Current signals can now be compared with historical patterns from the entire trained Futures universe. Re-run training periodically to add newer market behaviour.")
else:
    st.warning("The learning database is not populated yet. Run the training button once before relying on historical pattern confirmation.")

# =============================================================================
# V7 — STRUCTURE RADAR + EMA20/100 DUMP CONFIRMATION + CROSS-COIN LEARNING
# =============================================================================
# V7 keeps every earlier V5/V6/V6.1/V6.2 component.  This layer focuses on the
# exact market behaviour the trader wants to see first: newly forming HH/HL for
# LONGs, LH/LL for SHORTs, and fresh EMA20/EMA100 transitions on 15m and 4H.
# It does not place live orders.

V7_VERSION = "7.0-STRUCTURE-RADAR"
V7_DEFAULTS = {
    "min_score": 70,
    "pivot_left": 2,
    "pivot_right": 2,
    "min_structure_move_pct": 0.12,
    "fresh_cross_15m_bars": 6,
    "fresh_cross_4h_bars": 4,
}


def v71_early_structure_transition(d15, d1h=None):
    """Detect the *chronological* start of a trend reversal, not just the
    latest HH/HL or LH/LL pair.

    Bullish transition:
        prior LH/LL -> first HH -> HL -> second HH
    Bearish transition:
        prior HH/HL -> first LH -> LL -> second LH

    The first break is an early warning; the subsequent pullback and second
    break increase confidence. All pivots come from completed candles only.
    """
    out = {
        "state":"NO TRANSITION", "side":None, "score":0, "trigger":"",
        "sequence":"", "prior_state":"", "first_break_price":np.nan,
        "pullback_price":np.nan, "breakout_price":np.nan, "age_bars":None,
        "h1_state":""
    }
    if d15 is None or d15.empty:
        return out

    def _seq(d, tf_name, lookback):
        highs, lows = v71_pivots(d, V7_DEFAULTS["pivot_left"], V7_DEFAULTS["pivot_right"], lookback)
        if len(highs) < 3 or len(lows) < 3:
            return None
        # Merge pivots chronologically. A pivot is classified relative to the
        # previous pivot of the same type.
        events=[]
        for i,h in enumerate(highs):
            prev = highs[i-1]["price"] if i else np.nan
            if i:
                kind = "HH" if h["price"] > prev else "LH"
                events.append({"idx":h["idx"],"price":h["price"],"kind":kind})
        for i,l in enumerate(lows):
            prev = lows[i-1]["price"] if i else np.nan
            if i:
                kind = "HL" if l["price"] > prev else "LL"
                events.append({"idx":l["idx"],"price":l["price"],"kind":kind})
        events.sort(key=lambda z:z["idx"])
        # Keep the most recent meaningful event window. We deliberately require
        # chronology rather than merely having all four labels somewhere.
        events = events[-14:]
        if len(events) < 4:
            return None

        # Search from newest backwards for the strongest completed transition.
        best=None
        for start in range(max(0,len(events)-10), len(events)-3):
            e=events[start:]
            kinds=[x["kind"] for x in e]
            # Bullish: a bearish regime must precede the first HH, then HL,
            # then a higher HH. Allow unrelated same-type pivots between steps.
            hh_positions=[i for i,k in enumerate(kinds) if k=="HH"]
            for p in hh_positions:
                prior_bear = any(k in ("LH","LL") for k in kinds[:p])
                if not prior_bear: continue
                hl = next((i for i in range(p+1,len(kinds)) if kinds[i]=="HL"), None)
                if hl is None: continue
                hh2 = next((i for i in range(hl+1,len(kinds)) if kinds[i]=="HH"), None)
                if hh2 is None:
                    # First HH + subsequent HL = early developing transition.
                    cand=("EARLY LONG", "LONG", 72, e[p], e[hl], None,
                          f"LH/LL → HH → HL", "Prior bearish structure")
                else:
                    cand=("CONFIRMED EARLY LONG", "LONG", 92, e[p], e[hl], e[hh2],
                          f"LH/LL → HH → HL → HH", "Prior bearish structure")
                if best is None or cand[2] > best[2] or e[hl]["idx"] > best[4]["idx"]:
                    best=cand

            # Bearish mirror: bullish regime -> LH -> LL -> LH.
            lh_positions=[i for i,k in enumerate(kinds) if k=="LH"]
            for p in lh_positions:
                prior_bull = any(k in ("HH","HL") for k in kinds[:p])
                if not prior_bull: continue
                ll = next((i for i in range(p+1,len(kinds)) if kinds[i]=="LL"), None)
                if ll is None: continue
                lh2 = next((i for i in range(ll+1,len(kinds)) if kinds[i]=="LH"), None)
                if lh2 is None:
                    cand=("EARLY SHORT", "SHORT", 72, e[p], e[ll], None,
                          f"HH/HL → LH → LL", "Prior bullish structure")
                else:
                    cand=("CONFIRMED EARLY SHORT", "SHORT", 92, e[p], e[ll], e[lh2],
                          f"HH/HL → LH → LL → LH", "Prior bullish structure")
                if best is None or cand[2] > best[2] or e[ll]["idx"] > best[4]["idx"]:
                    best=cand
        if best is None:
            return None
        state,side,score,first,pull,second,sequence,prior=best
        age=max(0, len(completed(d))-1-first["idx"])
        return {"state":state,"side":side,"score":score,"sequence":sequence,
                "prior_state":prior,"first_break_price":first["price"],
                "pullback_price":pull["price"],"breakout_price":second["price"] if second else np.nan,
                "age_bars":age,"tf":tf_name}

    s15=_seq(d15,"15m",180)
    s4=_seq(d1h,"1H",120) if d1h is not None and not d1h.empty else None
    if s15 is None:
        if s4: out.update(s4); out["h1_state"]=s4["state"]
        return out

    out.update(s15)
    if s4:
        out["h1_state"]=s4["state"]
        # Higher-timeframe transition agreement is a confirmation, not a
        # requirement for the first 15m warning.
        if s4["side"] == s15["side"]:
            out["score"]=min(100, int(out["score"])+8)
            out["state"] += " + 1H ALIGNED"
        elif s4["side"] is not None and s4["side"] != s15["side"]:
            out["score"]=max(0, int(out["score"])-12)
            out["state"] += " / 1H CONFLICT"
    return out


def v71_pivots(d, left=2, right=2, lookback=160):
    """Confirmed swing pivots from completed candles only."""
    if d is None or d.empty:
        return [], []
    x = completed(d).tail(lookback).reset_index(drop=True)
    if len(x) < left + right + 10:
        return [], []
    h = pd.to_numeric(x.high, errors="coerce").to_numpy(float)
    l = pd.to_numeric(x.low, errors="coerce").to_numpy(float)
    highs, lows = [], []
    for i in range(left, len(x)-right):
        hs = h[i-left:i+right+1]
        ls = l[i-left:i+right+1]
        if np.isfinite(h[i]) and h[i] >= np.nanmax(hs) and h[i] > h[i-1] and h[i] >= h[i+1]:
            highs.append({"idx": i, "price": float(h[i])})
        if np.isfinite(l[i]) and l[i] <= np.nanmin(ls) and l[i] < l[i-1] and l[i] <= l[i+1]:
            lows.append({"idx": i, "price": float(l[i])})
    return highs, lows


def v71_structure_tf(d, tf_name):
    """Classify the latest swing sequence on one timeframe."""
    highs, lows = v71_pivots(d, V7_DEFAULTS["pivot_left"], V7_DEFAULTS["pivot_right"], 180 if tf_name == "15m" else 120)
    out = {
        "tf": tf_name, "state": "INSUFFICIENT", "side": None, "score": 0,
        "hh": False, "hl": False, "lh": False, "ll": False,
        "last_high": np.nan, "previous_high": np.nan,
        "last_low": np.nan, "previous_low": np.nan,
        "high_change_pct": np.nan, "low_change_pct": np.nan,
        "pivot_high_age": None, "pivot_low_age": None,
    }
    if len(highs) < 2 or len(lows) < 2:
        return out
    ph0, ph1 = highs[-2], highs[-1]
    pl0, pl1 = lows[-2], lows[-1]
    high_change = (ph1["price"] / ph0["price"] - 1) * 100
    low_change = (pl1["price"] / pl0["price"] - 1) * 100
    threshold = max(V7_DEFAULTS["min_structure_move_pct"], 0.18 * (v61_atr(d) / max(v6_num(completed(d).iloc[-1].close), 1) * 100 if np.isfinite(v61_atr(d)) else 0))
    hh, hl = high_change >= threshold, low_change >= threshold
    lh, ll = high_change <= -threshold, low_change <= -threshold
    age_base = len(completed(d)) - 1
    out.update({
        "hh": bool(hh), "hl": bool(hl), "lh": bool(lh), "ll": bool(ll),
        "last_high": ph1["price"], "previous_high": ph0["price"],
        "last_low": pl1["price"], "previous_low": pl0["price"],
        "high_change_pct": high_change, "low_change_pct": low_change,
        "pivot_high_age": max(0, age_base - ph1["idx"]),
        "pivot_low_age": max(0, age_base - pl1["idx"]),
    })
    if hh and hl:
        out["state"], out["side"], out["score"] = "STARTED / CONFIRMED HH/HL", "LONG", 75
        if high_change >= threshold * 2: out["score"] += 8
        if low_change >= threshold * 2: out["score"] += 8
    elif lh and ll:
        out["state"], out["side"], out["score"] = "STARTED / CONFIRMED LH/LL", "SHORT", 75
        if abs(high_change) >= threshold * 2: out["score"] += 8
        if abs(low_change) >= threshold * 2: out["score"] += 8
    elif hh or hl:
        out["state"], out["side"], out["score"] = "EARLY BULLISH STRUCTURE", "LONG", 48
    elif lh or ll:
        out["state"], out["side"], out["score"] = "EARLY BEARISH STRUCTURE", "SHORT", 48
    else:
        out["state"] = "MIXED STRUCTURE"
    return out


def v71_ema_transition(d, recent_bars):
    """Fresh EMA20/EMA100 transition using completed candles only."""
    if d is None or d.empty:
        return {"state":"NO DATA", "bearish":False, "bullish":False, "fresh_bearish":False, "fresh_bullish":False, "age":None, "spread_pct":np.nan}
    x = indicators(completed(d)).dropna(subset=["ema20", "ema100"]).reset_index(drop=True)
    if len(x) < 105:
        return {"state":"NO DATA", "bearish":False, "bullish":False, "fresh_bearish":False, "fresh_bullish":False, "age":None, "spread_pct":np.nan}
    diff = (x.ema20.astype(float) - x.ema100.astype(float)).to_numpy()
    fresh_bear = fresh_bull = False
    bear_age = bull_age = None
    for j in range(1, min(int(recent_bars), len(diff)-1) + 1):
        prev, cur = diff[-j-1], diff[-j]
        if not (np.isfinite(prev) and np.isfinite(cur)):
            continue
        age = j - 1
        if prev >= 0 and cur < 0 and not fresh_bear:
            fresh_bear, bear_age = True, age
        if prev <= 0 and cur > 0 and not fresh_bull:
            fresh_bull, bull_age = True, age
    price = v6_num(x.iloc[-1].close)
    spread = diff[-1] / price * 100 if np.isfinite(price) and price > 0 else np.nan
    if fresh_bear:
        state = "FRESH BEARISH EMA20/100 CROSS"
    elif fresh_bull:
        state = "FRESH BULLISH EMA20/100 CROSS"
    elif diff[-1] < 0:
        state = "EMA20 BELOW EMA100"
    elif diff[-1] > 0:
        state = "EMA20 ABOVE EMA100"
    else:
        state = "FLAT"
    return {"state":state, "bearish":bool(diff[-1] < 0), "bullish":bool(diff[-1] > 0),
            "fresh_bearish":fresh_bear, "fresh_bullish":fresh_bull,
            "age":bear_age if fresh_bear else bull_age, "spread_pct":spread}


def v71_structure_trade_score(struct15, struct4, ema15, ema4, r15, r4, vol_ratio, price):
    """Score a structure-led directional setup; score is not a guarantee."""
    results = []
    for side in ("LONG", "SHORT"):
        score = 0
        reasons = []
        confirmations = []
        blockers = []
        s = struct15 if side == "LONG" else struct15
        wanted = (s.get("hh") and s.get("hl")) if side == "LONG" else (s.get("lh") and s.get("ll"))
        if wanted:
            score += 38
            reasons.append("15m has both required swing components")
        elif (s.get("hh") or s.get("hl")) if side == "LONG" else (s.get("lh") or s.get("ll")):
            score += 18
            reasons.append("15m structure is starting to form")
        if side == "LONG":
            htf = struct4.get("hh") and struct4.get("hl")
            if htf: score += 20; reasons.append("4H also confirms HH/HL")
            elif struct4.get("hh") or struct4.get("hl"): score += 8; reasons.append("4H is beginning to improve")
            if ema15.get("fresh_bullish"): score += 12; confirmations.append("fresh 15m EMA20 > EMA100")
            elif ema15.get("bullish"): score += 5
            if ema4.get("fresh_bullish"): score += 18; confirmations.append("fresh 4H EMA20 > EMA100")
            elif ema4.get("bullish"): score += 7
            if v6_num(r15.rsi, 50) >= 50: score += 5; reasons.append("15m momentum is bullish")
            if v6_num(r15.macd, 0) > v6_num(r15.macd_signal, 0): score += 5
            if v6_num(vol_ratio, 0) >= 1.2: score += 7; reasons.append("volume confirms participation")
            if v6_num(r15.rsi, 50) >= 82: blockers.append("15m RSI is overextended")
        else:
            htf = struct4.get("lh") and struct4.get("ll")
            if htf: score += 20; reasons.append("4H also confirms LH/LL")
            elif struct4.get("lh") or struct4.get("ll"): score += 8; reasons.append("4H is beginning to weaken")
            if ema15.get("fresh_bearish"): score += 12; confirmations.append("fresh 15m EMA20 < EMA100")
            elif ema15.get("bearish"): score += 5
            if ema4.get("fresh_bearish"): score += 18; confirmations.append("fresh 4H EMA20 < EMA100")
            elif ema4.get("bearish"): score += 7
            if v6_num(r15.rsi, 50) <= 50: score += 5; reasons.append("15m momentum is bearish")
            if v6_num(r15.macd, 0) < v6_num(r15.macd_signal, 0): score += 5
            if v6_num(vol_ratio, 0) >= 1.2: score += 7; reasons.append("volume confirms participation")
            if v6_num(r15.rsi, 50) <= 18: blockers.append("15m RSI is overextended")
        results.append((side, min(100, int(score)), reasons, confirmations, blockers))
    return results


def v71_build_radar(a):
    """Convert an existing V6.1 market result into a compact V7 radar row."""
    pair = a.get("pair") or ""
    candidates = a.get("candidates", [])
    symbol = next((x.get("symbol") for x in candidates if x.get("symbol")), pair)
    price = v6_num(a.get("price"))
    s15 = v71_structure_tf((a.get("_d15") if a.get("_d15") is not None else pd.DataFrame()), "15m")
    return {"pair":pair, "symbol":symbol, "price":price, "structure15":s15}


def v71_scan_from_existing(scan):
    """Use V6.1's already-fetched market analysis without another 500-contract API scan.

    V6.1 normally does not retain candle frames, so this function uses the structure
    and EMA results already attached to each market object.  If a V6.1 object lacks
    the detailed fields, it is marked unavailable rather than inventing a signal.
    """
    rows = []
    for a in scan or []:
        pair = a.get("pair") or next((t.get("pair") for t in a.get("candidates",[]) if t.get("pair")), "")
        symbol = next((t.get("symbol") for t in a.get("candidates",[]) if t.get("symbol")), pair)
        price = v6_num(a.get("price"))
        stx = a.get("structure") or {}
        early = a.get("early_structure") or {}
        ec = a.get("ema_cross") or {}
        c15 = ec.get("15m") or {}
        c4 = ec.get("4H") or {}
        long_structure = bool(stx.get("hh") and stx.get("hl"))
        short_structure = bool(stx.get("lh") and stx.get("ll"))
        long_score = 0; short_score = 0
        long_reasons=[]; short_reasons=[]
        if long_structure:
            long_score += 45; long_reasons.append("15m HH + HL")
        elif stx.get("hh") or stx.get("hl"):
            long_score += 22; long_reasons.append("15m early HH/HL")
        if short_structure:
            short_score += 45; short_reasons.append("15m LH + LL")
        elif stx.get("lh") or stx.get("ll"):
            short_score += 22; short_reasons.append("15m early LH/LL")
        # Chronological trend-change detector. This is intentionally separate
        # from the static HH/HL/LH/LL state so the first reversal leg can surface earlier.
        if early.get("side") == "LONG":
            long_score += min(25, int(early.get("score",0)*0.25))
            long_reasons.append(early.get("sequence", "early bullish transition"))
        elif early.get("side") == "SHORT":
            short_score += min(25, int(early.get("score",0)*0.25))
            short_reasons.append(early.get("sequence", "early bearish transition"))
        if stx.get("h1_hh") and stx.get("h1_hl"):
            long_score += 20; long_reasons.append("1H HH + HL")
        if stx.get("h1_lh") and stx.get("h1_ll"):
            short_score += 20; short_reasons.append("1H LH + LL")
        if c15.get("fresh_bullish"): long_score += 10; long_reasons.append("fresh 15m EMA20/100 bullish cross")
        if c4.get("fresh_bullish"): long_score += 15; long_reasons.append("fresh 4H EMA20/100 bullish cross")
        if c15.get("fresh_bearish"): short_score += 10; short_reasons.append("fresh 15m EMA20/100 bearish cross")
        if c4.get("fresh_bearish"): short_score += 15; short_reasons.append("fresh 4H EMA20/100 bearish cross")
        if c15.get("bearish"): short_score += 4
        if c4.get("bearish"): short_score += 6
        if c15.get("bullish"): long_score += 4
        if c4.get("bullish"): long_score += 6
        momentum = a.get("momentum") or {}
        vol = v6_num(momentum.get("vol_ratio"), 1)
        rsi = v6_num(momentum.get("rsi"), 50)
        ret = v6_num(momentum.get("return_5h_pct"), 0)
        if vol >= 1.3:
            if long_score >= short_score: long_score += 8; long_reasons.append(f"volume {vol:.1f}x")
            if short_score >= long_score: short_score += 8; short_reasons.append(f"volume {vol:.1f}x")
        if ret >= 3: long_score += 5; long_reasons.append(f"5h +{ret:.1f}%")
        if ret <= -3: short_score += 5; short_reasons.append(f"5h {ret:.1f}%")
        # Historical learner, if V6.2 already blended it into candidates.
        learn_long = max([v6_num(t.get("learned_probability"), np.nan) for t in candidates if t.get("side")=="LONG" and np.isfinite(v6_num(t.get("learned_probability")))], default=np.nan)
        learn_short = max([v6_num(t.get("learned_probability"), np.nan) for t in candidates if t.get("side")=="SHORT" and np.isfinite(v6_num(t.get("learned_probability")))], default=np.nan)
        if np.isfinite(learn_long) and learn_long >= 65: long_score += 8; long_reasons.append(f"history {learn_long:.0f}%")
        if np.isfinite(learn_short) and learn_short >= 65: short_score += 8; short_reasons.append(f"history {learn_short:.0f}%")
        rows.append({
            "symbol":symbol, "pair":pair, "price":price,
            "long_score":min(100,int(long_score)), "short_score":min(100,int(short_score)),
            "long_reasons":long_reasons, "short_reasons":short_reasons,
            "structure":stx.get("state","UNKNOWN"),
            "early_state":early.get("state","NO TRANSITION"), "early_side":early.get("side"),
            "early_score":int(early.get("score",0) or 0), "early_sequence":early.get("sequence",""),
            "early_first_break":early.get("first_break_price",np.nan),
            "early_pullback":early.get("pullback_price",np.nan), "early_breakout":early.get("breakout_price",np.nan),
            "early_age_bars":early.get("age_bars"), "h1_transition":early.get("h1_state",""),
            "hh":bool(stx.get("hh")), "hl":bool(stx.get("hl")), "lh":bool(stx.get("lh")), "ll":bool(stx.get("ll")),
            "h1_hh":bool(stx.get("h1_hh")), "h1_hl":bool(stx.get("h1_hl")), "h1_lh":bool(stx.get("h1_lh")), "h1_ll":bool(stx.get("h1_ll")),
            "ema15":c15.get("state","NO DATA"), "ema4h":c4.get("state","NO DATA"),
            "fresh_ema_bear":bool(c15.get("fresh_bearish") or c4.get("fresh_bearish")),
            "fresh_ema_bull":bool(c15.get("fresh_bullish") or c4.get("fresh_bullish")),
            "rsi":rsi, "vol_ratio":vol, "return_5h_pct":ret,
            "regime":a.get("regime","UNKNOWN"),
        })
    return rows


# --------------------------- V7 UI -------------------------------------------
st.divider()
st.header("🎯 V7 — HH/HL • LH/LL Structure Radar")
st.caption("Finds coins that are beginning to build higher highs/higher lows for LONGs, lower highs/lower lows for SHORTs, and highlights fresh EMA20/EMA100 transitions on 15m and 4H. Existing V5/V6/V6.1/V6.2 remain intact.")

with st.expander("How V7 decides what matters", expanded=False):
    st.markdown("""
**LONG radar**
- 15m **HH + HL** is the core structure signal.
- 1H HH + HL strengthens it.
- Fresh 15m/4H **EMA20 crossing above EMA100** strengthens the trend transition.
- Volume and momentum are confirmations.

**SHORT / dump radar**
- 15m **LH + LL** is the core bearish structure signal.
- 1H LH + LL strengthens it.
- Fresh 15m/4H **EMA20 crossing below EMA100** is a high-impact bearish confirmation.
- A bearish EMA cross alone is **not** treated as an automatic SHORT.

**Important:** PUMP/DUMP describes what price is doing. LONG/SHORT describes whether the structure and confirmations provide a tradeable direction.
""")

v7_min = st.slider("V7 minimum structure score", 50, 95, V7_DEFAULTS["min_score"], 1, key="v7_min_score")

if st.button("🚀 RUN V7 STRUCTURE RADAR", type="primary", key="v7_run_structure_radar"):
    existing = st.session_state.get("v61_scan_results", [])
    if not existing:
        st.warning("Run **SCAN ALL COINDCX FUTURES** above first. V7 reuses that market-wide scan so it does not make another 500+ contract API request.")
    else:
        with st.spinner("Building HH/HL, LH/LL and EMA20/100 radar from the market-wide scan…"):
            radar = v71_scan_from_existing(existing)
        st.session_state["v7_radar_results"] = radar
        st.session_state["v7_radar_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

radar = st.session_state.get("v7_radar_results", [])
if radar:
    longs = sorted([x for x in radar if x["long_score"] >= v7_min], key=lambda x:x["long_score"], reverse=True)
    shorts = sorted([x for x in radar if x["short_score"] >= v7_min], key=lambda x:x["short_score"], reverse=True)
    hhhl = sorted([x for x in radar if x["hh"] and x["hl"]], key=lambda x:(x["long_score"], x["h1_hh"] and x["h1_hl"]), reverse=True)
    lhll = sorted([x for x in radar if x["lh"] and x["ll"]], key=lambda x:(x["short_score"], x["h1_lh"] and x["h1_ll"]), reverse=True)
    early_longs = sorted([x for x in radar if x.get("early_side")=="LONG" and x.get("early_score",0)>=70], key=lambda x:(x.get("early_score",0), x.get("long_score",0)), reverse=True)
    early_shorts = sorted([x for x in radar if x.get("early_side")=="SHORT" and x.get("early_score",0)>=70], key=lambda x:(x.get("early_score",0), x.get("short_score",0)), reverse=True)
    bear_cross = sorted([x for x in radar if x["fresh_ema_bear"]], key=lambda x:x["short_score"], reverse=True)
    bull_cross = sorted([x for x in radar if x["fresh_ema_bull"]], key=lambda x:x["long_score"], reverse=True)

    st.caption(f"V7 radar built from {len(radar)} market records | {st.session_state.get('v7_radar_time','—')}")
    q1,q2,q3,q4,q5,q6 = st.columns(6)
    q1.metric("Tradeable LONG", len(longs))
    q2.metric("Tradeable SHORT", len(shorts))
    q3.metric("EARLY LONG", len(early_longs))
    q4.metric("EARLY SHORT", len(early_shorts))
    q5.metric("HH + HL", len(hhhl))
    q6.metric("LH + LL", len(lhll))

    t1,t2,t3,t4,t5,t6,t7 = st.tabs(["🟢 LONG", "🔴 SHORT / DUMP", "🟡 EARLY LONG", "🟠 EARLY SHORT", "📈 HH + HL", "📉 LH + LL", "⚠️ EMA20/100"])
    with t1:
        if longs:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "Score":x["long_score"], "Price":x["price"], "Structure":x["structure"],
                "EMA15":x["ema15"], "EMA4H":x["ema4h"], "RSI":x["rsi"], "Vol":x["vol_ratio"],
                "Why":" | ".join(x["long_reasons"])
            } for x in longs[:20]]), use_container_width=True, hide_index=True)
        else: st.info("No LONG setup reached the selected structure score. No trade is the correct result.")
    with t2:
        if shorts:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "Score":x["short_score"], "Price":x["price"], "Structure":x["structure"],
                "EMA15":x["ema15"], "EMA4H":x["ema4h"], "RSI":x["rsi"], "Vol":x["vol_ratio"],
                "Why":" | ".join(x["short_reasons"])
            } for x in shorts[:20]]), use_container_width=True, hide_index=True)
        else: st.info("No SHORT/LH-LL setup reached the selected score.")
    with t3:
        if early_longs:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "Early score":x.get("early_score",0), "LONG score":x["long_score"],
                "Price":x["price"], "Transition":x.get("early_sequence",""),
                "First HH":x.get("early_first_break",np.nan), "HL":x.get("early_pullback",np.nan),
                "2nd HH":x.get("early_breakout",np.nan), "Age 15m bars":x.get("early_age_bars"),
                "1H":x.get("h1_transition",""), "EMA15":x["ema15"], "EMA4H":x["ema4h"]
            } for x in early_longs[:30]]), use_container_width=True, hide_index=True)
            st.caption("EARLY LONG = prior bearish structure followed chronologically by first HH → HL; a second HH upgrades it to confirmed early transition. This is an alert, not an automatic entry.")
        else: st.info("No early bullish transition detected.")
    with t4:
        if early_shorts:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "Early score":x.get("early_score",0), "SHORT score":x["short_score"],
                "Price":x["price"], "Transition":x.get("early_sequence",""),
                "First LH":x.get("early_first_break",np.nan), "LL":x.get("early_pullback",np.nan),
                "2nd LH":x.get("early_breakout",np.nan), "Age 15m bars":x.get("early_age_bars"),
                "1H":x.get("h1_transition",""), "EMA15":x["ema15"], "EMA4H":x["ema4h"]
            } for x in early_shorts[:30]]), use_container_width=True, hide_index=True)
            st.caption("EARLY SHORT = prior bullish structure followed chronologically by first LH → LL; a second LH upgrades it to confirmed early transition. This is an alert, not an automatic entry.")
        else: st.info("No early bearish transition detected.")
    with t5:
        if hhhl:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "LONG score":x["long_score"], "Price":x["price"],
                "15m":"HH + HL", "1H":"HH + HL" if x["h1_hh"] and x["h1_hl"] else "Partial / mixed",
                "EMA15":x["ema15"], "EMA4H":x["ema4h"], "Why":" | ".join(x["long_reasons"])
            } for x in hhhl[:30]]), use_container_width=True, hide_index=True)
        else: st.info("No current 15m HH + HL sequence found in the market scan.")
    with t6:
        if lhll:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "SHORT score":x["short_score"], "Price":x["price"],
                "15m":"LH + LL", "1H":"LH + LL" if x["h1_lh"] and x["h1_ll"] else "Partial / mixed",
                "EMA15":x["ema15"], "EMA4H":x["ema4h"], "Why":" | ".join(x["short_reasons"])
            } for x in lhll[:30]]), use_container_width=True, hide_index=True)
        else: st.info("No current 15m LH + LL sequence found in the market scan.")
    with t7:
        cross = bear_cross + bull_cross
        if cross:
            st.dataframe(pd.DataFrame([{
                "Symbol":x["symbol"], "Direction":"🔴 BEARISH" if x["fresh_ema_bear"] else "🟢 BULLISH",
                "SHORT score":x["short_score"], "LONG score":x["long_score"], "15m":x["ema15"], "4H":x["ema4h"],
                "Structure":x["structure"], "5h %":x["return_5h_pct"]
            } for x in cross[:40]]), use_container_width=True, hide_index=True)
        else: st.info("No fresh EMA20/EMA100 crossover detected in the existing market scan.")
else:
    st.info("Run **SCAN ALL COINDCX FUTURES** above, then run **V7 STRUCTURE RADAR**.")

st.caption(f"V{V7_VERSION}: V5 retained + V6 intraday + V6.1 market-wide scan + V6.2 learning + HH/HL/LH/LL and EMA20/100 structure radar. Analysis only; live orders remain disabled.")
