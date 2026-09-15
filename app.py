import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone

# =============================================================================
# APP
# =============================================================================
st.set_page_config(page_title="CoinDCX Futures Trading Agent", page_icon="🎯", layout="wide")
st.title("🎯 CoinDCX Futures Trading Agent")
st.caption("Simple daily trading workflow: find today's structure, check multi-timeframe support/resistance, and investigate extreme pump/dump reversals.")

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
def simple_today_structure(d15, bars=96):
    """Simple trader-facing structure for the latest 24 hours of completed 15m candles.

    Returns plain-language labels such as:
      - HIGHER HIGH + HIGHER LOW -> LONG structure
      - LOWER HIGH + LOWER LOW -> SHORT structure
      - HIGHER HIGH / HIGHER LOW -> developing bullish structure
      - LOWER HIGH / LOWER LOW -> developing bearish structure
      - MIXED / NO CLEAR STRUCTURE

    Uses confirmed swing pivots only, so the latest forming candle is never used.
    """
    out = {
        "label": "⚪ NO CLEAR STRUCTURE", "side": "WAIT", "hh": False, "hl": False,
        "lh": False, "ll": False, "high_now": np.nan, "high_prev": np.nan,
        "low_now": np.nan, "low_prev": np.nan, "high_change_pct": np.nan,
        "low_change_pct": np.nan, "lookback_bars": 0,
    }
    try:
        d = completed(d15)
        if d is None or d.empty:
            return out
        d = d.tail(int(bars)).reset_index(drop=True)
        out["lookback_bars"] = len(d)
        if len(d) < 12:
            out["label"] = "⚪ NOT ENOUGH DATA"
            return out

        h = pd.to_numeric(d["high"], errors="coerce").to_numpy(float)
        l = pd.to_numeric(d["low"], errors="coerce").to_numpy(float)
        highs, lows = [], []
        left = right = 2
        for i in range(left, len(d) - right):
            hs = h[i-left:i+right+1]
            ls = l[i-left:i+right+1]
            if np.isfinite(h[i]) and h[i] >= np.nanmax(hs) and h[i] > h[i-1] and h[i] >= h[i+1]:
                highs.append(float(h[i]))
            if np.isfinite(l[i]) and l[i] <= np.nanmin(ls) and l[i] < l[i-1] and l[i] <= l[i+1]:
                lows.append(float(l[i]))

        if len(highs) >= 2:
            hp, hn = highs[-2], highs[-1]
            out["high_prev"], out["high_now"] = hp, hn
            out["high_change_pct"] = (hn / hp - 1) * 100 if hp > 0 else np.nan
        if len(lows) >= 2:
            lp, ln = lows[-2], lows[-1]
            out["low_prev"], out["low_now"] = lp, ln
            out["low_change_pct"] = (ln / lp - 1) * 100 if lp > 0 else np.nan

        threshold = 0.05  # avoid calling tiny pivot noise a structure change
        hc = out["high_change_pct"]
        lc = out["low_change_pct"]
        out["hh"] = bool(np.isfinite(hc) and hc >= threshold)
        out["lh"] = bool(np.isfinite(hc) and hc <= -threshold)
        out["hl"] = bool(np.isfinite(lc) and lc >= threshold)
        out["ll"] = bool(np.isfinite(lc) and lc <= -threshold)

        if out["hh"] and out["hl"]:
            out["label"], out["side"] = "🟢 HIGHER HIGH + HIGHER LOW", "LONG"
        elif out["lh"] and out["ll"]:
            out["label"], out["side"] = "🔴 LOWER HIGH + LOWER LOW", "SHORT"
        elif out["hh"] or out["hl"]:
            out["label"], out["side"] = "🟡 DEVELOPING HIGHER HIGH / HIGHER LOW", "WATCH LONG"
        elif out["lh"] or out["ll"]:
            out["label"], out["side"] = "🟠 DEVELOPING LOWER HIGH / LOWER LOW", "WATCH SHORT"
        else:
            out["label"], out["side"] = "⚪ MIXED / NO CLEAR STRUCTURE", "WAIT"
    except Exception:
        return out
    return out


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
        "structure4":structure(d4),"structure1":structure(d1),"structure15":structure(d15) if not d15.empty else "Mixed",
        "today_structure": simple_today_structure(tf_data.get("15m")),
        "mtf_sr": v13_mtf_support_resistance(tf_data, current) if "v13_mtf_support_resistance" in globals() else {}
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
                # IMPORTANT: keep the analyzer outside the invalid-price guard.
                # The previous build accidentally indented this call underneath
                # `continue`, making every valid contract skip the analyzer and
                # leaving the V7/V61 result set empty.
                a = v61_analyze_candidate(pair, base[1], price, d15, d1h, d4h)
                if a:
                    a["today_structure"] = simple_today_structure(d15)
                    a["price"] = price
                    a["price_source"] = "LIVE_FEED" if np.isfinite(base[2]) and base[2] > 0 else "15M_CANDLE_FALLBACK"
                    a["symbol"] = base[1]
                    a["pair"] = pair
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
st.title("🧠 CoinDCX Futures Trading Agent — Simple Daily Dashboard")
st.caption("Simple daily workflow: Analyze a coin • See today's HH/HL/LH/LL • Check 15m/4H/1D/1W S/R • Hunt extreme pump/dump reversals. Analysis only — no live orders.")

margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Use meme-focused learning universe",value=False)
peer_limit=st.slider("Historical comparison universe",20,150,100,10,help="More contracts provide more historical examples but require more CoinDCX API calls.")
st.info("V5 rule: an extreme pump is NOT treated as an immediate short. Reversal now requires multiple confirmations; EMA weakness alone moves the setup to REVERSAL DEVELOPING, not REVERSAL CONFIRMED.")

st.success("Daily workflow: ⭐ TODAY scan → 📐 MTF S/R → 🎯 Confirmed signals. Use 🧨 Extreme Move Hunter only when hunting large pump/dump reversals.")

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

            _today_struct = current.get("today_structure", {}) or {}
            st.markdown("### ⭐ Simple answer — what is happening today?")
            st.write(f"**{_today_struct.get('label','⚪ NO CLEAR STRUCTURE')}**")
            _ts1,_ts2,_ts3,_ts4=st.columns(4)
            _ts1.metric("Higher High", "YES" if _today_struct.get("hh") else "NO")
            _ts2.metric("Higher Low", "YES" if _today_struct.get("hl") else "NO")
            _ts3.metric("Lower High", "YES" if _today_struct.get("lh") else "NO")
            _ts4.metric("Lower Low", "YES" if _today_struct.get("ll") else "NO")
            st.caption("🟢 HH + HL = bullish structure / LONG bias | 🔴 LH + LL = bearish structure / SHORT bias | ⚪ otherwise WAIT. Uses the latest completed 15m candles over roughly 24 hours.")

            st.markdown("### 🧭 Trend vs. reversal")
            t1,t2,t3,t4=st.columns(4)
            t1.metric("Current trend", "BULLISH" if current["bull"]>=current["bear"] else "MIXED")
            t2.metric("Momentum", "EXTREME" if event_is_extreme(current["target"]) else "NORMAL")
            t3.metric("Reversal stage",s15["reversal_stage"])
            t4.metric("15m structure",current.get("structure15","Mixed"))

            # V14: Support/resistance is shown for EVERY coin analyzed,
            # regardless of whether the final decision is LONG, SHORT or WAIT.
            st.markdown("### 📐 Multi-Timeframe Support & Resistance")
            _coin_sr = current.get("mtf_sr", {}) or {}
            _sr_summary = v14_sr_summary(_coin_sr, current.get("current")) if _coin_sr else {}
            if _coin_sr:
                st.dataframe(
                    pd.DataFrame(v13_sr_columns(_coin_sr)),
                    use_container_width=True,
                    hide_index=True,
                )
                if _sr_summary:
                    sr1,sr2,sr3,sr4=st.columns(4)
                    sr1.metric("Nearest Support", v13_format_price(_sr_summary.get("nearest_support")))
                    sr2.metric("Support distance", f"{_sr_summary.get('support_dist_pct'):.2f}%" if _sr_summary.get('support_dist_pct') is not None else "—")
                    sr3.metric("Nearest Resistance", v13_format_price(_sr_summary.get("nearest_resistance")))
                    sr4.metric("Resistance distance", f"+{_sr_summary.get('resistance_dist_pct'):.2f}%" if _sr_summary.get('resistance_dist_pct') is not None else "—")
                    st.caption(
                        f"Nearest support: {_sr_summary.get('support_tf','—')} | "
                        f"Nearest resistance: {_sr_summary.get('resistance_tf','—')}"
                    )
            else:
                st.warning("Support/resistance data could not be calculated for this coin from the available completed candles.")

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
    # Keep the V7 probability calculation self-contained in Streamlit Cloud.
    # A stale module namespace must never make the structure radar crash.
    import numpy as _np
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
        # Historical learner, if V6.2 already blended it into the V6.1
        # market object's candidate list.  The previous V15 build referenced
        # a non-existent local variable named `candidates`, which caused the
        # V7 radar to fail with NameError.
        candidates = a.get("candidates") or []
        _long_probs = []
        _short_probs = []
        for _t in candidates:
            _p = v6_num(_t.get("learned_probability"), _np.nan)
            if not _np.isfinite(_p):
                continue
            if _t.get("side") == "LONG":
                _long_probs.append(_p)
            elif _t.get("side") == "SHORT":
                _short_probs.append(_p)
        learn_long = max(_long_probs) if _long_probs else _np.nan
        learn_short = max(_short_probs) if _short_probs else _np.nan
        if _np.isfinite(learn_long) and learn_long >= 65: long_score += 8; long_reasons.append(f"history {learn_long:.0f}%")
        if _np.isfinite(learn_short) and learn_short >= 65: short_score += 8; short_reasons.append(f"history {learn_short:.0f}%")
        rows.append({
            "symbol":symbol, "pair":pair, "price":price,
            "long_score":min(100,int(long_score)), "short_score":min(100,int(short_score)),
            "long_reasons":long_reasons, "short_reasons":short_reasons,
            "structure":stx.get("state","UNKNOWN"),
            "today_structure": (a.get("today_structure") or {}).get("label", "⚪ NO CLEAR STRUCTURE"),
            "today_side": (a.get("today_structure") or {}).get("side", "WAIT"),
            "today_hh": bool((a.get("today_structure") or {}).get("hh")),
            "today_hl": bool((a.get("today_structure") or {}).get("hl")),
            "today_lh": bool((a.get("today_structure") or {}).get("lh")),
            "today_ll": bool((a.get("today_structure") or {}).get("ll")),
            "today_high": (a.get("today_structure") or {}).get("high_now", np.nan),
            "today_low": (a.get("today_structure") or {}).get("low_now", np.nan),
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



# --------------------------- V8 EXTREME EDGE UI -------------------------------
if st.session_state.get("v6_scan_results"):
    _all_v8 = st.session_state.get("v6_scan_results", [])
    _extreme_watch = [r for r in _all_v8 if r.get("source") == "V8 EXTREME EDGE"]
    if _extreme_watch:
        st.subheader("🎯 ATH / ATL Extreme Edge Radar")
        st.caption(
            "ATH is not automatically a SHORT and ATL is not automatically a LONG. "
            "The agent waits for breakout/continuation or rejection/breakdown structure."
        )
        _ew = []
        for r in _extreme_watch:
            _ew.append({
                "Coin": r.get("Coin"),
                "Extreme": r.get("Extreme"),
                "Side": r.get("Direction"),
                "Score": r.get("Score"),
                "Price": r.get("Entry"),
                "Distance ATH %": r.get("Distance ATH %"),
                "Distance ATL %": r.get("Distance ATL %"),
                "15m Structure": r.get("15m Structure"),
                "Decision": "PAPER TRADE" if r.get("Valid") == "✅ TRADE CANDIDATE" else "WAIT",
                "Why": r.get("Reasons"),
                "Blockers": r.get("Blockers"),
            })
        st.dataframe(pd.DataFrame(_ew), use_container_width=True, hide_index=True)




# -------------------- V13 MULTI-TIMEFRAME SUPPORT / RESISTANCE -----------------
V13_SR_TIMEFRAMES = ("15m", "4H", "1D", "1W")

def v13_sr_pivots(df, left=3, right=3):
    d = completed(df)
    if d is None or d.empty or len(d) < left + right + 5:
        return [], []
    highs = d["high"].astype(float).values
    lows = d["low"].astype(float).values
    supports, resistances = [], []
    for i in range(left, len(d) - right):
        if np.isfinite(highs[i]) and highs[i] >= np.max(highs[i-left:i+right+1]):
            resistances.append(float(highs[i]))
        if np.isfinite(lows[i]) and lows[i] <= np.min(lows[i-left:i+right+1]):
            supports.append(float(lows[i]))
    return supports, resistances

def v13_cluster_levels(levels, tolerance_pct=0.004):
    vals = sorted(float(x) for x in levels if np.isfinite(x) and x > 0)
    if not vals:
        return []
    clusters = [[vals[0]]]
    for x in vals[1:]:
        center = float(np.mean(clusters[-1]))
        if abs(x - center) / max(center, 1e-12) <= tolerance_pct:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [float(np.mean(c)) for c in clusters]

def v13_weekly_from_daily(daily):
    d = completed(daily)
    if d is None or d.empty:
        return None
    idx = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    w = d.assign(_week=idx.dt.to_period("W-SUN").astype(str))
    rows = []
    for _, g in w.groupby("_week", sort=True):
        rows.append({
            "timestamp": int(g["timestamp"].iloc[-1]),
            "open": float(g["open"].iloc[0]),
            "high": float(g["high"].max()),
            "low": float(g["low"].min()),
            "close": float(g["close"].iloc[-1]),
            "volume": float(g["volume"].sum()),
        })
    return pd.DataFrame(rows)

def v13_mtf_support_resistance(tf_data, current):
    try:
        current = float(current)
    except Exception:
        return {}
    data = dict(tf_data)
    if data.get("1W") is None or getattr(data.get("1W"), "empty", True):
        data["1W"] = v13_weekly_from_daily(data.get("1D"))

    out = {}
    for tf in V13_SR_TIMEFRAMES:
        d = completed(data.get(tf))
        if d is None or d.empty or len(d) < 10:
            out[tf] = {"S1": None, "S2": None, "S3": None,
                       "R1": None, "R2": None, "R3": None}
            continue

        supports, resistances = v13_sr_pivots(d)
        for n in (20, 50):
            if len(d) >= n:
                recent = d.iloc[-n:]
                supports.append(float(recent["low"].min()))
                resistances.append(float(recent["high"].max()))

        supports = v13_cluster_levels(supports)
        resistances = v13_cluster_levels(resistances)
        below = sorted([x for x in supports if x < current], reverse=True)
        above = sorted([x for x in resistances if x > current])

        out[tf] = {
            "S1": below[0] if len(below) > 0 else None,
            "S2": below[1] if len(below) > 1 else None,
            "S3": below[2] if len(below) > 2 else None,
            "R1": above[0] if len(above) > 0 else None,
            "R2": above[1] if len(above) > 1 else None,
            "R3": above[2] if len(above) > 2 else None,
        }
    return out

def v13_format_price(x):
    if x is None:
        return "—"
    try:
        x = float(x)
        if abs(x) >= 100: return f"{x:,.2f}"
        if abs(x) >= 1: return f"{x:.4f}"
        if abs(x) >= 0.01: return f"{x:.6f}"
        return f"{x:.10f}".rstrip("0").rstrip(".")
    except Exception:
        return "—"

def v13_sr_columns(sr):
    return [{
        "Timeframe": tf,
        "Support 1": v13_format_price(sr.get(tf, {}).get("S1")),
        "Support 2": v13_format_price(sr.get(tf, {}).get("S2")),
        "Support 3": v13_format_price(sr.get(tf, {}).get("S3")),
        "Resistance 1": v13_format_price(sr.get(tf, {}).get("R1")),
        "Resistance 2": v13_format_price(sr.get(tf, {}).get("R2")),
        "Resistance 3": v13_format_price(sr.get(tf, {}).get("R3")),
    } for tf in V13_SR_TIMEFRAMES]

def v13_attach_sr_to_records(records):
    """Attach MTF S/R to scanner records. Uses Current, then Entry as fallback.

    The original V13 expected a Current field that V10 did not emit, which
    caused S/R to be skipped. This version accepts either field.
    """
    enriched = []
    for r in records or []:
        rr = dict(r)
        pair = rr.get("Pair") or rr.get("pair")
        try:
            current = float(rr.get("Current", rr.get("Entry")))
        except Exception:
            rr["MTF_SR"] = {}
            enriched.append(rr)
            continue
        try:
            tf_data = {
                "15m": get_tf(pair, "15m", 12),
                "4H": get_tf(pair, "4H", 120),
                "1D": get_tf(pair, "1D", 180),
                "1W": get_tf(pair, "1W", 365),
            }
            rr["Current"] = v13_format_price(current)
            rr["MTF_SR"] = v13_mtf_support_resistance(tf_data, current)
        except Exception:
            rr["MTF_SR"] = {}
        enriched.append(rr)
    return enriched


def v14_sr_distance(current, level):
    try:
        current = float(current); level = float(level)
        if current <= 0 or level <= 0:
            return None
        return (level-current)/current*100.0
    except Exception:
        return None

def v14_sr_summary(sr, current):
    """Create nearest multi-timeframe S/R and trading-room summary."""
    out = {"nearest_support": None, "nearest_resistance": None, "support_tf": None, "resistance_tf": None,
           "support_dist_pct": None, "resistance_dist_pct": None}
    supports=[]; resistances=[]
    for tf in V13_SR_TIMEFRAMES:
        z=sr.get(tf,{}) or {}
        for k in ("S1","S2","S3"):
            v=z.get(k)
            if v is not None:
                supports.append((float(v),tf,k))
        for k in ("R1","R2","R3"):
            v=z.get(k)
            if v is not None:
                resistances.append((float(v),tf,k))
    if supports:
        # Highest support below current = nearest support.
        v,tf,k=max((x for x in supports if x[0] < float(current)), key=lambda x:x[0], default=(None,None,None))
        if v is not None:
            out.update(nearest_support=v,support_tf=f"{tf} {k}",support_dist_pct=v14_sr_distance(current,v))
    if resistances:
        v,tf,k=min((x for x in resistances if x[0] > float(current)), key=lambda x:x[0], default=(None,None,None))
        if v is not None:
            out.update(nearest_resistance=v,resistance_tf=f"{tf} {k}",resistance_dist_pct=v14_sr_distance(current,v))
    return out


# =============================================================================
# V10 EXTREME MOVE REVERSAL + NEXT-LEG RADAR
# =============================================================================
# This strategy is intentionally independent of ATH/ATL.  It hunts coins that
# have been repriced violently over several days and then classifies the next
# opportunity as: LONG next-leg, SHORT reversal, or WAIT.
V10_EXTREME_PUMP_3D = 150.0
V10_EXTREME_PUMP_5D = 250.0
V10_EXTREME_PUMP_7D = 400.0
V10_EXTREME_DUMP_3D = -65.0
V10_EXTREME_DUMP_5D = -75.0
V10_EXTREME_DUMP_7D = -85.0
V10_MIN_SCORE = 72
V10_MAX_SHORT_FROM_PEAK = 22.0       # don't chase a mature dump
V10_MAX_LONG_FROM_LOW = 22.0         # don't chase a mature rebound
V10_NEAR_HIGH_PULLBACK = 18.0
V10_NEAR_LOW_REBOUND = 18.0


def _v10_pct(a, b):
    try:
        a, b = float(a), float(b)
        return (a / b - 1.0) * 100.0 if b else np.nan
    except Exception:
        return np.nan


def v10_extreme_move_signal(pair, symbol, tf_data, current):
    """Detect massive multi-day pumps/dumps and trade the next phase.

    Four states are deliberately separated:
      1) PUMP -> HH/HL / consolidation -> LONG next leg
      2) PUMP -> rejection -> LH/LL -> SHORT
      3) DUMP -> LH/LL / continuation -> SHORT next leg
      4) DUMP -> stabilization -> HH/HL -> LONG reversal

    A large percentage move by itself never creates a trade.
    """
    try:
        d1 = completed(tf_data.get("1D"))
        d15 = completed(tf_data.get("15m"))
        d1h = completed(tf_data.get("1H"))
        d4 = completed(tf_data.get("4H"))
        if any(x is None or x.empty for x in (d1, d15, d1h, d4)):
            return None
        if len(d1) < 12 or len(d15) < 35 or len(d1h) < 35 or len(d4) < 35:
            return None
        price = float(current)
        if not np.isfinite(price) or price <= 0:
            return None

        # Daily closes are used for multi-day move detection.  The last row may
        # be the current forming day, so use completed history and current price
        # only for the live leg.
        c = pd.to_numeric(d1["close"], errors="coerce").dropna()
        h = pd.to_numeric(d1["high"], errors="coerce").dropna()
        l = pd.to_numeric(d1["low"], errors="coerce").dropna()
        if len(c) < 8:
            return None
        p3 = float(c.iloc[-4])
        p5 = float(c.iloc[-6])
        p7 = float(c.iloc[-8])
        ret3 = _v10_pct(price, p3)
        ret5 = _v10_pct(price, p5)
        ret7 = _v10_pct(price, p7)

        # Peak/trough of the recent completed daily window, excluding today's
        # forming candle.  This gives us "how far has the dump already traveled?"
        hist = d1.iloc[:-1] if len(d1) > 2 else d1
        recent_hi = float(pd.to_numeric(hist.high, errors="coerce").tail(8).max())
        recent_lo = float(pd.to_numeric(hist.low, errors="coerce").tail(8).min())
        from_peak = _v10_pct(price, recent_hi)       # negative after a peak
        from_low = _v10_pct(price, recent_lo)        # positive after a low

        s15 = v71_structure_tf(d15, "15m")
        s1h = v71_structure_tf(d1h, "1H")
        e15 = v71_ema_transition(d15, 8)
        e4 = v71_ema_transition(d4, 6)
        i15 = indicators(d15)
        i4 = indicators(d4)
        if i15.empty or i4.empty:
            return None
        r15, r4 = i15.iloc[-1], i4.iloc[-1]
        rsi = v6_num(r15.get("rsi"), 50)
        vol = v6_num(r15.get("vol_ratio"), 1)
        macd = v6_num(r15.get("macd"), 0)
        sig = v6_num(r15.get("macd_signal"), 0)

        # Detect whether the multi-day move is exceptional in either direction.
        pump_strength = max(
            ret3 / V10_EXTREME_PUMP_3D if ret3 > 0 else 0,
            ret5 / V10_EXTREME_PUMP_5D if ret5 > 0 else 0,
            ret7 / V10_EXTREME_PUMP_7D if ret7 > 0 else 0,
        )
        dump_strength = max(
            abs(ret3) / abs(V10_EXTREME_DUMP_3D) if ret3 < 0 else 0,
            abs(ret5) / abs(V10_EXTREME_DUMP_5D) if ret5 < 0 else 0,
            abs(ret7) / abs(V10_EXTREME_DUMP_7D) if ret7 < 0 else 0,
        )
        side_extreme = "PUMP" if pump_strength >= 1 and pump_strength >= dump_strength else "DUMP" if dump_strength >= 1 else None
        if not side_extreme:
            return None

        results = []

        # ---------------- MASSIVE PUMP ----------------
        if side_extreme == "PUMP":
            # LONG = healthy consolidation / next leg.
            ls, lr, lb = 0, [], []
            ls += 25; lr.append("massive multi-day pump")
            if ret7 >= V10_EXTREME_PUMP_7D: ls += 15; lr.append(f"7D +{ret7:.0f}%")
            elif ret5 >= V10_EXTREME_PUMP_5D: ls += 12; lr.append(f"5D +{ret5:.0f}%")
            elif ret3 >= V10_EXTREME_PUMP_3D: ls += 9; lr.append(f"3D +{ret3:.0f}%")
            if s15.get("hh") and s15.get("hl"): ls += 25; lr.append("15m HH + HL intact")
            elif s15.get("hh") or s15.get("hl"): ls += 12; lr.append("15m early HH/HL")
            if s1h.get("hh") and s1h.get("hl"): ls += 15; lr.append("1H HH + HL")
            elif s1h.get("hh") or s1h.get("hl"): ls += 7; lr.append("1H improving structure")
            if e15.get("bullish"): ls += 6; lr.append("15m bullish EMA")
            if e4.get("bullish"): ls += 6; lr.append("4H bullish EMA")
            if macd > sig: ls += 6; lr.append("MACD bullish")
            if vol >= 1.2: ls += 6; lr.append(f"volume {vol:.1f}x")
            if 2 <= max(0, -from_peak) <= V10_NEAR_HIGH_PULLBACK: ls += 10; lr.append(f"controlled pullback {abs(from_peak):.1f}% from peak")
            if from_peak < -V10_MAX_SHORT_FROM_PEAK: lb.append("pump has already retraced too far for a clean next-leg entry")
            if s15.get("lh") and s15.get("ll"): ls -= 25; lb.append("bearish reversal structure; prefer SHORT analysis")
            results.append(("LONG", ls, lr, lb, "PUMP NEXT-LEG"))

            # SHORT = blow-off/reversal after the pump.
            ss, sr, sb = 0, [], []
            ss += 25; sr.append("massive multi-day pump")
            if ret7 >= V10_EXTREME_PUMP_7D: ss += 15; sr.append(f"7D +{ret7:.0f}%")
            elif ret5 >= V10_EXTREME_PUMP_5D: ss += 12; sr.append(f"5D +{ret5:.0f}%")
            elif ret3 >= V10_EXTREME_PUMP_3D: ss += 9; sr.append(f"3D +{ret3:.0f}%")
            if s15.get("lh") and s15.get("ll"): ss += 32; sr.append("15m LH + LL")
            elif s15.get("lh") or s15.get("ll"): ss += 14; sr.append("15m early LH/LL")
            if s1h.get("lh") and s1h.get("ll"): ss += 15; sr.append("1H LH + LL")
            elif s1h.get("lh") or s1h.get("ll"): ss += 7; sr.append("1H weakening structure")
            if e15.get("bearish"): ss += 7; sr.append("15m bearish EMA")
            if e15.get("fresh_bearish"): ss += 9; sr.append("fresh 15m EMA transition")
            if e4.get("bearish"): ss += 6; sr.append("4H bearish EMA")
            if macd < sig: ss += 7; sr.append("MACD bearish")
            if vol >= 1.5: ss += 8; sr.append(f"volume {vol:.1f}x")
            if from_peak <= -2: ss += 10; sr.append(f"{abs(from_peak):.1f}% off recent peak")
            if from_peak < -V10_MAX_SHORT_FROM_PEAK: sb.append("dump already too far from peak; short may be late")
            if not (s15.get("lh") and s15.get("ll")): sb.append("wait for confirmed LH + LL before shorting the blow-off")
            results.append(("SHORT", ss, sr, sb, "PUMP REVERSAL"))

        # ---------------- MASSIVE DUMP ----------------
        if side_extreme == "DUMP":
            # LONG = capitulation/reversal.
            ls, lr, lb = 0, [], []
            ls += 25; lr.append("massive multi-day dump")
            if ret7 <= V10_EXTREME_DUMP_7D: ls += 15; lr.append(f"7D {ret7:.0f}%")
            elif ret5 <= V10_EXTREME_DUMP_5D: ls += 12; lr.append(f"5D {ret5:.0f}%")
            elif ret3 <= V10_EXTREME_DUMP_3D: ls += 9; lr.append(f"3D {ret3:.0f}%")
            if s15.get("hh") and s15.get("hl"): ls += 32; lr.append("15m HH + HL")
            elif s15.get("hh") or s15.get("hl"): ls += 14; lr.append("15m early HH/HL")
            if s1h.get("hh") and s1h.get("hl"): ls += 15; lr.append("1H HH + HL")
            elif s1h.get("hh") or s1h.get("hl"): ls += 7; lr.append("1H improving structure")
            if e15.get("bullish"): ls += 7; lr.append("15m bullish EMA")
            if e15.get("fresh_bullish"): ls += 9; lr.append("fresh 15m EMA transition")
            if e4.get("bullish"): ls += 6; lr.append("4H bullish EMA")
            if macd > sig: ls += 7; lr.append("MACD bullish")
            if vol >= 1.5: ls += 8; lr.append(f"volume {vol:.1f}x")
            if from_low <= V10_MAX_LONG_FROM_LOW: ls += 10; lr.append(f"near capitulation low (+{from_low:.1f}%)")
            if from_low > V10_MAX_LONG_FROM_LOW: lb.append("rebound already too far for a clean capitulation entry")
            if s15.get("lh") and s15.get("ll"): ls -= 25; lb.append("bearish continuation; prefer SHORT analysis")
            results.append(("LONG", ls, lr, lb, "DUMP REVERSAL"))

            # SHORT = continuation after a massive dump.
            ss, sr, sb = 0, [], []
            ss += 25; sr.append("massive multi-day dump")
            if ret7 <= V10_EXTREME_DUMP_7D: ss += 15; sr.append(f"7D {ret7:.0f}%")
            elif ret5 <= V10_EXTREME_DUMP_5D: ss += 12; sr.append(f"5D {ret5:.0f}%")
            elif ret3 <= V10_EXTREME_DUMP_3D: ss += 9; sr.append(f"3D {ret3:.0f}%")
            if s15.get("lh") and s15.get("ll"): ss += 32; sr.append("15m LH + LL")
            elif s15.get("lh") or s15.get("ll"): ss += 14; sr.append("15m early LH/LL")
            if s1h.get("lh") and s1h.get("ll"): ss += 15; sr.append("1H LH + LL")
            elif s1h.get("lh") or s1h.get("ll"): ss += 7; sr.append("1H weakening structure")
            if e15.get("bearish"): ss += 7; sr.append("15m bearish EMA")
            if e4.get("bearish"): ss += 6; sr.append("4H bearish EMA")
            if macd < sig: ss += 7; sr.append("MACD bearish")
            if vol >= 1.2: ss += 6; sr.append(f"volume {vol:.1f}x")
            if not (s15.get("lh") and s15.get("ll")): sb.append("wait for confirmed LH + LL before shorting continued weakness")
            if from_low > V10_MAX_LONG_FROM_LOW: ss -= 15; sb.append("bounce is already large; avoid chasing")
            results.append(("SHORT", ss, sr, sb, "DUMP CONTINUATION"))

        valid = []
        for side, score, reasons, blockers, state in results:
            hard = any("wait for confirmed" in b.lower() or "already too far" in b.lower() for b in blockers)
            if score >= V10_MIN_SCORE and not hard:
                valid.append((side, score, reasons, blockers, state))
        best = max(results, key=lambda z: z[1]) if results else None
        if not best:
            return None
        chosen = max(valid, key=lambda z: z[1]) if valid else best
        side, score, reasons, blockers, state = chosen
        return {
            "pair": pair, "symbol": symbol, "price": price,
            "side": side if valid else "WAIT", "score": int(max(0, min(100, score))),
            "valid": bool(valid), "state": state, "reasons": reasons[:10], "blockers": blockers[:8],
            "ret3d": ret3, "ret5d": ret5, "ret7d": ret7,
            "from_peak_pct": from_peak, "from_low_pct": from_low,
            "structure15": s15.get("state"), "structure1h": s1h.get("state"),
            "rsi": rsi, "volume": vol,
            "entry": price,
        }
    except Exception:
        return None


def v10_extreme_record(x):
    if not x:
        return None
    return {
        "Coin": x.get("symbol"), "Pair": x.get("pair"), "Direction": x.get("side"),
        "Score": x.get("score", 0),
        "Valid": "✅ TRADE CANDIDATE" if x.get("valid") else "WAIT",
        "Regime": x.get("state", ""), "15m Structure": x.get("structure15", ""),
        "Volume": f"{x.get('volume', 0):.1f}x",
        "Current": fmt(x.get("price", x.get("entry", 0))),
        "Entry": fmt(x.get("entry", x.get("price", 0))),
        "Stop": "—", "TP1": "—", "TP2": "—", "TP3": "—",
        "Blockers": "; ".join(x.get("blockers", [])) or "—",
        "Reasons": " | ".join(x.get("reasons", [])[:5]),
        "Extreme Move": x.get("state", ""), "3D %": x.get("ret3d"), "5D %": x.get("ret5d"),
        "7D %": x.get("ret7d"), "From Peak %": x.get("from_peak_pct"),
        "From Low %": x.get("from_low_pct"), "source": "V10 EXTREME MOVE",
        "setup": x,
    }


def v10_scan_all_extreme(progress=None, max_workers=6):
    """Scan the ENTIRE active USDT Futures universe for V10 extreme-move setups.

    This is deliberately separate from the normal 25/60-contract V6 scan.
    Every active contract is considered; a missing live-price-feed key falls
    back to the latest completed 15m close instead of discarding the contract.
    """
    instruments = active_instruments("USDT")
    try:
        prices = futures_prices()
    except Exception:
        prices = {}

    items, seen = [], set()
    for inst in instruments:
        pair = v61_instrument_pair(inst)
        if not pair:
            continue
        canonical = str(pair).strip().upper()
        if canonical in seen:
            continue
        seen.add(canonical)

        live = v61_price_for_pair(prices, pair)
        symbol = v61_symbol(inst, pair)
        if meme_only and not any(w in symbol or w in canonical for w in MEME_WORDS):
            continue
        items.append((pair, symbol, live))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _fetch(item):
        pair, symbol, live = item
        try:
            # V10 needs daily history for 3D/5D/7D plus intraday structure.
            tf_data = {
                "15m": get_tf(pair, "15m", 12),
                "1H": get_tf(pair, "1H", 45),
                "4H": get_tf(pair, "4H", 120),
                "1D": get_tf(pair, "1D", 180),
            }
            p = live
            if not np.isfinite(p) or p <= 0:
                c = completed(tf_data["15m"])
                if c is not None and not c.empty:
                    p = v6_num(c.iloc[-1].get("close"))
            if not np.isfinite(p) or p <= 0:
                return None
            sig = v10_extreme_move_signal(pair, symbol, tf_data, p)
            if not sig:
                return None
            rec = v10_extreme_record(sig)
            if rec:
                rec["price_source"] = "LIVE_FEED" if np.isfinite(live) and live > 0 else "15M_CANDLE_FALLBACK"
            return rec
        except Exception:
            return None

    results, done = [], 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_fetch, item) for item in items]
        for fut in as_completed(futs):
            done += 1
            if progress:
                progress(done, len(items))
            r = fut.result()
            if r:
                results.append(r)

    results.sort(key=lambda r: (r.get("Valid") != "✅ TRADE CANDIDATE",
                                -float(r.get("Score", 0))))
    return results, len(items)




# ----------------------- V12 ACTIONABLE SIGNAL BOARD ---------------------------
def v12_signal_row(r):
    direction = str(r.get("Direction", "")).upper()
    valid = str(r.get("Valid", ""))
    if valid == "✅ TRADE CANDIDATE" and direction == "LONG":
        decision = "🟢 LONG"
    elif valid == "✅ TRADE CANDIDATE" and direction == "SHORT":
        decision = "🔴 SHORT"
    else:
        decision = "⚪ WAIT"

    return {
        "Signal": decision,
        "Coin": r.get("Coin"),
        "Setup": r.get("Extreme Move"),
        "Score": r.get("Score"),
        "Current": r.get("Current"),
        "3D %": r.get("3D %"),
        "5D %": r.get("5D %"),
        "7D %": r.get("7D %"),
        "From Peak %": r.get("From Peak %"),
        "From Low %": r.get("From Low %"),
        "15m Structure": r.get("15m Structure"),
        "1H Structure": r.get("1H Structure"),
        "Entry Trigger": r.get("Entry Trigger"),
        "Invalidation": r.get("Invalidation"),
        "15m S1": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("S1"),
        "15m S2": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("S2"),
        "15m S3": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("S3"),
        "15m R1": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("R1"),
        "15m R2": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("R2"),
        "15m R3": (r.get("MTF_SR", {}).get("15m", {}) or {}).get("R3"),
        "4H S1": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("S1"),
        "4H S2": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("S2"),
        "4H S3": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("S3"),
        "4H R1": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("R1"),
        "4H R2": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("R2"),
        "4H R3": (r.get("MTF_SR", {}).get("4H", {}) or {}).get("R3"),
        "1D S1": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("S1"),
        "1D S2": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("S2"),
        "1D S3": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("S3"),
        "1D R1": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("R1"),
        "1D R2": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("R2"),
        "1D R3": (r.get("MTF_SR", {}).get("1D", {}) or {}).get("R3"),
        "1W S1": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("S1"),
        "1W S2": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("S2"),
        "1W S3": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("S3"),
        "1W R1": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("R1"),
        "1W R2": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("R2"),
        "1W R3": (r.get("MTF_SR", {}).get("1W", {}) or {}).get("R3"),
        "Why": r.get("Reasons"),
    }


def v12_actionable_candidates(records, minimum_score=78):
    """Return only actionable LONG/SHORT signals for manual trading."""
    out = []
    for r in records or []:
        try:
            score = float(r.get("Score", 0))
        except Exception:
            score = 0
        if r.get("Valid") != "✅ TRADE CANDIDATE":
            continue
        if score < minimum_score:
            continue
        if str(r.get("Direction", "")).upper() not in ("LONG", "SHORT"):
            continue
        out.append(r)
    return sorted(out, key=lambda x: float(x.get("Score", 0)), reverse=True)


def v12_render_signal_cards(records, title="🎯 ACTIONABLE MANUAL TRADING SIGNALS"):
    candidates = v12_actionable_candidates(records)
    st.subheader(title)
    if not candidates:
        st.info("No confirmed LONG/SHORT signals right now. WAIT is the correct decision.")
        return

    longs = [r for r in candidates if str(r.get("Direction")).upper() == "LONG"]
    shorts = [r for r in candidates if str(r.get("Direction")).upper() == "SHORT"]

    a, b, c = st.columns(3)
    a.metric("🟢 LONG signals", len(longs))
    b.metric("🔴 SHORT signals", len(shorts))
    c.metric("Total actionable", len(candidates))

    def render_side(side_records, heading, emoji):
        if not side_records:
            st.caption(f"{emoji} {heading}: none")
            return
        st.markdown(f"### {emoji} {heading}")
        for r in side_records[:15]:
            score = r.get("Score", 0)
            st.markdown(
                f"**{r.get('Coin','—')} — {r.get('Extreme Move','—')} — "
                f"Score {score}/100**"
            )
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Current", r.get("Current", "—"))
            x2.metric("3D", r.get("3D %", "—"))
            x3.metric("5D", r.get("5D %", "—"))
            x4.metric("7D", r.get("7D %", "—"))
            st.write(
                f"**Entry trigger:** {r.get('Entry Trigger','—')}  |  "
                f"**Invalidation:** {r.get('Invalidation','—')}"
            )
            st.write(
                f"**Structure:** 15m {r.get('15m Structure','—')} | "
                f"1H {r.get('1H Structure','—')}"
            )
            if r.get("MTF_SR"):
                st.markdown("**📐 Multi-timeframe Support & Resistance**")
                st.dataframe(
                    pd.DataFrame(v13_sr_columns(r["MTF_SR"])),
                    use_container_width=True,
                    hide_index=True,
                )
            st.write(f"**Why:** {r.get('Reasons','—')}")
            st.divider()

    render_side(longs, "LONG — manual candidates", "🟢")
    render_side(shorts, "SHORT — manual candidates", "🔴")

    st.markdown("#### 📋 Compact signal table")
    st.dataframe(
        pd.DataFrame([v12_signal_row(r) for r in candidates]),
        use_container_width=True,
        hide_index=True,
    )


st.divider()
st.subheader("🎯 CONFIRMED LONG / SHORT SIGNALS")
st.caption(
    "Scans the whole Futures universe for higher-confidence LONG/SHORT setups. "
    "Use this after the simple TODAY structure view when you want a stricter trade filter."
)

v12_min = st.slider(
    "Minimum actionable signal score",
    70, 95, 78, 1,
    key="v12_min_signal",
)

if st.button(
    "🎯 SCAN ALL COINS & GENERATE LONG / SHORT SIGNALS",
    type="primary",
    key="v12_signal_scan",
):
    bar_v12 = st.progress(0, text="Scanning all active Futures for actionable signals…")
    def _v12_progress(done, total):
        bar_v12.progress(
            int(done / max(total, 1) * 100),
            text=f"Analyzing {done}/{total} Futures contracts…"
        )
    with st.spinner("Finding extreme moves and deciding LONG / SHORT / WAIT…"):
        _v12_records, _v12_total = v10_scan_all_extreme(
            progress=_v12_progress,
            max_workers=6,
        )
    _v12_records = v13_attach_sr_to_records(_v12_records)
    bar_v12.progress(100, text=f"Complete — {_v12_total} contracts scanned + 15m/4H/1D/1W S/R")
    st.session_state["v12_signal_records"] = _v12_records
    st.session_state["v12_signal_total"] = _v12_total
    st.session_state["v12_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_v12_records = st.session_state.get("v12_signal_records", [])
if _v12_records:
    st.caption(
        f"Last signal scan: {st.session_state.get('v12_signal_time','—')} | "
        f"Contracts scanned: {st.session_state.get('v12_signal_total','—')}"
    )
    # Render only actual LONG/SHORT candidates prominently.
    _v12_filtered = [
        r for r in _v12_records
        if float(r.get("Score", 0) or 0) >= v12_min
    ]
    v12_render_signal_cards(_v12_filtered)


# ---------------- TODAY SIMPLE STRUCTURE DASHBOARD ----------------------------
st.divider()
st.subheader("⭐ TODAY — SIMPLE HH / HL / LH / LL MARKET VIEW")
st.caption(
    "Simple answer from the latest completed 15m structure: HH + HL = bullish; LH + LL = bearish. "
    "This section does not require a 78/100 trade score. It is designed to show you what is happening today first."
)

if st.button("⭐ SCAN ALL COINS — WHAT IS HAPPENING TODAY?", type="secondary", key="v18_today_scan_button"):
    bar_today = st.progress(0, text="Scanning all Futures for today's HH/HL/LH/LL structure…")
    def _today_progress(done, total):
        bar_today.progress(int(done / max(total, 1) * 100), text=f"Analyzing {done}/{total} Futures…")
    with st.spinner("Fetching 15m/1H/4H data and finding today's structure…"):
        _today_scan, _today_total = v61_scan_all(_today_progress, max_workers=6)
    st.session_state["v18_today_results"] = _today_scan
    st.session_state["v18_today_total"] = _today_total
    st.session_state["v18_today_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bar_today.progress(100, text=f"Complete — {_today_total} contracts scanned")

_today_scan = st.session_state.get("v18_today_results", [])
if _today_scan:
    def _today_rows(side=None):
        rows=[]
        for a in _today_scan:
            ts = a.get("today_structure") or {}
            if side and ts.get("side") != side:
                continue
            rows.append({
                "Coin": a.get("symbol", a.get("pair", "—")),
                "Current": v61_fmt_price(a.get("price")),
                "Today": ts.get("label", "⚪ NO CLEAR STRUCTURE"),
                "Higher High": "YES" if ts.get("hh") else "NO",
                "Higher Low": "YES" if ts.get("hl") else "NO",
                "Lower High": "YES" if ts.get("lh") else "NO",
                "Lower Low": "YES" if ts.get("ll") else "NO",
                "High change": f"{ts.get('high_change_pct', float('nan')):.2f}%" if np.isfinite(v6_num(ts.get('high_change_pct'), np.nan)) else "—",
                "Low change": f"{ts.get('low_change_pct', float('nan')):.2f}%" if np.isfinite(v6_num(ts.get('low_change_pct'), np.nan)) else "—",
            })
        return rows

    simple = _today_scan
    longs_today = [a for a in simple if (a.get("today_structure") or {}).get("side") == "LONG"]
    shorts_today = [a for a in simple if (a.get("today_structure") or {}).get("side") == "SHORT"]
    watch_long = [a for a in simple if (a.get("today_structure") or {}).get("side") == "WATCH LONG"]
    watch_short = [a for a in simple if (a.get("today_structure") or {}).get("side") == "WATCH SHORT"]
    st.caption(f"Last simple scan: {st.session_state.get('v18_today_time','—')} | Contracts scanned: {st.session_state.get('v18_today_total','—')}")
    z1,z2,z3,z4 = st.columns(4)
    z1.metric("🟢 HH + HL", len(longs_today))
    z2.metric("🔴 LH + LL", len(shorts_today))
    z3.metric("🟡 Developing LONG", len(watch_long))
    z4.metric("🟠 Developing SHORT", len(watch_short))

    if longs_today:
        st.markdown("### 🟢 LONG TODAY — HIGHER HIGH + HIGHER LOW")
        st.dataframe(pd.DataFrame(_today_rows("LONG")), use_container_width=True, hide_index=True)
    else:
        st.info("No coin currently has both Higher High + Higher Low in the latest completed 24h 15m structure.")

    if shorts_today:
        st.markdown("### 🔴 SHORT TODAY — LOWER HIGH + LOWER LOW")
        st.dataframe(pd.DataFrame(_today_rows("SHORT")), use_container_width=True, hide_index=True)
    else:
        st.info("No coin currently has both Lower High + Lower Low in the latest completed 24h 15m structure.")

    with st.expander("🟡/🟠 Developing structures", expanded=False):
        dev_rows = _today_rows("WATCH LONG") + _today_rows("WATCH SHORT")
        if dev_rows:
            st.dataframe(pd.DataFrame(dev_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No developing structure detected.")
else:
    st.info("Click **SCAN ALL COINS — WHAT IS HAPPENING TODAY?** to get the simple market-wide HH/HL/LH/LL answer.")


# --------------------- V19 STANDALONE MTF S/R ---------------------------------
st.divider()
st.subheader("📐 MTF SUPPORT & RESISTANCE — ANY COIN")
st.caption(
    "Enter any active CoinDCX Futures pair to see Support 1/2/3 and Resistance 1/2/3 "
    "on 15m, 4H, 1D and 1W. This is independent of LONG/SHORT signal scoring."
)

_v19_sr_coin = st.text_input(
    "Coin / Futures pair for MTF S/R",
    placeholder="LSK_USDT, B-LSK_USDT, DOGE_USDT",
    key="v19_mtf_sr_coin",
)

if st.button("📐 SHOW 15m / 4H / 1D / 1W SUPPORT & RESISTANCE", key="v19_mtf_sr_button"):
    try:
        with st.spinner("Finding the Futures contract and calculating multi-timeframe S/R…"):
            _req = normalize(_v19_sr_coin)
            if not _req:
                st.error("Enter a coin or Futures pair first.")
                st.stop()

            _prices = futures_prices()
            _matches = []
            for _q in (margin, "USDT", "INR"):
                for _raw in active_instruments(_q):
                    _pair = v61_instrument_pair(_raw)
                    if not _pair:
                        continue
                    _active_symbol = _pair
                    if isinstance(_raw, dict):
                        _active_symbol = str(
                            _raw.get("symbol") or _raw.get("display_name") or
                            _raw.get("market") or _raw.get("pair") or _pair
                        )
                    if not coin_matches(_pair, _active_symbol, _req, _q):
                        continue
                    _prec = price_record_for_pair(_prices, _pair) if "price_record_for_pair" in globals() else None
                    if _prec is None:
                        _target = str(_pair).upper().replace("-", "").replace("_", "")
                        for _key, _rec in (_prices or {}).items():
                            _ident = str(_key).upper().replace("-", "").replace("_", "")
                            if _ident == _target or (_ident.startswith("B") and _ident[1:] == _target):
                                _prec = _rec if isinstance(_rec, dict) else {"pair": _key, "price": _rec}
                                break
                    if _prec is None:
                        try:
                            _probe = completed(get_tf(_pair, "15m", 2))
                            if not _probe.empty:
                                _lc = float(_probe.iloc[-1]["close"])
                                _prec = {"pair": _pair, "price": _lc, "ls": _lc}
                        except Exception:
                            _prec = None
                    if _prec is not None:
                        _price = _prec.get("price", _prec.get("ls", _prec.get("last_price")))
                        try:
                            _price = float(_price)
                        except Exception:
                            _price = None
                        if _price and _price > 0:
                            _matches.append((_pair, _price, _active_symbol))
                if _matches:
                    break

            if not _matches:
                st.error(f"Could not find an active CoinDCX Futures contract matching '{_v19_sr_coin}'.")
                st.stop()

            _sr_pair, _sr_current, _sr_symbol = _matches[0]
            _sr_tf_data = {
                "15m": get_tf(_sr_pair, "15m", 12),
                "4H": get_tf(_sr_pair, "4H", 120),
                "1D": get_tf(_sr_pair, "1D", 180),
                "1W": get_tf(_sr_pair, "1W", 365),
            }
            _sr = v13_mtf_support_resistance(_sr_tf_data, _sr_current)
            _sr_summary = v14_sr_summary(_sr, _sr_current)
            st.session_state["v19_mtf_sr_result"] = {
                "pair": _sr_pair,
                "symbol": _sr_symbol,
                "current": _sr_current,
                "sr": _sr,
                "summary": _sr_summary,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
    except Exception as _e:
        st.error(f"MTF S/R calculation failed: {_e}")

_v19_sr_result = st.session_state.get("v19_mtf_sr_result")
if _v19_sr_result:
    st.markdown(f"### {_v19_sr_result.get('symbol', _v19_sr_result.get('pair', '—'))}")
    _sr_cur = _v19_sr_result["current"]
    _sr_sum = _v19_sr_result.get("summary", {})
    _c1, _c2, _c3 = st.columns(3)
    _c1.metric("Current Price", v13_format_price(_sr_cur))
    _c2.metric(
        "Nearest Support",
        v13_format_price(_sr_sum.get("nearest_support")),
        f"{_sr_sum.get('support_dist_pct'):+.2f}%" if _sr_sum.get("support_dist_pct") is not None else "—",
    )
    _c3.metric(
        "Nearest Resistance",
        v13_format_price(_sr_sum.get("nearest_resistance")),
        f"{_sr_sum.get('resistance_dist_pct'):+.2f}%" if _sr_sum.get("resistance_dist_pct") is not None else "—",
    )
    st.dataframe(
        pd.DataFrame(v13_sr_columns(_v19_sr_result.get("sr", {}))),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Nearest support: {_sr_sum.get('support_tf','—')} | "
        f"Nearest resistance: {_sr_sum.get('resistance_tf','—')} | "
        f"Calculated: {_v19_sr_result.get('time','—')}"
    )


# --------------------- V10 WHOLE-MARKET SCAN CONTROL ---------------------------
st.divider()
st.subheader("🧨 V10 Whole-Market Extreme Move Hunter")
st.caption(
    "Scans every active CoinDCX USDT Futures contract and shows only coins matching "
    "the massive-pump / massive-dump conditions. This is separate from the normal "
    "25/60-contract intraday shortlist."
)
v10w1, v10w2 = st.columns([1, 3])
with v10w1:
    v10_workers = st.slider("V10 concurrent workers", 2, 10, 6, 1, key="v10_workers")
with v10w2:
    st.info(
        "The hunter looks for 3D/5D/7D extreme moves, then classifies the current "
        "state as LONG next-leg, SHORT reversal, LONG dump-reversal, SHORT dump-continuation, or WAIT."
    )

if st.button("🧨 SCAN ALL COINDCX FUTURES FOR EXTREME MOVES", type="primary", key="v10_all_scan"):
    bar_v10 = st.progress(0, text="Starting whole-market extreme-move scan…")
    def _v10_progress(done, total):
        bar_v10.progress(int(done / max(total, 1) * 100),
                         text=f"Scanning extreme moves: {done}/{total}")
    with st.spinner("Scanning every active Futures contract…"):
        _v10_scan, _v10_total = v10_scan_all_extreme(_v10_progress, max_workers=v10_workers)
    bar_v10.progress(100, text=f"Extreme scan complete: {_v10_total} active contracts checked")
    st.session_state["v10_all_scan_results"] = _v10_scan
    st.session_state["v10_all_scan_total"] = _v10_total
    st.session_state["v10_all_scan_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

_v10_all = st.session_state.get("v10_all_scan_results", [])
if _v10_all:
    _v10_trades = [x for x in _v10_all if x.get("Valid") == "✅ TRADE CANDIDATE"]
    _v10_longs = [x for x in _v10_trades if x.get("Direction") == "LONG"]
    _v10_shorts = [x for x in _v10_trades if x.get("Direction") == "SHORT"]

    st.caption(
        f"Last whole-market scan: {st.session_state.get('v10_all_scan_time','—')} | "
        f"Contracts checked: {st.session_state.get('v10_all_scan_total','—')}"
    )
    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Extreme candidates", len(_v10_trades))
    q2.metric("🟢 LONG", len(_v10_longs))
    q3.metric("🔴 SHORT", len(_v10_shorts))
    q4.metric("WAIT / watch", max(0, len(_v10_all) - len(_v10_trades)))

    _rows = []
    for r in _v10_all:
        _rows.append({
            "Coin": r.get("Coin"),
            "State": r.get("Extreme Move"),
            "Direction": r.get("Direction"),
            "Score": r.get("Score"),
            "3D %": r.get("3D %"),
            "5D %": r.get("5D %"),
            "7D %": r.get("7D %"),
            "From Peak %": r.get("From Peak %"),
            "From Low %": r.get("From Low %"),
            "15m Structure": r.get("15m Structure"),
            "Volume": r.get("Volume"),
            "Decision": "🟢 PAPER LONG" if r.get("Direction") == "LONG" and r.get("Valid") == "✅ TRADE CANDIDATE"
                       else "🔴 PAPER SHORT" if r.get("Direction") == "SHORT" and r.get("Valid") == "✅ TRADE CANDIDATE"
                       else "⚪ WAIT",
            "Reasons": r.get("Reasons"),
            "Blockers": r.get("Blockers"),
        })
    st.dataframe(
        pd.DataFrame(_rows).sort_values(["Decision", "Score"], ascending=[True, False]),
        use_container_width=True, hide_index=True
    )

# --------------------------- V10 EXTREME MOVE UI -----------------------------
if st.session_state.get("v6_scan_results"):
    _all_v10 = st.session_state.get("v6_scan_results", [])
    _extreme10 = [r for r in _all_v10 if r.get("source") == "V10 EXTREME MOVE"]
    if _extreme10:
        st.subheader("🧨 V10 Extreme Move Reversal + Next-Leg Hunter")
        st.caption(
            "Finds multi-day +150%/+250%/+400% pumps and -65%/-75%/-85% dumps. "
            "A pump can become LONG next-leg or SHORT reversal; a dump can become LONG reversal or SHORT continuation."
        )
        _v10_rows = []
        for r in _extreme10:
            _v10_rows.append({
                "Coin": r.get("Coin"), "State": r.get("Extreme Move"), "Side": r.get("Direction"),
                "Score": r.get("Score"), "3D %": round(r.get("3D %"),1) if isinstance(r.get("3D %"),(int,float)) else r.get("3D %"),
                "5D %": round(r.get("5D %"),1) if isinstance(r.get("5D %"),(int,float)) else r.get("5D %"),
                "7D %": round(r.get("7D %"),1) if isinstance(r.get("7D %"),(int,float)) else r.get("7D %"),
                "From Peak %": round(r.get("From Peak %"),1) if isinstance(r.get("From Peak %"),(int,float)) else r.get("From Peak %"),
                "From Low %": round(r.get("From Low %"),1) if isinstance(r.get("From Low %"),(int,float)) else r.get("From Low %"),
                "15m": r.get("15m Structure"), "Decision": "PAPER TRADE" if r.get("Valid")=="✅ TRADE CANDIDATE" else "WAIT",
                "Why": r.get("Reasons"), "Blockers": r.get("Blockers"),
            })
        st.dataframe(pd.DataFrame(_v10_rows).sort_values(["Score"], ascending=False), use_container_width=True, hide_index=True)


st.divider()
st.caption("Core engines retained internally for the remaining workflows. Legacy V6/V6.1/V6.2/V7/Hot-ATH scanner panels have been removed from the visible app to keep the trading workflow simple.")
