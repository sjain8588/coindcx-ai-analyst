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
st.caption("Automatically fetches CoinDCX Futures history, finds similar historical setups, and explains what happened next.")

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
    if i < 40:
        return None
    row = x.iloc[i]
    close = safe(row.close)
    atr = safe(row.atr)
    if not np.isfinite(close) or close <= 0:
        return None
    look = x.iloc[max(0,i-24):i+1]
    ret4 = (close / safe(x.iloc[i-4].close)-1)*100 if i >= 4 else np.nan
    ret12 = (close / safe(x.iloc[i-12].close)-1)*100 if i >= 12 else np.nan
    ret24 = (close / safe(x.iloc[i-24].close)-1)*100 if i >= 24 else np.nan
    hi24 = safe(look.high.max())
    lo24 = safe(look.low.min())
    range_pct = ((hi24-lo24)/close*100) if hi24 > 0 else np.nan
    body = abs(safe(row.close)-safe(row.open))/close*100
    upper_wick = (safe(row.high)-max(safe(row.open),safe(row.close)))/close*100
    lower_wick = (min(safe(row.open),safe(row.close))-safe(row.low))/close*100
    ema20 = safe(row.ema20); ema50 = safe(row.ema50); ema100 = safe(row.ema100)
    stack = 1 if ema20 > ema50 > ema100 else -1 if ema20 < ema50 < ema100 else 0
    return {
        "ret4":ret4, "ret12":ret12, "ret24":ret24,
        "range_pct":range_pct,
        "body_pct":body, "upper_wick_pct":upper_wick, "lower_wick_pct":lower_wick,
        "rsi":safe(row.rsi), "adx":safe(row.adx), "vol_ratio":safe(row.vol_ratio),
        "atr_pct":safe(row.atr_pct), "ema_stack":stack,
        "above20":1 if close > ema20 else 0,
        "above50":1 if close > ema50 else 0,
        "above100":1 if close > ema100 else 0,
        "structure":structure(x.iloc[:i+1]),
    }

def feature_vector(f):
    if f is None: return None
    # Normalized features make BTC, DOGE and tiny coins comparable.
    vals = [
        safe(f["ret4"]), safe(f["ret12"]), safe(f["ret24"]), safe(f["range_pct"]),
        safe(f["body_pct"]), safe(f["upper_wick_pct"]), safe(f["lower_wick_pct"]),
        safe(f["rsi"]), safe(f["adx"]), safe(f["vol_ratio"]), safe(f["atr_pct"]),
        safe(f["ema_stack"]), safe(f["above20"]), safe(f["above50"]), safe(f["above100"])
    ]
    return np.array([0 if not np.isfinite(v) else v for v in vals], dtype=float)

def scaled_distance(a,b):
    # Robust scale: percentage/indicator ranges are intentionally normalized.
    scales = np.array([10,20,35,20,5,5,5,20,20,3,10,1,1,1,1], dtype=float)
    return float(np.sqrt(np.mean(((a-b)/scales)**2)))

def event_outcome(x, i, horizon, direction="UP"):
    if i+1 >= len(x): return None
    future = x.iloc[i+1:min(len(x), i+1+horizon)]
    if future.empty: return None
    entry = safe(x.iloc[i].close)
    if entry <= 0: return None
    end_ret = (safe(future.iloc[-1].close)/entry-1)*100
    best = (safe(future.high.max())/entry-1)*100
    worst = (safe(future.low.min())/entry-1)*100
    # A practical classification, not a promise. For an UP event, continuation
    # means the move kept working; for a DOWN event, continuation means the
    # breakdown kept working.
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
    return {"end":end_ret,"best":best,"worst":worst,"label":label}

def find_pump_events(d, horizon=24, min_pump=15):
    """Find completed historical pump events without using future data to define the event."""
    if d is None or len(d) < 90:
        return []
    x = indicators(completed(d))
    events = []
    last_event = -999
    for i in range(40, len(x)-horizon-1):
        f = rolling_features(x,i)
        if not f: continue
        # Event is a completed 4H/1D move already visible at time i.
        if safe(f["ret24"]) >= min_pump:
            # Keep distinct events separated so one long pump does not dominate.
            if i-last_event < 12: continue
            outcome = event_outcome(x,i,horizon,"UP")
            if outcome:
                events.append({"i":i,"time":x.iloc[i].time, "features":f, "vector":feature_vector(f), "outcome":outcome})
                last_event=i
    return events

def find_breakout_events(d, mode="ATH", horizon=24):
    if d is None or len(d) < 90: return []
    x = indicators(completed(d))
    events=[]
    last=-999
    for i in range(40, len(x)-horizon-1):
        hist=x.iloc[:i]
        price=safe(x.iloc[i].close)
        if price <= 0: continue
        extreme=safe(hist.high.max() if mode=="ATH" else hist.low.min())
        if extreme <= 0: continue
        event=(price>extreme) if mode=="ATH" else (price<extreme)
        if event and i-last>=8:
            f=rolling_features(x,i)
            outcome=event_outcome(x,i,horizon,"UP" if mode=="ATH" else "DOWN")
            if f and outcome:
                events.append({"i":i,"time":x.iloc[i].time,"features":f,"vector":feature_vector(f),"outcome":outcome})
                last=i
    return events

def current_pattern(d):
    x=indicators(completed(d))
    if len(x)<50: return None
    return rolling_features(x,len(x)-1)

def classify_current_event(d):
    x=completed(d)
    if len(x)<40: return "NORMAL"
    ind=indicators(x); last=ind.iloc[-1]; hist=ind.iloc[:-1]
    price=safe(last.close)
    prior_ath=safe(hist.high.max())
    prior_atl=safe(hist.low.min())
    if prior_ath>0 and price>prior_ath: return "ATH BREAKOUT"
    if prior_atl>0 and price<prior_atl: return "ATL BREAKDOWN"
    ret24=(price/safe(x.iloc[-24].close)-1)*100 if len(x)>24 else 0
    if ret24>=15: return "HOT / PUMP"
    if ret24<=-15: return "FAST DUMP"
    return "NORMAL"

def similar_events(target_features, event_pool, max_matches=40):
    tv=feature_vector(target_features)
    if tv is None: return []
    scored=[]
    for e in event_pool:
        if e.get("vector") is None: continue
        dist=scaled_distance(tv,e["vector"])
        similarity=max(0,100*(1-dist))
        if similarity>=45:
            scored.append((similarity,e))
    scored.sort(key=lambda z:z[0],reverse=True)
    return scored[:max_matches]

def outcome_summary(matches):
    if not matches: return None
    labels=[e["outcome"]["label"] for _,e in matches]
    total=len(labels)
    counts={k:labels.count(k) for k in ["CONTINUED","SIDEWAYS / PULLBACK","DUMPED","REVERSED / BOUNCED"]}
    best=np.mean([e["outcome"]["best"] for _,e in matches])
    worst=np.mean([e["outcome"]["worst"] for _,e in matches])
    end=np.mean([e["outcome"]["end"] for _,e in matches])
    return {"total":total,"counts":counts,"continue_pct":counts["CONTINUED"]/total*100,"dump_pct":counts["DUMPED"]/total*100,"reverse_pct":counts["REVERSED / BOUNCED"]/total*100,"side_pct":counts["SIDEWAYS / PULLBACK"]/total*100,"avg_end":end,"avg_best":best,"avg_worst":worst}

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
    # pairs_signature is a tuple of pairs, making the cache key deterministic.
    pool=[]; failures=[]; now=int(time.time())
    for pair in pairs_signature:
        try:
            # 4H history is the main pattern-learning timeframe.
            d=get_tf(pair,"4H",240)
            if len(d)<100: continue
            if event_mode=="PUMP": events=find_pump_events(d,horizon=24,min_pump=15)
            elif event_mode=="ATH": events=find_breakout_events(d,"ATH",horizon=24)
            else: events=find_breakout_events(d,"ATL",horizon=24)
            for e in events:
                e["pair"]=pair
                pool.append(e)
        except Exception as exc:
            failures.append(f"{pair}: {type(exc).__name__}: {exc}")
    return pool, failures

# =============================================================================
# SIMPLE PREDICTION LANGUAGE
# =============================================================================
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
# UI SETTINGS
# =============================================================================
margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Use meme-focused learning universe",value=False,help="Turn this on if you only want meme-style contracts in the comparison universe.")
peer_limit=st.slider("Historical comparison universe",20,120,60,10,help="More coins = more historical examples but more CoinDCX API work.")

st.divider()
st.header("🔎 Analyze a Coin")
coin=st.text_input("Coin / Futures pair",placeholder="MARSCOIN, DOGE, PEPE, B-DOGE_USDT")

if st.button("🧠 Analyze Coin & Learn From CoinDCX",type="primary"):
    try:
        with st.spinner("Fetching CoinDCX history and studying the pattern..."):
            prices=futures_prices(); req=normalize(coin)
            found=[]
            for q in [margin]+[x for x in ("USDT","INR") if x!=margin]:
                for pair in active_instruments(q):
                    p=prices.get(pair)
                    if not p: continue
                    symbol=str(p.get("mkt",pair)).upper()
                    if coin_matches(pair,symbol,req,q): found.append((pair,p,symbol,q))
            if not found: st.error(f"No active CoinDCX Futures contract found for '{coin}'."); st.stop()
            found.sort(key=lambda z:(0 if z[3]==margin else 1,len(z[0])))
            pair,p,symbol,_=found[0]
            current=analyze_current_coin(pair,p)

            st.subheader(f"{symbol} — Simple Prediction")
            event=current["event"]
            st.write(f"**What the tool sees:** {event}")
            a,b,c,d=st.columns(4)
            a.metric("Current",fmt(current["current"]))
            b.metric("24h",f"{safe(p.get('pc',0),0):+.2f}%")
            b4=current["4h"]
            c.metric("4H RSI",f"{safe(b4.rsi):.1f}" if pd.notna(b4.rsi) else "—")
            d.metric("4H Volume",f"{safe(b4.vol_ratio):.1f}x" if pd.notna(b4.vol_ratio) else "—")

            # Build historical pool according to current event type.
            mode="ATH" if event=="ATH BREAKOUT" else "ATL" if event=="ATL BREAKDOWN" else "PUMP"
            universe=universe_rows(margin,meme_only,peer_limit)
            pairs_sig=tuple(z[0] for z in universe if z[0]!=pair)
            pool,failures=build_learning_pool(pairs_sig,margin,peer_limit,mode)
            matches=similar_events(current["target"],pool,max_matches=40)
            summary=outcome_summary(matches)
            title,text=human_result(summary,current)
            if title.startswith("🟢"): st.success(title)
            elif title.startswith("🔴"): st.error(title)
            else: st.warning(title)
            st.markdown(f"### {title}")
            st.write(text)

            if summary:
                st.write(f"**Similar historical cases:** {summary['total']}")
                a,b,c,d=st.columns(4)
                a.metric("Continued",f"{summary['continue_pct']:.0f}%")
                b.metric("Sideways / Pullback",f"{summary['side_pct']:.0f}%")
                c.metric("Dumped",f"{summary['dump_pct']:.0f}%")
                d.metric("Avg next-period move",f"{summary['avg_end']:+.1f}%")

            st.markdown("### 📌 What this means in simple language")
            if summary and summary["total"]>=8:
                if summary["continue_pct"]>summary["dump_pct"]:
                    st.write("Historically, coins that looked like this more often kept moving in the same direction than completely reversed. That does **not** mean this coin must do the same.")
                else:
                    st.write("Historically, similar setups often struggled after the initial move. It is better to wait for confirmation instead of chasing the move.")
            else:
                st.write("There is not enough historical evidence yet. Treat this as an observation, not a prediction.")

            st.markdown("### 👀 What to watch now")
            for item in confirmation_text(current):
                st.write("• "+item)

            st.markdown("### 📊 7-Timeframe EMA picture")
            ema_table=[]
            for tf in ["1m","5m","15m","1H","4H","1D","1W"]:
                r=current["ema_rows"].get(tf,{})
                ema_table.append({"Timeframe":tf,"EMA20/50/100":r.get("state","NO DATA"),"Alignment":f"{r.get('count',0)}/3"})
            st.dataframe(pd.DataFrame(ema_table),use_container_width=True,hide_index=True)
            st.write(f"**Full 21-condition EMA alignment:** {current['bull']*3}/21 bullish conditions, {current['bear']*3}/21 bearish conditions.")

            if matches:
                st.markdown("### 🔎 Closest historical examples")
                rows=[]
                for sim,e in matches[:15]:
                    f=e["features"]; o=e["outcome"]
                    rows.append({
                        "Similarity":f"{sim:.0f}%","Coin":e.get("pair","—"),"Date":str(e["time"])[:16],
                        "Pump 24 bars":f"{safe(f['ret24']):+.1f}%","RSI":f"{safe(f['rsi']):.0f}",
                        "Volume":f"{safe(f['vol_ratio']):.1f}x","EMA stack":"Bullish" if f["ema_stack"]==1 else "Bearish" if f["ema_stack"]==-1 else "Mixed",
                        "What happened":o["label"],"Next-period":f"{o['end']:+.1f}%"
                    })
                st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

            with st.expander("Advanced details"):
                st.write(f"**4H structure:** {current['structure4']} | **1D structure:** {current['structure1']} | **15m structure:** {current['structure15']}")
                st.write(f"**4H ADX:** {safe(b4.adx):.1f} | **4H MACD:** {'Bullish' if safe(b4.macd)>safe(b4.macd_signal) else 'Bearish'}")
                st.write(f"**Historical learning pool:** {len(pool)} events from {len(pairs_sig)} comparison contracts.")
                st.write("The engine compares normalized historical conditions and then checks what happened in the following 24 four-hour candles. It does not guarantee future price movement.")
                if failures: st.code("\n".join(failures[:50]))

            # Save results in session state for optional further inspection.
            st.session_state["last_analysis"]={"symbol":symbol,"pair":pair,"current":current,"summary":summary,"matches":matches}

    except Exception as e:
        st.error(f"Analysis failed: {type(e).__name__}: {e}")

# =============================================================================
# CURRENT HOT / ATH / ATL DISCOVERY
# =============================================================================
st.divider()
st.header("🔥 Hot / ATH / ATL Discovery")
st.caption("This section finds current events first. Click a coin above to perform the deeper historical similarity study.")

if st.button("🔍 Scan Current Hot / ATH / ATL Coins"):
    try:
        with st.spinner("Reading current CoinDCX Futures prices..."):
            prices=futures_prices(); rows=[]
            active=active_instruments(margin)
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
                    ind=indicators(dc); last=ind.iloc[-1]
                    tag=None
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
            if out:
                st.dataframe(pd.DataFrame(out),use_container_width=True,hide_index=True)
                st.info("Copy a coin name from this table into the Analyze a Coin box to run the historical pattern study.")
            else: st.warning("No current hot/ATH/ATL candidates were found in the scanned universe.")
            if failures:
                with st.expander("Scan diagnostics"): st.code("\n".join(failures[:50]))
    except Exception as e:
        st.error(f"Discovery scan failed: {type(e).__name__}: {e}")

st.divider()
st.caption("Analysis only. No orders, balances, API keys or withdrawals are used. Historical similarity is evidence, not a guarantee or financial advice.")
