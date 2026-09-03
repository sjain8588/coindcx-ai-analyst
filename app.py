import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"

st.set_page_config(page_title="CoinDCX Futures Scanner", page_icon="🎯", layout="wide")
st.title("🎯 CoinDCX Futures Momentum Scanner")
st.caption("Analysis only • Top 5 futures candidates • Built for volatile/meme-coin momentum • No trading")

@st.cache_data(ttl=30)
def get_active_instruments(margin="USDT"):
    url = f"{API}/exchange/v1/derivatives/futures/data/active_instruments"
    r = requests.get(url, params=[("margin_currency_short_name[]", margin)], timeout=20)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected active-instruments response: {data}")
    return data

@st.cache_data(ttl=5)
def get_futures_prices():
    r = requests.get(f"{PUBLIC}/market_data/v3/current_prices/futures/rt", timeout=20)
    r.raise_for_status()
    data = r.json()
    return data.get("prices", {}) if isinstance(data, dict) else {}

@st.cache_data(ttl=30)
def get_futures_candles(pair, resolution, from_ts, to_ts):
    params = {
        "pair": pair,
        "from": int(from_ts),
        "to": int(to_ts),
        "resolution": resolution,
        "pcode": "f",
    }
    r = requests.get(f"{PUBLIC}/market_data/candlesticks", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    rows = data.get("data", []) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected candle response: {data}")
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    for c in ["open","high","low","close","volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d["time"], unit="ms", errors="coerce")
    return d.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=5)
def get_futures_orderbook(pair):
    r = requests.get(f"{PUBLIC}/market_data/v3/orderbook/{pair}-futures/50", timeout=15)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=5)
def get_futures_trades(pair):
    r = requests.get(
        f"{API}/exchange/v1/derivatives/futures/data/trades",
        params={"pair": pair}, timeout=15
    )
    r.raise_for_status()
    x = r.json()
    return x if isinstance(x, list) else []

def indicators(d):
    x = d.copy()
    for n in [20,50,100,200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()

    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0,np.nan)
    x["rsi"] = 100 - 100/(1+rs)

    e12 = x.close.ewm(span=12,adjust=False).mean()
    e26 = x.close.ewm(span=26,adjust=False).mean()
    x["macd"] = e12-e26
    x["macd_signal"] = x.macd.ewm(span=9,adjust=False).mean()

    tr = pd.concat([
        x.high-x.low,
        (x.high-x.close.shift()).abs(),
        (x.low-x.close.shift()).abs()
    ],axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14,adjust=False).mean()
    x["volma"] = x.volume.rolling(20).mean()

    # Bollinger Bands
    x["bbmid"] = x.close.rolling(20).mean()
    x["bbstd"] = x.close.rolling(20).std()
    x["bbup"] = x.bbmid + 2*x.bbstd
    x["bblow"] = x.bbmid - 2*x.bbstd

    # ADX / DI
    up = x.high.diff()
    dn = -x.low.diff()
    plus = pd.Series(np.where((up>dn)&(up>0),up,0.0), index=x.index)
    minus = pd.Series(np.where((dn>up)&(dn>0),dn,0.0), index=x.index)
    atr = x.atr.replace(0,np.nan)
    pdi = 100*plus.ewm(alpha=1/14,adjust=False).mean()/atr
    mdi = 100*minus.ewm(alpha=1/14,adjust=False).mean()/atr
    dx = 100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    x["adx"] = dx.ewm(alpha=1/14,adjust=False).mean()
    x["pdi"],x["mdi"] = pdi,mdi
    return x

def structure(d):
    if len(d)<40: return "Mixed"
    recent=d.tail(12)
    prior=d.iloc[-36:-12]
    if recent.high.max()>prior.high.max() and recent.low.min()>prior.low.min():
        return "Bullish"
    if recent.high.max()<prior.high.max() and recent.low.min()<prior.low.min():
        return "Bearish"
    return "Mixed"

def technical_side(d):
    x=d.iloc[-1]
    long_points = 0
    short_points = 0
    ema_pairs = [
        (x.close, x.ema20),
        (x.ema20, x.ema50),
        (x.ema50, x.ema100),
        (x.ema100, x.ema200),
    ]
    for a,b in ema_pairs:
        if pd.notna(a) and pd.notna(b):
            long_points += int(a>b)
            short_points += int(a<b)

    long_points += int(50<x.rsi<70) + int(x.macd>x.macd_signal) + int(x.pdi>x.mdi) + int(x.volume>x.volma)
    short_points += int(30<x.rsi<50) + int(x.macd<x.macd_signal) + int(x.mdi>x.pdi) + int(x.volume>x.volma)
    return long_points, short_points

def micro(pair):
    ob = None
    flow = None
    try:
        b=get_futures_orderbook(pair)
        bids=b.get("bids",{})
        asks=b.get("asks",{})
        B=[(float(p),float(q)) for p,q in bids.items()]
        A=[(float(p),float(q)) for p,q in asks.items()]
        if B and A:
            best_bid=max(p for p,_ in B)
            best_ask=min(p for p,_ in A)
            mid=(best_bid+best_ask)/2
            near_b=sum(q for p,q in B if p>=mid*.995)
            near_a=sum(q for p,q in A if p<=mid*1.005)
            ob=(near_b-near_a)/(near_b+near_a) if near_b+near_a else 0
    except Exception:
        pass

    try:
        trs=get_futures_trades(pair)
        buy=sell=0.0
        for t in trs:
            q=float(t.get("quantity",0) or 0)
            # CoinDCX futures docs define is_maker; use it only as a flow proxy.
            if bool(t.get("is_maker",False)): sell += q
            else: buy += q
        flow=(buy-sell)/(buy+sell) if buy+sell else 0
    except Exception:
        pass
    return ob,flow

def calculate_support_resistance(d):
    """Identify practical horizontal support/resistance from daily swing levels.
    Levels are clustered using ATR so multiple nearby touches become one zone.
    Monthly highs/lows are also included as major reference levels.
    """
    if d.empty:
        return {
            "support1": np.nan, "support2": np.nan,
            "resistance1": np.nan, "resistance2": np.nan,
            "monthly_support": np.nan, "monthly_resistance": np.nan,
            "ath": np.nan, "atl": np.nan,
        }

    x = d.sort_values("time").reset_index(drop=True).copy()
    price = float(x.iloc[-1].close)
    atr_series = x["atr"] if "atr" in x else pd.Series(dtype=float)
    atr = float(atr_series.iloc[-1]) if len(atr_series) and pd.notna(atr_series.iloc[-1]) else price * 0.02
    tolerance = max(atr * 0.60, price * 0.002)

    levels = []
    lookback = 2
    for i in range(lookback, len(x) - lookback):
        hi = float(x.iloc[i].high)
        lo = float(x.iloc[i].low)
        if hi >= float(x.iloc[i-lookback:i+lookback+1].high.max()):
            levels.append({"price": hi, "kind": "R", "time": x.iloc[i].time, "monthly": False})
        if lo <= float(x.iloc[i-lookback:i+lookback+1].low.min()):
            levels.append({"price": lo, "kind": "S", "time": x.iloc[i].time, "monthly": False})

    # Monthly extremes are strong reference levels.
    try:
        monthly = (
            x.set_index("time")
             .resample("ME")
             .agg({"high":"max", "low":"min"})
             .dropna()
        )
        for t, row in monthly.iterrows():
            levels.append({"price": float(row.high), "kind": "R", "time": t, "monthly": True})
            levels.append({"price": float(row.low), "kind": "S", "time": t, "monthly": True})
    except Exception:
        pass

    def cluster(kind):
        raw = [z for z in levels if z["kind"] == kind]
        raw.sort(key=lambda z: z["price"])
        clusters = []
        for z in raw:
            if not clusters or abs(z["price"] - clusters[-1]["price"]) > tolerance:
                clusters.append({"price": z["price"], "touches": 1, "last_time": z["time"], "monthly": z["monthly"]})
            else:
                c = clusters[-1]
                c["price"] = (c["price"] * c["touches"] + z["price"]) / (c["touches"] + 1)
                c["touches"] += 1
                c["last_time"] = max(c["last_time"], z["time"])
                c["monthly"] = c["monthly"] or z["monthly"]

        now = x.iloc[-1].time
        for c in clusters:
            age_days = max(0.0, (now - c["last_time"]).total_seconds() / 86400.0)
            recency = 1.0 / (1.0 + age_days / 30.0)
            c["score"] = c["touches"] * 3 + recency * 4 + (4 if c["monthly"] else 0)
        return clusters

    supports = [c for c in cluster("S") if c["price"] < price * 0.9995]
    resistances = [c for c in cluster("R") if c["price"] > price * 1.0005]

    supports.sort(key=lambda c: c["price"], reverse=True)
    resistances.sort(key=lambda c: c["price"])

    ath = float(x.high.max())
    atl = float(x.low.min())

    monthly_support = np.nan
    monthly_resistance = np.nan
    try:
        monthly = (
            x.set_index("time")
             .resample("ME")
             .agg({"high":"max", "low":"min"})
             .dropna()
        )
        below = monthly.low[monthly.low < price]
        above = monthly.high[monthly.high > price]
        if not below.empty:
            monthly_support = float(below.max())
        if not above.empty:
            monthly_resistance = float(above.min())
    except Exception:
        pass

    return {
        "support1": float(supports[0]["price"]) if supports else np.nan,
        "support2": float(supports[1]["price"]) if len(supports) > 1 else np.nan,
        "resistance1": float(resistances[0]["price"]) if resistances else np.nan,
        "resistance2": float(resistances[1]["price"]) if len(resistances) > 1 else np.nan,
        "monthly_support": monthly_support if pd.notna(monthly_support) else atl,
        "monthly_resistance": monthly_resistance if pd.notna(monthly_resistance) else ath,
        "ath": ath,
        "atl": atl,
    }


def score_candidate(pair, price_row, d1, h1, m15):
    x1,xh,x15=d1.iloc[-1],h1.iloc[-1],m15.iloc[-1]
    change=float(price_row.get("pc",0) or 0)
    move=abs(change)

    b1,s1=technical_side(d1)
    bh,sh=technical_side(h1)
    b15,s15=technical_side(m15)

    ob,flow=micro(pair)

    # Direction follows the user's strategy: strongest gainers are long candidates,
    # strongest losers are short candidates. Technicals decide whether to trade or wait.
    side="LONG" if change>0 else "SHORT"

    if side=="LONG":
        raw = move*1.5 + (b1-s1)*7 + (bh-sh)*5 + (b15-s15)*3
        if ob is not None: raw += max(0,ob)*8
        if flow is not None: raw += max(0,flow)*6
    else:
        raw = move*1.5 + (s1-b1)*7 + (sh-bh)*5 + (s15-b15)*3
        if ob is not None: raw += max(0,-ob)*8
        if flow is not None: raw += max(0,-flow)*6

    # Penalize chasing a move that is already very far from the daily EMA20.
    ext=abs(float(x1.close-x1.ema20))/max(float(x1.atr),1e-12)
    chase_penalty=max(0,(ext-2.0))*7
    raw-=min(25,chase_penalty)

    confidence=max(35,min(95,50+raw*.45))

    # History quality matters. New listings can still be analyzed, but their
    # confidence is capped because long-term EMA/support context is incomplete.
    history_factor = min(1.0, min(len(d1)/120.0, len(h1)/240.0, len(m15)/240.0))
    confidence = min(confidence, 50 + 45*history_factor)
    if history_factor < 0.5:
        confidence = min(confidence, 65)

    aligned = (
        (side=="LONG" and b1>s1 and bh>=sh and b15>=s15 and x15.macd>x15.macd_signal) or
        (side=="SHORT" and s1>b1 and sh>=bh and s15>=b15 and x15.macd<x15.macd_signal)
    )

    extreme = (side=="LONG" and x1.rsi>=78) or (side=="SHORT" and x1.rsi<=22)
    signal = side if aligned and not extreme and confidence>=68 else "WAIT"

    sr = calculate_support_resistance(d1)

    price=float(x15.close)
    atr=float(x15.atr)
    if signal=="LONG":
        sl=price-1.4*atr
        risk=price-sl
        tp1=price+1.5*risk
        tp2=price+2.5*risk
    elif signal=="SHORT":
        sl=price+1.4*atr
        risk=sl-price
        tp1=price-1.5*risk
        tp2=price-2.5*risk
    else:
        sl=tp1=tp2=np.nan

    return {
        "pair":pair,"change":change,"side":side,"signal":signal,"confidence":confidence,
        "price":price,"support1":sr["support1"],"support2":sr["support2"],
        "resistance1":sr["resistance1"],"resistance2":sr["resistance2"],
        "monthly_support":sr["monthly_support"],"monthly_resistance":sr["monthly_resistance"],
        "ath":sr["ath"],"atl":sr["atl"],
        "rsi":float(x1.rsi),"adx":float(xh.adx),"structure":structure(d1),
        "ob":ob,"flow":flow,"ext":ext,"score":raw,
        "entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,
        "d1":d1,"h1":h1,"m15":m15
    }

def fmt(v):
    if v is None or pd.isna(v): return "—"
    return f"{v:,.8f}".rstrip("0").rstrip(".")

margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Meme-focused scan",value=True)

def normalize_coin_input(text):
    q = text.strip().upper().replace("/", "").replace("-", "").replace("_", "")
    # Remove common futures quote suffixes so "VELVETUSDT" becomes "VELVET".
    for quote in ("USDT", "INR", "USDC"):
        if q.endswith(quote) and len(q) > len(quote):
            q = q[:-len(quote)]
            break
    if q.startswith("B") and len(q) > 2:
        # Do not blindly strip B from normal tickers. Only strip it when the
        # remaining token looks like a contract symbol used by CoinDCX.
        pass
    return q
if meme_only:
    st.caption("Meme focus uses a broad symbol/name keyword filter; turn it off to scan every active futures instrument.")

MEME_WORDS={"DOGE","SHIB","PEPE","BONK","FLOKI","WIF","BOME","MEME","BRETT","MOG","TURBO","MEW","NEIRO","BABYDOGE","1000SHIB","1000PEPE","1000BONK","1000FLOKI","1000LUNC","PONKE","MYRO","SLERF","LADYS","DEGEN","MOTHER","MAGA","TRUMP"}


st.divider()
st.header("💼 My Invested Coin — Long / Short Check")
st.caption("Enter a coin you already hold. The scanner will find its CoinDCX Futures contract and evaluate whether the current setup favors LONG, SHORT, or WAIT.")

coin_input = st.text_input(
    "Coin / Futures pair",
    placeholder="Example: DOGE, SHIB, PEPE, B-DOGE_USDT",
    help="You can enter DOGE, DOGEUSDT, or the CoinDCX futures pair such as B-DOGE_USDT."
)
avg_price_input = st.number_input(
    "Optional: your average entry price",
    min_value=0.0,
    value=0.0,
    step=0.00000001,
    format="%.8f",
    help="Enter your actual average price if you want the scanner to compare the current price with your cost."
)

if st.button("📊 Analyze My Coin"):
    try:
        if not coin_input.strip():
            st.warning("Enter a coin first, for example DOGE or SHIB.")
            st.stop()

        prices = get_futures_prices()

        # Search both common quote markets for the requested coin. This is
        # important because a coin can exist in Futures USDT even when the
        # currently selected margin market is different.
        requested = normalize_coin_input(coin_input)
        margin_list = [margin] + [q for q in ("USDT","INR") if q != margin]

        candidates = []
        seen = set()

        for qmargin in margin_list:
            try:
                active = get_active_instruments(qmargin)
            except Exception:
                continue

            for pair in active:
                pair_u = str(pair).upper()
                p = prices.get(pair) if 'prices' in locals() else None
                if not p:
                    continue

                symbol = str(p.get("mkt", "")).upper()
                # Build clean base-token candidates from pair and market symbol.
                names = {
                    pair_u.replace("-", "").replace("_", ""),
                    symbol.replace("-", "").replace("_", "")
                }
                matched = False
                for name in names:
                    n = name
                    for prefix in ("B", "I"):
                        # B-/I- are removed by the compact normalization only
                        # when the resulting name contains the requested token.
                        if n.startswith(prefix):
                            n2 = n[1:]
                        else:
                            n2 = n
                        if n2 == requested + qmargin or n2 == requested:
                            matched = True
                        if n2.startswith(requested + qmargin):
                            matched = True
                if matched and pair not in seen:
                    candidates.append((pair, p, symbol, qmargin))
                    seen.add(pair)

        if not candidates:
            # Fallback: if the price feed uses a slightly different market
            # label, show the closest symbols to make troubleshooting easy.
            available = []
            for qmargin in margin_list:
                try:
                    active = get_active_instruments(qmargin)
                    for pair in active:
                        s = str(pair).upper()
                        if requested in s.replace("-", "").replace("_",""):
                            available.append(s)
                except Exception:
                    pass
            hint = ", ".join(available[:8])
            extra = f" Possible matching contracts: {hint}" if hint else ""
            st.error(
                f"No active CoinDCX Futures contract was found for '{coin_input}'. "
                f"Tried {', '.join(margin_list)} margin markets.{extra}"
            )
            st.stop()

        # Prefer the user's selected margin market, then USDT.
        candidates.sort(key=lambda z: (0 if z[3] == margin else 1, len(z[0])))
        pair, p, symbol, found_margin = candidates[0]
        if found_margin != margin:
            st.info(f"Found {symbol} in the **{found_margin}** Futures market; analyzing that contract.")

        now = int(time.time())
        d1 = get_futures_candles(pair, "1D", now-400*86400, now)
        h1 = get_futures_candles(pair, "60", now-120*86400, now)
        m15 = get_futures_candles(pair, "15", now-30*86400, now)

        # Meme/futures listings can be very new. Do not require 210 daily
        # candles just to produce an analysis. Use the available history and
        # clearly reduce confidence when history is limited.
        if len(d1) < 10 or len(h1) < 30 or len(m15) < 30:
            st.error(
                f"CoinDCX returned too little history for {symbol}: "
                f"1D={len(d1)}, 1H={len(h1)}, 15m={len(m15)}. "
                f"Need at least 10 / 30 / 30 candles."
            )
            st.stop()

        x = score_candidate(
            pair, p,
            indicators(d1),
            indicators(h1),
            indicators(m15)
        )

        st.subheader(f"{symbol}  •  {pair}")

        current = float(x["price"])
        if avg_price_input > 0:
            pnl_pct = (current - avg_price_input) / avg_price_input * 100
            st.metric("Your position vs current price", f"{pnl_pct:+.2f}%")

        # Give a dedicated interpretation for an already-held spot position.
        if x["signal"] == "LONG":
            st.success(
                "🟢 LONG BIAS — Current Futures structure supports the bullish direction. "
                "For an existing spot holding, this means the trend is currently favorable to the long side."
            )
        elif x["signal"] == "SHORT":
            st.error(
                "🔴 SHORT BIAS — Current Futures structure supports the bearish direction. "
                "For an existing spot holding, this is a warning that downside momentum is stronger."
            )
        else:
            st.warning(
                "🟡 WAIT — The coin is not giving a sufficiently clean long or short setup right now. "
                "Avoid making a directional decision from momentum alone."
            )

        a,b,c,d = st.columns(4)
        a.metric("Current price", fmt(current))
        b.metric("24h move", f'{x["change"]:+.2f}%')
        c.metric("Signal confidence", f'{x["confidence"]:.0f}%')
        d.metric("1D RSI", f'{x["rsi"]:.1f}')

        st.caption(
            f"History available: 1D {len(x['d1'])} candles • "
            f"1H {len(x['h1'])} candles • 15m {len(x['m15'])} candles"
        )
        if min(len(x["d1"]), len(x["h1"]), len(x["m15"])) < 60:
            st.info("This is a newer/limited-history Futures contract. The signal is intentionally given a lower confidence ceiling.")
        st.write(
            f"**1D structure:** {x['structure']}  |  "
            f"**1H ADX:** {x['adx']:.1f}  |  "
            f"**Support:** {fmt(x['support1'])} / {fmt(x['support2'])}  |  "
            f"**Resistance:** {fmt(x['resistance1'])} / {fmt(x['resistance2'])}"
        )
        a,b,c,d = st.columns(4)
        a.metric("Monthly Support", fmt(x["monthly_support"]))
        b.metric("Monthly Resistance", fmt(x["monthly_resistance"]))
        c.metric("ATH", fmt(x["ath"]))
        d.metric("ATL", fmt(x["atl"]))

        if x["signal"] in ("LONG", "SHORT"):
            a,b,c = st.columns(3)
            a.metric("Reference entry", fmt(x["entry"]))
            b.metric("Stop loss", fmt(x["sl"]))
            c.metric("TP1 / TP2", f'{fmt(x["tp1"])} / {fmt(x["tp2"])}')

        with st.expander("Why did the scanner choose this direction?"):
            st.write(
                "The decision combines the 1D, 1H and 15m trend with EMA 20/50/100/200, "
                "RSI, MACD, ADX/DI, ATR, volume, Bollinger Bands, market structure, "
                "support/resistance and futures microstructure."
            )
            st.write(
                f"Daily RSI: {x['rsi']:.1f} | 1H ADX: {x['adx']:.1f} | "
                f"Order-book imbalance: "
                f"{('not available' if x['ob'] is None else f'{x['ob']*100:.2f}%')} | "
                f"Trade-flow proxy: "
                f"{('not available' if x['flow'] is None else f'{x['flow']*100:.2f}%')}"
            )
            st.line_chart(
                x["d1"].set_index("time")[["close","ema20","ema50","ema100","ema200"]].tail(250)
            )

    except Exception as e:
        st.error(f"My Coin analysis failed: {e}")
        st.code(str(e))

if st.button("🔍 Scan Top Futures",type="primary"):
    try:
        active=get_active_instruments(margin)
        prices=get_futures_prices()

        # Keep instruments that have a current price record.
        rows=[]
        for pair in active:
            p=prices.get(pair)
            if not p: continue
            symbol=str(p.get("mkt",pair)).upper()
            if meme_only and not any(word in symbol or word in pair.upper() for word in MEME_WORDS):
                continue
            pc=float(p.get("pc",0) or 0)
            if pc==0: continue
            rows.append((pair,p,symbol))

        if not rows:
            st.error("No matching futures instruments were returned. Try turning off Meme-focused scan.")
            st.stop()

        rows.sort(key=lambda z:abs(float(z[1].get("pc",0) or 0)),reverse=True)

        # Analyze the strongest movers first. This avoids making hundreds of API calls.
        results=[]
        now=int(time.time())
        for pair,p,symbol in rows[:12]:
            try:
                d1=get_futures_candles(pair,"1D",now-400*86400,now)
                h1=get_futures_candles(pair,"60",now-120*86400,now)
                m15=get_futures_candles(pair,"15",now-30*86400,now)
                if len(d1)<10 or len(d1h)<30 or len(m15)<30: continue
                results.append((symbol,score_candidate(pair,p,indicators(d1),indicators(h1),indicators(m15))))
            except Exception:
                continue

        if not results:
            st.error("Futures instruments were found, but no candidate returned enough futures candle data for analysis.")
            st.stop()

        results.sort(key=lambda z:z[1]["score"],reverse=True)
        results=results[:5]

        st.header("🎯 Today's Top 5 Futures Candidates")
        st.caption("First filter: largest 24h movers. Second filter: trend, EMA 20/50/100/200, RSI, MACD, ADX/DI, ATR, volume, Bollinger Bands, daily support/resistance and futures liquidity.")

        for i,(symbol,x) in enumerate(results,1):
            icon="🟢" if x["signal"]=="LONG" else "🔴" if x["signal"]=="SHORT" else "🟡"
            with st.container(border=True):
                st.subheader(f"{i}. {symbol}   {icon} {x['signal']}")
                a,b,c,d=st.columns(4)
                a.metric("24h move",f'{x["change"]:.2f}%')
                b.metric("Confidence",f'{x["confidence"]:.0f}%')
                c.metric("Price",fmt(x["price"]))
                d.metric("Daily RSI",f'{x["rsi"]:.1f}')

                st.write(f"**Support:** {fmt(x['support2'])} / {fmt(x['support1'])}   |   **Resistance:** {fmt(x['resistance1'])} / {fmt(x['resistance2'])}")
                st.caption(f"Monthly S/R: {fmt(x['monthly_support'])} / {fmt(x['monthly_resistance'])}   |   ATH: {fmt(x['ath'])}   |   ATL: {fmt(x['atl'])}")
                st.caption(
                    f"History available: 1D {len(x['d1'])} candles • "
                    f"1H {len(x['h1'])} candles • 15m {len(x['m15'])} candles"
                )

                if x["signal"] in ("LONG","SHORT"):
                    a,b,c,d=st.columns(4)
                    a.metric("Entry reference",fmt(x["entry"]))
                    b.metric("Stop Loss",fmt(x["sl"]))
                    c.metric("TP1",fmt(x["tp1"]))
                    d.metric("TP2",fmt(x["tp2"]))
                else:
                    direction="long" if x["side"]=="LONG" else "short"
                    st.warning(f"🟡 WAIT — {symbol} is a strong {direction} mover, but the confirmation/entry quality is not strong enough. Do not chase.")

                with st.expander("Advanced analysis"):
                    st.write(f"1D structure: **{x['structure']}**")
                    st.write(f"EMA20/50/100/200: {fmt(x['d1'].iloc[-1].ema20)} / {fmt(x['d1'].iloc[-1].ema50)} / {fmt(x['d1'].iloc[-1].ema100)} / {fmt(x['d1'].iloc[-1].ema200)}")
                    if pd.isna(x['d1'].iloc[-1].ema200):
                        st.info("This futures contract does not yet have 200 daily candles. EMA200 is therefore not used as a bearish/bullish vote for this coin.")
                    st.write(f"1H ADX: {x['adx']:.1f} | 1D RSI: {x['rsi']:.1f}")
                    if x["ob"] is not None: st.write(f"Near-price futures order-book imbalance: {x['ob']*100:.2f}%")
                    if x["flow"] is not None: st.write(f"Futures trade-flow proxy: {x['flow']*100:.2f}%")
                    st.line_chart(x["d1"].set_index("time")[["close","ema20","ema50","ema100","ema200"]].tail(250))

    except Exception as e:
        st.error(f"Scanner failed: {e}")
        st.code(str(e))

st.divider()
st.caption("The My Invested Coin tool is an analysis signal, not a command to open a leveraged position. For a spot holding, a SHORT BIAS is best interpreted as a downside-risk warning unless you intentionally hedge with futures.\n\nAnalysis only. CoinDCX futures endpoints used: active instruments, futures current prices, futures candlesticks, futures trades and futures order book. No API keys, orders, balances or withdrawals are used.")

# -----------------------------------------------------------------------------
# ATH / ATL scanners
# -----------------------------------------------------------------------------

def extreme_status(current, historical_extreme, mode):
    if historical_extreme is None or pd.isna(historical_extreme) or historical_extreme <= 0:
        return None, np.nan
    distance = (current - historical_extreme) / historical_extreme * 100.0
    if mode == "ATH":
        if distance > 0:
            return "🔥 ATH BREAKOUT", distance
        if distance >= -0.5:
            return "🟢 AT ATH", distance
        if distance >= -5:
            return "🟡 NEAR ATH", distance
    else:
        if distance < 0:
            return "🔴 ATL BREAKDOWN", distance
        if distance <= 0.5:
            return "🟠 AT ATL", distance
        if distance <= 5:
            return "🟡 NEAR ATL", distance
    return None, distance


def scan_extreme_coins(active, prices, margin, mode, meme_only=False, scan_limit=60):
    rows = []
    for pair in active:
        p = prices.get(pair)
        if not p:
            continue
        symbol = str(p.get("mkt", pair)).upper()
        if meme_only and not any(word in symbol or word in pair.upper() for word in MEME_WORDS):
            continue
        try:
            pc = float(p.get("pc", 0) or 0)
            last = float(p.get("lp", p.get("last_price", p.get("price", 0))) or 0)
        except Exception:
            continue
        if last <= 0:
            continue
        rows.append((pair, p, symbol, pc, last))

    if not rows:
        return []

    # Pre-filter to the most active movers. This keeps CoinDCX API traffic practical.
    if mode == "ATH":
        rows.sort(key=lambda z: z[3], reverse=True)
    else:
        rows.sort(key=lambda z: z[3])
    rows = rows[:scan_limit]

    now = int(time.time())
    results = []
    for pair, p, symbol, pc, current in rows:
        try:
            d1 = get_futures_candles(pair, "1D", now - 400*86400, now)
            if len(d1) < 2:
                continue

            # Exclude the latest daily candle from the historical extreme.
            # This prevents the current incomplete candle from defining its own ATH/ATL.
            history = d1.iloc[:-1]
            if history.empty:
                continue
            historical_ath = float(history.high.max())
            historical_atl = float(history.low.min())
            extreme = historical_ath if mode == "ATH" else historical_atl
            status, distance = extreme_status(current, extreme, mode)
            if status is None:
                continue

            ind = indicators(d1)
            last = ind.iloc[-1]
            results.append({
                "symbol": symbol,
                "pair": pair,
                "price": current,
                "extreme": extreme,
                "distance": distance,
                "change": pc,
                "status": status,
                "rsi": float(last.rsi) if pd.notna(last.rsi) else np.nan,
                "volume": float(last.volume) if pd.notna(last.volume) else np.nan,
                "volma": float(last.volma) if pd.notna(last.volma) else np.nan,
                "candles": len(d1),
            })
        except Exception:
            continue

    if mode == "ATH":
        # Breakouts first, then closest to the previous ATH, then momentum.
        results.sort(key=lambda z: (0 if z["distance"] > 0 else 1, abs(z["distance"]), -z["change"]))
    else:
        # Breakdowns first, then closest to/below ATL, then strongest downside momentum.
        results.sort(key=lambda z: (0 if z["distance"] < 0 else 1, abs(z["distance"]), z["change"]))
    return results[:10]


st.divider()
st.header("🔥 / 🩸 Historical Extreme Scanner")
st.caption(
    "Find the Top 10 Futures coins near or beyond their available-history ATH or ATL. "
    "The latest daily candle is excluded when defining the previous ATH/ATL, so a current breakout/breakdown is detected rather than counted as the extreme itself."
)

ath_col, atl_col = st.columns(2)
with ath_col:
    scan_ath = st.button("🔥 Scan Top 10 Near / Above ATH")
with atl_col:
    scan_atl = st.button("🩸 Scan Top 10 Near / Below ATL")

if scan_ath or scan_atl:
    mode = "ATH" if scan_ath else "ATL"
    try:
        active = get_active_instruments(margin)
        prices = get_futures_prices()
        with st.spinner(f"Scanning Futures for {mode} candidates..."):
            extreme_results = scan_extreme_coins(
                active, prices, margin, mode,
                meme_only=meme_only,
                scan_limit=60
            )

        if not extreme_results:
            if meme_only:
                st.warning(
                    f"No {mode} candidates were found within the 5% threshold among the first 60 matching movers. "
                    "Turn off Meme-focused scan and try again for the full Futures universe."
                )
            else:
                st.info(f"No Futures coins were found within the 5% {mode} threshold in the scanned universe.")
        else:
            title = "🔥 Top 10 Near / Above ATH" if mode == "ATH" else "🩸 Top 10 Near / Below ATL"
            st.subheader(title)
            st.caption(
                "ATH/ATL is based on the available CoinDCX Futures daily history requested by this scanner (up to 400 days). "
                "It is not guaranteed to be the lifetime exchange-wide ATH/ATL for older contracts."
            )

            table = []
            for i, z in enumerate(extreme_results, 1):
                table.append({
                    "#": i,
                    "Coin": z["symbol"],
                    "Current": fmt(z["price"]),
                    "ATH" if mode == "ATH" else "ATL": fmt(z["extreme"]),
                    "Distance": f'{z["distance"]:+.2f}%',
                    "24h": f'{z["change"]:+.2f}%',
                    "RSI": f'{z["rsi"]:.1f}' if pd.notna(z["rsi"]) else "—",
                    "Status": z["status"],
                })
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

            for i, z in enumerate(extreme_results, 1):
                direction = "above" if z["distance"] > 0 else "below"
                st.write(
                    f"**{i}. {z['symbol']} — {z['status']}** | "
                    f"Price {fmt(z['price'])} | "
                    f"Previous {mode} {fmt(z['extreme'])} | "
                    f"{abs(z['distance']):.2f}% {direction} | "
                    f"24h {z['change']:+.2f}% | RSI {z['rsi']:.1f}"
                )
    except Exception as e:
        st.error(f"{mode} scanner failed: {e}")
        st.code(str(e))
