import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# CLEAN COINDCX TRADE AGENT
# ============================================================
# Strategy:
# SHORT = hard pump -> extension -> bearish transition -> EMA20 rejection
#         -> local-low break. Avoid mature dumps.
# LONG  = hard dump -> stabilization -> bullish transition -> EMA20 hold
#         -> local-high break. Avoid chasing extended rebounds.
#
# Analysis only. No private API, no orders.
# Uses completed candles only.
# ============================================================

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"
MAX_WORKERS = 12
DEEP_POOL = 50

def get_json(url, params=None, timeout=25):
    r = requests.get(url, params=params, timeout=timeout,
                     headers={"User-Agent": "CoinDCX-Trade-Agent/1.0"})
    r.raise_for_status()
    return r.json()

def active_usdt_pairs():
    # CoinDCX's hardened public universe endpoint returns the ACTIVE
    # futures contracts directly.  The older instruments/contracts endpoints
    # can return catalogue data or fail for the current API version.
    url = API + "/exchange/v1/derivatives/futures/data/active_instruments"
    try:
        r = requests.get(
            url,
            params=[("margin_currency_short_name[]", "USDT")],
            timeout=25,
            headers={"User-Agent": "CoinDCX-Trade-Agent/2.0"},
        )
        r.raise_for_status()
        payload = r.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected active_instruments response: {payload}")

        pairs = []
        for x in payload:
            if isinstance(x, str):
                p = x
            elif isinstance(x, dict):
                p = (x.get("pair") or x.get("symbol") or x.get("instrument")
                     or x.get("symbol_id") or x.get("market"))
            else:
                p = None
            if p:
                p = str(p).upper().strip()
                # Keep only actual USDT-margined contracts.
                if "USDT" in p:
                    pairs.append(p)

        pairs = sorted(set(pairs))
        if not pairs:
            raise RuntimeError("CoinDCX returned no active USDT Futures contracts.")
        return pairs
    except Exception as e:
        raise RuntimeError(f"CoinDCX active Futures API error: {e}") from e

def candles(pair, resolution, days):
    now = int(time.time())
    payload = get_json(
        f"{PUBLIC}/market_data/candlesticks",
        {"pair": pair, "from": now-int(days*86400), "to": now,
         "resolution": {"1m":"1", "5m":"5", "15":"15", "15m":"15",
                        "1H":"60", "4H":"240", "1D":"1D"}.get(resolution, resolution),
         "pcode": "f"},
    )
    rows = payload.get("data", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    tcol = "time" if "time" in d.columns else "timestamp"
    if tcol not in d.columns:
        return pd.DataFrame()
    for c in ["open","high","low","close","volume"]:
        if c not in d.columns:
            return pd.DataFrame()
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["time"] = pd.to_datetime(d[tcol], unit="ms", utc=True, errors="coerce")
    d = d.dropna(subset=["time","open","high","low","close"]).sort_values("time")
    d = d.drop_duplicates("time").reset_index(drop=True)
    if len(d) > 2:
        d = d.iloc[:-1].copy()
    return d

def indicators(d):
    x = d.copy()
    for n in [20,50,100,200]:
        x[f"ema{n}"] = x.close.ewm(span=n, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    x["rsi"] = 100 - 100/(1 + gain/loss.replace(0,np.nan))
    tr = pd.concat([x.high-x.low,
                    (x.high-x.close.shift()).abs(),
                    (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["atr_pct"] = x.atr/x.close*100
    x["vol_ma"] = x.volume.rolling(20).mean()
    x["vol_ratio"] = x.volume/x.vol_ma.replace(0,np.nan)
    return x

def pivots(d, left=3, right=3):
    highs, lows = [], []
    if len(d) < left+right+10:
        return highs,lows
    for i in range(left, len(d)-right):
        if d.high.iloc[i] >= d.high.iloc[i-left:i+right+1].max():
            highs.append((i,float(d.high.iloc[i])))
        if d.low.iloc[i] <= d.low.iloc[i-left:i+right+1].min():
            lows.append((i,float(d.low.iloc[i])))
    return highs,lows

def structure(d):
    h,l = pivots(d)
    if len(h)<2 or len(l)<2:
        return "MIXED"
    h1,h2=h[-2][1],h[-1][1]
    l1,l2=l[-2][1],l[-1][1]
    if h2>h1 and l2>l1: return "HH + HL"
    if h2<h1 and l2<l1: return "LH + LL"
    if h2>h1 and l2<=l1: return "BULLISH DEVELOPING"
    if h2<=h1 and l2>l1: return "BEARISH DEVELOPING"
    return "MIXED"

def pct(d,bars):
    if len(d)<=bars: return np.nan
    return (d.close.iloc[-1]/d.close.iloc[-1-bars]-1)*100

def levels(d,current):
    h,l=pivots(d)
    s=sorted({round(v,12) for _,v in l if v<current}, reverse=True)
    r=sorted({round(v,12) for _,v in h if v>current})
    # add recent extremes
    recent=d.tail(min(160,len(d)))
    s += [float(v) for v in recent.low.nsmallest(8) if v<current]
    r += [float(v) for v in recent.high.nlargest(8) if v>current]
    def cluster(vals):
        vals=sorted(set(vals))
        out=[]
        for v in vals:
            if not out or abs(v-out[-1])/max(abs(out[-1]),1e-12)>0.007:
                out.append(v)
            else:
                out[-1]=(out[-1]+v)/2
        return out
    s=cluster(s)
    r=cluster(r)
    return sorted(s,reverse=True)[:3], sorted(r)[:3]

def deep_scan(pair):
    try:
        d15=indicators(candles(pair,"15",10))
        d4=indicators(candles(pair,"240",120))
        d1=indicators(candles(pair,"1D",500))
        if len(d15)<120 or len(d4)<40 or len(d1)<30:
            return None
        current=float(d15.close.iloc[-1])
        s15=structure(d15.tail(160)); s4=structure(d4.tail(80)); s1=structure(d1.tail(50))
        ema15=float(d15.ema20.iloc[-1]); ema4=float(d4.ema20.iloc[-1])
        e15=(current/ema15-1)*100; e4=(current/ema4-1)*100
        r15=float(d15.rsi.iloc[-1]); r4=float(d4.rsi.iloc[-1])
        vol=float(d15.vol_ratio.iloc[-1]) if np.isfinite(d15.vol_ratio.iloc[-1]) else np.nan
        m24=pct(d15,96); m3=pct(d15,min(288,len(d15)-1)); m7=pct(d15,min(672,len(d15)-1))
        peak=float(d15.tail(96).high.max())
        low=float(d15.tail(96).low.min())
        draw=(current/peak-1)*100 if peak else np.nan
        recovery=(current/low-1)*100 if low else np.nan
        h,l=pivots(d15.tail(160))
        local_hi=h[-1][1] if h else np.nan
        local_lo=l[-1][1] if l else np.nan
        s15l,r15l=levels(d15,current)
        s4l,r4l=levels(d4,current)
        s1l,r1l=levels(d1,current)

        pump=0
        if m24>=15: pump+=2
        if m3>=30: pump+=3
        if m7>=50: pump+=3
        if r4>=70: pump+=2
        if e15>=6: pump+=2

        dump=0
        if m24<=-15: dump+=2
        if m3<=-30: dump+=3
        if m7<=-50: dump+=3
        if r4<=30: dump+=2
        if e15<=-6: dump+=2

        short_score=pump
        if s15=="LH + LL": short_score+=4
        elif s15=="BEARISH DEVELOPING": short_score+=2
        if current<ema15: short_score+=2
        if -12<=draw<=-2: short_score+=2
        if draw<-20: short_score-=4
        if r15<40: short_score-=2

        long_score=dump
        if s15=="HH + HL": long_score+=4
        elif s15=="BULLISH DEVELOPING": long_score+=2
        if current>=ema15: long_score+=2
        if 2<=recovery<=15: long_score+=2
        if recovery>18: long_score-=3
        if r15>70: long_score-=2

        short_ready=(pump>=5 and s15 in ("LH + LL","BEARISH DEVELOPING")
                     and current<=ema15*1.01 and draw>-18)
        long_ready=(dump>=5 and s15 in ("HH + HL","BULLISH DEVELOPING")
                    and current>=ema15*0.99 and recovery<=18)

        if short_ready:
            setup="SHORT READY"; side="SHORT"; score=short_score
            trigger=f"Break below {local_lo:.8g}" if np.isfinite(local_lo) else "Break recent 15m low"
            invalid=f"Reclaim 15m EMA20 {ema15:.8g}"
            thesis="Hard pump is transitioning into fresh bearish structure."
        elif long_ready:
            setup="LONG READY"; side="LONG"; score=long_score
            trigger=f"Break above {local_hi:.8g}" if np.isfinite(local_hi) else "Break recent 15m high"
            invalid=f"Loss of 15m EMA20 {ema15:.8g}"
            thesis="Hard dump is transitioning into fresh bullish structure."
        elif s4=="LH + LL" and s15 in ("LH + LL","BEARISH DEVELOPING"):
            setup="SHORT WATCH"; side="SHORT"; score=short_score
            trigger="Wait for EMA20 rejection + local-low break"
            invalid="15m HH/HL + EMA20 reclaim"
            thesis="Bearish trend; wait for a fresh bounce/rejection instead of chasing."
        elif s4=="HH + HL" and s15 in ("HH + HL","BULLISH DEVELOPING"):
            setup="LONG WATCH"; side="LONG"; score=long_score
            trigger="Wait for EMA20 hold + local-high break"
            invalid="15m LH/LL + EMA20 loss"
            thesis="Bullish trend; wait for a fresh pullback instead of chasing."
        else:
            setup="WAIT"; side="WAIT"; score=max(short_score,long_score)
            trigger="No fresh trigger"; invalid="Structure unresolved"
            thesis="No sufficiently fresh confirmed setup."

        return dict(pair=pair,current=current,move24=m24,move3d=m3,move7d=m7,
                    draw24=draw,recovery24=recovery,rsi15=r15,rsi4=r4,
                    ema15dist=e15,ema4dist=e4,vol=vol,s15=s15,s4=s4,s1=s1,
                    support15=s15l,resistance15=r15l,support4=s4l,resistance4=r4l,
                    support1=s1l,resistance1=r1l,local_hi=local_hi,local_lo=local_lo,
                    pump=pump,dump=dump,score=score,setup=setup,side=side,
                    trigger=trigger,invalidation=invalid,thesis=thesis)
    except Exception:
        return None

def scan_market():
    pairs=active_usdt_pairs()
    if not pairs: raise RuntimeError("Could not retrieve active USDT Futures.")
    # First-pass 15m scan keeps the deep scan focused and fast.
    def quick(p):
        try:
            d=candles(p,"15",8)
            if len(d)<100: return None
            return (p,pct(d,96),pct(d,min(288,len(d)-1)),pct(d,min(672,len(d)-1)))
        except Exception: return None
    quick_rows=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs=[ex.submit(quick,p) for p in pairs]
        for f in as_completed(fs):
            x=f.result()
            if x: quick_rows.append(x)
    q=pd.DataFrame(quick_rows,columns=["pair","m24","m3","m7"])
    if q.empty: raise RuntimeError("No market data returned.")
    candidates=list(dict.fromkeys(
        q.sort_values("m3",ascending=False).head(DEEP_POOL).pair.tolist()+
        q.sort_values("m3",ascending=True).head(DEEP_POOL).pair.tolist()+
        q.assign(a=q.m24.abs()).sort_values("a",ascending=False).head(DEEP_POOL).pair.tolist()
    ))
    rows=[]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        fs=[ex.submit(deep_scan,p) for p in candidates]
        for f in as_completed(fs):
            x=f.result()
            if x: rows.append(x)
    return q,pd.DataFrame(rows)

def fp(v):
    if not np.isfinite(v): return "—"
    if abs(v)>=1000: return f"{v:,.2f}"
    if abs(v)>=1: return f"{v:,.4f}"
    if abs(v)>=.01: return f"{v:,.6f}"
    return f"{v:.10f}"

def lvl(vals):
    return " | ".join(fp(v) for v in vals) if vals else "—"

def table(df, side):
    x=df[df.setup.str.contains(side,na=False)].copy()
    if x.empty: return pd.DataFrame()
    x["rank"]=x.score+x.pump*.5 if side=="SHORT" else x.score+x.dump*.5
    x=x.sort_values("rank",ascending=False).head(5)
    return pd.DataFrame([{
        "Coin":r.pair,"Setup":r.setup,"Score":int(r.score),
        "24H":f"{r.move24:+.1f}%","3D":f"{r.move3d:+.1f}%","7D":f"{r.move7d:+.1f}%",
        "15m":r.s15,"4H":r.s4,"4H RSI":f"{r.rsi4:.1f}",
        "EMA20":f"{r.ema15dist:+.1f}%","Trigger":r.trigger,
        "Invalidation":r.invalidation,"4H S":lvl(r.support4),"4H R":lvl(r.resistance4)
    } for _,r in x.iterrows()])

st.set_page_config(page_title="CoinDCX Trade Agent",page_icon="🎯",layout="wide")
st.title("🎯 CoinDCX Trade Agent — Fresh LONG / SHORT")
st.caption("Clean strategy only: hard-move exhaustion + structure transition + fresh trigger. No order placement.")

with st.sidebar:
    st.header("Scan")
    workers=st.slider("Scan workers",4,20,12)
    run=st.button("🔎 SCAN MARKET NOW",type="primary",use_container_width=True)
    st.markdown("---")
    st.write("🔴 SHORT: hard pump → LH/LL → EMA20 rejection → low break")
    st.write("🟢 LONG: hard dump → HH/HL → EMA20 hold → high break")
    st.write("🟡 WATCH: trend exists, but entry is not fresh")
    st.write("⚪ WAIT: mixed structure")

if run:
    with st.spinner("Scanning active USDT Futures..."):
        try:
            MAX_WORKERS=workers
            q,df=scan_market()
            st.session_state["q"]=q
            st.session_state["df"]=df
            st.session_state["scan_time"]=time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"Scan failed: {e}")

df=st.session_state.get("df")
q=st.session_state.get("q")
if df is None or df.empty:
    st.info("Click SCAN MARKET NOW.")
    st.stop()

st.success(f"Scan: {st.session_state.get('scan_time','—')} | Movers reviewed: {len(q)} | Deep candidates: {len(df)}")

shorts=table(df,"SHORT")
longs=table(df,"LONG")
a,b=st.columns(2)
with a:
    st.subheader("🔴 TOP 5 SHORT")
    st.dataframe(shorts,use_container_width=True,hide_index=True) if not shorts.empty else st.info("No fresh SHORT setup.")
with b:
    st.subheader("🟢 TOP 5 LONG")
    st.dataframe(longs,use_container_width=True,hide_index=True) if not longs.empty else st.info("No fresh LONG setup.")

st.subheader("🔍 Detailed Setup")
names=sorted(set((shorts.Coin.tolist() if not shorts.empty else [])+
                 (longs.Coin.tolist() if not longs.empty else [])))
if names:
    selected=st.selectbox("Select coin",names)
    r=df[df.pair==selected].iloc[0]
    st.markdown(f"### {r.pair} — {r.setup}")
    st.write(f"**Thesis:** {r.thesis}")
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric("Current",fp(r.current)); c2.metric("24H",f"{r.move24:+.2f}%")
    c3.metric("3D",f"{r.move3d:+.2f}%"); c4.metric("7D",f"{r.move7d:+.2f}%")
    c5.metric("4H RSI",f"{r.rsi4:.1f}")
    st.write(f"**15m:** {r.s15} | **4H:** {r.s4} | **1D:** {r.s1}")
    st.write(f"**15m EMA20:** {r.ema15dist:+.2f}% | **4H EMA20:** {r.ema4dist:+.2f}% | **15m volume:** {r.vol:.1f}x")
    rows=[]
    for tf,supports,resists in [("15m",r.support15,r.resistance15),("4H",r.support4,r.resistance4),("1D",r.support1,r.resistance1)]:
        for i,p in enumerate(supports,1): rows.append({"TF":tf,"Level":f"S{i}","Price":fp(p),"Distance":f"{(p/r.current-1)*100:+.2f}%"})
        for i,p in enumerate(resists,1): rows.append({"TF":tf,"Level":f"R{i}","Price":fp(p),"Distance":f"{(p/r.current-1)*100:+.2f}%"})
    st.markdown("#### Support / Resistance")
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("#### Trade Plan")
    st.write(f"**Trigger:** {r.trigger}")
    st.write(f"**Invalidation:** {r.invalidation}")
    st.write("**Rule:** do not chase a mature pump/dump; wait for the trigger.")
else:
    st.info("No fresh LONG/SHORT setup passed the filters.")

st.markdown("---")
st.caption("A hard pump/dump alone is not a signal. The agent requires a structure transition and a fresh trigger. This is technical analysis, not a guarantee of future price movement.")
