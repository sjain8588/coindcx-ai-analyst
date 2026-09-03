import streamlit as st
import pandas as pd
import numpy as np
import requests
import time

API = "https://api.coindcx.com"
PUBLIC = "https://public.coindcx.com"

st.set_page_config(page_title="CoinDCX Futures Scanner", page_icon="🎯", layout="wide")
st.title("🎯 CoinDCX Futures Momentum Scanner")
st.caption("Analysis only • Momentum + ATH/ATL • No trading")

MEME_WORDS={"DOGE","SHIB","PEPE","BONK","FLOKI","WIF","BOME","MEME","BRETT","MOG","TURBO","MEW","NEIRO","BABYDOGE","1000SHIB","1000PEPE","1000BONK","1000FLOKI","1000LUNC","PONKE","MYRO","SLERF","LADYS","DEGEN","MOTHER","MAGA","TRUMP"}

@st.cache_data(ttl=30)
def active_instruments(margin="USDT"):
    r=requests.get(f"{API}/exchange/v1/derivatives/futures/data/active_instruments",params=[("margin_currency_short_name[]",margin)],timeout=20)
    r.raise_for_status(); x=r.json()
    if not isinstance(x,list): raise RuntimeError(f"Unexpected instruments response: {x}")
    return x

@st.cache_data(ttl=5)
def futures_prices():
    r=requests.get(f"{PUBLIC}/market_data/v3/current_prices/futures/rt",timeout=20); r.raise_for_status(); x=r.json()
    return x.get("prices",{}) if isinstance(x,dict) else {}

@st.cache_data(ttl=30)
def candles(pair,resolution,start_ts,end_ts):
    p={"pair":pair,"from":int(start_ts),"to":int(end_ts),"resolution":resolution,"pcode":"f"}
    r=requests.get(f"{PUBLIC}/market_data/candlesticks",params=p,timeout=30); r.raise_for_status(); x=r.json()
    rows=x.get("data",[]) if isinstance(x,dict) else x
    if not isinstance(rows,list): raise RuntimeError(f"Unexpected candle response for {pair}: {x}")
    d=pd.DataFrame(rows)
    if d.empty:return d
    for c in ["open","high","low","close","volume"]:
        if c not in d: raise RuntimeError(f"{pair} candle response missing {c}")
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["time"]=pd.to_datetime(d["time"],unit="ms",errors="coerce")
    return d.dropna(subset=["time","open","high","low","close","volume"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)

@st.cache_data(ttl=5)
def orderbook(pair):
    r=requests.get(f"{PUBLIC}/market_data/v3/orderbook/{pair}-futures/50",timeout=15); r.raise_for_status(); return r.json()

@st.cache_data(ttl=5)
def trades(pair):
    r=requests.get(f"{API}/exchange/v1/derivatives/futures/data/trades",params={"pair":pair},timeout=15); r.raise_for_status(); x=r.json(); return x if isinstance(x,list) else []

def indicators(d):
    x=d.copy()
    for n in (20,50,100,200): x[f"ema{n}"]=x.close.ewm(span=n,adjust=False).mean()
    delta=x.close.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean(); rs=gain/loss.replace(0,np.nan); x["rsi"]=100-100/(1+rs)
    e12=x.close.ewm(span=12,adjust=False).mean(); e26=x.close.ewm(span=26,adjust=False).mean(); x["macd"]=e12-e26; x["macd_signal"]=x.macd.ewm(span=9,adjust=False).mean()
    tr=pd.concat([x.high-x.low,(x.high-x.close.shift()).abs(),(x.low-x.close.shift()).abs()],axis=1).max(axis=1); x["atr"]=tr.ewm(alpha=1/14,adjust=False).mean(); x["volma"]=x.volume.rolling(20).mean()
    x["bbmid"]=x.close.rolling(20).mean(); x["bbstd"]=x.close.rolling(20).std(); x["bbup"]=x.bbmid+2*x.bbstd; x["bblow"]=x.bbmid-2*x.bbstd
    up=x.high.diff(); dn=-x.low.diff(); plus=pd.Series(np.where((up>dn)&(up>0),up,0.0),index=x.index); minus=pd.Series(np.where((dn>up)&(dn>0),dn,0.0),index=x.index); atr=x.atr.replace(0,np.nan); pdi=100*plus.ewm(alpha=1/14,adjust=False).mean()/atr; mdi=100*minus.ewm(alpha=1/14,adjust=False).mean()/atr; dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan); x["adx"]=dx.ewm(alpha=1/14,adjust=False).mean(); x["pdi"]=pdi; x["mdi"]=mdi
    return x

def structure(d):
    if len(d)<40:return "Mixed"
    a=d.tail(12); b=d.iloc[-36:-12]
    if a.high.max()>b.high.max() and a.low.min()>b.low.min():return "Bullish"
    if a.high.max()<b.high.max() and a.low.min()<b.low.min():return "Bearish"
    return "Mixed"

def tech(d):
    x=d.iloc[-1]; L=S=0
    for a,b in [(x.close,x.ema20),(x.ema20,x.ema50),(x.ema50,x.ema100),(x.ema100,x.ema200)]:
        if pd.notna(a) and pd.notna(b): L+=a>b; S+=a<b
    L+=int(50<x.rsi<70)+int(x.macd>x.macd_signal)+int(x.pdi>x.mdi)+int(pd.notna(x.volma) and x.volume>x.volma)
    S+=int(30<x.rsi<50)+int(x.macd<x.macd_signal)+int(x.mdi>x.pdi)+int(pd.notna(x.volma) and x.volume>x.volma)
    return int(L),int(S)

def micro(pair):
    ob=flow=None
    try:
        b=orderbook(pair); B=[(float(p),float(q)) for p,q in b.get("bids",{}).items()]; A=[(float(p),float(q)) for p,q in b.get("asks",{}).items()]
        if B and A:
            bid=max(p for p,q in B); ask=min(p for p,q in A); mid=(bid+ask)/2; nb=sum(q for p,q in B if p>=mid*.995); na=sum(q for p,q in A if p<=mid*1.005); ob=(nb-na)/(nb+na) if nb+na else 0
    except Exception: pass
    try:
        buy=sell=0
        for t in trades(pair):
            q=float(t.get("quantity",0) or 0); sell+=q if bool(t.get("is_maker",False)) else 0; buy+=q if not bool(t.get("is_maker",False)) else 0
        flow=(buy-sell)/(buy+sell) if buy+sell else 0
    except Exception: pass
    return ob,flow

def sr_levels(d):
    x=d.sort_values("time").reset_index(drop=True).copy(); price=float(x.iloc[-1].close); atr=float(x.iloc[-1].atr) if "atr" in x and pd.notna(x.iloc[-1].atr) else price*.02; tol=max(atr*.6,price*.002); levels=[]
    for i in range(2,len(x)-2):
        if x.iloc[i].high>=x.iloc[i-2:i+3].high.max(): levels.append((float(x.iloc[i].high),"R",x.iloc[i].time,False))
        if x.iloc[i].low<=x.iloc[i-2:i+3].low.min(): levels.append((float(x.iloc[i].low),"S",x.iloc[i].time,False))
    try:
        m=x.set_index("time").resample("ME").agg({"high":"max","low":"min"}).dropna()
    except Exception:
        m=x.set_index("time").resample("M").agg({"high":"max","low":"min"}).dropna()
    for t,r in m.iterrows(): levels += [(float(r.high),"R",t,True),(float(r.low),"S",t,True)]
    def cluster(kind):
        raw=sorted([z for z in levels if z[1]==kind],key=lambda z:z[0]); out=[]
        for p,k,t,mon in raw:
            if not out or abs(p-out[-1]["price"])>tol: out.append({"price":p,"touches":1,"last":t,"monthly":mon})
            else:
                c=out[-1]; c["price"]=(c["price"]*c["touches"]+p)/(c["touches"]+1); c["touches"]+=1; c["last"]=max(c["last"],t); c["monthly"]|=mon
        return out
    s=sorted([c for c in cluster("S") if c["price"]<price*.9995],key=lambda c:c["price"],reverse=True); r=sorted([c for c in cluster("R") if c["price"]>price*1.0005],key=lambda c:c["price"])
    below=m.low[m.low<price]; above=m.high[m.high>price]
    return {"support1":s[0]["price"] if s else np.nan,"support2":s[1]["price"] if len(s)>1 else np.nan,"resistance1":r[0]["price"] if r else np.nan,"resistance2":r[1]["price"] if len(r)>1 else np.nan,"monthly_support":float(below.max()) if not below.empty else np.nan,"monthly_resistance":float(above.min()) if not above.empty else np.nan,"ath":float(x.high.max()),"atl":float(x.low.min())}

def score(pair,p,d1,h1,m15):
    x1,xh,x15=d1.iloc[-1],h1.iloc[-1],m15.iloc[-1]; ch=float(p.get("pc",0) or 0); move=abs(ch); b1,s1=tech(d1); bh,sh=tech(h1); b15,s15=tech(m15); ob,flow=micro(pair); side="LONG" if ch>0 else "SHORT"
    raw=move*1.5+(b1-s1)*7+(bh-sh)*5+(b15-s15)*3 if side=="LONG" else move*1.5+(s1-b1)*7+(sh-bh)*5+(s15-b15)*3
    if side=="LONG": raw+=(max(0,ob)*8 if ob is not None else 0)+(max(0,flow)*6 if flow is not None else 0)
    else: raw+=(max(0,-ob)*8 if ob is not None else 0)+(max(0,-flow)*6 if flow is not None else 0)
    atr=max(float(x1.atr) if pd.notna(x1.atr) else float(x1.close)*.01,1e-12); ext=abs(float(x1.close-x1.ema20))/atr; raw-=min(25,max(0,(ext-2)*7)); conf=max(35,min(95,50+raw*.45)); hf=min(1,min(len(d1)/120,len(h1)/240,len(m15)/240)); conf=min(conf,50+45*hf); conf=min(conf,65) if hf<.5 else conf
    aligned=(side=="LONG" and b1>s1 and bh>=sh and b15>=s15 and x15.macd>x15.macd_signal) or (side=="SHORT" and s1>b1 and sh>=bh and s15>=b15 and x15.macd<x15.macd_signal); extreme=(side=="LONG" and x1.rsi>=78) or (side=="SHORT" and x1.rsi<=22); signal=side if aligned and not extreme and conf>=68 else "WAIT"
    price=float(x15.close); atr15=max(float(x15.atr) if pd.notna(x15.atr) else price*.01,1e-12); sl=tp1=tp2=np.nan
    if signal=="LONG": sl=price-1.4*atr15; tp1=price+1.5*(price-sl); tp2=price+2.5*(price-sl)
    elif signal=="SHORT": sl=price+1.4*atr15; tp1=price-1.5*(sl-price); tp2=price-2.5*(sl-price)
    return {"pair":pair,"change":ch,"side":side,"signal":signal,"confidence":conf,"price":price,"rsi":float(x1.rsi),"adx":float(xh.adx),"structure":structure(d1),"ob":ob,"flow":flow,"score":raw,"entry":price,"sl":sl,"tp1":tp1,"tp2":tp2,**sr_levels(d1),"d1":d1,"h1":h1,"m15":m15}

def fmt(v):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):,.8f}".rstrip("0").rstrip(".")
    except Exception:return "—"

def coin_matches(pair,symbol,requested,quote):
    req=normalize(requested); names=[str(pair).upper().replace("-","").replace("_",""),str(symbol).upper().replace("-","").replace("_","")]
    for n in names:
        for v in (n,n[1:] if n.startswith("B") else n,n[1:] if n.startswith("I") else n):
            if v==req or v==req+quote or v.startswith(req+quote): return True
    return False

def normalize(s):
    q=s.strip().upper().replace("/","").replace("-","").replace("_","")
    for quote in ("USDT","INR","USDC"):
        if q.endswith(quote) and len(q)>len(quote): return q[:-len(quote)]
    return q

margin=st.selectbox("Futures margin market",["USDT","INR"],index=0)
meme_only=st.checkbox("Meme-focused scan",value=True)

# ============================== MY COIN ====================================
st.divider(); st.header("💼 My Invested Coin — Long / Short Check")
st.caption("Analyze a coin you already hold using Futures trend, momentum, structure and liquidity.")
coin=st.text_input("Coin / Futures pair",placeholder="DOGE, SHIB, PEPE, B-DOGE_USDT")
avg=st.number_input("Optional: your average entry price",min_value=0.0,value=0.0,step=0.00000001,format="%.8f")
if st.button("📊 Analyze My Coin"):
    try:
        prices=futures_prices(); req=normalize(coin); markets=[margin]+[q for q in ("USDT","INR") if q!=margin]; found=[]
        for q in markets:
            for pair in active_instruments(q):
                p=prices.get(pair)
                if p and coin_matches(pair,p.get("mkt",pair),req,q): found.append((pair,p,str(p.get("mkt",pair)).upper(),q))
        if not found: st.error(f"No active CoinDCX Futures contract found for '{coin}'."); st.stop()
        found.sort(key=lambda z:(0 if z[3]==margin else 1,len(z[0]))); pair,p,symbol,found_margin=found[0]
        now=int(time.time()); d1=candles(pair,"1D",now-400*86400,now); h1=candles(pair,"60",now-120*86400,now); m15=candles(pair,"15",now-30*86400,now)
        if len(d1)<10 or len(h1)<30 or len(m15)<30: st.error(f"Too little history for {symbol}: 1D={len(d1)}, 1H={len(h1)}, 15m={len(m15)}"); st.stop()
        x=score(pair,p,indicators(d1),indicators(h1),indicators(m15)); st.subheader(f"{symbol} • {pair}")
        if avg>0: st.metric("Your position vs current price",f"{(x['price']-avg)/avg*100:+.2f}%")
        (st.success if x["signal"]=="LONG" else st.error if x["signal"]=="SHORT" else st.warning)("🟢 LONG BIAS" if x["signal"]=="LONG" else "🔴 SHORT BIAS" if x["signal"]=="SHORT" else "🟡 WAIT")
        a,b,c,d=st.columns(4); a.metric("Current",fmt(x["price"])); b.metric("24h",f"{x['change']:+.2f}%"); c.metric("Confidence",f"{x['confidence']:.0f}%"); d.metric("1D RSI",f"{x['rsi']:.1f}")
        st.write(f"**1D structure:** {x['structure']} | **1H ADX:** {x['adx']:.1f} | **S1/S2:** {fmt(x['support1'])} / {fmt(x['support2'])} | **R1/R2:** {fmt(x['resistance1'])} / {fmt(x['resistance2'])}")
        a,b,c,d=st.columns(4); a.metric("Monthly Support",fmt(x["monthly_support"])); b.metric("Monthly Resistance",fmt(x["monthly_resistance"])); c.metric("ATH",fmt(x["ath"])); d.metric("ATL",fmt(x["atl"]))
        if x["signal"] in ("LONG","SHORT"):
            a,b,c,d=st.columns(4); a.metric("Entry",fmt(x["entry"])); b.metric("Stop",fmt(x["sl"])); c.metric("TP1",fmt(x["tp1"])); d.metric("TP2",fmt(x["tp2"]))
        with st.expander("Advanced analysis"):
            st.write(f"EMA20/50/100/200: {fmt(x['d1'].iloc[-1].ema20)} / {fmt(x['d1'].iloc[-1].ema50)} / {fmt(x['d1'].iloc[-1].ema100)} / {fmt(x['d1'].iloc[-1].ema200)}")
            st.write(f"Order-book imbalance: {('N/A' if x['ob'] is None else f'{x["ob"]*100:.2f}%')} | Trade-flow proxy: {('N/A' if x['flow'] is None else f'{x["flow"]*100:.2f}%')}")
            st.line_chart(x["d1"].set_index("time")[["close","ema20","ema50","ema100","ema200"]].tail(250))
    except Exception as e: st.error(f"My Coin analysis failed: {type(e).__name__}: {e}")

# ============================ TOP MOMENTUM =================================
if st.button("🔍 Scan Top Futures",type="primary"):
    try:
        active=active_instruments(margin); prices=futures_prices(); rows=[]
        for pair in active:
            p=prices.get(pair)
            if not p: continue
            symbol=str(p.get("mkt",pair)).upper()
            if meme_only and not any(w in symbol or w in pair.upper() for w in MEME_WORDS): continue
            try: pc=float(p.get("pc",0) or 0)
            except Exception: continue
            if pc: rows.append((pair,p,symbol,pc))
        rows.sort(key=lambda z:abs(z[3]),reverse=True); results=[]; failures=[]; now=int(time.time())
        for pair,p,symbol,pc in rows[:80]:
            try:
                d1=candles(pair,"1D",now-400*86400,now); h1=candles(pair,"60",now-120*86400,now); m15=candles(pair,"15",now-30*86400,now)
                if len(d1)<10 or len(h1)<30 or len(m15)<30:
                    failures.append(f"{symbol}: insufficient candles 1D={len(d1)}, 1H={len(h1)}, 15m={len(m15)}"); continue
                results.append((symbol,score(pair,p,indicators(d1),indicators(h1),indicators(m15))))
                if len(results)>=5: break
            except Exception as e: failures.append(f"{symbol}: {type(e).__name__}: {e}")
        if not results:
            st.error("No Futures candidate could be analyzed.")
            with st.expander("🔧 Scan diagnostics",expanded=True): st.code("\n".join(failures[:80]) or "No diagnostics returned.")
            st.stop()
        st.header("🎯 Today's Top 5 Futures Candidates"); results.sort(key=lambda z:z[1]["score"],reverse=True)
        for i,(symbol,x) in enumerate(results[:5],1):
            icon="🟢" if x["signal"]=="LONG" else "🔴" if x["signal"]=="SHORT" else "🟡"
            with st.container(border=True):
                st.subheader(f"{i}. {symbol} {icon} {x['signal']}"); a,b,c,d=st.columns(4); a.metric("24h",f"{x['change']:+.2f}%"); b.metric("Confidence",f"{x['confidence']:.0f}%"); c.metric("Price",fmt(x['price'])); d.metric("RSI",f"{x['rsi']:.1f}")
                st.write(f"**S1/S2:** {fmt(x['support1'])} / {fmt(x['support2'])} | **R1/R2:** {fmt(x['resistance1'])} / {fmt(x['resistance2'])} | **1D:** {x['structure']} | **1H ADX:** {x['adx']:.1f}")
                if x["signal"] in ("LONG","SHORT"):
                    a,b,c,d=st.columns(4); a.metric("Entry",fmt(x["entry"])); b.metric("Stop",fmt(x["sl"])); c.metric("TP1",fmt(x["tp1"])); d.metric("TP2",fmt(x["tp2"]))
                with st.expander("Advanced analysis"): st.line_chart(x["d1"].set_index("time")[["close","ema20","ema50","ema100","ema200"]].tail(250))
    except Exception as e: st.error(f"Scanner failed: {type(e).__name__}: {e}")

# =============================== ATH / ATL =================================
def current_price(p):
    for k in ("ls","lp","last_price","price","mp","mark_price"):
        try:
            v=float(p.get(k,0) or 0)
            if v>0:return v
        except Exception: pass
    return 0.0

def extreme_status(cur,ext,mode):
    if ext<=0:return None,np.nan
    dist=(cur-ext)/ext*100
    if mode=="ATH":
        if dist>0:return "🔥 ATH BREAKOUT",dist
        if dist>=-.5:return "🟢 AT ATH",dist
        if dist>=-5:return "🟡 NEAR ATH",dist
    else:
        if dist<0:return "🔴 ATL BREAKDOWN",dist
        if dist<=.5:return "🟠 AT ATL",dist
        if dist<=5:return "🟡 NEAR ATL",dist
    return None,dist

def scan_extremes(active,prices,mode,limit=60):
    rows=[]
    for pair in active:
        p=prices.get(pair)
        if not p:continue
        sym=str(p.get("mkt",pair)).upper()
        if meme_only and not any(w in sym or w in pair.upper() for w in MEME_WORDS):continue
        try: pc=float(p.get("pc",0) or 0); cur=current_price(p)
        except Exception:continue
        if cur>0:rows.append((pair,p,sym,pc,cur))
    rows.sort(key=lambda z:z[3],reverse=(mode=="ATH")); rows=rows[:limit]; now=int(time.time()); out=[]; failures=[]
    for pair,p,sym,pc,cur in rows:
        try:
            d=candles(pair,"1D",now-400*86400,now)
            if len(d)<2:continue
            hist=d.iloc[:-1]; ext=float(hist.high.max()) if mode=="ATH" else float(hist.low.min()); status,dist=extreme_status(cur,ext,mode); c7=float(d.iloc[-8].close) if len(d)>=8 else float(d.iloc[0].close); ch7=(cur-c7)/c7*100 if c7 else np.nan
            if mode=="ATL" and status is None and pc<=-8: status="🔻 STRONG DOWN"
            if status is None:continue
            ind=indicators(d); last=ind.iloc[-1]; vr=float(last.volume)/float(last.volma) if pd.notna(last.volma) and last.volma>0 else np.nan
            out.append({"symbol":sym,"pair":pair,"price":cur,"extreme":ext,"distance":dist,"change":pc,"change7":ch7,"status":status,"rsi":float(last.rsi) if pd.notna(last.rsi) else np.nan,"vr":vr})
        except Exception as e: failures.append(f"{sym}: {type(e).__name__}: {e}")
    if mode=="ATH":out.sort(key=lambda z:(0 if z["distance"]>0 else 1,abs(z["distance"]),-z["change"]))
    else:out.sort(key=lambda z:(0 if z["status"]=="🔴 ATL BREAKDOWN" else 1 if z["status"]=="🟠 AT ATL" else 2 if z["status"]=="🟡 NEAR ATL" else 3,abs(z["distance"]),z["change"],z["change7"]))
    return out[:10],failures

st.divider(); st.header("🔥 / 🩸 Historical Extreme Scanner"); st.caption("Previous ATH/ATL excludes the latest daily candle. History is up to 400 days. ATL scan also includes 24h losers ≤ -8%.")
a,b=st.columns(2); do_ath=a.button("🔥 Scan Top 10 Near / Above ATH"); do_atl=b.button("🩸 Scan Top 10 Near / Below ATL")
if do_ath or do_atl:
    mode="ATH" if do_ath else "ATL"
    try:
        res,fail=scan_extremes(active_instruments(margin),futures_prices(),mode)
        if not res:
            st.warning(f"No {mode} candidates found in the scanned universe.")
            if fail:
                with st.expander("🔧 Extreme scanner diagnostics"):st.code("\n".join(fail[:60]))
        else:
            st.subheader("🔥 Top 10 Near / Above ATH" if mode=="ATH" else "🩸 Top 10 Near / Below ATL")
            table=[]
            for i,z in enumerate(res,1):table.append({"#":i,"Coin":z["symbol"],"Current":fmt(z["price"]),"ATH" if mode=="ATH" else "ATL":fmt(z["extreme"]),"Distance":f'{z["distance"]:+.2f}%',"24h":f'{z["change"]:+.2f}%',"7d":f'{z["change7"]:+.2f}%',"RSI":f'{z["rsi"]:.1f}',"Vol/20D":f'{z["vr"]:.1f}x' if pd.notna(z["vr"]) else "—","Status":z["status"]})
            st.dataframe(pd.DataFrame(table),use_container_width=True,hide_index=True)
    except Exception as e:st.error(f"{mode} scanner failed: {type(e).__name__}: {e}")

st.divider(); st.caption("Analysis only. No API keys, orders, balances or withdrawals are used. For a spot holding, SHORT BIAS is a downside-risk warning, not automatically a sell instruction.")
