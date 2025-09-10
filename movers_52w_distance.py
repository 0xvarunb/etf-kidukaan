# src/movers_52w_distance.py
import os, math, datetime as dt, pathlib
import numpy as np, pandas as pd, yfinance as yf, requests

TOP_VOLUME_COUNT = 20
VOLUME_LOOKBACK_DAYS = 60
TOP_N = 10
ROLLING_DAYS_52W = 252
HIST_PADDING_DAYS = 400
DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "universe_nse.txt"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLE    = True

def load_universe(path: pathlib.Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out, seen = [], set()
    for s in lines:
        sym = s.split(":")[-1].strip().upper()
        if not sym: continue
        yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        if yf_sym not in seen:
            out.append(yf_sym); seen.add(yf_sym)
    return out

def last_non_nan(s: pd.Series) -> float:
    try: return float(s.dropna().iloc[-1])
    except: return float("nan")

def fetch_avg_volume_topN(tickers, lookback_days=60, topN=20):
    df = yf.download(tickers, period=f"{max(lookback_days*2, 90)}d", interval="1d",
                     auto_adjust=False, progress=False, group_by="ticker")
    avg_vol = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try: v = df[t]["Volume"].tail(lookback_days)
            except: v = pd.Series(dtype=float)
            m = v.replace(0, np.nan).mean()
            if pd.notna(m) and m > 0: avg_vol[t] = float(m)
    else:
        v = df.get("Volume", pd.Series(dtype=float)).tail(lookback_days)
        m = v.replace(0, np.nan).mean()
        if len(tickers)==1 and pd.notna(m) and m>0: avg_vol[tickers[0]] = float(m)
    s = pd.Series(avg_vol).dropna().sort_values(ascending=False)
    return s.head(min(topN, len(s))).index.tolist()

def fetch_latest_price(tickers):
    def latest_from(interval):
        d = yf.download(tickers, period="1d", interval=interval, auto_adjust=True, progress=False)
        if d is None or len(d)==0: return pd.Series(dtype=float)
        if isinstance(d.columns, pd.MultiIndex):
            cols = {}
            for t in set(d.columns.get_level_values(0)):
                try: cols[t] = d[t]["Close"]
                except KeyError: pass
            if not cols: return pd.Series(dtype=float)
            wide = pd.concat(cols, axis=1)
        else:
            wide = d[["Close"]].rename(columns={"Close": tickers[0]})
        return pd.Series({t: last_non_nan(wide[t]) for t in wide.columns})
    s = latest_from("1m")
    if s.empty or s.isna().all(): s = latest_from("5m")
    if s.empty or s.isna().all():
        dd = yf.download(tickers, period="2d", interval="1d", auto_adjust=True, progress=False)
        wide = dd["Close"] if isinstance(dd.columns, pd.MultiIndex) else dd[["Close"]].rename(columns={"Close": tickers[0]})
        s = pd.Series({t: last_non_nan(wide[t]) for t in wide.columns})
    return s

def fetch_52w_low(tickers, days_52w=252, pad_days=400):
    d = yf.download(tickers, period=f"{pad_days}d", interval="1d",
                    auto_adjust=True, progress=False, group_by="ticker")
    lows, dates = {}, {}
    if isinstance(d.columns, pd.MultiIndex):
        for t in tickers:
            try: s = d[t]["Close"].dropna()
            except: s = pd.Series(dtype=float)
            tail = s.tail(days_52w)
            if not tail.empty:
                lows[t] = float(tail.min()); dates[t] = tail.idxmin()
    else:
        s = d.get("Close", pd.Series(dtype=float)).dropna().tail(days_52w)
        if not s.empty and len(tickers)==1:
            lows[tickers[0]] = float(s.min()); dates[tickers[0]] = s.idxmin()
    return pd.DataFrame({"low_52w": pd.Series(lows), "low_date": pd.Series(dates)})

def tg_send_message(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    chunks = []
    while text:
        chunks.append(text[:3900]); text = text[3900:]
    ok = True
    for c in chunks:
        r = requests.post(url, data={"chat_id": chat_id, "text": c})
        ok = ok and r.ok
    return ok

def ist_now():
    try:
        import pytz
        return dt.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except:
        return dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def main():
    universe = load_universe(DATA_PATH)
    selected = fetch_avg_volume_topN(universe, VOLUME_LOOKBACK_DAYS, TOP_VOLUME_COUNT)
    if not selected: selected = universe[:TOP_VOLUME_COUNT]

    latest = fetch_latest_price(selected)
    lowdf  = fetch_52w_low(selected, ROLLING_DAYS_52W, HIST_PADDING_DAYS)
    df = pd.concat([latest.rename("last"), lowdf], axis=1)
    df["dist_from_52w_low_pct"] = (df["last"] / df["low_52w"] - 1.0) * 100.0
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["last","low_52w","dist_from_52w_low_pct"])

    far = df.sort_values("dist_from_52w_low_pct", ascending=False).head(TOP_N)
    clo = df.sort_values("dist_from_52w_low_pct", ascending=True ).head(TOP_N)

    def block(title, tdf):
        lines = [title]
        for i,(sym,r) in enumerate(tdf.iterrows(),1):
            ld = r["low_date"].strftime("%Y-%m-%d") if pd.notna(r["low_date"]) else "—"
            lines.append(f"{i:2d}. {sym}  {r['dist_from_52w_low_pct']:+.2f}%  "
                         f"(last {r['last']:.2f} / 52wLow {r['low_52w']:.2f} on {ld})")
        return "\n".join(lines)

    header = f"Universe: Top {TOP_VOLUME_COUNT} by {VOLUME_LOOKBACK_DAYS}-day Avg Volume\nSelected: {', '.join(selected)}"
    msg = f"""ETF Distance vs 52-Week Low (IST {ist_now()})

{header}

{block("Farthest ABOVE 52w Low:", far)}

{block("Closest TO 52w Low:", clo)}
"""
    print(msg)
    if TELEGRAM_ENABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        ok = tg_send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg)
        print(f"Telegram sendMessage: {'OK' if ok else 'FAIL'}")
    else:
        print("Telegram disabled or credentials missing; skipping push.")

if __name__ == "__main__":
    main()
