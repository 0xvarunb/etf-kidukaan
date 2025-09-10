# src/movers_52w_distance.py
# Schedules: run at 09:30 & 15:15 IST via GitHub Actions
# Features:
# - Universe from data/universe_nse.txt (robust path + env override) with safe fallback
# - Select Top-20 by 60d average Volume (up to today, no look-ahead in scanner context)
# - Rank Farthest ABOVE / Closest TO 52-week low (252 trading days)
# - Telegram message only, formatted as neat <pre> tables (copy-friendly)

import os, math, datetime as dt, pathlib
import numpy as np
import pandas as pd
import yfinance as yf
import requests

# -------------------- Config --------------------
TOP_VOLUME_COUNT = 20
VOLUME_LOOKBACK_DAYS = 60
TOP_N = 10
ROLLING_DAYS_52W = 252
HIST_PADDING_DAYS = 400  # days of history to download to ensure 252 rows exist

# Telegram (message only)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_ENABLE    = True

# Optional override for the universe file via env
UNIVERSE_FILE_ENV = os.getenv("UNIVERSE_FILE", "")

# -------------------- Path resolution --------------------
def resolve_universe_path() -> pathlib.Path | None:
    """Find data/universe_nse.txt in common locations or from UNIVERSE_FILE env."""
    if UNIVERSE_FILE_ENV:
        p = pathlib.Path(UNIVERSE_FILE_ENV).expanduser().resolve()
        return p if p.is_file() else None

    here = pathlib.Path(__file__).resolve()
    candidates = [
        here.parent.parent / "data" / "universe_nse.txt",  # repo_root/data (when script in src/)
        here.parent / "data" / "universe_nse.txt",         # repo_root/data (if script at root)
        pathlib.Path(os.getenv("GITHUB_WORKSPACE", "")) / "data" / "universe_nse.txt",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None

# -------------------- Universe loader --------------------
def load_universe(path: pathlib.Path | None) -> list[str]:
    # Built-in fallback so the job never crashes
    default_list = [
        "NSE:MOM30IETF","NSE:NIFTYQLITY","NSE:VAL30IETF","NSE:ABSLPSE","NSE:UTISXN50","NSE:CPSEETF",
        "NSE:GOLDBEES","NSE:HNGSNGBEES","NSE:MAHKTECH","NSE:HDFCGROWTH","NSE:LOWVOLIETF","NSE:HDFCQUAL",
        "NSE:BSE500IETF","NSE:COMMOIETF","NSE:FINIETF","NSE:INFRAIETF","NSE:MNC","NSE:ALPHAETF",
        "NSE:MIDSMALL","NSE:SMALLCAP","NSE:MONIFTY500","NSE:MOREALTY","NSE:MOSMALL250","NSE:MOVALUE",
        "NSE:MONQ50","NSE:MON100","NSE:TOP100CASE","NSE:NIFTYBEES","NSE:MOMENTUM50","NSE:ALPHA",
        "NSE:ALPL30IETF","NSE:AUTOIETF","NSE:BANKBEES","NSE:DIVOPPBEES","NSE:EVINDIA","NSE:BFSI",
        "NSE:FMCGIETF","NSE:HEALTHY","NSE:MOHEALTH","NSE:CONSUMBEES","NSE:MODEFENCE","NSE:TNIDETF",
        "NSE:MAKEINDIA","NSE:ITBEES","NSE:METALIETF","NSE:MOM100","NSE:MIDCAPETF","NSE:MIDQ50ADD",
        "NSE:MIDCAP","NSE:NEXT50IETF","NSE:OILIETF","NSE:PHARMABEES","NSE:PVTBANIETF","NSE:PSUBNKBEES",
        "NSE:TOP10ADD","NSE:ESG","NSE:NV20IETF","NSE:MULTICAP","NSE:EMULTIMQ","NSE:MAFANG",
        "NSE:MASPTOP50","NSE:ICICIB22","NSE:MIDSELIETF","NSE:SILVERBEES","NSE:SENSEXIETF","NSE:SHARIABEES",
    ]

    if path is None:
        print("[Info] data/universe_nse.txt not found; using built-in default list.")
        lines = default_list
    else:
        try:
            lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as e:
            print(f"[Warn] Failed to read {path}: {e}. Using built-in default list.")
            lines = default_list

    # normalize to Yahoo (SYMBOL.NS)
    out, seen = [], set()
    for s in lines:
        sym = s.split(":")[-1].strip().upper()
        if not sym:
            continue
        yf_sym = sym if sym.endswith(".NS") else f"{sym}.NS"
        if yf_sym not in seen:
            out.append(yf_sym); seen.add(yf_sym)
    return out

# -------------------- Helpers --------------------
def last_non_nan(s: pd.Series) -> float:
    try:
        return float(s.dropna().iloc[-1])
    except Exception:
        return float("nan")

def fetch_avg_volume_topN(tickers, lookback_days=60, topN=20):
    df = yf.download(
        tickers, period=f"{max(lookback_days*2, 90)}d", interval="1d",
        auto_adjust=False, progress=False, group_by="ticker"
    )
    avg_vol = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            try:
                v = df[t]["Volume"].tail(lookback_days)
            except Exception:
                v = pd.Series(dtype=float)
            m = v.replace(0, np.nan).mean()
            if pd.notna(m) and m > 0:
                avg_vol[t] = float(m)
    else:
        v = df.get("Volume", pd.Series(dtype=float)).tail(lookback_days)
        m = v.replace(0, np.nan).mean()
        if len(tickers) == 1 and pd.notna(m) and m > 0:
            avg_vol[tickers[0]] = float(m)

    s = pd.Series(avg_vol).dropna().sort_values(ascending=False)
    return s.head(min(topN, len(s))).index.tolist()

def fetch_latest_price(tickers):
    def latest_from(interval):
        d = yf.download(tickers, period="1d", interval=interval,
                        auto_adjust=True, progress=False)
        if d is None or len(d) == 0:
            return pd.Series(dtype=float)
        if isinstance(d.columns, pd.MultiIndex):
            cols = {}
            for t in set(d.columns.get_level_values(0)):
                try:
                    cols[t] = d[t]["Close"]
                except KeyError:
                    pass
            if not cols:
                return pd.Series(dtype=float)
            wide = pd.concat(cols, axis=1)
        else:
            wide = d[["Close"]].rename(columns={"Close": tickers[0]})
        return pd.Series({t: last_non_nan(wide[t]) for t in wide.columns})

    s = latest_from("1m")
    if s.empty or s.isna().all():
        s = latest_from("5m")
    if s.empty or s.isna().all():
        dd = yf.download(tickers, period="2d", interval="1d",
                         auto_adjust=True, progress=False)
        wide = dd["Close"] if isinstance(dd.columns, pd.MultiIndex) \
               else dd[["Close"]].rename(columns={"Close": tickers[0]})
        s = pd.Series({t: last_non_nan(wide[t]) for t in wide.columns})
    return s

def fetch_52w_low(tickers, days_52w=252, pad_days=400):
    d = yf.download(tickers, period=f"{pad_days}d", interval="1d",
                    auto_adjust=True, progress=False, group_by="ticker")
    lows, dates = {}, {}
    if isinstance(d.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = d[t]["Close"].dropna()
            except Exception:
                s = pd.Series(dtype=float)
            tail = s.tail(days_52w)
            if not tail.empty:
                lows[t] = float(tail.min())
                dates[t] = tail.idxmin()
    else:
        s = d.get("Close", pd.Series(dtype=float)).dropna().tail(days_52w)
        if not s.empty and len(tickers) == 1:
            lows[tickers[0]] = float(s.min())
            dates[tickers[0]] = s.idxmin()
    return pd.DataFrame({"low_52w": pd.Series(lows), "low_date": pd.Series(dates)})

def tg_send_message(token, chat_id, text, parse_mode=None):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    if parse_mode:
        data["parse_mode"] = parse_mode  # "HTML" or "MarkdownV2"
    resp = requests.post(url, data=data)
    return resp.ok

def ist_now():
    try:
        import pytz
        return dt.datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    except Exception:
        return dt.datetime.utcnow().strftime("%Y-%m-%d %I:%M:%S %p UTC")

def make_pre_table(title: str, df: pd.DataFrame) -> str:
    """
    Build a clean ASCII table in a <pre> block:
    columns: ETF | Price | Δ vs 52W Low
    """
    if df is None or df.empty:
        return f"<i>{title}</i>\n<pre>(no data)</pre>"

    # Prepare display strings
    rows = []
    for sym, r in df.iterrows():
        try:
            price_val = float(r["last"])
            pct_val   = float(r["dist_from_52w_low_pct"])
        except Exception:
            continue
        price_str = f"₹{price_val:,.2f}"
        pct_str   = f"{pct_val:+.2f}%"
        rows.append((sym, price_str, pct_str))

    # Column widths
    sym_w = max(3, max(len(x[0]) for x in rows)) if rows else 3
    pr_w  = max(5, max(len(x[1]) for x in rows)) if rows else 5
    pc_w  = max(7, max(len(x[2]) for x in rows)) if rows else 7

    # Header & rule
    header = f"{'ETF':<{sym_w}} | {'Price':>{pr_w}} | {'Δ vs 52W Low':>{pc_w}}"
    rule   = f"{'-'*sym_w}-+-{'-'*pr_w}-+-{'-'*pc_w}"

    # Body
    body_lines = [f"{sym:<{sym_w}} | {pr:>{pr_w}} | {pc:>{pc_w}}" for sym, pr, pc in rows]

    table = "\n".join([header, rule, *body_lines])
    return f"<i>{title}</i>\n<pre>{table}</pre>"

# -------------------- Main --------------------
def main():
    universe_path = resolve_universe_path()
    universe = load_universe(universe_path)

    # 1) Select Top-20 by 60d Avg Volume
    selected = fetch_avg_volume_topN(universe, VOLUME_LOOKBACK_DAYS, TOP_VOLUME_COUNT)
    if not selected:
        selected = universe[:TOP_VOLUME_COUNT]

    # 2) Latest price & 52w low stats on that selected universe
    latest = fetch_latest_price(selected)
    lowdf  = fetch_52w_low(selected, ROLLING_DAYS_52W, HIST_PADDING_DAYS)

    df = pd.concat([latest.rename("last"), lowdf], axis=1)
    df["dist_from_52w_low_pct"] = (df["last"] / df["low_52w"] - 1.0) * 100.0
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["last","low_52w","dist_from_52w_low_pct"])

    # 3) Rank lists
    far = df.sort_values("dist_from_52w_low_pct", ascending=False).head(TOP_N)
    clo = df.sort_values("dist_from_52w_low_pct", ascending=True ).head(TOP_N)

    # 4) Pretty Telegram message (HTML + <pre> tables)
    header_html = (
        f"<b>ETF 52W Distance Scan @ {ist_now()}</b>\n"
        f"<i>Universe: Top {TOP_VOLUME_COUNT} by {VOLUME_LOOKBACK_DAYS}-day Avg Volume</i>\n"
    )
    table_far = make_pre_table(f"Top {TOP_N} farthest ABOVE 52W low:", far)
    table_clo = make_pre_table(f"Top {TOP_N} closest TO 52W low:",  clo)

    msg = f"{header_html}\n{table_far}\n\n{table_clo}"
    print(msg)  # visible in Actions logs

    # 5) Telegram push (message only)
    if TELEGRAM_ENABLE and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        ok = tg_send_message(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, parse_mode="HTML")
        print(f"Telegram sendMessage: {'OK' if ok else 'FAIL'}")
    else:
        print("Telegram disabled or credentials missing; skipping push.")

if __name__ == "__main__":
    main()
