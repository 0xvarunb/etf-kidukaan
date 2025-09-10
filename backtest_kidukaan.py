# src/backtest_kidukaan.py
import math, datetime as dt, pathlib
import numpy as np, pandas as pd, yfinance as yf, matplotlib.pyplot as plt

START_DATE = "2018-01-01"
END_DATE   = None
DAILY_BUDGET_RUPEES = 10_000
INITIAL_CASH = 1_000_000.0
PROFIT_EXIT_PCT = 0.0
TOP_VOLUME_COUNT = 20
VOLUME_LOOKBACK_DAYS = 60
ROLLING_LOW_DAYS = 252
NIFTY_TICKER = "^NSEI"
SMA_LENGTH = 200
DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "universe_nse.txt"

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

def fetch_panel(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="ticker")
    if isinstance(data.columns, pd.MultiIndex):
        close = pd.concat({t: data[t]["Close"] for t in tickers if (t in data.columns.levels[0] and "Close" in data[t])}, axis=1).dropna(how="all")
        volraw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, group_by="ticker")
        volume = pd.concat({t: volraw[t]["Volume"] for t in tickers if (t in volraw.columns.levels[0] and "Volume" in volraw[t])}, axis=1).dropna(how="all")
    else:
        only = tickers[0]
        close = data[["Close"]].rename(columns={"Close": only}).dropna(how="all")
        volraw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
        volume = volraw[["Volume"]].rename(columns={"Volume": only}).dropna(how="all")
    return close, volume

def fetch_nifty_series(start, end):
    raw = yf.download([NIFTY_TICKER,"NIFTYBEES.NS"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(raw, pd.DataFrame):
        s = raw[NIFTY_TICKER] if NIFTY_TICKER in raw.columns and raw[NIFTY_TICKER].dropna().size>0 else raw["NIFTYBEES.NS"]
    else:
        s = raw
    return s.dropna()

def rolling_min_inclusive(df, window):
    return df.rolling(window, min_periods=5).min()

def backtest():
    if END_DATE is None:
        end = dt.date.today().isoformat()
    else:
        end = END_DATE

    universe_all = load_universe(DATA_PATH)
    close, volume = fetch_panel(universe_all, START_DATE, end)
    dates = close.index
    volume = volume.reindex(dates)

    nifty = fetch_nifty_series(START_DATE, end).reindex(dates).ffill()
    sma200 = nifty.rolling(SMA_LENGTH, min_periods=SMA_LENGTH).mean()
    roll_low = rolling_min_inclusive(close, ROLLING_LOW_DAYS)

    start_idx = max(ROLLING_LOW_DAYS, VOLUME_LOOKBACK_DAYS+1, SMA_LENGTH)
    start_pos = start_idx if start_idx < len(dates) else 0

    cash = INITIAL_CASH
    positions = []
    trades = []
    equity_curve = []

    def mtm(prices):
        return sum(p["qty"] * prices.get(p["ticker"], np.nan) for p in positions)

    for i in range(start_pos, len(dates)):
        d = dates[i]
        prev_d = dates[i-1] if i>0 else d
        px = close.loc[d]

        # SELL (one in profit)
        sell_idx = -1; best_gain = 0.0
        for j, pos in enumerate(positions):
            price = px.get(pos["ticker"], np.nan)
            if not np.isnan(price):
                gain = (price - pos["cost"]) / pos["cost"]
                if gain >= PROFIT_EXIT_PCT and gain > best_gain:
                    best_gain = gain; sell_idx = j
        if sell_idx >= 0:
            pos = positions.pop(sell_idx)
            sell_px = px[pos["ticker"]]; proceeds = sell_px*pos["qty"]
            pnl = (sell_px - pos["cost"]) * pos["qty"]
            cash += proceeds
            trades.append({"date": d, "side":"SELL", "ticker":pos["ticker"], "price": float(sell_px), "qty": float(pos["qty"]), "pnl": float(pnl)})

        # BUY (if Nifty >= 200-DMA)
        can_buy = (not np.isnan(nifty.loc[d])) and (not np.isnan(sma200.loc[d])) and nifty.loc[d] >= sma200.loc[d]
        if can_buy and cash >= DAILY_BUDGET_RUPEES:
            vol_win = volume.loc[:prev_d].tail(VOLUME_LOOKBACK_DAYS)
            avg_vol = vol_win.mean(axis=0).replace(0, np.nan).dropna()
            topN = avg_vol.sort_values(ascending=False).head(TOP_VOLUME_COUNT).index.tolist()

            today_close = close.loc[d, topN].dropna()
            today_low   = roll_low.loc[d, topN].dropna()
            common = today_close.index.intersection(today_low.index)
            if len(common) > 0:
                ratio = (today_close.loc[common] / today_low.loc[common]).replace([np.inf,-np.inf], np.nan).dropna()
                if not ratio.empty:
                    tgt = ratio.idxmin()
                    buy_px = today_close[tgt]
                    qty = DAILY_BUDGET_RUPEES / buy_px
                    positions.append({"ticker": tgt, "qty": qty, "cost": float(buy_px), "buy_date": d})
                    cash -= DAILY_BUDGET_RUPEES
                    trades.append({"date": d, "side":"BUY", "ticker": tgt, "price": float(buy_px), "qty": float(qty), "pnl": 0.0})

        equity = cash + mtm(px)
        equity_curve.append({"date": d, "equity": float(equity), "cash": float(cash), "positions": len(positions)})

    eq = pd.DataFrame(equity_curve).set_index("date")
    tr = pd.DataFrame(trades)

    total_return = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) - 1.0
    days = (eq.index[-1] - eq.index[0]).days
    yrs = max(1e-9, days / 365.25)
    cagr = (1 + total_return) ** (1/yrs) - 1 if total_return > -0.999 else -1.0
    roll_max = eq["equity"].cummax()
    dd = eq["equity"]/roll_max - 1.0; max_dd = float(dd.min())

    wins = tr[tr["side"]=="SELL"]["pnl"] > 0
    win_rate = float(wins.mean()) if not wins.empty else float("nan")

    # Outputs
    eq["equity"].plot(figsize=(10,5), title="ETF Ki Dukaan — Daily Backtest"); plt.tight_layout()
    plt.savefig("equity_curve.png"); print("Saved: equity_curve.png")
    tr.to_csv("trades.csv", index=False); print("Saved: trades.csv")

    print("\n=== SUMMARY ===")
    print(f"Period: {eq.index[0].date()} → {eq.index[-1].date()}  ({days} days)")
    print(f"Final Equity: ₹{eq['equity'].iloc[-1]:,.0f}")
    print(f"Total Return: {100*total_return:,.2f}%  |  CAGR: {100*cagr:,.2f}%")
    print(f"Max Drawdown: {100*max_dd:,.2f}%  |  Win-rate: {np.nan if np.isnan(win_rate) else round(100*win_rate,2)}%")
    print(f"Buys: {(tr['side']=='BUY').sum()}  |  Sells: {(tr['side']=='SELL').sum()}")

if __name__ == "__main__":
    backtest()
