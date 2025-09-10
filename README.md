# ETF Ki Dukaan (NSE) – 52W Distance Scanner + Backtester

Daily scanner:
- Selects **Top-20 ETFs by 60-day average volume**
- Ranks **Farthest ABOVE 52-week low** and **Closest TO 52-week low**
- Sends a **Telegram message** at **09:30 IST** and **15:15 IST** (no attachments)

Backtester:
- Daily strategy: **Buy the ETF closest to its 52-week low** (from the daily Top-20 by 60d Avg Volume)
- **Pause buys when NIFTY < 200-DMA**
- Allocation: ₹10,000 per eligible day; **sell one position** per day if in profit

## Quickstart (local)
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/backtest_kidukaan.py
# To run the scanner once:
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
python src/movers_52w_distance.py
