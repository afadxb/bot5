# Trading Bot

Modular algorithmic trading bot for equities with IBKR data/execution, robust
universe loading, throttled data requests, bracket/OCO order wiring, and
rotating logs. Targets Python 3.12 with a small, production‑friendly footprint.

## Features

- IBKR data with interval/duration mapping, throttling and retry
- Vectorized indicators (SMA, EMA, RSI, MACD, ATR, Supertrend, OBV) with NaN guards
- Scoring engine with sentiment gate and near‑support confirmation
- Bracket orders (entry + stop + TP1/TP2) using IB parentId + OCA (one‑cancels‑all)
- SQLite persistence (positions, orders, snapshots) using WAL and helpful indices
- Rotating logs to file and console; clear env validation on startup
- Universe loaded from `sp100.csv` (headerless or with `Symbol` column)
- Dry‑run scan mode to score the whole universe without placing orders
- Optional push notifications via Pushover

## Repository Layout

| Module | Description |
| ------ | ----------- |
| `config.py` | Loads environment variables and defines global constants |
| `models.py` | Dataclasses and enums for orders, positions, regimes |
| `data_access.py` | SQLite persistence for orders, positions and analytics |
| `order_management.py` | Build/manage bracket orders, stops and trails |
| `brokers.py` | Broker abstractions (IBKR with parentId + OCA wiring) |
| `data_providers.py` | External market data provider interfaces (IBKR) |
| `strategy.py` | Indicators, scoring, sentiment, support, TradingBot orchestration |
| `alerts.py` | Pushover notifications |
| `main.py` | Entry point, rotating logs, env validation, scan mode |

## Architecture

See `docs/architecture.md` for a high‑level overview of components and data flow.

## Universe CSV

`SP100_CSV` can be either of the following:
- Headerless file with one symbol per line:
  
  AAPL
  MSFT
  AMD

- CSV with a `Symbol` column:
  
  Symbol,Name
  AAPL,Apple Inc.
  MSFT,Microsoft Corp

`SPY` and `VIX` are excluded from trading and used for regime detection only.

## Configuration (.env)

```
DB_PATH=trading_bot.db
LOG_FILE=trading_bot.log
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1
OPERATING_SYSTEM=windows
PYTHON_VERSION=3.12
PUSHOVER_USER=
PUSHOVER_TOKEN=
DEBUG=0
SP100_CSV=sp100.csv

# Entry and sentiment
ENTRY_SCORE_THRESHOLD=70
SUPPORT_DISTANCE_ATR=0.5
SUPPORT_DISTANCE_PCT=0.5
FG_HARD_BLOCK_THRESHOLD=25
FG_PENALTY_THRESHOLD=45
FG_OVERHEAT_THRESHOLD=80
FG_PENALTY=5
FG_OVERHEAT_PENALTY=5
NEWS_POSITIVE_THRESHOLD=0.7
NEWS_NEGATIVE_THRESHOLD=-0.7
NEWS_POSITIVE_BONUS=3
NEWS_NEGATIVE_PENALTY=5
RISK_OFF_MIN_SCORE=85
RISK_OFF_NEWS_THRESHOLD=0

# Risk
ACCOUNT_EQUITY=100000
RISK_PER_TRADE=0.01
MAX_POSITION_PCT=0.1

# Safety / modes
ENABLE_TRADING=0
SCAN_ONLY=0
```

Notes:
- Account equity for sizing prefers IBKR `NetLiquidation`; falls back to `ACCOUNT_EQUITY`.
- Set `ENABLE_TRADING=1` to submit orders to IBKR. When `0`, orders are persisted but not sent.
- `DEBUG=1` increases logging verbosity.

## Installation

Create a Python 3.12 virtual environment and install dependencies:

```
pip install -r requirements.txt
```

## Usage

Run a dry‑run scan (no orders placed):

```
python main.py scan
# or
set SCAN_ONLY=1 & python main.py   # Windows
export SCAN_ONLY=1 && python main.py  # macOS/Linux
```

Run the hourly strategy (obeys market hours; can place orders if enabled):

```
python main.py
```

Enable trading only when connected to IBKR and comfortable with the configuration:

```
ENABLE_TRADING=1
```

## Operational Notes

- Data provider maps intervals to IBKR bar sizes (e.g., `60min` -> `1 hour`, `daily` -> `1 day`) and
  selects durations based on `outputsize` (`3 D/1 M` intraday; `3 M/2 Y` daily). Requests are throttled
  and retried with exponential backoff to respect pacing limits.
- Indicators and scoring handle NaNs during warmup; logic checks for sufficient history before use.
- Bracket orders wire IBKR `parentId` for child legs and set an OCA group, so one fill cancels the others.
- SQLite runs in WAL mode; indices are added on common columns for snappy reads.
- Logging uses a rotating file handler (5 x 5MB) and console stream.

## Development

Run tests and linters as needed:

```
pytest -q
```

---

This software interacts with real markets and brokers. Use paper trading first and
proceed carefully in live environments.

