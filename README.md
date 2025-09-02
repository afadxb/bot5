# Trading Bot

This repository contains a fully modular algorithmic trading bot aimed at
automating equity trades.  Each piece of functionality – from data retrieval to
order execution – lives in a small, well documented module.  The code targets
Python 3.12 and has minimal external dependencies, making it suitable for local
experimentation or further extension.

## Features

- Pulls historical and real‑time market data from IBKR
- Persists orders, positions and analytics snapshots to a local SQLite database
- Scores symbols using a rich set of technical indicators and market regime
  filters
- Places bracket orders and manages stops through a pluggable broker interface
- Optional push notifications via the Pushover service

## Repository Layout

The project is intentionally split into small modules to keep concerns isolated:

| Module | Description |
| ------ | ----------- |
| `config.py` | Loads environment variables and defines global constants |
| `models.py` | Dataclasses and enums representing orders and positions |
| `data_access.py` | SQLite persistence for orders, positions and analytics |
| `order_management.py` | High level order submission and modification logic |
| `brokers.py` | Lightweight abstractions for broker integrations (IBKR) |
| `data_providers.py` | Interfaces for external market data sources |
| `strategy.py` | Trading strategy, indicator helpers and `TradingBot` orchestration |
| `alerts.py` | Convenience wrapper for Pushover notifications |
| `main.py` | Command line entry point to run the bot |

## Architecture

For a high-level view of how these components interact, see [docs/architecture.md](docs/architecture.md).

## Configuration

Runtime configuration is supplied through a `.env` file located in the project
root.  The following variables are recognised:

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
SP100_CSV=
```

`PUSHOVER_USER` and `PUSHOVER_TOKEN` enable optional push notifications via the
[Pushover](https://pushover.net/) service.  Set `DEBUG` to ``1`` for verbose
logging.  `SP100_CSV` should point to a CSV file containing the S&P 100
constituents.  Defaults are provided when the variables are absent.

## Installation

Create a Python 3.12 virtual environment and install dependencies:

```
pip install -r requirements.txt
```

## Usage

Populate a `.env` file with the desired configuration, then run the bot:

```
python main.py
```

On start-up the bot fetches account details, loads any persisted positions and
evaluates the trading universe defined by `SP100_CSV`.

## Development

Run the test suite to validate changes:

```
pytest -q
```

## Notes

The bot integrates with real broker APIs and market data providers.  By default
IBKR supplies both market data and order execution.  The included
`ibkr_client.py` demonstrates how to request historical data and submit basic
orders while respecting IBKR pacing limitations.

