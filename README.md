# Trading Bot

This project provides a modular algorithmic trading bot. The code has been
refactored into dedicated modules for data access, order management and
strategy logic to improve readability and maintainability. It targets a
Windows environment running Python 3.12.

## Project Structure

- `config.py` – loads environment variables and defines common constants.
- `models.py` – dataclasses and enums that describe orders and positions.
- `data_access.py` – database layer for persisting orders and positions.
- `order_management.py` – routines for placing and managing orders.
- `brokers.py` – lightweight abstractions for broker integrations.
- `data_providers.py` – pluggable market data provider interfaces, including
  real-time streaming via the Alpha Vantage API.
- `strategy.py` – trading strategy, indicators and bot orchestration.
- `main.py` – entry point that wires everything together.

## Configuration

Runtime configuration is provided through a `.env` file. The following
parameters are supported:

```
DB_PATH=trading_bot.db
LOG_FILE=trading_bot.log
IBKR_HOST=127.0.0.1
IBKR_PORT=7496
IBKR_CLIENT_ID=1
OPERATING_SYSTEM=windows
PYTHON_VERSION=3.12
ALPHAVANTAGE_API_KEY=
PUSHOVER_USER=
PUSHOVER_TOKEN=
DEBUG=0
SP100_CSV=
```

`ALPHAVANTAGE_API_KEY` is optional and enables the included Alpha Vantage
data provider. When supplied the bot can fetch both historical and
real-time quote data. `PUSHOVER_USER` and `PUSHOVER_TOKEN` enable optional
push notifications via the [Pushover](https://pushover.net/) service. Set
`DEBUG` to ``1`` for verbose logging. `SP100_CSV` should point to a CSV
file containing the S&P 100 constituents. Additional variables can be
added as needed. Defaults are provided when the variables are absent.

## Development

Create a Python 3.12 virtual environment and install dependencies:

```
pip install -r requirements.txt  # if available
```

Run the test suite:

```
pytest -q
```

## Notes

The bot integrates with real broker APIs and market data providers. By
default it retrieves market data from the Alpha Vantage API, while IBKR
handles account management and order execution. The included
`ibkr_client.py` demonstrates how to request historical data and submit
basic orders while respecting their pacing limitations. Upon startup the bot
logs the current IBKR cash balance, buying power and any open positions
loaded from the local database.

