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
- `data_providers.py` – pluggable market data provider interfaces.
- `strategy.py` – trading strategy, indicators and bot orchestration.
- `main.py` – entry point that wires everything together.

## Configuration

Runtime configuration is provided through a `.env` file. The following
parameters are supported:

```
DB_PATH=trading_bot.db
LOG_FILE=trading_bot.log
IBKR_HOST=127.0.0.1
IBKR_PORT=7497
IBKR_CLIENT_ID=1
OPERATING_SYSTEM=windows
PYTHON_VERSION=3.12
ALPHAVANTAGE_API_KEY=
```

`ALPHAVANTAGE_API_KEY` is optional and enables the included Alpha Vantage
data provider. Additional variables can be added as needed. Defaults are
provided when the variables are absent.

## Development

Create a Python 3.12 virtual environment and install dependencies:

```
pip install -r requirements.txt  # if available
```

Run the (currently empty) test suite:

```
pytest -q
```

## Notes

The strategy logic and data handling routines are illustrative. In a real
trading environment these modules integrate with broker APIs and data
providers. The included `ibkr_client.py` demonstrates how to request
historical data from Interactive Brokers while respecting their pacing
limitations, following their [TWS API notes and limitations](https://www.interactivebrokers.com/campus/ibkr-api-page/twsapi-doc/#notes-and-limitations).
It mirrors the approach shown in the
"[Using technical indicators with TWS API](https://www.interactivebrokers.com/campus/ibkr-quant-news/using-technical-indicators-with-tws-api/)" article.
