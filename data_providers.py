"""External market data provider interfaces."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import requests


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_data(self, symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
        """Return historical market data for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol to request.
        interval:
            Bar interval string, e.g. ``"1min"`` or ``"60min"``.
        outputsize:
            Number of data points to return.
        """


class AlphaVantageDataProvider(DataProvider):
    """Fetch historical data from the Alpha Vantage REST API.

    Only a tiny subset of the API is implemented to keep the example light.
    The provider returns an empty :class:`pandas.DataFrame` if the request
    fails for any reason which allows the calling code to gracefully fall
    back to other data sources.
    """

    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def get_historical_data(self, symbol: str, interval: str, outputsize: int = 100) -> pd.DataFrame:
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "full" if outputsize > 100 else "compact",
            "datatype": "json",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json().get(f"Time Series ({interval})", {})
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("Alpha Vantage request failed: %s", exc)
            return pd.DataFrame()

        records = []
        for ts, vals in sorted(payload.items()):
            records.append(
                {
                    "date": pd.to_datetime(ts),
                    "open": float(vals.get("1. open", 0)),
                    "high": float(vals.get("2. high", 0)),
                    "low": float(vals.get("3. low", 0)),
                    "close": float(vals.get("4. close", 0)),
                    "volume": float(vals.get("5. volume", 0)),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame.from_records(records).set_index("date")
        return df
