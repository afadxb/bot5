"""External market data provider interfaces."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import pandas as pd
import requests

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover - allow import failure until installed
    yf = None


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_data(self, symbol: str, interval: str, outputsize: str) -> pd.DataFrame:
        """Return historical market data for *symbol*.

        Parameters
        ----------
        symbol:
            Ticker symbol to request.
        interval:
            Data interval such as ``"60min"`` or ``"15min"``.
        outputsize:
            Either ``"compact"`` (latest 100 points) or ``"full"`` as defined
            by the Alpha Vantage API documentation.
        """

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[float]:
        """Return the latest traded price for *symbol* or ``None`` on failure."""

    def stream_quotes(
        self, symbol: str, poll_interval: int = 60
    ) -> Iterator[Tuple[pd.Timestamp, float]]:
        """Yield ``(timestamp, price)`` tuples by polling :meth:`get_quote`.

        This naive implementation provides a simple real-time data stream by
        repeatedly requesting the latest price from the underlying provider.
        Subclasses can override this with a more efficient approach such as a
        WebSocket feed.
        """

        logger = logging.getLogger(__name__)
        while True:  # pragma: no cover - infinite loop
            try:
                price = self.get_quote(symbol)
            except Exception as exc:  # pragma: no cover - network dependent
                logger.error("Quote streaming failed for %s: %s", symbol, exc)
                price = None
            if price is not None:
                yield pd.Timestamp.utcnow(), price
            time.sleep(poll_interval)


class AlphaVantageDataProvider(DataProvider):
    """Fetch historical and real-time data from the Alpha Vantage REST API.

    The provider returns empty results if the request fails for any reason,
    allowing the calling code to gracefully fall back to other data sources.
    """

    def __init__(self, api_key: str, base_url: str = "https://www.alphavantage.co/query") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)

    def get_historical_data(
        self, symbol: str, interval: str, outputsize: str = "compact"
    ) -> pd.DataFrame:
        outputsize = outputsize if outputsize in {"compact", "full"} else "compact"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": "json",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("Alpha Vantage request failed: %s", exc)
            return pd.DataFrame()

        series_key = f"Time Series ({interval})"
        payload = data.get(series_key)
        if not payload:
            note = data.get("Note")
            error = data.get("Error Message")
            if note:
                self.logger.warning("Alpha Vantage notice: %s", note)
            if error:
                self.logger.error("Alpha Vantage error: %s", error)
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

    def get_quote(self, symbol: str) -> Optional[float]:
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "datatype": "json",
            "apikey": self.api_key,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("Alpha Vantage quote request failed: %s", exc)
            return None

        quote = data.get("Global Quote")
        if not quote:
            note = data.get("Note")
            error = data.get("Error Message")
            if note:
                self.logger.warning("Alpha Vantage notice: %s", note)
            if error:
                self.logger.error("Alpha Vantage error: %s", error)
            return None

        price = quote.get("05. price")
        return float(price) if price is not None else None


class YahooDataProvider(DataProvider):
    """Fetch historical and real-time data from Yahoo Finance using ``yfinance``."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _map_interval(interval: str) -> str:
        mapping = {
            "1min": "1m",
            "5min": "5m",
            "15min": "15m",
            "30min": "30m",
            "60min": "60m",
        }
        return mapping.get(interval, "1h")

    def get_historical_data(
        self, symbol: str, interval: str, outputsize: str = "compact"
    ) -> pd.DataFrame:
        if yf is None:  # pragma: no cover - dependency missing
            self.logger.error("yfinance is not available")
            return pd.DataFrame()

        yahoo_interval = self._map_interval(interval)
        period = "7d" if outputsize == "compact" else "max"

        try:
            df = yf.download(
                symbol,
                period=period,
                interval=yahoo_interval,
                progress=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("Yahoo Finance download failed: %s", exc)
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        df.index.name = "date"
        return df[["open", "high", "low", "close", "volume"]]

    def get_quote(self, symbol: str) -> Optional[float]:
        if yf is None:  # pragma: no cover - dependency missing
            self.logger.error("yfinance is not available")
            return None

        try:
            ticker = yf.Ticker(symbol)
            price = ticker.fast_info.get("last_price")
            return float(price) if price is not None else None
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("Yahoo Finance quote request failed: %s", exc)
            return None
