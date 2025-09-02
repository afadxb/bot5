"""External market data provider interfaces."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple

import pandas as pd

from ibkr_client import IBKRClient, stock_contract, index_contract


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
            Either ``"compact"`` or ``"full"``. Providers may interpret these
            values differently; the :class:`IBKRDataProvider` maps "compact" to
            a 1-day lookback and "full" to a 1-month lookback.
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


class IBKRDataProvider(DataProvider):
    """Fetch historical and real-time data from Interactive Brokers."""

    def __init__(self, client: IBKRClient) -> None:
        self.client = client
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _contract_for_symbol(symbol: str):
        """Return an appropriate IBKR contract for *symbol*.

        VIX is traded as an index on CBOE rather than a stock.  Handling it
        here prevents ``No security definition`` errors when requesting
        historical data or quotes.
        """
        normalized = symbol.lstrip("^").upper()
        if normalized == "VIX":
            return index_contract("VIX")
        return stock_contract(symbol)

    @staticmethod
    def _map_interval(interval: str) -> str:
        mapping = {
            "1min": "1 min",
            "5min": "5 mins",
            "15min": "15 mins",
            "30min": "30 mins",
            "60min": "1 hour",
        }
        return mapping.get(interval, "1 min")

    @staticmethod
    def _map_duration(outputsize: str) -> str:
        return "1 M" if outputsize == "full" else "1 D"

    def get_historical_data(
        self, symbol: str, interval: str, outputsize: str = "compact"
    ) -> pd.DataFrame:
        contract = self._contract_for_symbol(symbol)
        bar_size = self._map_interval(interval)
        duration = self._map_duration(outputsize)

        try:
            return self.client.request_historical_data(
                contract, duration=duration, bar_size=bar_size
            )
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("IBKR historical data request failed: %s", exc)
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[float]:
        contract = self._contract_for_symbol(symbol)
        try:
            df = self.client.request_historical_data(
                contract, duration="1 D", bar_size="1 min"
            )
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("IBKR quote request failed: %s", exc)
            return None

        if df.empty:
            return None
        return float(df["close"].iloc[-1])
