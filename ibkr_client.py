"""Lightweight IBKR client for requesting historical data and indicators.

This module follows Interactive Brokers' TWS API notes and limitations by
making one blocking historical data request at a time. It is inspired by the
official "Using technical indicators with TWS API" example.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from ibapi.client import EClient
    from ibapi.wrapper import EWrapper
    from ibapi.contract import Contract
    from ibapi.common import BarData
except ModuleNotFoundError:  # pragma: no cover - allow compilation without ibapi
    class IBAPIUnavailableError(RuntimeError):
        """Raised when the optional `ibapi` package is required but missing."""

        def __init__(self) -> None:
            super().__init__(
                "Interactive Brokers 'ibapi' package is required. Install it to enable IBKR functionality."
            )

    class EClient:  # type: ignore
        """Lightweight stand-in for the real IB API client."""

        def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - empty stub
            """Accept arbitrary arguments to mirror the real client's signature."""

        def _raise(self, *args, **kwargs) -> None:
            raise IBAPIUnavailableError()

        disconnect = _raise
        connect = _raise
        reqHistoricalData = _raise
        run = _raise

    class EWrapper:  # type: ignore
        """Stub wrapper class used when `ibapi` is not available."""

        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover - stub
            pass

    @dataclass
    class Contract:  # type: ignore
        """Minimal contract representation for environments without `ibapi`."""

        symbol: str = ""
        secType: str = ""
        exchange: str = ""
        currency: str = ""

    @dataclass
    class BarData:  # type: ignore
        """Simple container replicating fields of IBKR bar data."""

        date: str = ""
        open: float = 0.0
        high: float = 0.0
        low: float = 0.0
        close: float = 0.0
        volume: float = 0.0


from config import IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID


class IBKRClient(EWrapper, EClient):
    """Minimal client focusing on historical data retrieval.

    The client issues a single request at a time and waits for the
    :func:`historicalDataEnd` callback before returning, keeping within the
    IBKR pacing limits.
    """

    def __init__(self, host: str = IBKR_HOST, port: int = IBKR_PORT, client_id: int = IBKR_CLIENT_ID):
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.logger = logging.getLogger(__name__)
        self.host = host
        self.port = port
        self.client_id = client_id
        self._historical: List[List] = []
        self._finished = threading.Event()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect_and_run(self) -> None:
        """Connect to TWS/Gateway and start the message loop in a thread."""
        self.connect(self.host, self.port, self.client_id)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def disconnect(self) -> None:  # pragma: no cover - network side effect
        super().disconnect()
        self._finished.set()

    # ------------------------------------------------------------------
    # Historical data handling
    # ------------------------------------------------------------------
    def historicalData(self, reqId: int, bar: BarData) -> None:  # pragma: no cover
        self._historical.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId: int, start: str, end: str) -> None:  # pragma: no cover
        self._finished.set()

    def request_historical_data(self, contract: Contract, duration: str = "2 M", bar_size: str = "1 hour") -> pd.DataFrame:
        """Request historical bars and return them as a :class:`pandas.DataFrame`."""

        self._historical.clear()
        self._finished.clear()

        # One request at a time to respect IBKR pacing limits (50 msgs/s)
        self.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=1,
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[],
        )
        self._finished.wait()

        df = pd.DataFrame(self._historical, columns=["date", "open", "high", "low", "close", "volume"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df


def stock_contract(symbol: str) -> Contract:
    """Create a SMART-routed US stock contract."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def simple_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """Convenience SMA calculation mirroring the IBKR example."""
    return data.rolling(window=window).mean()

