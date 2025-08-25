"""Lightweight IBKR client for requesting historical and account data.

This module follows Interactive Brokers' TWS API notes and limitations by
making one blocking request at a time.  While minimal, the implementation is
geared for production use with a real IBKR Gateway/TWS connection and can
retrieve both historical price information and basic account summaries.
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
except Exception:  # pragma: no cover - allow compilation without ibapi
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
        reqAccountSummary = _raise
        cancelAccountSummary = _raise
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

    logger.warning("`ibapi` package not available; IBKRClient methods will raise IBAPIUnavailableError")

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
        self._account_summary: dict[str, str] = {}
        self._summary_done = threading.Event()

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

    def request_historical_data(
        self,
        contract: Contract,
        duration: str = "2 M",
        bar_size: str = "1 hour",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
    ) -> pd.DataFrame:
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
            whatToShow=what_to_show,
            useRTH=1 if use_rth else 0,
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

    # ------------------------------------------------------------------
    # Account information
    # ------------------------------------------------------------------
    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str) -> None:  # pragma: no cover
        self._account_summary[tag] = value

    def accountSummaryEnd(self, reqId: int) -> None:  # pragma: no cover
        self._summary_done.set()

    def get_account_summary(self) -> dict:  # pragma: no cover - network dependent
        """Retrieve basic account summary such as cash balance."""

        self._account_summary.clear()
        self._summary_done.clear()

        # Request a couple of common fields. Users can extend this as needed.
        self.reqAccountSummary(1, "All", "TotalCashValue,BuyingPower")
        self._summary_done.wait()
        self.cancelAccountSummary(1)
        return dict(self._account_summary)


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

