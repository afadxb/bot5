"""Lightweight IBKR client for requesting historical data, account details
and placing simple orders.

This module follows Interactive Brokers' TWS API notes and limitations by
making one blocking request at a time. While minimal, the implementation is
geared for production use with a real IBKR Gateway/TWS connection and can
retrieve historical prices, account summaries and open positions, as well as
submit or cancel basic orders.
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
    from ibapi.order import Order as IBOrder
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
        reqPositions = _raise
        cancelPositions = _raise
        placeOrder = _raise
        cancelOrder = _raise
        reqIds = _raise

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

    @dataclass
    class IBOrder:  # type: ignore
        """Simplified stand-in for :class:`ibapi.order.Order`."""

        action: str = "BUY"
        orderType: str = "MKT"
        totalQuantity: int = 0
        lmtPrice: float = 0.0
        auxPrice: float = 0.0
        trailingPercent: float = 0.0
        trailStopPrice: float = 0.0
        # Bracket/OCO fields
        transmit: bool = True
        parentId: int = 0
        ocaGroup: str = ""
        ocaType: int = 1

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
        self._next_order_id: int | None = None
        self._next_id_ready = threading.Event()
        self._positions: List[dict] = []
        self._positions_done = threading.Event()


    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------
    def connect_and_run(self) -> None:
        """Connect to TWS/Gateway and start the message loop in a thread."""
        self.connect(self.host, self.port, self.client_id)
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        # Request the next valid order id for order placement
        self.reqIds(-1)

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
        # Wait for completion with a timeout to avoid deadlocks
        finished = self._finished.wait(timeout=30)
        if not finished:
            self.logger.warning("Historical data request timed out: %s %s", duration, bar_size)

        df = pd.DataFrame(self._historical, columns=["date", "open", "high", "low", "close", "volume"])
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Order handling
    # ------------------------------------------------------------------
    def nextValidId(self, orderId: int) -> None:  # pragma: no cover
        """Callback providing the next valid order id."""
        self._next_order_id = orderId
        self._next_id_ready.set()

    def place_order(self, contract: Contract, order: IBOrder) -> int:  # pragma: no cover
        """Submit an order and return the IBKR order id."""
        if self._next_order_id is None:
            self.reqIds(-1)
            self._next_id_ready.wait(timeout=10)
        oid = self._next_order_id
        super().placeOrder(oid, contract, order)
        self._next_order_id += 1
        return oid

    def cancel_order(self, order_id: int) -> None:  # pragma: no cover
        """Cancel an existing order."""
        super().cancelOrder(order_id)

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
        self.reqAccountSummary(1, "All", "TotalCashValue,BuyingPower,NetLiquidation")
        self._summary_done.wait()
        self.cancelAccountSummary(1)
        return dict(self._account_summary)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------
    def position(self, account: str, contract: Contract, position: float, avgCost: float) -> None:  # pragma: no cover
        self._positions.append({
            "account": account,
            "contract": contract.symbol,
            "position": position,
            "avg_cost": avgCost,
        })

    def positionEnd(self) -> None:  # pragma: no cover
        self._positions_done.set()

    def get_positions(self) -> List[dict]:  # pragma: no cover - network dependent
        """Return open positions for the account."""
        self._positions.clear()
        self._positions_done.clear()
        self.reqPositions()
        self._positions_done.wait()
        self.cancelPositions()
        return list(self._positions)

def stock_contract(symbol: str) -> Contract:
    """Create a SMART-routed US stock contract."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "STK"
    contract.exchange = "SMART"
    contract.currency = "USD"
    return contract


def index_contract(symbol: str, exchange: str = "CBOE") -> Contract:
    """Create an index contract such as the CBOE VIX."""
    contract = Contract()
    contract.symbol = symbol
    contract.secType = "IND"
    contract.exchange = exchange
    contract.currency = "USD"
    return contract


def simple_sma(data: pd.Series, window: int = 20) -> pd.Series:
    """Convenience SMA calculation mirroring the IBKR example."""
    return data.rolling(window=window).mean()

