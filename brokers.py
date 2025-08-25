"""Simple broker API abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from models import Order, OrderType
from ibkr_client import IBKRClient, IBOrder, stock_contract


class BrokerAPI(ABC):
    """Abstract broker interface used by :class:`OrderManager`."""

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """Submit *order* to the broker and return the broker order id."""

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order and return ``True`` on success."""


class IBKRBroker(BrokerAPI):
    """Tiny wrapper integrating :class:`IBKRClient` with the order system.

    The implementation only logs the requests.  Integrating the full IBKR
    order management API would require additional infrastructure that is
    outside the scope of this repository, but this interface makes it easy
    to plug in a real implementation later.
    """

    def __init__(self, client: IBKRClient) -> None:
        self.client = client
        self.logger = logging.getLogger(__name__)

    def place_order(self, order: Order) -> str:  # pragma: no cover - network dependent
        contract = stock_contract(order.symbol)
        ib_order = IBOrder()
        ib_order.action = order.side
        ib_order.totalQuantity = order.quantity
        if order.order_type == OrderType.LMT:
            ib_order.orderType = "LMT"
            ib_order.lmtPrice = order.limit_price or 0.0
        elif order.order_type == OrderType.STP:
            ib_order.orderType = "STP"
            ib_order.auxPrice = order.stop_price or 0.0
        elif order.order_type == OrderType.TRAIL:
            ib_order.orderType = "TRAIL"
            ib_order.trailingPercent = order.trail_amount or 0.0
        self.logger.info("Submitting %s to IBKR", order.order_id)
        order_id = self.client.place_order(contract, ib_order)
        return str(order_id)

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - network dependent
        self.logger.info("Cancelling order %s", order_id)
        try:
            self.client.cancel_order(int(order_id))
            return True
        except Exception:
            return False
