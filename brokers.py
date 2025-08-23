"""Simple broker API abstractions."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from models import Order
from ibkr_client import IBKRClient


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
        self.logger.info("Submitting %s to IBKR", order.order_id)
        # A real implementation would call ``self.client.placeOrder`` here.
        return order.order_id

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - network dependent
        self.logger.info("Cancelling order %s", order_id)
        # A real implementation would call ``self.client.cancelOrder`` here.
        return True
