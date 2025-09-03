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
        # Map our strategy order ids and groups (position ids) to IB ids
        self._ib_ids: dict[str, int] = {}
        self._group_parent: dict[str, int] = {}

    def place_order(self, order: Order) -> str:  # pragma: no cover - network dependent
        contract = stock_contract(order.symbol)
        ib_order = IBOrder()
        ib_order.action = order.side
        # Explicitly disable unsupported broker attributes.
        # Some IBKR API versions default ``eTradeOnly`` to ``True``,
        # which can trigger error 10268 ("'EtradeOnly' order attribute is not supported")
        # when the field is sent to the server.  Setting it to ``False`` prevents
        # the attribute from being transmitted.
        if hasattr(ib_order, "eTradeOnly"):
            ib_order.eTradeOnly = False

        # Ensure we don't submit fractional share sizes which older API
        # versions cannot handle.  Trim to an integer and log the adjustment
        # so the caller is aware of the change.
        qty = int(order.quantity)
        if qty != order.quantity:
            self.logger.warning(
                "Rounded fractional quantity %s to %s for %s",
                order.quantity,
                qty,
                order.order_id,
            )
        order.quantity = qty
        ib_order.totalQuantity = qty
        if order.order_type == OrderType.LMT:
            ib_order.orderType = "LMT"
            ib_order.lmtPrice = order.limit_price or 0.0
        elif order.order_type == OrderType.STP:
            ib_order.orderType = "STP"
            ib_order.auxPrice = order.stop_price or 0.0
        elif order.order_type == OrderType.TRAIL:
            ib_order.orderType = "TRAIL"
            ib_order.trailingPercent = order.trail_amount or 0.0
        # Bracket/OCO setup: if this order belongs to a position group, wire parent/oca
        group = order.parent_order_id
        is_parent = order.order_type == OrderType.LMT and order.side.upper() == "BUY"
        if group:
            if is_parent:
                # Parent entry; record mapping after placement
                ib_order.transmit = True
            else:
                # Child legs reference parent's IB id and form an OCA group
                parent_ib = self._group_parent.get(group)
                if parent_ib:
                    ib_order.parentId = parent_ib
                ib_order.ocaGroup = str(group)
                ib_order.ocaType = 1  # Cancel remaining when one fills
                ib_order.transmit = True

        self.logger.info("Submitting %s to IBKR", order.order_id)
        ib_id = self.client.place_order(contract, ib_order)
        self._ib_ids[order.order_id] = ib_id
        if group and is_parent:
            self._group_parent[group] = ib_id
        return str(ib_id)

    def cancel_order(self, order_id: str) -> bool:  # pragma: no cover - network dependent
        self.logger.info("Cancelling order %s", order_id)
        try:
            ib_id = self._ib_ids.get(order_id)
            if ib_id is None:
                # Fallback: attempt to parse as int
                ib_id = int(order_id)
            self.client.cancel_order(int(ib_id))
            return True
        except Exception:
            return False
