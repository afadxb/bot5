"""Core domain models and enumerations used by the trading bot.

The module defines lightweight :class:`dataclasses.dataclass` and
``Enum`` types that describe orders, positions and market regimes.  These
structures are intentionally simple and free of behaviour so they can be
shared between modules without creating circular dependencies.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from config import EASTERN_TZ


class MarketRegime(Enum):
    """High level classification of prevailing market conditions."""

    TRENDING = "TR"
    RANGING = "RG"
    RISK_OFF = "RO"


class OrderStatus(Enum):
    """Lifecycle states for submitted orders."""

    PENDING_NEW = "PENDING_NEW"
    LIVE = "LIVE"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class OrderType(Enum):
    """Supported order types."""

    LMT = "LMT"
    STP = "STP"
    TRAIL = "TRAIL"


class PositionStatus(Enum):
    """State machine representing a trade's lifecycle."""

    INIT = "INIT"
    ARMED = "ARMED"
    FILLED = "FILLED"
    MANAGED = "MANAGED"
    SCALE_OUT_25 = "SCALE_OUT_25"
    SCALE_OUT_50 = "SCALE_OUT_50"
    SCALE_OUT_75 = "SCALE_OUT_75"
    EXITED = "EXITED"


@dataclass
class Order:
    """Representation of an order submitted to a broker."""

    order_id: str
    symbol: str
    order_type: OrderType
    side: str
    quantity: int
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_amount: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING_NEW
    filled_quantity: int = 0
    avg_fill_price: float = 0.0
    parent_order_id: Optional[str] = None
    child_orders: List[str] = None
    timestamp: datetime = None

    def __post_init__(self) -> None:
        """Ensure optional fields have sensible defaults."""

        if self.child_orders is None:
            self.child_orders = []
        if self.timestamp is None:
            self.timestamp = datetime.now(EASTERN_TZ)


@dataclass
class Position:
    """Represents an open or closed trading position."""

    position_id: str
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: int
    status: PositionStatus
    current_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_price: float = 0.0
    exit_price: float = 0.0
    exit_time: datetime = None
    score_components: Dict = None
    risk_per_share: float = 0.0

    def __post_init__(self) -> None:
        """Populate mutable default fields after initialization."""

        if self.score_components is None:
            self.score_components = {}
