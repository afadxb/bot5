import logging
from datetime import datetime
from typing import Dict, List

from models import Order, OrderType, OrderStatus
from brokers import BrokerAPI

class OrderManager:
    def __init__(self, data_manager, broker: BrokerAPI | None = None):
        self.orders: Dict[str, Order] = {}
        self.position_orders: Dict[str, List[str]] = {}
        self.data_manager = data_manager
        self.broker = broker
        self.logger = logging.getLogger(f"{__name__}.OrderManager")
    
    def place_bracket_order(self, symbol, quantity, entry_price, stop_loss, take_profit1, take_profit2=None):
        """
        Place a bracket order (entry + OCO stop loss/take profit)
        """
        # Generate unique IDs
        entry_id = f"{symbol}_{datetime.now().timestamp()}_ENTRY"
        stop_id = f"{symbol}_{datetime.now().timestamp()}_STOP"
        tp1_id = f"{symbol}_{datetime.now().timestamp()}_TP1"
        
        # Create entry order
        entry_order = Order(
            order_id=entry_id,
            symbol=symbol,
            order_type=OrderType.LMT,
            side="BUY",
            quantity=quantity,
            limit_price=entry_price,
            status=OrderStatus.PENDING_NEW
        )
        
        # Create stop loss order
        stop_order = Order(
            order_id=stop_id,
            symbol=symbol,
            order_type=OrderType.STP,
            side="SELL",
            quantity=quantity,
            stop_price=stop_loss,
            parent_order_id=entry_id,
            status=OrderStatus.PENDING_NEW
        )
        
        # Create take profit 1 order
        tp1_order = Order(
            order_id=tp1_id,
            symbol=symbol,
            order_type=OrderType.LMT,
            side="SELL",
            quantity=quantity // 2 if take_profit2 else quantity,
            limit_price=take_profit1,
            parent_order_id=entry_id,
            status=OrderStatus.PENDING_NEW
        )
        
        # Add child orders to entry order
        entry_order.child_orders = [stop_id, tp1_id]
        
        # Store orders
        self.orders[entry_id] = entry_order
        self.orders[stop_id] = stop_order
        self.orders[tp1_id] = tp1_order
        self.position_orders.setdefault(entry_id, [entry_id]).extend([stop_id, tp1_id])
        
        # Create TP2 if specified
        if take_profit2:
            tp2_id = f"{symbol}_{datetime.now().timestamp()}_TP2"
            tp2_order = Order(
                order_id=tp2_id,
                symbol=symbol,
                order_type=OrderType.LMT,
                side="SELL",
                quantity=quantity - (quantity // 2),
                limit_price=take_profit2,
                parent_order_id=entry_id,
                status=OrderStatus.PENDING_NEW
            )
            entry_order.child_orders.append(tp2_id)
            self.orders[tp2_id] = tp2_order
            self.position_orders[entry_id].append(tp2_id)
        
        # Save orders to database
        self.data_manager.save_order(entry_order)
        self.data_manager.save_order(stop_order)
        self.data_manager.save_order(tp1_order)
        if take_profit2:
            self.data_manager.save_order(tp2_order)

        if self.broker:
            self.broker.place_order(entry_order)
            self.broker.place_order(stop_order)
            self.broker.place_order(tp1_order)
            if take_profit2:
                self.broker.place_order(tp2_order)
        
        self.logger.info(f"Bracket order placed for {symbol}: entry@{entry_price}, stop@{stop_loss}")
        return entry_id
    
    def promote_to_trailing_stop(self, position_id, trail_amount, trail_type="ATR"):
        """
        Replace a take profit order with a trailing stop
        """
        position_orders = self.position_orders.get(position_id, [])
        
        # Find the take profit order to cancel
        tp_order_id = None
        for order_id in position_orders:
            order = self.orders.get(order_id)
            if order and order.order_type == OrderType.LMT and order.side == "SELL":
                tp_order_id = order_id
                break
        
        if not tp_order_id:
            self.logger.warning(f"No take profit order found for position {position_id}")
            return False
        
        # Cancel the TP order
        if self.cancel_order(tp_order_id):
            # Create trailing stop order
            trail_id = f"{position_id}_TRAIL_{datetime.now().timestamp()}"
            original_order = self.orders[position_orders[0]]  # Get the entry order
            
            trail_order = Order(
                order_id=trail_id,
                symbol=original_order.symbol,
                order_type=OrderType.TRAIL,
                side="SELL",
                quantity=original_order.quantity - original_order.filled_quantity,
                trail_amount=trail_amount,
                parent_order_id=position_id,
                status=OrderStatus.PENDING_NEW
            )
            
            self.orders[trail_id] = trail_order
            self.position_orders[position_id].append(trail_id)
            self.data_manager.save_order(trail_order)
            if self.broker:
                self.broker.place_order(trail_order)
            
            self.logger.info(f"Trailing stop order placed for {original_order.symbol}")
            return True
        
        return False
    
    def cancel_order(self, order_id):
        """Cancel an order"""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status not in [OrderStatus.LIVE, OrderStatus.PENDING_NEW]:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False
        
        try:
            if self.broker:
                self.broker.cancel_order(order_id)
            order.status = OrderStatus.PENDING_CANCEL
            self.data_manager.save_order(order)
            self.logger.info(f"Cancel requested for order {order_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def adjust_stop_loss(self, position_id: str, new_stop: float, quantity: int | None = None) -> bool:
        """Ratchet the existing stop loss to *new_stop*.

        The current stop order is cancelled and replaced with a new one at the
        requested price.  Returns ``True`` when a stop order was updated.
        """
        position_orders = self.position_orders.get(position_id, [])

        # Locate existing stop order
        stop_order_id = None
        for oid in position_orders:
            order = self.orders.get(oid)
            if order and order.order_type == OrderType.STP:
                stop_order_id = oid
                break

        if not stop_order_id:
            self.logger.warning(f"No stop order found for position {position_id}")
            return False

        # Cancel the old stop
        if not self.cancel_order(stop_order_id):
            return False
        position_orders.remove(stop_order_id)
        self.position_orders[position_id] = position_orders

        original_order = self.orders[position_orders[0]]  # entry order
        qty = quantity if quantity is not None else (
            original_order.quantity - original_order.filled_quantity
        )
        new_id = f"{original_order.symbol}_{datetime.now().timestamp()}_STOP"
        stop_order = Order(
            order_id=new_id,
            symbol=original_order.symbol,
            order_type=OrderType.STP,
            side="SELL",
            quantity=qty,
            stop_price=new_stop,
            parent_order_id=position_id,
            status=OrderStatus.PENDING_NEW,
        )

        self.orders[new_id] = stop_order
        self.position_orders[position_id].append(new_id)
        self.data_manager.save_order(stop_order)
        if self.broker:
            self.broker.place_order(stop_order)

        self.logger.info(
            f"Stop loss adjusted for {original_order.symbol} to {new_stop}"
        )
        return True

    def upgrade_trailing_stop(self, position_id: str, new_trail: float) -> bool:
        """Upgrade an existing trailing stop to ``new_trail`` amount."""
        position_orders = self.position_orders.get(position_id, [])
        for oid in position_orders:
            order = self.orders.get(oid)
            if order and order.order_type == OrderType.TRAIL:
                order.trail_amount = new_trail
                self.data_manager.save_order(order)
                if self.broker:
                    self.broker.place_order(order)
                self.logger.info(
                    f"Trailing stop for {order.symbol} upgraded to {new_trail}"
                )
                return True

        self.logger.warning(
            f"No trailing stop order found to upgrade for position {position_id}"
        )
        return False

