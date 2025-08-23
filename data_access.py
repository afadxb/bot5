import json
import logging
import sqlite3
from datetime import datetime
from typing import Dict

from models import Order, Position, OrderStatus, OrderType, PositionStatus
from config import DB_PATH


class DataManager:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Position table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    position_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    entry_time DATETIME,
                    entry_price REAL,
                    quantity INTEGER,
                    status TEXT,
                    current_pnl REAL,
                    realized_pnl REAL,
                    stop_price REAL,
                    exit_price REAL,
                    exit_time DATETIME,
                    score_components TEXT,
                    risk_per_share REAL
                )
            ''')

            # Order table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    symbol TEXT,
                    order_type TEXT,
                    side TEXT,
                    quantity INTEGER,
                    limit_price REAL,
                    stop_price REAL,
                    trail_amount REAL,
                    status TEXT,
                    filled_quantity INTEGER,
                    avg_fill_price REAL,
                    parent_order_id TEXT,
                    child_orders TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY (position_id) REFERENCES positions (position_id)
                )
            ''')

            # Score snapshot table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS score_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    regime TEXT,
                    entry_total REAL,
                    entry_components TEXT,
                    exit_total REAL,
                    exit_components TEXT,
                    sentiment TEXT
                )
            ''')

            # Signal log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    action TEXT,
                    reason_codes TEXT,
                    details TEXT
                )
            ''')

            # Regime snapshot table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS regime_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    spy_metrics TEXT,
                    vix REAL,
                    regime_label TEXT
                )
            ''')

            conn.commit()

    def save_position(self, position: Position):
        """Save position to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO positions
                (position_id, symbol, entry_time, entry_price, quantity, status,
                 current_pnl, realized_pnl, stop_price, exit_price, exit_time,
                 score_components, risk_per_share)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                position.position_id,
                position.symbol,
                position.entry_time,
                position.entry_price,
                position.quantity,
                position.status.value,
                position.current_pnl,
                position.realized_pnl,
                position.stop_price,
                position.exit_price,
                position.exit_time,
                json.dumps(position.score_components),
                position.risk_per_share
            ))
            conn.commit()

    def save_order(self, order: Order):
        """Save order to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO orders
                (order_id, position_id, symbol, order_type, side, quantity,
                 limit_price, stop_price, trail_amount, status, filled_quantity,
                 avg_fill_price, parent_order_id, child_orders, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order.order_id,
                order.parent_order_id,
                order.symbol,
                order.order_type.value,
                order.side,
                order.quantity,
                order.limit_price,
                order.stop_price,
                order.trail_amount,
                order.status.value,
                order.filled_quantity,
                order.avg_fill_price,
                order.parent_order_id,
                json.dumps(order.child_orders),
                order.timestamp
            ))
            conn.commit()

    def load_positions(self) -> Dict[str, Position]:
        """Load all positions from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM positions")
            positions = {}
            for row in cursor.fetchall():
                positions[row["position_id"]] = Position(
                    position_id=row["position_id"],
                    symbol=row["symbol"],
                    entry_time=datetime.fromisoformat(row["entry_time"]) if row["entry_time"] else None,
                    entry_price=row["entry_price"],
                    quantity=row["quantity"],
                    status=PositionStatus(row["status"]),
                    current_pnl=row["current_pnl"],
                    realized_pnl=row["realized_pnl"],
                    stop_price=row["stop_price"],
                    exit_price=row["exit_price"],
                    exit_time=datetime.fromisoformat(row["exit_time"]) if row["exit_time"] else None,
                    score_components=json.loads(row["score_components"] or "{}"),
                    risk_per_share=row["risk_per_share"],
                )
        return positions

    def load_orders(self) -> Dict[str, Order]:
        """Load all orders from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM orders")
            orders = {}
            for row in cursor.fetchall():
                orders[row["order_id"]] = Order(
                    order_id=row["order_id"],
                    symbol=row["symbol"],
                    order_type=OrderType(row["order_type"]),
                    side=row["side"],
                    quantity=row["quantity"],
                    limit_price=row["limit_price"],
                    stop_price=row["stop_price"],
                    trail_amount=row["trail_amount"],
                    status=OrderStatus(row["status"]),
                    filled_quantity=row["filled_quantity"],
                    avg_fill_price=row["avg_fill_price"],
                    parent_order_id=row["parent_order_id"],
                    child_orders=json.loads(row["child_orders"] or "[]"),
                    timestamp=datetime.fromisoformat(row["timestamp"]) if row["timestamp"] else None,
                )
        return orders
