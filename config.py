"""Centralized configuration and environment settings for the trading bot.

This module loads environment variables from a ``.env`` file and exposes
constants used throughout the system such as database paths, logging
locations and market session times.  Only lightweight logic lives here so the
rest of the codebase can import configuration values without side effects.
"""

import os
from datetime import time
import pytz
from dotenv import load_dotenv

load_dotenv()

# Timezone and market schedule
EASTERN_TZ = pytz.timezone('America/New_York')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
FOUR_HOUR_TIMES = [time(9, 30), time(13, 30)]

# Environment configurable settings
DB_PATH = os.getenv('DB_PATH', 'trading_bot.db')
LOG_FILE = os.getenv('LOG_FILE', 'trading_bot.log')
IBKR_HOST = os.getenv('IBKR_HOST', '127.0.0.1')
IBKR_PORT = int(os.getenv('IBKR_PORT', '7496'))
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
OPERATING_SYSTEM = os.getenv('OPERATING_SYSTEM', 'windows')
PYTHON_VERSION = os.getenv('PYTHON_VERSION', '3.12')
PUSHOVER_USER = os.getenv('PUSHOVER_USER')
PUSHOVER_TOKEN = os.getenv('PUSHOVER_TOKEN')
DEBUG = os.getenv('DEBUG', '0') == '1'

# Entry condition tuning
ENTRY_SCORE_THRESHOLD = float(os.getenv('ENTRY_SCORE_THRESHOLD', '70'))
SUPPORT_DISTANCE_ATR = float(os.getenv('SUPPORT_DISTANCE_ATR', '0.5'))
SUPPORT_DISTANCE_PCT = float(os.getenv('SUPPORT_DISTANCE_PCT', '0.5'))

# Sentiment gating thresholds
FG_HARD_BLOCK_THRESHOLD = float(os.getenv('FG_HARD_BLOCK_THRESHOLD', '25'))
FG_PENALTY_THRESHOLD = float(os.getenv('FG_PENALTY_THRESHOLD', '45'))
FG_OVERHEAT_THRESHOLD = float(os.getenv('FG_OVERHEAT_THRESHOLD', '80'))
FG_PENALTY = float(os.getenv('FG_PENALTY', '5'))
FG_OVERHEAT_PENALTY = float(os.getenv('FG_OVERHEAT_PENALTY', '5'))
NEWS_POSITIVE_THRESHOLD = float(os.getenv('NEWS_POSITIVE_THRESHOLD', '0.7'))
NEWS_NEGATIVE_THRESHOLD = float(os.getenv('NEWS_NEGATIVE_THRESHOLD', '-0.7'))
NEWS_POSITIVE_BONUS = float(os.getenv('NEWS_POSITIVE_BONUS', '3'))
NEWS_NEGATIVE_PENALTY = float(os.getenv('NEWS_NEGATIVE_PENALTY', '5'))
RISK_OFF_MIN_SCORE = float(os.getenv('RISK_OFF_MIN_SCORE', '85'))
RISK_OFF_NEWS_THRESHOLD = float(os.getenv('RISK_OFF_NEWS_THRESHOLD', '0'))

# Risk management
ACCOUNT_EQUITY = float(os.getenv('ACCOUNT_EQUITY', '100000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.01'))
MAX_POSITION_PCT = float(os.getenv('MAX_POSITION_PCT', '0.1'))
