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
IBKR_PORT = int(os.getenv('IBKR_PORT', '7497'))
IBKR_CLIENT_ID = int(os.getenv('IBKR_CLIENT_ID', '1'))
