import logging

from config import LOG_FILE
from strategy import TradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)


def main():
    """Entry point for running the trading bot."""
    bot = TradingBot()
    return bot


if __name__ == "__main__":
    main()
