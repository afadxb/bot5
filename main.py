import logging

from alerts import send_alert
from config import DEBUG, LOG_FILE
from strategy import TradingBot

logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)


def main():
    """Entry point for running the trading bot."""
    try:
        send_alert("Trading bot starting")
        bot = TradingBot()
        return bot
    except Exception as exc:  # pragma: no cover - integration dependent
        logging.exception("Fatal error starting trading bot")
        send_alert(f"Trading bot error: {exc}")
        raise


if __name__ == "__main__":
    main()
