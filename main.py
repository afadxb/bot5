"""Executable entry point for running the trading bot locally."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from alerts import send_alert
from config import DEBUG, LOG_FILE
from strategy import TradingBot

def _configure_logging():
    level = logging.DEBUG if DEBUG else logging.INFO
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    root.setLevel(level)
    # Clear existing handlers to avoid duplicate logs in interactive shells
    root.handlers = []
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    file_handler.setFormatter(logging.Formatter(fmt))
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

_configure_logging()

# Silence verbose logs from the underlying IB API client and wrapper
logging.getLogger("ibapi.wrapper").setLevel(logging.WARNING)
logging.getLogger("ibapi.client").setLevel(logging.WARNING)


def _validate_env() -> None:
    csv_path = os.getenv("SP100_CSV", "sp100.csv")
    if not os.path.exists(csv_path):
        logging.error("SP100_CSV file not found at %s", csv_path)
        raise SystemExit(2)
    if os.getenv("ENABLE_TRADING", "0") == "1":
        # Minimal sanity checks
        host = os.getenv("IBKR_HOST")
        port = os.getenv("IBKR_PORT")
        if not host or not port:
            logging.error("ENABLE_TRADING=1 requires IBKR_HOST and IBKR_PORT configured")
            raise SystemExit(2)


def main():
    """Entry point for running the trading bot."""
    try:
        mode = None
        if len(sys.argv) > 1:
            mode = sys.argv[1].lower()
        elif os.getenv("SCAN_ONLY", "0") == "1":
            mode = "scan"

        if mode == "scan":
            send_alert("Trading bot scan starting")
            _validate_env()
            bot = TradingBot()
            results = bot.scan_universe()
            # Also echo a compact console summary
            printable = [
                f"{r['symbol']}: score={r.get('entry_score'):.1f} near={r.get('near_support')}"
                for r in results if r.get("status") == "ok"
            ]
            print("\n".join(printable))
            return results

        send_alert("Trading bot starting")
        _validate_env()
        bot = TradingBot()
        bot.run_hourly()  # Run one full cycle before returning
        return bot  # Wrap in a scheduler for continuous execution
    except Exception as exc:  # pragma: no cover - integration dependent
        logging.exception("Fatal error starting trading bot")
        send_alert(f"Trading bot error: {exc}")
        raise


if __name__ == "__main__":
    main()
