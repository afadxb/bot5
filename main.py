"""Executable entry point for running the trading bot locally."""

import logging
import os
import sys

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

# Silence verbose logs from the underlying IB API client and wrapper
logging.getLogger("ibapi.wrapper").setLevel(logging.WARNING)
logging.getLogger("ibapi.client").setLevel(logging.WARNING)


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
        bot = TradingBot()
        bot.run_hourly()  # Run one full cycle before returning
        return bot  # Wrap in a scheduler for continuous execution
    except Exception as exc:  # pragma: no cover - integration dependent
        logging.exception("Fatal error starting trading bot")
        send_alert(f"Trading bot error: {exc}")
        raise


if __name__ == "__main__":
    main()
