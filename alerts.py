"""Utility helpers for sending notifications to external services."""

import logging

import requests

from config import PUSHOVER_USER, PUSHOVER_TOKEN


def send_alert(message: str, title: str = "Trading Bot") -> bool:
    """Send a notification via the Pushover API.

    Returns ``True`` if the alert was successfully dispatched, ``False``
    otherwise. If the required environment variables are not configured the
    function exits silently and returns ``False``.
    """
    if not PUSHOVER_USER or not PUSHOVER_TOKEN:
        logging.debug("Pushover credentials not set; alert not sent")
        return False

    payload = {
        "token": PUSHOVER_TOKEN,
        "user": PUSHOVER_USER,
        "message": message,
        "title": title,
    }
    try:
        response = requests.post(
            "https://api.pushover.net/1/messages.json", data=payload, timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as exc:  # pragma: no cover - network dependent
        logging.error("Failed to send Pushover alert: %s", exc)
        return False
