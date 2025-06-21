import logging
import os
import sys
import json
from typing import Optional
from prompthelix.config import LOGGING_CONFIG

_configured = False


class JsonFormatter(logging.Formatter):
    """Simple JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "level": record.levelname,
            "message": record.getMessage(),
        }
        return json.dumps(log_record)


def setup_logging(
    debug: Optional[bool] = None,
    log_file: Optional[str] = None,
    json_format: bool = False
) -> logging.Logger:
    """
    Configure the root logger once using LOGGING_CONFIG.

    Parameters
    ----------
    debug : bool | None
        If True, force DEBUG level; if False, force INFO; if None, read
        PROMPTHELIX_DEBUG env var. Env var values "1"/"true"/"yes" enable debug.
    log_file : str | None
        If provided, also writes logs to this file (appending).
    json_format : bool
        If True, formats logs as JSON, otherwise uses the configured format string.
    """
    global _configured
    root = logging.getLogger()
    if _configured:
        return root

    # Determine debug setting
    if debug is None:
        debug_env = os.getenv("PROMPTHELIX_DEBUG", "").lower()
        debug = debug_env in {"1", "true", "yes"}

    # Determine level
    default_level = LOGGING_CONFIG.get("level", "INFO")
    level_name = "DEBUG" if debug else default_level
    level = getattr(logging, level_name.upper(), logging.INFO)

    # Determine formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        fmt = LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        datefmt = LOGGING_CONFIG.get("datefmt", None)
        formatter = logging.Formatter(fmt, datefmt)

    # Clear existing handlers
    if root.handlers:
        root.handlers.clear()

    root.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True
    return root
