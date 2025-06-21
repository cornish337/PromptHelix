import logging
import os
from typing import Optional
from prompthelix.config import LOGGING_CONFIG


_configured = False


def setup_logging(debug: bool | None = None, log_file: Optional[str] = None) -> None:
    """Configure the root logger once using ``LOGGING_CONFIG``.

    Parameters
    ----------
    debug : bool | None
        If ``True`` the logging level is set to ``DEBUG``. If ``None``
        the ``PROMPTHELIX_DEBUG`` environment variable is checked.
    log_file : str | None
        Optional path to a log file. If provided, logs are written to this
        file in addition to the console.
    """
    global _configured
    if _configured:
        return

    if debug is None:
        debug_env = os.getenv("PROMPTHELIX_DEBUG", "").lower()
        debug = debug_env in {"1", "true", "yes"}

    level_name = "DEBUG" if debug else LOGGING_CONFIG.get("level", "INFO")
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOGGING_CONFIG.get("format"), filename=log_file)
    _configured = True

