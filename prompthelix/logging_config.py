import logging
import logging.config


def configure_logging(debug: bool = False) -> None:
    """Configure application-wide logging."""
    level = logging.DEBUG if debug else logging.INFO
    config = {
        "version": 1,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(module)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stdout",
            }
        },
        "root": {"level": level, "handlers": ["console"]},
    }
    logging.config.dictConfig(config)

    # Tweak noisy library log levels
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
