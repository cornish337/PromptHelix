import json  # For custom JSON formatter
import logging
import logging.config
import os
import sys
from datetime import datetime  # For custom JSON formatter

from prompthelix.config import (
    LOG_DIR,
    LOG_FILE_NAME,
    LOG_FORMAT,
    LOG_LEVEL,
    LOGGING_CONFIG,
    ensure_directories_exist,
)

# Get a logger for this module itself
logger = logging.getLogger(__name__)


# Custom JSON Formatter
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "name": record.name,
            # "module": record.module, # Often same as name for specific loggers
            # "funcName": record.funcName,
            # "lineno": record.lineno,
        }
        if isinstance(record.msg, dict):
            log_record.update(record.msg)
        else:
            log_record["message"] = record.getMessage()

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record)


def setup_logging():
    """
    Configures logging for the application.
    Uses settings from prompthelix.config.
    """
    # Ensure log directory exists before trying to write log files
    # This call was in config.py, but it's better to call it explicitly before logging setup
    # if file logging is enabled. ensure_directories_exist() itself uses print statements,
    # which is fine before logging is fully set up.
    ensure_directories_exist()

    # Determine if file logging is enabled
    enable_file_logging = bool(LOG_FILE_NAME and LOG_DIR)

    # Create a basic configuration first, which can be customized
    # This uses the new LOGGING_CONFIG structure from config.py
    config_to_use = LOGGING_CONFIG.copy()  # Make a copy to modify

    # Update levels and formatters from individual settings as primary source
    # for simplicity, overriding parts of LOGGING_CONFIG if they differ.
    config_to_use["root"]["level"] = LOG_LEVEL
    if (
        "console" in config_to_use["handlers"]
    ):  # Ensure console handler exists before configuring
        config_to_use["handlers"]["console"]["level"] = LOG_LEVEL
    if "standard" in config_to_use["formatters"]:  # Ensure standard formatter exists
        config_to_use["formatters"]["standard"]["format"] = LOG_FORMAT

    # Add JSON formatter to the config
    config_to_use["formatters"]["json"] = {
        "()": JSONFormatter,  # Use the custom class
    }

    # Update levels for specific noisy loggers if defined in LOGGING_CONFIG['loggers']
    # but ensure their handlers also use the standard formatter and appropriate level
    for logger_name, logger_cfg in config_to_use.get("loggers", {}).items():
        if "handlers" in logger_cfg:  # Ensure handlers are defined
            for handler_name in logger_cfg["handlers"]:
                if handler_name in config_to_use["handlers"]:  # e.g., 'console'
                    config_to_use["handlers"][handler_name]["formatter"] = "standard"
        # The level for these specific loggers is already set in LOGGING_CONFIG

    if enable_file_logging:
        log_file_path = os.path.join(LOG_DIR, LOG_FILE_NAME)

        # Add file handler to the config
        config_to_use["handlers"]["file"] = {
            "level": LOG_LEVEL,
            "formatter": "standard",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": log_file_path,
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 2,
            "encoding": "utf-8",
        }
        # Add file handler to the root logger's handlers list
        if "file" not in config_to_use["root"]["handlers"]:
            config_to_use["root"]["handlers"].append("file")

        # Also add file handler to any specific loggers if they are defined
        # to ensure they also log to file, unless they have propagate=False
        # and their own specific handlers.
        # For now, root logger handles propagation.

        print(
            f"Logging Config: Standard file logging enabled. Path: {log_file_path}, Level: {LOG_LEVEL}"
        )
    else:
        print(
            f"Logging Config: Standard file logging is disabled (LOG_FILE_NAME or LOG_DIR not set). Level: {LOG_LEVEL}"
        )

    # Setup for GA Metrics JSONL logger
    # Metrics will always be logged to a file if LOG_DIR is set.
    # The level for metrics logger will typically be INFO.
    if LOG_DIR:
        ga_metrics_log_file_path = os.path.join(LOG_DIR, "ga_metrics.jsonl")
        config_to_use["handlers"]["ga_metrics_file"] = {
            "level": "INFO",  # Metrics are generally INFO level
            "formatter": "json",  # Use the JSON formatter
            "class": "logging.handlers.RotatingFileHandler",
            "filename": ga_metrics_log_file_path,
            "maxBytes": 1024 * 1024 * 10,  # 10 MB
            "backupCount": 5,
            "encoding": "utf-8",
        }
        # Configure the 'prompthelix.ga_metrics' logger
        config_to_use["loggers"]["prompthelix.ga_metrics"] = {
            "handlers": ["ga_metrics_file"],
            "level": "INFO",
            "propagate": False,  # Do not pass metrics logs to the root logger's handlers
        }
        print(
            f"Logging Config: GA Metrics JSONL logging enabled. Path: {ga_metrics_log_file_path}"
        )
    else:
        print("Logging Config: GA Metrics JSONL logging disabled (LOG_DIR not set).")

    try:
        logging.config.dictConfig(config_to_use)
        logger.info("Logging setup complete using dictConfig.")
        if enable_file_logging:
            logger.info(
                f"Logging to console and file: {os.path.join(LOG_DIR, LOG_FILE_NAME)}"
            )
        else:
            logger.info(f"Logging to console only.")

    except Exception as e:
        # Fallback to basicConfig if dictConfig fails, printing the error.
        # This is crucial for ensuring the application can still log something.
        print(f"Error setting up logging with dictConfig: {e}", file=sys.stderr)
        print("Falling back to basicConfig for console logging.", file=sys.stderr)
        logging.basicConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
            handlers=[logging.StreamHandler(sys.stdout)],  # Ensure output to stdout
        )
        # Get a new logger instance after basicConfig might have reset things
        logging.getLogger(__name__).error(
            "Fell back to basicConfig due to dictConfig error.", exc_info=True
        )

    # After setup, explicitly set levels for very noisy libraries if not handled by dictConfig loggers section
    # This is a bit redundant if dictConfig's loggers section works as expected, but serves as a safeguard.
    libraries_to_quieten = {
        "httpx": logging.WARNING,
        "openai": logging.WARNING,
        "watchfiles": logging.WARNING,  # Often used by uvicorn --reload
        "uvicorn.access": logging.WARNING,  # Access logs can be noisy
    }
    for lib_name, lib_level in libraries_to_quieten.items():
        lib_logger = logging.getLogger(lib_name)
        if (
            lib_logger.level < lib_level or lib_logger.level == 0
        ):  # only change if current level is more verbose or not set
            # Check if specific logger config was applied by dictConfig
            if not (
                lib_name in config_to_use.get("loggers", {})
                and config_to_use["loggers"][lib_name].get("level")
            ):
                lib_logger.setLevel(lib_level)
                # Ensure it has a handler if it's not propagating to root, or if root has no handlers
                if not lib_logger.propagate and not lib_logger.handlers:
                    # Add console handler to ensure its messages are seen if it's not propagating
                    # This part is tricky; usually, we want libraries to propagate to root.
                    # If `propagate: False` was set in dictConfig, this won't add handlers here.
                    pass

    # Test message
    logger.debug("This is a DEBUG message from logging_config after setup.")
    logger.info("This is an INFO message from logging_config after setup.")
    logger.warning("This is a WARNING message from logging_config after setup.")


# Example of how to use it:
# if __name__ == '__main__':
#     setup_logging()
#     # Example usage in other modules:
#     # import logging
#     # logger = logging.getLogger(__name__)
#     # logger.info("This is a test log message from main execution.")
#     # logger.debug("This is a debug message that should only appear if LOG_LEVEL is DEBUG.")
#     # logging.getLogger("another_module").info("Log from another module.")
#     # logging.getLogger("httpx").info("This httpx INFO should be suppressed by default.")
#     # logging.getLogger("httpx").warning("This httpx WARNING should be visible.")
