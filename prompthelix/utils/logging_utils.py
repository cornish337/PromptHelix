import logging
import sys
import json

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


def setup_logging(debug: bool = False, json_format: bool = False) -> logging.Logger:
    """Configure root logger with optional JSON formatting."""
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter: logging.Formatter
    if json_format:
        formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    if root.handlers:
        root.handlers.clear()
    root.setLevel(level)
    root.addHandler(handler)
    return root
