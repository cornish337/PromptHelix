import logging
import html  # Import the html module
import asyncio  # Moved import to module level
from prompthelix.websocket_manager import ConnectionManager

class WebSocketLogHandler(logging.Handler):
    def __init__(self, connection_manager: ConnectionManager):
        super().__init__()
        self.connection_manager = connection_manager

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = self.format(record)
            # Escape HTML characters in the log message
            escaped_log_entry = html.escape(log_entry)
            log_data = {
                "type": "debug_log",
                "data": {
                    "timestamp": record.created,
                    "level": record.levelname,
                    "message": escaped_log_entry,  # Use the escaped message
                    "module": record.module,
                    "funcName": record.funcName,
                    "lineno": record.lineno,
                }
            }
            # Schedule async broadcast; fall back to running synchronously if no loop
            coro = self.connection_manager.broadcast_json(log_data)
            try:
                asyncio.create_task(coro)
            except RuntimeError:
                # If no running loop, run the coroutine synchronously
                asyncio.run(coro)
        except Exception:
            self.handleError(record)
