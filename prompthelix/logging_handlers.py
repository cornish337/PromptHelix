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
            # Schedule async broadcast; handle cases where no loop is running
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(self.connection_manager.broadcast_json(log_data))
                else:
                    # Fallback if loop exists but isn't running (e.g., during test collection)
                    print(
                        f"LOGGING_HANDLER_NO_LOOP (loop not running): "
                        f"{log_data['data']['message'][:100]}..."
                    )
            except RuntimeError:
                # Fallback if get_running_loop() fails entirely
                print(
                    f"LOGGING_HANDLER_NO_LOOP (RuntimeError): "
                    f"{log_data['data']['message'][:100]}..."
                )
        except Exception:
            self.handleError(record)
