import logging
import html # Import the html module
import asyncio # Moved import to module level
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
                    "message": escaped_log_entry, # Use the escaped message
                    "module": record.module,
                    "funcName": record.funcName,
                    "lineno": record.lineno,
                }
            }
            # Use asyncio.create_task to run the async broadcast_json method
            # import asyncio # No longer imported locally
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    asyncio.create_task(self.connection_manager.broadcast_json(log_data))
                else:
                    # Fallback or log differently if no loop is running (e.g., during test collection)
                    print(f"LOGGING_HANDLER_NO_LOOP (loop not running): {log_data['data']['message'][:100]}...") # Print truncated message
            except RuntimeError: # Catches "no running event loop"
                # Fallback or log differently if get_running_loop() itself fails
                print(f"LOGGING_HANDLER_NO_LOOP (RuntimeError): {log_data['data']['message'][:100]}...") # Print truncated message
        except Exception:
            self.handleError(record)
