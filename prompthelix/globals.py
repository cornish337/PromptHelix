# prompthelix/globals.py
"""
Global shared instances for the PromptHelix application.
This module should be kept lightweight and free of complex imports
to avoid circular dependencies.
"""
from prompthelix.websocket_manager import ConnectionManager

# Global WebSocket connection manager instance
websocket_manager = ConnectionManager()

# You can add other global instances here if needed, e.g., a global MessageBus
# from prompthelix.database import SessionLocal
# from prompthelix.message_bus import MessageBus
# message_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
# Note: If adding MessageBus here, ensure SessionLocal import doesn't create cycles.
# For now, only websocket_manager is strictly needed to break the current cycle.

print("PromptHelix Config: GLOBALS - websocket_manager initialized.")
