import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch, ANY
from datetime import datetime

from prompthelix.message_bus import MessageBus
# Assuming ConnectionManager is imported for type hinting in MessageBus,
# but we'll mock it here.

class TestMessageBusBroadcasting(unittest.TestCase):

    def setUp(self):
        self.mock_db_session_factory = MagicMock()
        self.mock_connection_manager = AsyncMock() # Use AsyncMock if its methods are async
        self.mock_connection_manager.broadcast_json = AsyncMock()

    @patch('asyncio.create_task') # Patch asyncio.create_task
    def test_log_message_to_db_with_connection_manager_creates_broadcast_task(self, mock_create_task):
        bus = MessageBus(
            db_session_factory=self.mock_db_session_factory,
            connection_manager=self.mock_connection_manager
        )

        # Mock the database part to avoid actual DB operations
        mock_session = MagicMock()
        self.mock_db_session_factory.return_value = mock_session

        test_payload = {"key": "value"}
        bus._log_message_to_db(
            session_id="test_session",
            sender_id="sender_agent",
            recipient_id="recipient_agent",
            message_type="test_type",
            content_payload=test_payload
        )

        # Assert that asyncio.create_task was called
        mock_create_task.assert_called_once()

        # Further check: The argument to create_task should be the coroutine
        # self.manager._broadcast_log_async(log_data_dict_expected)
        # This is harder to check directly without also capturing the coro.
        # For now, checking create_task was called is a good first step.
        # We can also test _broadcast_log_async directly.

    async def test_broadcast_log_async_calls_connection_manager(self):
        bus = MessageBus(
            db_session_factory=self.mock_db_session_factory,
            connection_manager=self.mock_connection_manager
        )

        log_data = {
            "session_id": "s1", "sender_id": "a1", "recipient_id": "a2",
            "message_type": "m_type", "content": {"data": "content"},
            "timestamp": datetime.utcnow().isoformat(), # Ensure timestamp is similar format
            "db_log_status": "success" # Or whatever status is set
        }

        # Directly call the async helper method that create_task would run
        await bus._broadcast_log_async(log_data)

        self.mock_connection_manager.broadcast_json.assert_called_once_with({
            "type": "new_conversation_log",
            "data": log_data
        })

    @patch('asyncio.create_task')
    def test_log_message_to_db_without_connection_manager(self, mock_create_task):
        bus = MessageBus(db_session_factory=self.mock_db_session_factory, connection_manager=None)

        mock_session = MagicMock()
        self.mock_db_session_factory.return_value = mock_session

        bus._log_message_to_db(
            session_id="test_session",
            sender_id="sender_agent",
            recipient_id="recipient_agent",
            message_type="test_type",
            content_payload={"key": "value"}
        )

        mock_create_task.assert_not_called()
        self.mock_connection_manager.broadcast_json.assert_not_called()

    # Test that dispatch_message and broadcast_message trigger the logging (and thus broadcasting)
    @patch.object(MessageBus, '_log_message_to_db') # Patch the internal logging method
    def test_dispatch_message_triggers_logging(self, mock_log_db):
        bus = MessageBus(
            db_session_factory=self.mock_db_session_factory,
            connection_manager=self.mock_connection_manager # CM is present
        )

        # Mock an agent registry for dispatch_message to proceed
        mock_agent = MagicMock()
        mock_agent.receive_message = MagicMock(return_value={"status": "ok"})
        bus.register("recipient_agent", mock_agent)

        message_payload = {"data": "test_dispatch", "session_id": "s_dispatch"} # Added session_id here
        message = {
            "sender_id": "sender_agent",
            "recipient_id": "recipient_agent",
            "message_type": "direct_request",
            "payload": message_payload
            # "session_id": "s_dispatch" # Assuming session_id might be in payload - moved to payload
        }

        bus.dispatch_message(message)

        # Check that _log_message_to_db was called with expected args (content can be tricky)
        mock_log_db.assert_called_once_with(
            session_id='s_dispatch', # Or how it's extracted
            sender_id="sender_agent",
            recipient_id="recipient_agent",
            message_type="direct_request",
            content_payload=message_payload
        )

    @patch.object(MessageBus, '_log_message_to_db') # Patch the internal logging method
    async def test_broadcast_message_triggers_logging(self, mock_log_db):
        bus = MessageBus(
            db_session_factory=self.mock_db_session_factory,
            connection_manager=self.mock_connection_manager # CM is present
        )

        payload = {"info": "broadcast_info", "session_id": "s_broadcast"}
        message_type = "system_announcement"
        sender_id = "system_broadcaster"

        await bus.broadcast_message(message_type, payload, sender_id=sender_id)

        # Check that _log_message_to_db was called with expected args
        # The recipient_id for broadcasts in _log_message_to_db is "BROADCAST"
        mock_log_db.assert_called_once_with(
            session_id='s_broadcast', # Or how it's extracted
            sender_id=sender_id,
            recipient_id="BROADCAST", # Based on current implementation detail
            message_type=message_type,
            content_payload=payload
        )

if __name__ == '__main__':
    # Use pytest: `pytest path/to/your/test_message_bus_broadcasting.py`
    unittest.main()
