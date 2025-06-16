import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from typing import List

from sqlalchemy.orm import Session as DbSession

# Adjust imports based on actual project structure
from prompthelix.models import ConversationLog
from prompthelix.schemas import ConversationSession, ConversationLogEntry # ConversationLogEntry might not be directly returned by service methods, they return models
from prompthelix.services.conversation_service import ConversationService

class TestConversationService(unittest.TestCase):

    def setUp(self):
        self.service = ConversationService()
        self.mock_db_session = MagicMock(spec=DbSession)

        # Sample log data - these are ConversationLog model instances
        self.log1_sess1 = ConversationLog(id=1, session_id="session1", sender_id="agentA", content='{"key": "value1"}', timestamp=datetime(2023, 1, 1, 10, 0, 0), message_type="type1", recipient_id="agentB")
        self.log2_sess1 = ConversationLog(id=2, session_id="session1", sender_id="agentB", content='{"key": "value2"}', timestamp=datetime(2023, 1, 1, 10, 1, 0), message_type="type2", recipient_id="agentA")
        self.log1_sess2 = ConversationLog(id=3, session_id="session2", sender_id="agentC", content='{"key": "value3"}', timestamp=datetime(2023, 1, 1, 11, 0, 0), message_type="type3", recipient_id="agentD")
        self.log2_sess2_older = ConversationLog(id=4, session_id="session2", sender_id="agentE", content='{"key": "value4"}', timestamp=datetime(2023, 1, 1, 10, 50, 0), message_type="type4", recipient_id="agentC")


    def test_get_conversation_sessions_empty(self):
        self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        result = self.service.get_conversation_sessions(self.mock_db_session)
        self.assertEqual(result, [])
        self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.return_value.limit.return_value.all.assert_called_once()


    def test_get_conversation_sessions_with_data(self):
        # Mock the complex query result for sessions
        # Each row needs to have session_id, message_count, first_message_at, last_message_at
        # These are SQLAlchemy Row objects or similar mockable objects
        mock_session_row1_data = MagicMock()
        mock_session_row1_data.session_id = "session1"
        mock_session_row1_data.message_count = 2
        mock_session_row1_data.first_message_at = datetime(2023, 1, 1, 10, 0, 0)
        mock_session_row1_data.last_message_at = datetime(2023, 1, 1, 10, 1, 0)

        mock_session_row2_data = MagicMock()
        mock_session_row2_data.session_id = "session2"
        mock_session_row2_data.message_count = 2 # Updated count for session2
        mock_session_row2_data.first_message_at = datetime(2023, 1, 1, 10, 50, 0) # Updated first_message_at for session2
        mock_session_row2_data.last_message_at = datetime(2023, 1, 1, 11, 0, 0)

        # Order should be most recent session first (session2's last message is later)
        self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [mock_session_row2_data, mock_session_row1_data]

        result = self.service.get_conversation_sessions(self.mock_db_session, skip=0, limit=5)

        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ConversationSession) # Service converts rows to Schema objects
        self.assertEqual(result[0].session_id, "session2")
        self.assertEqual(result[0].message_count, 2)
        self.assertEqual(result[0].last_message_at, datetime(2023, 1, 1, 11, 0, 0))

        self.assertEqual(result[1].session_id, "session1")
        self.assertEqual(result[1].message_count, 2)
        self.assertEqual(result[1].last_message_at, datetime(2023, 1, 1, 10, 1, 0))

    def test_get_conversation_sessions_pagination(self):
        mock_session_row1_data = MagicMock() # Only one session for pagination test
        mock_session_row1_data.session_id = "session1"
        mock_session_row1_data.message_count = 2
        mock_session_row1_data.first_message_at = datetime(2023, 1, 1, 10, 0, 0)
        mock_session_row1_data.last_message_at = datetime(2023, 1, 1, 10, 1, 0)

        # Mock for skip=0, limit=1
        query_chain_limit1 = self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.return_value.limit
        query_chain_limit1.return_value.all.return_value = [mock_session_row1_data]

        result_limit1 = self.service.get_conversation_sessions(self.mock_db_session, skip=0, limit=1)
        self.assertEqual(len(result_limit1), 1)
        self.assertEqual(result_limit1[0].session_id, "session1")
        query_chain_limit1.assert_called_with(1)
        self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.assert_called_with(0)


        # Mock for skip=1, limit=1 (should be empty if only one session total)
        query_chain_skip1_limit1 = self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.return_value.limit
        query_chain_skip1_limit1.return_value.all.return_value = []

        result_skip1_limit1 = self.service.get_conversation_sessions(self.mock_db_session, skip=1, limit=1)
        self.assertEqual(len(result_skip1_limit1), 0)
        query_chain_skip1_limit1.assert_called_with(1)
        self.mock_db_session.query.return_value.group_by.return_value.order_by.return_value.offset.assert_called_with(1)


    def test_get_messages_by_session_id_found(self):
        # Service returns model instances which API will convert to schemas
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [self.log1_sess1, self.log2_sess1]
        result = self.service.get_messages_by_session_id(self.mock_db_session, "session1", skip=0, limit=10)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], ConversationLog)
        self.assertEqual(result[0].id, 1)
        self.assertEqual(result[1].id, 2) # Ordered by timestamp (setup data is already ordered)

    def test_get_messages_by_session_id_not_found(self):
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        result = self.service.get_messages_by_session_id(self.mock_db_session, "non_existent_session")
        self.assertEqual(result, [])

    def test_get_messages_by_session_id_pagination(self):
        # Test limit
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [self.log1_sess1]
        result = self.service.get_messages_by_session_id(self.mock_db_session, "session1", skip=0, limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.assert_called_with(0)
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.assert_called_with(1)

        # Test skip
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [self.log2_sess1]
        result = self.service.get_messages_by_session_id(self.mock_db_session, "session1", skip=1, limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 2) # Assuming log2_sess1 would be the second item
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.assert_called_with(1)
        self.mock_db_session.query.return_value.filter.return_value.order_by.return_value.offset.return_value.limit.assert_called_with(1)


    def test_get_all_logs_empty(self):
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = []
        result = self.service.get_all_logs(self.mock_db_session)
        self.assertEqual(result, [])

    def test_get_all_logs_with_data(self):
        # Assuming descending order of timestamp for get_all_logs
        # log1_sess2 (11:00), log2_sess1 (10:01), log2_sess2_older (10:50), log1_sess1 (10:00)
        # Correct order by timestamp desc: log1_sess2, log2_sess2_older, log2_sess1, log1_sess1
        ordered_logs = [self.log1_sess2, self.log2_sess2_older, self.log2_sess1, self.log1_sess1]
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = ordered_logs

        result = self.service.get_all_logs(self.mock_db_session, skip=0, limit=5)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], ConversationLog)
        self.assertEqual(result[0].id, 3) # log1_sess2 is the latest
        self.assertEqual(result[1].id, 4) # log2_sess2_older
        self.assertEqual(result[2].id, 2) # log2_sess1
        self.assertEqual(result[3].id, 1) # log1_sess1

    def test_get_all_logs_pagination(self):
        ordered_logs = [self.log1_sess2, self.log2_sess2_older, self.log2_sess1, self.log1_sess1]

        # Test limit
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [ordered_logs[0]]
        result = self.service.get_all_logs(self.mock_db_session, skip=0, limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.log1_sess2.id)
        self.mock_db_session.query.return_value.order_by.return_value.offset.assert_called_with(0)
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.assert_called_with(1)

        # Test skip
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [ordered_logs[1]]
        result = self.service.get_all_logs(self.mock_db_session, skip=1, limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, self.log2_sess2_older.id)
        self.mock_db_session.query.return_value.order_by.return_value.offset.assert_called_with(1)
        self.mock_db_session.query.return_value.order_by.return_value.offset.return_value.limit.assert_called_with(1)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False) # Added argv and exit=False for notebook/REPL environments
                                                            # In a standard CLI run, unittest.main() is fine.
