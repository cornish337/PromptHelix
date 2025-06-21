import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch, ANY
from datetime import datetime

from prompthelix.agents.base import BaseAgent # The class to test
from prompthelix.message_bus import MessageBus # For type hinting and mocking structure

# A minimal concrete agent for testing BaseAgent functionality
class DummyAgent(BaseAgent):
    def __init__(self, agent_id: str, message_bus=None, settings=None):
        super().__init__(agent_id, message_bus, settings)
        self.should_raise_error_on_process = False

    def process_request(self, request_data: dict) -> dict:
        if self.should_raise_error_on_process:
            raise ValueError("Simulated processing error")
        return {"status": "processed", "data_received": request_data}

class TestAgentBaseMetrics(unittest.TestCase):

    def setUp(self):
        self.mock_message_bus = MagicMock(spec=MessageBus)
        self.mock_message_bus.connection_manager = AsyncMock() # Mock the nested attribute
        self.mock_message_bus.connection_manager.broadcast_json = AsyncMock()

        self.agent_id = "test_dummy_agent"
        self.dummy_agent = DummyAgent(
            agent_id=self.agent_id,
            message_bus=self.mock_message_bus
        )

    @patch('asyncio.create_task') # Patch where it's called in BaseAgent.publish_metrics
    def test_receive_message_increments_processed_count_and_publishes(self, mock_create_task):
        self.dummy_agent.last_operation_type = "ping"
        self.dummy_agent._total_fitness_change = 0.4
        self.dummy_agent._fitness_change_events = 2

        self.dummy_agent.receive_message({
            "sender_id": "another_agent",
            "recipient_id": self.agent_id,
            "message_type": "ping",
            "payload": {"ping_data": "hello"}
        })

        self.assertEqual(self.dummy_agent.messages_processed, 1)
        self.assertEqual(self.dummy_agent.errors_encountered, 0)

        # Check that publish_metrics (which calls create_task) was triggered
        # In BaseAgent, publish_metrics calls create_task which then calls broadcast_json.
        # So, we check the mock on broadcast_json.

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args

        self.assertEqual(args[0]['type'], 'agent_metric_update')
        metric_data = args[0]['data']
        self.assertEqual(metric_data['agent_id'], self.agent_id)
        self.assertEqual(metric_data['messages_processed'], 1)
        self.assertEqual(metric_data['errors_encountered'], 0)
        self.assertEqual(metric_data['last_operation_type'], "ping")
        self.assertAlmostEqual(metric_data['average_fitness_change'], 0.2)
        self.assertIn('timestamp', metric_data)

        # Also assert create_task was called once, as it's the direct call from publish_metrics
        mock_create_task.assert_called_once()


    @patch('asyncio.create_task')
    def test_receive_message_increments_error_count_on_exception_and_publishes(self, mock_create_task):
        self.dummy_agent.should_raise_error_on_process = True  # Configure dummy to raise error
        self.dummy_agent.last_operation_type = "direct_request"
        self.dummy_agent._total_fitness_change = -0.3
        self.dummy_agent._fitness_change_events = 3

        # This call to receive_message should trigger an error in process_request
        response = self.dummy_agent.receive_message({
            "sender_id": "another_agent",
            "recipient_id": self.agent_id,
            "message_type": "direct_request", # This type will call process_request
            "payload": {"some_data": "trigger_error"}
        })

        self.assertEqual(self.dummy_agent.messages_processed, 1)
        self.assertEqual(self.dummy_agent.errors_encountered, 1)

        # Check error response structure
        self.assertIsNotNone(response)
        self.assertEqual(response.get('status'), 'error')
        self.assertEqual(response.get('agent_id'), self.agent_id)
        self.assertIn("Simulated processing error", response.get('error_message', ""))

        # Check that publish_metrics was still called
        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args

        self.assertEqual(args[0]['type'], 'agent_metric_update')
        metric_data = args[0]['data']
        self.assertEqual(metric_data['agent_id'], self.agent_id)
        self.assertEqual(metric_data['messages_processed'], 1)
        self.assertEqual(metric_data['errors_encountered'], 1)
        self.assertEqual(metric_data['last_operation_type'], "direct_request")
        self.assertAlmostEqual(metric_data['average_fitness_change'], -0.1)

        mock_create_task.assert_called_once()

    @patch('asyncio.create_task')
    def test_publish_metrics_no_message_bus(self, mock_create_task):
        agent_no_bus = DummyAgent(agent_id="agent_no_bus", message_bus=None)
        agent_no_bus.publish_metrics()
        mock_create_task.assert_not_called()
        # Also ensure broadcast_json on the setup's mock_message_bus was not called by this new agent
        self.mock_message_bus.connection_manager.broadcast_json.assert_not_called()


    @patch('asyncio.create_task')
    def test_publish_metrics_no_connection_manager(self, mock_create_task):
        mock_bus_no_cm = MagicMock(spec=MessageBus)
        mock_bus_no_cm.connection_manager = None # Explicitly set to None

        agent_bus_no_cm = DummyAgent(agent_id="agent_bus_no_cm", message_bus=mock_bus_no_cm)
        agent_bus_no_cm.publish_metrics()

        mock_create_task.assert_not_called()
        # Ensure broadcast_json on the setup's mock_message_bus was not called by this new agent
        self.mock_message_bus.connection_manager.broadcast_json.assert_not_called()

    @patch('asyncio.create_task')
    def test_publish_metrics_computes_average(self, mock_create_task):
        self.dummy_agent.last_operation_type = 'test_op'
        self.dummy_agent._total_fitness_change = 1.2
        self.dummy_agent._fitness_change_events = 3

        self.dummy_agent.publish_metrics()

        mock_create_task.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        data = args[0]['data']
        self.assertEqual(data['last_operation_type'], 'test_op')
        self.assertAlmostEqual(data['average_fitness_change'], 0.4)


    @patch('asyncio.create_task')
    def test_publish_metrics_asyncio_runtime_error(self, mock_create_task):
        # Simulate asyncio.create_task raising a RuntimeError (e.g., no loop)
        mock_create_task.side_effect = RuntimeError("Test: No event loop")

        # Logger is imported in base.py as logger = logging.getLogger(__name__)
        # We can patch this logger to capture its output.
        with patch('prompthelix.agents.base.logger') as mock_agent_logger:
            self.dummy_agent.publish_metrics()

            mock_create_task.assert_called_once() # create_task was attempted
            # Check that an error was logged
            mock_agent_logger.error.assert_called_once()
            args, _ = mock_agent_logger.error.call_args
            self.assertIn("Could not publish metrics due to asyncio RuntimeError", args[0])
            # Ensure broadcast_json was not successfully called if create_task failed early
            # This depends on where the error occurs. If create_task itself errors, broadcast_json isn't reached.
            # self.mock_message_bus.connection_manager.broadcast_json.assert_not_called() # This might be too strict if create_task is more complex


if __name__ == '__main__':
    unittest.main()
