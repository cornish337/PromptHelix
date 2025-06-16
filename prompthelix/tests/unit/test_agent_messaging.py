import unittest
from unittest.mock import MagicMock
from datetime import datetime

from prompthelix.message_bus import MessageBus
from prompthelix.agents.base import BaseAgent # For creating a mock agent

class MockAgent(BaseAgent):
    """A simple agent for testing messaging that inherits from BaseAgent."""
    def __init__(self, agent_id: str, message_bus=None):
        super().__init__(agent_id, message_bus)
        self.received_ping_payload = None

    def process_request(self, request_data: dict) -> dict:
        # Not the primary focus for ping tests, but good to have a basic implementation
        return {"status": "processed", "data_received": request_data}

    # Override receive_message if we want to capture specific things for asserts,
    # but for ping, BaseAgent's implementation is what we are testing.
    # However, if we want to check payload of ping, we might need to.
    # BaseAgent's receive_message already logs and returns the pong for "ping".
    # Let's assume BaseAgent's receive_message is sufficient for ping test.


class TestAgentMessaging(unittest.TestCase):

    def setUp(self):
        self.message_bus = MessageBus()
        self.agent1 = MockAgent(agent_id="Agent1", message_bus=self.message_bus)
        self.agent2 = MockAgent(agent_id="Agent2", message_bus=self.message_bus)
        self.message_bus.register(self.agent1.agent_id, self.agent1)
        self.message_bus.register(self.agent2.agent_id, self.agent2)

    def test_ping_functionality(self):
        """Test that agent1 can ping agent2 and receive a pong response."""
        ping_payload = {"data": "ping_from_agent1"}
        response = self.agent1.send_message(
            recipient_agent_id=self.agent2.agent_id,
            message_content=ping_payload,
            message_type="ping"
        )

        self.assertIsInstance(response, dict)
        self.assertEqual(response.get("status"), "pong")
        self.assertEqual(response.get("agent_id"), self.agent2.agent_id)
        self.assertIn("timestamp", response)
        try:
            datetime.fromisoformat(response["timestamp"]) # Check if timestamp is valid ISO format
        except ValueError:
            self.fail("Ping response timestamp is not a valid ISO format string.")

    def test_message_to_non_existent_agent(self):
        """Test dispatching a message to a non-existent agent via MessageBus."""
        message_to_non_existent = {
            "sender_id": self.agent1.agent_id,
            "recipient_id": "NonExistentAgent",
            "message_type": "test_message",
            "payload": {"data": "hello"}
        }
        response = self.message_bus.dispatch_message(message_to_non_existent)

        self.assertIsInstance(response, dict)
        self.assertEqual(response.get("status"), "error")
        self.assertIn("Recipient agent 'NonExistentAgent' not found", response.get("error", ""))

    def test_send_message_failure_no_bus(self):
        """Test that an agent without a message bus fails to send a message gracefully."""
        agent_no_bus = MockAgent(agent_id="AgentNoBus", message_bus=None)
        response = agent_no_bus.send_message(
            recipient_agent_id=self.agent1.agent_id,
            message_content={"data": "hello"},
            message_type="test_message"
        )

        self.assertIsInstance(response, dict)
        self.assertEqual(response.get("status"), "error")
        self.assertEqual(response.get("error"), "No message bus available to send message.")

    def test_send_message_recipient_no_receive_method(self):
        """Test sending a message to an agent that doesn't have receive_message."""
        faulty_agent_instance = object() # A plain object that doesn't have receive_message
        self.message_bus.register("FaultyAgent", faulty_agent_instance)

        response = self.agent1.send_message(
            recipient_agent_id="FaultyAgent",
            message_content={"data": "hello"},
            message_type="test_message"
        )
        self.assertIsInstance(response, dict)
        self.assertEqual(response.get("status"), "error")
        self.assertIn("does not have a callable 'receive_message' method", response.get("error", ""))

    def test_send_message_recipient_receive_method_exception(self):
        """Test sending a message to an agent whose receive_message raises an exception."""
        agent_with_faulty_receive = MockAgent(agent_id="FaultyReceiveAgent", message_bus=self.message_bus)

        # Mock the receive_message to raise an exception
        original_receive_message = agent_with_faulty_receive.receive_message
        def faulty_receive(*args, **kwargs):
            raise RuntimeError("Simulated receive error")
        agent_with_faulty_receive.receive_message = MagicMock(side_effect=faulty_receive)

        self.message_bus.register(agent_with_faulty_receive.agent_id, agent_with_faulty_receive)

        response = self.agent1.send_message(
            recipient_agent_id=agent_with_faulty_receive.agent_id,
            message_content={"data": "hello"},
            message_type="test_message"
        )
        self.assertIsInstance(response, dict)
        self.assertEqual(response.get("status"), "error")
        self.assertIn(f"Error delivering message to agent '{agent_with_faulty_receive.agent_id}'", response.get("error", ""))
        self.assertIn("Simulated receive error", response.get("error", ""))

        # Restore original method if necessary for other tests or cleanup, though for this test structure it's fine.
        agent_with_faulty_receive.receive_message = original_receive_message


if __name__ == '__main__':
    unittest.main()
