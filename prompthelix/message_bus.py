import logging

# Configure basic logging for the message bus
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageBus:
    """
    A simple message bus for inter-agent communication.
    """
    def __init__(self):
        self._registry = {}
        logger.info("MessageBus initialized.")

    def register(self, agent_id: str, agent_instance):
        """
        Registers an agent instance with the message bus.

        Args:
            agent_id (str): The unique identifier for the agent.
            agent_instance: The instance of the agent.
        """
        if agent_id in self._registry:
            logger.warning(f"Agent ID '{agent_id}' is already registered. Overwriting.")
        self._registry[agent_id] = agent_instance
        logger.info(f"Agent '{agent_id}' registered with the message bus.")

    def unregister(self, agent_id: str):
        """
        Unregisters an agent from the message bus.

        Args:
            agent_id (str): The unique identifier for the agent to unregister.
        """
        if agent_id in self._registry:
            del self._registry[agent_id]
            logger.info(f"Agent '{agent_id}' unregistered from the message bus.")
        else:
            logger.warning(f"Attempted to unregister agent ID '{agent_id}', which was not found.")

    def dispatch_message(self, message: dict):
        """
        Dispatches a message to the recipient agent.

        The message structure should be a dictionary including at least:
        - 'sender_id' (str): ID of the sending agent.
        - 'recipient_id' (str): ID of the receiving agent.
        - 'message_type' (str): Type of the message (e.g., "request_data", "critique_result").
        - 'payload' (dict): The actual content of the message.

        Args:
            message (dict): The message to dispatch.
        """
        if not all(key in message for key in ['sender_id', 'recipient_id', 'message_type', 'payload']):
            logger.error(f"Invalid message structure: {message}. Missing required keys.")
            return

        recipient_id = message.get('recipient_id')
        sender_id = message.get('sender_id')
        message_type = message.get('message_type')

        logger.info(f"Attempting to dispatch message of type '{message_type}' from '{sender_id}' to '{recipient_id}'.")

        recipient_agent = self._registry.get(recipient_id)

        if recipient_agent:
            try:
                # Assuming the agent has a 'receive_message' method
                if hasattr(recipient_agent, 'receive_message') and callable(recipient_agent.receive_message):
                    recipient_agent.receive_message(message)
                    logger.info(f"Message dispatched and received by '{recipient_id}'.")
                else:
                    logger.error(f"Recipient agent '{recipient_id}' does not have a callable 'receive_message' method.")
            except Exception as e:
                logger.error(f"Error delivering message to agent '{recipient_id}': {e}", exc_info=True)
        else:
            logger.warning(f"Recipient agent '{recipient_id}' not found in registry. Message from '{sender_id}' of type '{message_type}' could not be delivered.")

if __name__ == '__main__':
    # Example Usage (simple test within the file)
    class MockAgent:
        def __init__(self, agent_id, message_bus=None):
            self.agent_id = agent_id
            self.message_bus = message_bus # Agents should have a reference to the bus to send messages

        def send_message(self, recipient_id, message_type, payload):
            if self.message_bus:
                msg = {
                    "sender_id": self.agent_id,
                    "recipient_id": recipient_id,
                    "message_type": message_type,
                    "payload": payload
                }
                self.message_bus.dispatch_message(msg)
            else:
                print(f"{self.agent_id}: Message bus not available. Cannot send message.")

        def receive_message(self, message: dict):
            logger.info(f"Agent '{self.agent_id}' received message: {message['message_type']} from '{message['sender_id']}' with payload: {message['payload']}")

    # Test
    bus = MessageBus()
    agent_A = MockAgent(agent_id="AgentA", message_bus=bus)
    agent_B = MockAgent(agent_id="AgentB", message_bus=bus)

    bus.register(agent_A.agent_id, agent_A)
    bus.register(agent_B.agent_id, agent_B)

    # Agent A sends a message to Agent B
    agent_A.send_message(recipient_id="AgentB", message_type="test_ping", payload={"data": "Hello from A"})

    # Agent B sends a message to a non-existent agent
    agent_B.send_message(recipient_id="AgentC", message_type="test_echo", payload={"data": "Hello from B to C"})

    # Test unregister
    bus.unregister("AgentA")
    agent_B.send_message(recipient_id="AgentA", message_type="farewell", payload={"data": "Goodbye from B"})

    # Test invalid message
    bus.dispatch_message({"sender_id": "system", "payload": "test"})

    logger.info("MessageBus example usage finished.")
