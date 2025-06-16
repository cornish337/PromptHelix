import abc
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the PromptHelix system.
    Includes basic inter-agent communication capabilities.
    """

    def __init__(self, agent_id: str, message_bus=None):
        """
        Initializes the BaseAgent.

        Args:
            agent_id (str): A unique identifier for the agent.
            message_bus (object, optional): An optional message bus instance 
                                            for inter-agent communication. 
                                            If provided, it should have a 
                                            `dispatch_message(message)` method.
        """
        self.agent_id = agent_id
        self.message_bus = message_bus

    @abc.abstractmethod
    def process_request(self, request_data: dict) -> dict:
        """
        Processes an incoming request that is specific to the agent's core function.

        This method must be implemented by subclasses to define agent-specific
        request handling logic. It's typically invoked when a message of a 
        specific type (e.g., "direct_request") is received.

        Args:
            request_data (dict): A dictionary containing the request details, 
                                 usually from the payload of a received message.

        Returns:
            dict: A dictionary containing the response from the agent.
        """
        pass

    def send_message(self, recipient_agent_id: str, message_content: dict, message_type: str = "generic_message"):
        """
        Constructs and sends a message to another agent, potentially via a message bus.

        Args:
            recipient_agent_id (str): The ID of the agent to send the message to.
            message_content (dict): The payload of the message.
            message_type (str, optional): The type of the message, used by the recipient
                                          to determine handling. Defaults to "generic_message".

        Returns:
            The response from the recipient agent's `receive_message` method if successfully dispatched,
            or an error dictionary / None if dispatch fails or message bus is not available.
        """

        message = {
            "sender_id": self.agent_id,
            "recipient_id": recipient_agent_id,
            "message_type": message_type,
            "payload": message_content,
            "timestamp": datetime.utcnow().isoformat()
        }

        if self.message_bus and callable(getattr(self.message_bus, "dispatch_message", None)):
            try:
                response = self.message_bus.dispatch_message(message)
                logger.info(f"Agent '{self.agent_id}' sent message to '{recipient_agent_id}' via message bus: Type='{message_type}', Payload Snippet='{str(message_content)[:50]}...'. Response: {response}")
                return response
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}' failed to send message to '{recipient_agent_id}' via message bus: {e}", exc_info=True)
                return {"status": "error", "error": f"Exception during message dispatch by sender: {str(e)}"}
        else:
            logger.warning(f"Agent '{self.agent_id}' (no message bus) attempting to print message for '{recipient_agent_id}': Type='{message_type}', Payload='{message_content}'")
            return {"status": "error", "error": "No message bus available to send message."}


    def receive_message(self, message: dict):
        """
        Handles an incoming message received by this agent.

        This method would typically be called by a message bus when a message
        is routed to this agent. It logs the message and, based on the 
        message_type, may delegate processing to `self.process_request`.

        Args:
            message (dict): The message dictionary, expected to contain keys like
                            'sender_id', 'recipient_id', 'message_type', 'payload'.

        Returns:
            The result of `self.process_request(payload)` if the message_type
            is "direct_request", or None for other message types in this basic setup.
        """

        sender_id = message.get('sender_id', 'UnknownSender')
        msg_type = message.get('message_type', 'UnknownType')
        payload = message.get('payload', {})

        logger.info(f"Agent '{self.agent_id}' received message from '{sender_id}': Type='{msg_type}', Payload Snippet='{str(payload)[:50]}...'")

        if msg_type == "direct_request":
            logger.info(f"Agent '{self.agent_id}' processing 'direct_request' from '{sender_id}'.")
            return self.process_request(payload)
        elif msg_type == "ping":
            logger.info(f"Agent '{self.agent_id}' received ping from '{sender_id}'. Responding with pong.")
            return {
                "status": "pong",
                "agent_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        elif msg_type == "info_update": # Example of another message type
            logger.info(f"Agent '{self.agent_id}' received 'info_update' from '{sender_id}'. Logging payload: {payload}")
            # Potentially update internal state based on this info
            return {"status": "info_logged", "agent_id": self.agent_id}
        else:
            logger.info(f"Agent '{self.agent_id}' logged message of type '{msg_type}' from '{sender_id}'. No specific handler implemented in BaseAgent.")
            return {"status": "message_logged_by_base_agent", "type": msg_type, "agent_id": self.agent_id}

