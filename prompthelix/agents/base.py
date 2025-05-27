import abc
from datetime import datetime

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

    def send_message(self, recipient_agent_id: str, message_content: dict, message_type: str = "generic_message") -> bool:
        """
        Constructs and sends a message to another agent, potentially via a message bus.

        Args:
            recipient_agent_id (str): The ID of the agent to send the message to.
            message_content (dict): The payload of the message.
            message_type (str, optional): The type of the message, used by the recipient
                                          to determine handling. Defaults to "generic_message".

        Returns:
            bool: True if the message was sent or attempted to be sent, False otherwise.
                  In this implementation, it always returns True after attempting.
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
                self.message_bus.dispatch_message(message)
                print(f"{self.agent_id} sent message to {recipient_agent_id} via message bus: Type='{message_type}', Payload Snippet='{str(message_content)[:50]}...'")
                return True
            except Exception as e:
                print(f"{self.agent_id} failed to send message to {recipient_agent_id} via message bus: {e}")
                # Fallback to direct print if bus dispatch fails
                print(f"{self.agent_id} (bus dispatch failed) details for {recipient_agent_id}: Type='{message_type}', Payload='{message_content}'")
                return True # Still attempted
        else:
            print(f"{self.agent_id} (no bus) sending message to {recipient_agent_id}: Type='{message_type}', Payload='{message_content}'")
            # In a real scenario without a bus, this message wouldn't actually be delivered
            # unless a direct P2P mechanism was implemented here.
            return True

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

        print(f"{self.agent_id} received message from {sender_id}: Type='{msg_type}', Payload Snippet='{str(payload)[:50]}...'")

        if msg_type == "direct_request":
            print(f"{self.agent_id} processing 'direct_request' from {sender_id}.")
            return self.process_request(payload)
        elif msg_type == "info_update": # Example of another message type
            print(f"{self.agent_id} received 'info_update' from {sender_id}. Logging: {payload}")
            # Potentially update internal state based on this info
            return {"status": "info_logged"}
        else:
            print(f"{self.agent_id} logged message of type '{msg_type}' or will handle differently.")
            return {"status": "message_logged", "type": msg_type}
