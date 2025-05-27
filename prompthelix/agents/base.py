import abc

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the PromptHelix system.
    """

    def __init__(self, agent_id: str):
        """
        Initializes the BaseAgent.

        Args:
            agent_id: A unique identifier for the agent.
        """
        self.agent_id = agent_id

    @abc.abstractmethod
    def process_request(self, request_data: dict) -> dict:
        """
        Processes an incoming request.

        This method must be implemented by subclasses to define agent-specific
        request handling logic.

        Args:
            request_data: A dictionary containing the request details.

        Returns:
            A dictionary containing the response from the agent.
        """
        pass

    def send_message(self, recipient_agent_id: str, message_content: dict):
        """
        Sends a message to another agent.

        Placeholder for inter-agent communication.

        Args:
            recipient_agent_id: The ID of the agent to send the message to.
            message_content: A dictionary containing the message content.
        """
        # Placeholder for message sending logic
        print(f"Agent {self.agent_id} sending message to {recipient_agent_id}: {message_content}")
        pass

    def receive_message(self, sender_agent_id: str, message_content: dict):
        """
        Receives a message from another agent.

        Placeholder for inter-agent communication.

        Args:
            sender_agent_id: The ID of the agent that sent the message.
            message_content: A dictionary containing the message content.
        """
        # Placeholder for message receiving logic
        print(f"Agent {self.agent_id} received message from {sender_agent_id}: {message_content}")
        pass
