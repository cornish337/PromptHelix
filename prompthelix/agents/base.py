import abc
from datetime import datetime
import logging
import asyncio # Added
from typing import Optional, Dict # Keep this one

logger = logging.getLogger(__name__)

# Assuming the first definition is the one to be kept and augmented.
# The second one seems like a duplication.

class BaseAgent(abc.ABC):
    """
    Abstract base class for all agents in the PromptHelix system.
    Includes basic inter-agent communication capabilities.
    """

    def __init__(self, agent_id: str, message_bus=None, settings: Optional[Dict] = None):
        """
        Initializes the BaseAgent.

        Args:
            agent_id (str): A unique identifier for the agent.
            message_bus (object, optional): An optional message bus instance
                                            for inter-agent communication.
            settings (Optional[Dict], optional): Configuration settings for the agent.
                                                 Defaults to None, which means an empty dict.
        """
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.settings: Dict = settings if settings is not None else {}
        self.messages_processed: int = 0 # Added
        self.errors_encountered: int = 0 # Added

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

    async def broadcast_message(self, message_type: str, payload: dict):
        """Broadcasts a message to all subscribers via the message bus."""
        if self.message_bus and hasattr(self.message_bus, "broadcast_message"):
            await self.message_bus.broadcast_message(message_type, payload, sender_id=self.agent_id)
        else:
            logger.warning(f"Agent '{self.agent_id}': No message bus available to broadcast '{message_type}'.")

    def subscribe_to(self, message_type: str):
        """Subscribe this agent to a message type on the bus."""
        if self.message_bus and hasattr(self.message_bus, "subscribe"):
            self.message_bus.subscribe(self.agent_id, message_type)
        else:
            logger.warning(f"Agent '{self.agent_id}': No message bus available to subscribe to '{message_type}'.")

    def publish_metrics(self):
        # Publishes current agent metrics via the message bus's connection manager.
        if hasattr(self, 'message_bus') and self.message_bus and \
           hasattr(self.message_bus, 'connection_manager') and self.message_bus.connection_manager:

            payload_data = {
                "agent_id": self.agent_id,
                "messages_processed": self.messages_processed,
                "errors_encountered": self.errors_encountered,
                "timestamp": datetime.utcnow().isoformat() # Add a timestamp for the metric update
            }

            try:
                asyncio.create_task(
                    self.message_bus.connection_manager.broadcast_json({
                        "type": "agent_metric_update",
                        "data": payload_data
                    })
                )
            except RuntimeError as e:
                logger.error(f"Agent '{self.agent_id}': Could not publish metrics due to asyncio RuntimeError: {e}. Ensure an event loop is running.")
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}': Exception during metrics publishing: {e}", exc_info=True)
        else:
            logger.debug(f"Agent '{self.agent_id}': Message bus or connection manager not available. Skipping metrics publish.")

    def receive_message(self, message: dict):
        """
        Handles an incoming message received by this agent.
        """
        self.messages_processed += 1

        sender_id = message.get('sender_id', 'UnknownSender')
        msg_type = message.get('message_type', 'UnknownType')
        payload = message.get('payload', {})
        response = None

        logger.info(f"Agent '{self.agent_id}' received message from '{sender_id}': Type='{msg_type}', Payload Snippet='{str(payload)[:50]}...'")

        try:
            if msg_type == "direct_request":
                logger.info(f"Agent '{self.agent_id}' processing 'direct_request' from '{sender_id}'.")
                response = self.process_request(payload)
            elif msg_type == "ping":
                logger.info(f"Agent '{self.agent_id}' received ping from '{sender_id}'. Responding with pong.")
                response = {
                    "status": "pong",
                    "agent_id": self.agent_id,
                    "timestamp": datetime.utcnow().isoformat()
                }
            elif msg_type == "info_update":
                logger.info(
                    f"Agent '{self.agent_id}' received 'info_update' from '{sender_id}'. Logging payload: {payload}"
                )
                response = {"status": "info_logged", "agent_id": self.agent_id}
            elif msg_type in {"evaluation_result", "critique_result"}: # Specific for MetaLearnerAgent
                logger.info(f"Agent '{self.agent_id}' processing '{msg_type}' from '{sender_id}'.")
                # This assumes process_request can handle this structure, which it does for MetaLearnerAgent
                response = self.process_request({"data_type": msg_type, "data": payload})
            else:
                logger.info(
                    f"Agent '{self.agent_id}' logged message of type '{msg_type}' from '{sender_id}'. No specific handler implemented in BaseAgent."
                )
                response = {"status": "message_logged_by_base_agent", "type": msg_type, "agent_id": self.agent_id}

        except Exception as e:
            self.errors_encountered += 1
            logger.error(f"Agent '{self.agent_id}' encountered an error while processing message type '{msg_type}': {e}", exc_info=True)
            response = {
                "status": "error",
                "agent_id": self.agent_id,
                "error_message": f"Error processing message: {str(e)}",
                "original_message_type": msg_type
            }
        finally:
            self.publish_metrics()

        return response

