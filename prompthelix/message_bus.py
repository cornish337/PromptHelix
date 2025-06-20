import logging
import asyncio
import inspect
import json
from datetime import datetime # Added
from typing import TYPE_CHECKING, Optional, Dict, Set, Any # Added TYPE_CHECKING, Optional, Dict, Set, Any
from sqlalchemy.orm import Session as DbSession
from prompthelix.models import ConversationLog

if TYPE_CHECKING: # Added
    from prompthelix.websocket_manager import ConnectionManager # For type hinting

# Configure basic logging for the message bus
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageBus:
    """
    A simple message bus for inter-agent communication.
    """
    def __init__(self, db_session_factory=None, connection_manager: Optional['ConnectionManager'] = None): # Modified
        self._registry: Dict[str, Any] = {} # Added type hint
        self._subscriptions: Dict[str, Set[str]] = {} # Added type hint
        self._queue: asyncio.Queue = asyncio.Queue() # Added type hint
        self._task: Optional[asyncio.Task] = None # Added type hint
        self._running: bool = False # Added type hint
        self.db_session_factory = db_session_factory
        self.connection_manager = connection_manager # Added
        logger.info("MessageBus initialized.")

    async def _broadcast_log_async(self, log_data: dict): # Added method
        """Helper to broadcast log data asynchronously via WebSocket."""
        if self.connection_manager:
            try:
                await self.connection_manager.broadcast_json({"type": "new_conversation_log", "data": log_data})
                logger.debug("Log data broadcasted via WebSocket.")
            except Exception as e:
                logger.error(f"Error broadcasting log data via WebSocket: {e}", exc_info=True)

    def _log_message_to_db(self, session_id: str, sender_id: str, recipient_id: Optional[str], message_type: str, content_payload: dict): # recipient_id can be Optional
        db_logged_successfully = False
        # Safely serialize content payload for storage/broadcast
        try:
            serialized_content = json.dumps(content_payload, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize content payload: {e}", exc_info=True)
            serialized_content = json.dumps(str(content_payload))

        if self.db_session_factory:
            db: Optional[DbSession] = None  # type hint
            try:
                db = self.db_session_factory()
                log_entry = ConversationLog(
                    session_id=session_id,
                    sender_id=sender_id,
                    recipient_id=recipient_id,
                    message_type=message_type,
                    content=serialized_content,
                )
                db.add(log_entry)
                db.commit()
                db_logged_successfully = True
                logger.debug(
                    f"Message from {sender_id} to {recipient_id} logged to DB. Session: {session_id}, Type: {message_type}"
                )
            except Exception as e:
                logger.error(f"Failed to log message to DB: {e}", exc_info=True)
                if db:
                    db.rollback()
            finally:
                if db:
                    db.close()

        # WebSocket broadcast logic
        if self.connection_manager:
            # Use the serialized content for broadcasting to ensure JSON safety
            try:
                broadcast_content = json.loads(serialized_content)
            except Exception:
                broadcast_content = serialized_content

            log_data = {
                "session_id": session_id,
                "sender_id": sender_id,
                "recipient_id": recipient_id,
                "message_type": message_type,
                "content": broadcast_content,
                "timestamp": datetime.utcnow().isoformat(),
                "db_logged": db_logged_successfully,
            }
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._broadcast_log_async(log_data))
            except RuntimeError:
                asyncio.run(self._broadcast_log_async(log_data))

    def register(self, agent_id: str, agent_instance):
        """Registers an agent instance with the message bus.

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

    def subscribe(self, agent_id: str, message_type: str):
        """Subscribes an agent to a specific message type for broadcast."""
        if agent_id not in self._registry:
            logger.error(f"Cannot subscribe unknown agent '{agent_id}'.")
            return
        self._subscriptions.setdefault(message_type, set()).add(agent_id)
        logger.info(f"Agent '{agent_id}' subscribed to '{message_type}' messages.")

    def unsubscribe(self, agent_id: str, message_type: str):
        """Removes an agent subscription for a message type."""
        if message_type in self._subscriptions:
            self._subscriptions[message_type].discard(agent_id)
            logger.info(f"Agent '{agent_id}' unsubscribed from '{message_type}' messages.")

    def start(self):
        """Starts the async queue processing loop if not already running."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._process_queue())

    async def stop(self):
        """Stops the queue processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _deliver(self, agent, message):
        """Internal helper to deliver a message to an agent, awaiting if coroutine."""
        if hasattr(agent, 'receive_message') and callable(agent.receive_message):
            if inspect.iscoroutinefunction(agent.receive_message):
                await agent.receive_message(message)
            else:
                agent.receive_message(message)
        else:
            logger.error(f"Agent '{getattr(agent, 'agent_id', 'unknown')}' lacks a callable 'receive_message' method.")

    async def _process_queue(self):
        """Background task that processes broadcast messages."""
        while self._running:
            message = await self._queue.get()
            msg_type = message.get('message_type')
            recipients = self._subscriptions.get(msg_type, set()).copy()
            for agent_id in recipients:
                agent = self._registry.get(agent_id)
                if agent:
                    try:
                        await self._deliver(agent, message)
                    except Exception as e:
                        logger.error(f"Error delivering broadcast to agent '{agent_id}': {e}", exc_info=True)
                else:
                    logger.warning(f"Subscribed agent '{agent_id}' not registered. Skipping.")

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
            return {"status": "error", "error": "Invalid message structure, missing keys."}

        recipient_id = message.get('recipient_id')
        sender_id = message.get('sender_id')
        message_type = message.get('message_type')
        # Ensure recipient_id is a string, even if it's None from the message, for logging consistency.
        # However, _log_message_to_db now accepts Optional[str] for recipient_id.
        # So direct pass-through is fine.

        logger.info(f"Attempting to dispatch message of type '{message_type}' from '{sender_id}' to '{recipient_id}'.")

        recipient_agent = self._registry.get(recipient_id)
        response = None

        try:
            if recipient_agent:
                if hasattr(recipient_agent, 'receive_message') and callable(recipient_agent.receive_message):
                    response = recipient_agent.receive_message(message)
                    logger.info(f"Message dispatched and received by '{recipient_id}'. Agent response: {response}")
                else:
                    error_msg = f"Recipient agent '{recipient_id}' does not have a callable 'receive_message' method."
                    logger.error(error_msg)
                    response = {"status": "error", "error": error_msg}
            else:
                error_msg = f"Recipient agent '{recipient_id}' not found."
                logger.warning(f"Recipient agent '{recipient_id}' not found in registry. Message from '{sender_id}' of type '{message_type}' could not be delivered.")
                response = {"status": "error", "error": error_msg}
        except Exception as e:
            error_msg = f"Error delivering message to agent '{recipient_id}': {str(e)}"
            logger.error(f"Error delivering message to agent '{recipient_id}': {e}", exc_info=True)
            response = {"status": "error", "error": error_msg}
        finally:
            # Log the message interaction
            payload_data = message.get('payload', {})
            session_id_val = payload_data.get('session_id')

            if isinstance(session_id_val, (str, int, float)):
                session_id_str = str(session_id_val)
            elif session_id_val is None:
                session_id_str = 'unknown_session_dispatch_none' # Specific placeholder
            else: # For complex types like dicts or lists
                session_id_str = 'unknown_session_dispatch_complex_type' # Ensure this aligns with expectations

            # recipient_id for logging can be None if it's a broadcast or system message not targeting a specific agent directly in this context
            # However, dispatch_message implies a specific recipient. If recipient_id is None here, it's unusual.
            # For now, we pass recipient_id as is.
            self._log_message_to_db(
                session_id=session_id_str, # This needs to be robust
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                content_payload=payload_data # Log the original payload
            )
        return response

    async def broadcast_message(self, message_type: str, payload: dict, sender_id: str = "system"):
        """Places a broadcast message onto the internal queue and logs it."""
        message = {
            "sender_id": sender_id,
            "recipient_id": None, # Broadcasts don't have a single recipient at this stage
            "message_type": message_type,
            "payload": payload,
        }
        await self._queue.put(message)
        self.start() # Ensure the queue processor is running

        # Log the broadcast attempt
        session_id_val = payload.get('session_id')
        if isinstance(session_id_val, (str, int, float)): # Ensure session_id is a simple type or correctly stringified
            session_id_str = str(session_id_val)
        elif session_id_val is None:
            session_id_str = 'unknown_broadcast_session_none' # Specific placeholder
        else: # For complex types
            session_id_str = 'unknown_broadcast_session_complex_type' # Ensure this aligns

        self._log_message_to_db(
            session_id=session_id_str, # This needs to be robust
            sender_id=sender_id,
            recipient_id="BROADCAST", # Explicitly mark broadcast recipient for logs
            message_type=message_type,
            content_payload=payload
        )

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

    # Broadcast demonstration
    async def demo_broadcast():
        bus.subscribe(agent_B.agent_id, "broadcast_test")
        await bus.broadcast_message("broadcast_test", {"info": "Hello subscribers"}, sender_id="AgentA")
        await asyncio.sleep(0.1)
        await bus.stop()

    asyncio.run(demo_broadcast())

    # Test invalid message
    bus.dispatch_message({"sender_id": "system", "payload": "test"})

    logger.info("MessageBus example usage finished.")
