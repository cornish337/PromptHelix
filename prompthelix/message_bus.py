import logging
import asyncio
import inspect

# Configure basic logging for the message bus
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageBus:
    """
    A simple message bus for inter-agent communication.
    """
    def __init__(self):
        self._registry = {}
        self._subscriptions = {}
        self._queue = asyncio.Queue()
        self._task = None
        self._running = False
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

        logger.info(f"Attempting to dispatch message of type '{message_type}' from '{sender_id}' to '{recipient_id}'.")

        recipient_agent = self._registry.get(recipient_id)

        if recipient_agent:
            try:
                if hasattr(recipient_agent, 'receive_message') and callable(recipient_agent.receive_message):
                    response = recipient_agent.receive_message(message)
                    logger.info(f"Message dispatched and received by '{recipient_id}'. Agent response: {response}")
                    return response
                else:
                    logger.error(f"Recipient agent '{recipient_id}' does not have a callable 'receive_message' method.")
                    return {"status": "error", "error": f"Recipient agent '{recipient_id}' does not have a callable 'receive_message' method."}
            except Exception as e:
                logger.error(f"Error delivering message to agent '{recipient_id}': {e}", exc_info=True)
                return {"status": "error", "error": f"Error delivering message to agent '{recipient_id}': {str(e)}"}
        else:
            logger.warning(f"Recipient agent '{recipient_id}' not found in registry. Message from '{sender_id}' of type '{message_type}' could not be delivered.")
            return {"status": "error", "error": f"Recipient agent '{recipient_id}' not found."}

    async def broadcast_message(self, message_type: str, payload: dict, sender_id: str = "system"):
        """Places a broadcast message onto the internal queue."""
        message = {
            "sender_id": sender_id,
            "recipient_id": None,
            "message_type": message_type,
            "payload": payload,
        }
        await self._queue.put(message)
        self.start()

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
