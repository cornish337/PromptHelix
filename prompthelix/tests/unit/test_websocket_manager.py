import asyncio
import unittest
from unittest.mock import AsyncMock, patch

# Adjust import path based on your project structure
from prompthelix.websocket_manager import ConnectionManager

class TestConnectionManager(unittest.TestCase):

    def setUp(self):
        self.manager = ConnectionManager()
        # Create a new event loop for each test if running methods that use asyncio.create_task
        # self.loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(self.loop)

    # def tearDown(self):
    #     self.loop.close()

    async def test_connect(self):
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock() # Ensure accept is also an AsyncMock if called directly

        await self.manager.connect(mock_websocket)

        mock_websocket.accept.assert_called_once()
        self.assertIn(mock_websocket, self.manager.active_connections)

    async def test_disconnect(self):
        mock_websocket = AsyncMock()

        # First connect the websocket
        await self.manager.connect(mock_websocket)
        self.assertIn(mock_websocket, self.manager.active_connections)

        # Then disconnect
        self.manager.disconnect(mock_websocket)
        self.assertNotIn(mock_websocket, self.manager.active_connections)

    async def test_disconnect_not_connected(self):
        mock_websocket = AsyncMock()
        # Ensure disconnecting a non-connected websocket doesn't raise error and list remains empty
        self.manager.disconnect(mock_websocket)
        self.assertNotIn(mock_websocket, self.manager.active_connections)
        self.assertEqual(len(self.manager.active_connections), 0)

    async def test_send_personal_message_text(self):
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        message = "Hello there"

        await self.manager.send_personal_message(message, mock_websocket)
        mock_websocket.send_text.assert_called_once_with(message)

    async def test_send_personal_json(self):
        mock_websocket = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        data = {"key": "value", "num": 123}

        await self.manager.send_personal_json(data, mock_websocket)
        mock_websocket.send_json.assert_called_once_with(data)

    async def test_broadcast_text(self):
        mock_ws1 = AsyncMock()
        mock_ws1.send_text = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.send_text = AsyncMock()

        await self.manager.connect(mock_ws1)
        await self.manager.connect(mock_ws2)

        message = "Broadcast test"
        await self.manager.broadcast(message)

        mock_ws1.send_text.assert_called_once_with(message)
        mock_ws2.send_text.assert_called_once_with(message)

    async def test_broadcast_text_to_specific_websocket_still_works(self):
        # This test is slightly misnamed based on typical broadcast,
        # it's more like send_personal_message if it's to one.
        # If broadcast is meant for all, this test should check all active connections.
        # Assuming it's testing that send_text is called on a connected websocket.
        mock_websocket = AsyncMock()
        mock_websocket.send_text = AsyncMock() # send_text for string messages

        await self.manager.connect(mock_websocket) # Connect it first

        message = "Specific message"
        # To test broadcast to all, use self.manager.broadcast(message)
        # To test sending to one, use self.manager.send_personal_message(message, mock_websocket)
        # Let's assume the intent was to test broadcast to a single connected client (as part of the group)
        await self.manager.broadcast(message)

        mock_websocket.send_text.assert_called_once_with(message)


    async def test_broadcast_json(self):
        mock_ws1 = AsyncMock()
        mock_ws1.send_json = AsyncMock()
        mock_ws2 = AsyncMock()
        mock_ws2.send_json = AsyncMock()

        await self.manager.connect(mock_ws1)
        await self.manager.connect(mock_ws2)

        data = {"message": "Broadcast JSON test"}
        await self.manager.broadcast_json(data)

        mock_ws1.send_json.assert_called_once_with(data)
        mock_ws2.send_json.assert_called_once_with(data)

    async def test_broadcast_to_no_clients(self):
        # Ensure broadcasting to no clients doesn't raise an error
        try:
            await self.manager.broadcast("No one to send to")
            await self.manager.broadcast_json({"data": "empty"})
        except Exception as e:
            self.fail(f"Broadcast to no clients raised an exception: {e}")

# To run these tests using asyncio, you might need a runner:
async def main():
    # This is a simple way to run, for more complex suites, use pytest-asyncio or similar
    suite = unittest.TestSuite()
    # Need to use TestLoader to load async tests properly or use a test runner that supports async tests.
    # For simplicity, let's assume a runner like pytest-asyncio or similar will handle this.
    # If running directly:
    # suite.addTest(TestConnectionManager("test_connect")) # and so on for all async tests
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
    # This direct way won't work well with async def test_ methods without adaptation.
    # The tests are structured for pytest or a similar runner that handles async tests.
    pass


if __name__ == '__main__':
    # This basic execution won't correctly run async unittest methods.
    # Use pytest: `pytest path/to/your/test_websocket_manager.py`
    # Ensure you have pytest and pytest-asyncio installed:
    # pip install pytest pytest-asyncio
    unittest.main() # This will likely error for async def tests without pytest-asyncio
