# prompthelix/tests/unit/test_logging_handlers.py
import unittest
import logging
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Adjust the import path based on your project structure
from prompthelix.logging_handlers import WebSocketLogHandler
from prompthelix.websocket_manager import ConnectionManager # Assuming this is the correct path

class TestWebSocketLogHandler(unittest.TestCase):

    def setUp(self):
        # Create a mock ConnectionManager
        self.mock_connection_manager = MagicMock(spec=ConnectionManager)
        # Mock the broadcast_json method to be an AsyncMock
        self.mock_connection_manager.broadcast_json = AsyncMock()

        # Create an instance of the handler
        self.handler = WebSocketLogHandler(connection_manager=self.mock_connection_manager)

        # Create a logger and add the handler to it
        self.logger = logging.getLogger('test_websocket_logger')
        self.logger.setLevel(logging.DEBUG) # Process all levels for testing

        # Clear existing handlers and add our test handler
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.logger.addHandler(self.handler)

        # Set a simple formatter for consistent output testing if needed
        formatter = logging.Formatter('%(message)s') # Basic formatter for this test
        self.handler.setFormatter(formatter)


    @patch('asyncio.create_task')
    def test_emit_sends_correct_log_data_structure(self, mock_create_task):
        # Arrange
        log_message = "Test log message with <html> characters."
        expected_escaped_message = "Test log message with &lt;html&gt; characters."

        # Act
        # self.logger.info(log_message) # Original call, will be called again after side_effect modification

        # Assert
        # Check if asyncio.create_task was called
        # mock_create_task.assert_called_once() # This will be checked after the second log call

        # Get the coroutine passed to create_task (it's the first argument of the first call)
        # coro_call = mock_create_task.call_args[0][0]

        # To check what broadcast_json would have been called with, we need to simulate
        # the execution of the coroutine. Since broadcast_json is an AsyncMock,
        # we can check its call arguments directly after it's "awaited" by create_task.
        # However, create_task itself doesn't await here, it schedules.
        # For simplicity in a unit test, we can check the arguments passed to broadcast_json
        # by assuming create_task would eventually run it.

        # To properly test the call to an async method wrapped by create_task,
        # we can make the mock_connection_manager.broadcast_json a MagicMock (non-async)
        # if we don't want to involve asyncio loop in the test.
        # Or, if we keep it AsyncMock, we need to ensure the event loop runs.
        # Let's re-evaluate. The handler calls asyncio.create_task(self.connection_manager.broadcast_json(log_data))
        # So, mock_create_task will be called with broadcast_json(log_data).
        # The argument to mock_create_task is the coroutine object.

        # We need to check the arguments of the `broadcast_json` call.
        # The `broadcast_json` itself is an AsyncMock. When `create_task` is called
        # with `self.connection_manager.broadcast_json(log_data)`, the `log_data` is
        # evaluated at that point.
        # The call to create_task is `create_task(coroutine)`.
        # The coroutine is `self.mock_connection_manager.broadcast_json.func(log_data_arg)`.
        # This is a bit tricky to get the `log_data_arg` directly from `mock_create_task`.

        # Alternative: Check call_args on the mock_connection_manager.broadcast_json directly
        # This requires the task to have been run.
        # For unit testing, let's assume create_task will execute the passed coroutine.
        # The simplest way is to check the arguments of broadcast_json directly if create_task
        # is patched to call its argument.

        # Let's redefine mock_create_task to execute the coroutine immediately for testing
        def side_effect_for_create_task(coro):
            # This is a simplified way to run the coro for testing.
            # In a real scenario, an event loop would manage this.
            try:
                coro.send(None) # Start the coroutine
            except StopIteration:
                pass # Coroutine completed
            except Exception as e:
                print(f"Coroutine execution failed in test: {e}")
            return MagicMock() # Return a mock task object

        mock_create_task.side_effect = side_effect_for_create_task

        self.logger.info(log_message) # Log again with the new side_effect

        mock_create_task.assert_called_once() # ensure create_task was called
        self.mock_connection_manager.broadcast_json.assert_called_once()
        called_with_log_data = self.mock_connection_manager.broadcast_json.call_args[0][0]

        self.assertEqual(called_with_log_data['type'], 'debug_log')
        self.assertIn('data', called_with_log_data)
        log_payload = called_with_log_data['data']

        self.assertEqual(log_payload['level'], 'INFO')
        self.assertEqual(log_payload['message'], expected_escaped_message)
        self.assertEqual(log_payload['module'], 'test_logging_handlers') # This file
        self.assertIn('timestamp', log_payload)
        self.assertEqual(log_payload['funcName'], 'test_emit_sends_correct_log_data_structure')
        self.assertIn('lineno', log_payload)


    @patch('asyncio.create_task')
    def test_emit_html_escaping(self, mock_create_task):
        original_broadcast_json = self.mock_connection_manager.broadcast_json
        sync_broadcast_mock = MagicMock(name="sync_broadcast_mock_for_html_escaping")

        # Define a side effect for create_task that simply executes the coroutine.
        # The key is that self.mock_connection_manager.broadcast_json will be
        # the sync_broadcast_mock when the logger.info call happens.
        def run_coro_immediately_side_effect(coro):
            try:
                coro.send(None)  # Start/run the coroutine
            except StopIteration:
                pass  # Coroutine completed
            except Exception as e:
                print(f"Coroutine execution failed in test_emit_html_escaping: {e}")
            return MagicMock(name="mock_task_object")

        mock_create_task.side_effect = run_coro_immediately_side_effect

        test_cases = [
            ("simple", "simple"),
            ("<a>link</a>", "&lt;a&gt;link&lt;/a&gt;"),
            ("text with 'quotes' and \"double quotes\"", "text with &#x27;quotes&#x27; and &quot;double quotes&quot;"),
            ("<script>alert('xss')</script>", "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;")
        ]

        try:
            # Replace the ConnectionManager's broadcast_json with our sync mock BEFORE logging
            self.mock_connection_manager.broadcast_json = sync_broadcast_mock

            for raw_message, expected_escaped_message in test_cases:
                with self.subTest(raw_message=raw_message):
                    sync_broadcast_mock.reset_mock() # Reset for each sub-test
                    mock_create_task.reset_mock() # Reset create_task mock as well

                    self.logger.info(raw_message)

                    mock_create_task.assert_called_once() # Ensure create_task was involved
                    sync_broadcast_mock.assert_called_once() # Crucial: check the sync mock

                    called_with_log_data = sync_broadcast_mock.call_args[0][0]
                    self.assertEqual(called_with_log_data['data']['message'], expected_escaped_message)
        finally:
            # Restore the original broadcast_json mock AFTER all tests are done
            self.mock_connection_manager.broadcast_json = original_broadcast_json

    @patch('asyncio.create_task')
    def test_emit_create_task_runtime_error_falls_back_to_run(self, mock_create_task):
        # Simulate create_task raising RuntimeError
        mock_create_task.side_effect = RuntimeError("no event loop")

        self.logger.info("runtime error test")

        mock_create_task.assert_called_once()
        # broadcast_json should still be awaited via asyncio.run
        self.mock_connection_manager.broadcast_json.assert_awaited_once()


    def test_handler_level_respects_logger_level(self):
        # This test is more about logging configuration than the handler itself.
        # The handler itself doesn't filter by level initially if its own level is NOTSET;
        # that's the logger's job. If the handler has a specific level set, it filters further.

        # Set handler level to a low level (DEBUG) so it doesn't filter out messages passed by logger
        self.handler.setLevel(logging.DEBUG)
        # Ensure logger is also at DEBUG to pass messages through
        self.logger.setLevel(logging.DEBUG)

        # Temporarily remove and re-add handler to make sure level settings are applied if there's caching
        self.logger.removeHandler(self.handler)
        self.logger.addHandler(self.handler)

        def simple_coro_runner(coro):
            try:
                coro.send(None)
            except StopIteration:
                pass
            return MagicMock()

        with patch('asyncio.create_task', MagicMock(side_effect=simple_coro_runner)) as mock_task:
            self.mock_connection_manager.broadcast_json.reset_mock()
            self.logger.debug("A debug message.")
            self.mock_connection_manager.broadcast_json.assert_called_once()

            self.mock_connection_manager.broadcast_json.reset_mock()
            self.logger.info("An info message.")
            self.mock_connection_manager.broadcast_json.assert_called_once()

    def test_handler_filters_by_its_own_level(self):
        self.handler.setLevel(logging.WARNING) # Handler will only process WARNING and above
        self.logger.setLevel(logging.DEBUG) # Logger allows DEBUG and above (passes more to handler)

        # Ensure handler is fresh with this level
        self.logger.removeHandler(self.handler)
        self.logger.addHandler(self.handler)

        def simple_coro_runner(coro):
            try:
                coro.send(None)
            except StopIteration:
                pass
            return MagicMock()

        with patch('asyncio.create_task', MagicMock(side_effect=simple_coro_runner)) as mock_task:
            self.mock_connection_manager.broadcast_json.reset_mock()

            self.logger.debug("This should be filtered by handler.")
            self.mock_connection_manager.broadcast_json.assert_not_called()

            self.logger.info("This should also be filtered by handler.")
            self.mock_connection_manager.broadcast_json.assert_not_called()

            self.logger.warning("This should be processed.")
            self.mock_connection_manager.broadcast_json.assert_called_once()

            self.mock_connection_manager.broadcast_json.reset_mock()
            self.logger.error("This should also be processed.")
            self.mock_connection_manager.broadcast_json.assert_called_once()

if __name__ == '__main__':
    unittest.main()
