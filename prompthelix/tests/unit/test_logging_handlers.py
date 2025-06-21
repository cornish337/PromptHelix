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


    @patch('prompthelix.logging_handlers.asyncio.get_running_loop')
    def test_emit_sends_correct_log_data_structure(self, mock_get_running_loop):
        # Force the fallback path in emit() by making get_running_loop raise RuntimeError
        mock_get_running_loop.side_effect = RuntimeError("no running event loop")

        log_message = "Test log message with <html> characters."
        expected_escaped_message = "Test log message with &lt;html&gt; characters."

        # Act
        self.logger.info(log_message)

        # Assert
        # broadcast_json is an AsyncMock, so use assert_awaited_once_with
        self.mock_connection_manager.broadcast_json.assert_awaited_once()

        # Check the arguments it was called with
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


    @patch('prompthelix.logging_handlers.asyncio.get_running_loop')
    def test_emit_html_escaping(self, mock_get_running_loop):
        # Force the fallback path in emit()
        mock_get_running_loop.side_effect = RuntimeError("no running event loop")

        test_cases = [
            ("simple", "simple"),
            ("<a>link</a>", "&lt;a&gt;link&lt;/a&gt;"),
            ("text with 'quotes' and \"double quotes\"", "text with &#x27;quotes&#x27; and &quot;double quotes&quot;"),
            ("<script>alert('xss')</script>", "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;")
        ]

        for raw_message, expected_escaped_message in test_cases:
            with self.subTest(raw_message=raw_message):
                # Reset the AsyncMock for each sub-test
                self.mock_connection_manager.broadcast_json.reset_mock()

                self.logger.info(raw_message)

                # broadcast_json is an AsyncMock, check it was awaited
                self.mock_connection_manager.broadcast_json.assert_awaited_once()
                called_with_log_data = self.mock_connection_manager.broadcast_json.call_args[0][0]
                self.assertEqual(called_with_log_data['data']['message'], expected_escaped_message)

    @patch('prompthelix.logging_handlers.asyncio.get_running_loop') # Changed patch target
    @patch('prompthelix.logging_handlers.asyncio.create_task') # Keep this to ensure it's NOT called
    def test_emit_create_task_runtime_error_falls_back_to_run(self, mock_create_task, mock_get_running_loop):
        # Simulate get_running_loop raising RuntimeError to force fallback
        mock_get_running_loop.side_effect = RuntimeError("no event loop")

        self.logger.info("runtime error test")

        mock_create_task.assert_not_called()
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

        # Patch get_running_loop to force fallback and ensure broadcast_json is awaited
        with patch('prompthelix.logging_handlers.asyncio.get_running_loop', side_effect=RuntimeError("no running event loop")) as mock_get_loop, \
             patch('prompthelix.logging_handlers.asyncio.create_task') as mock_create_task_in_try: # Ensure create_task in try block isn't called

            self.mock_connection_manager.broadcast_json.reset_mock()
            self.logger.debug("A debug message.")
            self.mock_connection_manager.broadcast_json.assert_awaited_once()
            mock_create_task_in_try.assert_not_called() # Should use fallback

            self.mock_connection_manager.broadcast_json.reset_mock()
            mock_get_loop.side_effect = RuntimeError("no running event loop") # Reset side effect for next call
            self.logger.info("An info message.")
            self.mock_connection_manager.broadcast_json.assert_awaited_once()
            mock_create_task_in_try.assert_not_called() # Should use fallback

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

        with patch('prompthelix.logging_handlers.asyncio.get_running_loop', side_effect=RuntimeError("no running event loop")) as mock_get_loop, \
             patch('prompthelix.logging_handlers.asyncio.create_task') as mock_create_task_in_try:
            self.mock_connection_manager.broadcast_json.reset_mock() # Reset before any logging

            self.logger.debug("This should be filtered by handler.")
            self.mock_connection_manager.broadcast_json.assert_not_awaited() # Changed to not_awaited
            mock_create_task_in_try.assert_not_called()

            self.logger.info("This should also be filtered by handler.")
            self.mock_connection_manager.broadcast_json.assert_not_awaited() # Changed to not_awaited
            mock_create_task_in_try.assert_not_called()

            mock_get_loop.side_effect = RuntimeError("no running event loop") # Reset for next call
            self.logger.warning("This should be processed.")
            self.mock_connection_manager.broadcast_json.assert_awaited_once()
            mock_create_task_in_try.assert_not_called()

            self.mock_connection_manager.broadcast_json.reset_mock()
            mock_get_loop.side_effect = RuntimeError("no running event loop") # Reset for next call
            self.logger.error("This should also be processed.")
            self.mock_connection_manager.broadcast_json.assert_awaited_once()
            mock_create_task_in_try.assert_not_called()

if __name__ == '__main__':
    unittest.main()
