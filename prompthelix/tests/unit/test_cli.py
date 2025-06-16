import pytest
import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import io # For capturing stdout/stderr

from prompthelix.cli import main_cli
from prompthelix.enums import ExecutionMode # Needed for asserting calls

# Test Scenarios for prompthelix.cli.main_cli

@patch('sys.stderr', new_callable=io.StringIO) # Capture stderr
@patch('sys.stdout', new_callable=io.StringIO) # Capture stdout
class TestCli:

    def setup_method(self):
        """Reset stderr/stdout captures for each test method."""
        # This is more pytest style, if using unittest.TestCase, use setUp
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    # 1. `test` command
    @patch('unittest.TestLoader')
    @patch('unittest.TextTestRunner')
    def test_command_test_successful(self, MockTextTestRunner, MockTestLoader, mock_stdout, mock_stderr):
        mock_suite = MagicMock()
        MockTestLoader.return_value.discover.return_value = mock_suite

        mock_runner_instance = MockTextTestRunner.return_value
        mock_result = MagicMock()
        mock_result.wasSuccessful.return_value = True
        mock_runner_instance.run.return_value = mock_result

        with patch.object(sys, 'argv', ['prompthelix', 'test']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        stdout_val = sys.stdout.getvalue() # Use the module-level sys.stdout
        assert "CLI: Running all tests..." in stdout_val
        MockTestLoader.return_value.discover.assert_called_once()
        # The exact path for discover might be tricky due to how cli.py calculates it.
        # For unit test, we mostly care it's called.
        # Check that it's called with a path ending in 'prompthelix/tests' or 'prompthelix.tests'
        # For now, let's check it was called with some string for start_dir.
        # discover_args, _ = MockTestLoader.return_value.discover.call_args
        # assert 'tests' in discover_args[0] # Example, might need refinement based on cli.py logic

        mock_runner_instance.run.assert_called_once_with(mock_suite)
        assert "CLI: Tests completed successfully." in stdout_val
        mock_exit.assert_called_once_with(0)

    @patch('unittest.TestLoader')
    @patch('unittest.TextTestRunner')
    def test_command_test_failure(self, MockTextTestRunner, MockTestLoader, mock_stdout, mock_stderr):
        mock_suite = MagicMock()
        MockTestLoader.return_value.discover.return_value = mock_suite

        mock_runner_instance = MockTextTestRunner.return_value
        mock_result = MagicMock()
        mock_result.wasSuccessful.return_value = False # Simulate test failure
        mock_runner_instance.run.return_value = mock_result

        with patch.object(sys, 'argv', ['prompthelix', 'test']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        stdout_val = sys.stdout.getvalue()
        assert "CLI: Running all tests..." in stdout_val
        MockTestLoader.return_value.discover.assert_called_once()
        mock_runner_instance.run.assert_called_once_with(mock_suite)
        assert "CLI: Some tests failed." in stdout_val
        mock_exit.assert_called_once_with(1)

    # 2. `run ga` command (default mode TEST)
    @patch('prompthelix.orchestrator.main_ga_loop')
    def test_command_run_ga_default_test_mode(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        mock_chromosome = MagicMock()
        mock_chromosome.fitness_score = 0.95
        mock_main_ga_loop.return_value = mock_chromosome

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga']), \
             patch.object(sys, 'exit') as mock_exit: # To catch potential exits
            main_cli()

        stdout_val = sys.stdout.getvalue()
        assert "CLI: Running Genetic Algorithm..." in stdout_val
        mock_main_ga_loop.assert_called_once_with(
            task_desc=unittest.mock.ANY, # Default task_desc
            keywords=unittest.mock.ANY,  # Default keywords
            num_generations=unittest.mock.ANY,
            population_size=unittest.mock.ANY,
            elitism_count=unittest.mock.ANY,
            execution_mode=ExecutionMode.TEST, # Crucial check
            return_best=True
        )
        assert "CLI: Genetic Algorithm completed." in stdout_val
        assert f"Best prompt fitness: {mock_chromosome.fitness_score}" in stdout_val
        # Assert that sys.exit was not called with an error code.
        # If it exits with 0, that's fine for a successful run.
        # The main concern is it doesn't exit with 1.
        if mock_exit.called:
            assert mock_exit.call_args[0][0] == 0, "sys.exit called with non-zero for successful GA run"


    # 3. `run ga --mode REAL` command
    @patch('prompthelix.orchestrator.main_ga_loop')
    def test_command_run_ga_mode_real(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        mock_chromosome = MagicMock()
        mock_main_ga_loop.return_value = mock_chromosome

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--mode', 'REAL']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=unittest.mock.ANY,
            keywords=unittest.mock.ANY,
            num_generations=unittest.mock.ANY,
            population_size=unittest.mock.ANY,
            elitism_count=unittest.mock.ANY,
            execution_mode=ExecutionMode.REAL, # Crucial check
            return_best=True
        )
        if mock_exit.called:
            assert mock_exit.call_args[0][0] == 0

    # 4. `run ga` command when `main_ga_loop` raises an exception
    @patch('prompthelix.orchestrator.main_ga_loop', side_effect=Exception("GA Error"))
    def test_command_run_ga_exception(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        stderr_val = sys.stderr.getvalue()
        assert "CLI: Error running Genetic Algorithm: GA Error" in stderr_val
        mock_exit.assert_called_once_with(1)

    # 5. `run` command with an unknown module
    def test_command_run_unknown_module(self, mock_stdout, mock_stderr):
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'unknown_module']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        stderr_val = sys.stderr.getvalue()
        assert "Error: Unknown module 'unknown_module'." in stderr_val
        mock_exit.assert_called_once_with(1)

    # 6. No command provided
    @patch('argparse.ArgumentParser.print_help')
    def test_no_command_provided(self, mock_print_help_method, mock_stdout, mock_stderr):
        # This approach mocks the method on the class.
        # It assumes that main_cli() will instantiate ArgumentParser and call print_help() on it.
        with patch.object(sys, 'argv', ['prompthelix']), \
             patch.object(sys, 'exit') as mock_exit: # Argparse might exit after print_help
            main_cli()

        mock_print_help_method.assert_called_once()
        # Depending on argparse behavior, it might exit. If so, check exit code (usually 0 or not an error).
        # If it doesn't exit, mock_exit should not be called with 1.
        # main_cli itself calls parser.print_help() if no command, so this should work.

    # 7. `--version` command
    def test_version_command(self, mock_stdout, mock_stderr):
        with patch.object(sys, 'argv', ['prompthelix', '--version']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        mock_exit.assert_called_once_with(0)
        stdout_val = sys.stdout.getvalue()
        assert "prompthelix 0.1.0" in stdout_val
