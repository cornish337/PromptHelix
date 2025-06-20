import pytest
import unittest
from unittest.mock import patch, MagicMock, ANY
import sys
import io # For capturing stdout/stderr
import json # For mocking JSONDecodeError

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
    @patch('prompthelix.cli.main_ga_loop') # Corrected patch path
    def test_command_run_ga_all_defaults(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        mock_chromosome = MagicMock()
        mock_chromosome.fitness_score = 0.95
        mock_chromosome.to_prompt_string = MagicMock(return_value="Best prompt content")
        mock_main_ga_loop.return_value = mock_chromosome

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        stdout_val = sys.stdout.getvalue()
        # Check for new log message pattern from cli.py
        assert "CLI: Initializing Genetic Algorithm run..." in stdout_val

        expected_task_desc = "Generate a creative story about a space explorer." # Default from cli.py
        expected_keywords = ["space", "adventure", "discovery"] # Default from cli.py
        expected_num_generations = 2 # Default from cli.py
        expected_population_size = 5 # Default from cli.py
        expected_elitism_count = 1   # Default from cli.py

        mock_main_ga_loop.assert_called_once_with(
            task_desc=expected_task_desc,
            keywords=expected_keywords,
            num_generations=expected_num_generations,
            population_size=expected_population_size,
            elitism_count=expected_elitism_count,
            execution_mode=ExecutionMode.TEST, # Default from argparse
            initial_prompt_str=None,
            # agent_settings_override=None, # These will be passed from orchestrator.py in future
            # llm_settings_override=None,   # These will be passed from orchestrator.py in future
            return_best=True
        )
        # As per cli.py, if output_file is not specified, it prints fitness
        assert f"Best prompt fitness: {mock_chromosome.fitness_score}" in stdout_val
        # assert "CLI: Genetic Algorithm completed successfully." in stdout_val # Logger info
        # Assert that sys.exit was not called with an error code.
        # If it exits with 0, that's fine for a successful run.
        # The main concern is it doesn't exit with 1.
        if mock_exit.called:
            assert mock_exit.call_args[0][0] == 0, "sys.exit called with non-zero for successful GA run"


    # 3. `run ga --execution-mode REAL` command
    @patch('prompthelix.cli.main_ga_loop') # Corrected patch path
    def test_command_run_ga_execution_mode_real(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        mock_chromosome = MagicMock()
        mock_chromosome.fitness_score = 0.90
        mock_chromosome.to_prompt_string = MagicMock(return_value="Real prompt")
        mock_main_ga_loop.return_value = mock_chromosome

        # Note: The old test used '--mode REAL'. The new argument is '--execution-mode REAL'.
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--execution-mode', 'REAL']), \
             patch.object(sys, 'exit') as mock_exit:
            main_cli()

        expected_task_desc = "Generate a creative story about a space explorer."
        expected_keywords = ["space", "adventure", "discovery"]
        expected_num_generations = 2
        expected_population_size = 5
        expected_elitism_count = 1

        mock_main_ga_loop.assert_called_once_with(
            task_desc=expected_task_desc,
            keywords=expected_keywords,
            num_generations=expected_num_generations,
            population_size=expected_population_size,
            elitism_count=expected_elitism_count,
            execution_mode=ExecutionMode.REAL, # Crucial check
            initial_prompt_str=None,
            return_best=True
        )
        if mock_exit.called:
            assert mock_exit.call_args[0][0] == 0

    @patch('prompthelix.cli.main_ga_loop')
    def test_command_run_ga_custom_prompt(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        custom_prompt = "This is my custom starting prompt."
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--prompt', custom_prompt]):
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY, # Defaults
            execution_mode=ExecutionMode.TEST, # Default
            initial_prompt_str=custom_prompt, # Check this
            return_best=True
        )

    @patch('prompthelix.cli.main_ga_loop')
    def test_command_run_ga_custom_task_and_keywords(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        custom_task = "Design a new cookie recipe."
        custom_keywords = ["cookie", "baking", "sweet"]
        argv = ['prompthelix', 'run', 'ga', '--task-description', custom_task, '--keywords'] + custom_keywords
        with patch.object(sys, 'argv', argv):
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=custom_task, # Check this
            keywords=custom_keywords, # Check this
            num_generations=ANY, population_size=ANY, elitism_count=ANY, # Defaults
            execution_mode=ExecutionMode.TEST, # Default
            initial_prompt_str=None, # Default
            return_best=True
        )

    @patch('prompthelix.cli.main_ga_loop')
    def test_command_run_ga_custom_ga_params(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        custom_gens = 10
        custom_pop = 20
        custom_elitism = 3
        argv = [
            'prompthelix', 'run', 'ga',
            '--num-generations', str(custom_gens),
            '--population-size', str(custom_pop),
            '--elitism-count', str(custom_elitism)
        ]
        with patch.object(sys, 'argv', argv):
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, # Defaults
            num_generations=custom_gens, # Check this
            population_size=custom_pop, # Check this
            elitism_count=custom_elitism, # Check this
            execution_mode=ExecutionMode.TEST, # Default
            initial_prompt_str=None, # Default
            return_best=True
        )

    @patch('prompthelix.orchestrator.main_ga_loop')
    def test_command_run_ga_parallel_workers(self, mock_main_ga_loop, mock_stdout, mock_stderr):
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--parallel-workers', '4']):
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY,
            keywords=ANY,
            num_generations=ANY,
            population_size=ANY,
            elitism_count=ANY,
            execution_mode=ExecutionMode.TEST,
            initial_prompt_str=None,
            agent_settings_override=None,
            llm_settings_override=None,
            parallel_workers=4,
            return_best=True
        )

    @patch('prompthelix.cli.main_ga_loop')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_command_run_ga_output_file(self, mock_file_open, mock_main_ga_loop, mock_stdout, mock_stderr):
        output_filename = "test_output.txt"
        prompt_content = "This is the best prompt."

        mock_chromosome = MagicMock()
        mock_chromosome.fitness_score = 0.99
        mock_chromosome.to_prompt_string = MagicMock(return_value=prompt_content)
        mock_main_ga_loop.return_value = mock_chromosome

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--output-file', output_filename]):
            main_cli()

        # Defaults for other params
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None, return_best=True
        )
        mock_file_open.assert_called_once_with(output_filename, 'w')
        mock_file_open().write.assert_called_once_with(prompt_content)

    @patch('prompthelix.cli.main_ga_loop')
    @patch('json.loads') # To mock json.loads for string input
    @patch('os.path.isfile', return_value=False) # Ensure it's treated as a string
    @patch('prompthelix.cli.logger.info') # To check logging
    def test_command_run_ga_agent_settings_json_string(self, mock_logger_info, mock_os_isfile, mock_json_loads, mock_main_ga_loop, mock_stdout, mock_stderr):
        settings_str = '{"MetaLearnerAgent": {"key": "value"}}'
        parsed_settings = {"MetaLearnerAgent": {"key": "value"}}
        mock_json_loads.return_value = parsed_settings

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--agent-settings', settings_str]):
            main_cli()

        mock_json_loads.assert_called_once_with(settings_str)
        # cli.py currently loads/logs settings but doesn't pass them to main_ga_loop yet.
        # This will be changed when cli.py is updated to pass them.
        # For now, main_ga_loop is called with None for these override args.
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None,
            agent_settings_override=parsed_settings, # Check that parsed settings are passed
            llm_settings_override=None, # Not testing this override here
            return_best=True
        )
        mock_logger_info.assert_any_call("Loaded agent settings override from JSON string.")

    @patch('prompthelix.cli.main_ga_loop')
    @patch('os.path.isfile', return_value=True) # To simulate file existence
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"FileSystemAgent": {"file_key": "file_value"}}')
    @patch('json.load') # To mock json.load from file
    @patch('prompthelix.cli.logger.info') # To check logging
    def test_command_run_ga_agent_settings_json_file(self, mock_logger_info, mock_json_load_file, mock_open_file, mock_os_isfile, mock_main_ga_loop, mock_stdout, mock_stderr):
        settings_filepath = "fake_agent_config.json"
        parsed_settings_from_file = {"FileSystemAgent": {"file_key": "file_value"}}
        mock_json_load_file.return_value = parsed_settings_from_file

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--agent-settings', settings_filepath]):
            main_cli()

        mock_os_isfile.assert_called_once_with(settings_filepath)
        mock_open_file.assert_called_once_with(settings_filepath, 'r')
        mock_json_load_file.assert_called_once_with(mock_open_file())
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None,
            agent_settings_override=parsed_settings_from_file, # Check that parsed settings are passed
            llm_settings_override=None, # Not testing this override here
            return_best=True
        )
        mock_logger_info.assert_any_call(f"Loaded agent settings override from file: {settings_filepath}")

    @patch('prompthelix.cli.main_ga_loop')
    @patch('os.path.isfile', return_value=False) # Ensure it's treated as a string
    @patch('json.loads', side_effect=json.JSONDecodeError("Bad JSON", "doc", 0))
    @patch('prompthelix.cli.logger.error') # Mock the logger for error
    def test_command_run_ga_agent_settings_invalid_json_string(self, mock_logger_error, mock_json_loads, mock_os_isfile, mock_main_ga_loop, mock_stdout, mock_stderr):
        invalid_settings_str = '{"MetaLearnerAgent": "value"' # Missing closing brace

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--agent-settings', invalid_settings_str]):
            main_cli()

        mock_json_loads.assert_called_once_with(invalid_settings_str)
        mock_logger_error.assert_any_call(f"Invalid JSON for agent_settings: {mock_json_loads.side_effect}. Using default agent settings.")
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None, return_best=True
        )

    @patch('prompthelix.cli.main_ga_loop')
    @patch('json.loads')
    @patch('os.path.isfile', return_value=False)
    @patch('prompthelix.cli.logger.info')
    def test_command_run_ga_llm_settings_json_string(self, mock_logger_info, mock_os_isfile, mock_json_loads, mock_main_ga_loop, mock_stdout, mock_stderr):
        settings_str = '{"temperature": 0.8}'
        parsed_settings = {"temperature": 0.8}
        mock_json_loads.return_value = parsed_settings

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--llm-settings', settings_str]):
            main_cli()

        mock_json_loads.assert_called_once_with(settings_str)
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None,
            agent_settings_override=None, # Not testing this override here
            llm_settings_override=parsed_settings, # Check that parsed settings are passed
            return_best=True
        )
        mock_logger_info.assert_any_call("Loaded LLM settings override from JSON string.")

    @patch('prompthelix.cli.main_ga_loop')
    @patch('os.path.isfile', return_value=True)
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"max_tokens": 150}')
    @patch('json.load')
    @patch('prompthelix.cli.logger.info')
    def test_command_run_ga_llm_settings_json_file(self, mock_logger_info, mock_json_load_file, mock_open_file, mock_os_isfile, mock_main_ga_loop, mock_stdout, mock_stderr):
        settings_filepath = "fake_llm_config.json"
        parsed_settings_from_file = {"max_tokens": 150}
        mock_json_load_file.return_value = parsed_settings_from_file

        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--llm-settings', settings_filepath]):
            main_cli()

        mock_os_isfile.assert_called_once_with(settings_filepath)
        mock_open_file.assert_called_once_with(settings_filepath, 'r')
        mock_json_load_file.assert_called_once_with(mock_open_file())
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None,
            agent_settings_override=None, # Not testing this override here
            llm_settings_override=parsed_settings_from_file, # Check that parsed settings are passed
            return_best=True
        )
        mock_logger_info.assert_any_call(f"Loaded LLM settings override from file: {settings_filepath}")

    @patch('prompthelix.cli.main_ga_loop')
    @patch('os.path.isfile', return_value=False)
    @patch('json.loads', side_effect=json.JSONDecodeError("Bad LLM JSON", "doc", 0))
    @patch('prompthelix.cli.logger.error')
    def test_command_run_ga_llm_settings_invalid_json_string(self, mock_logger_error, mock_json_loads, mock_os_isfile, mock_main_ga_loop, mock_stdout, mock_stderr):
        invalid_settings_str = '{"temperature": "high"' # Missing closing brace
        with patch.object(sys, 'argv', ['prompthelix', 'run', 'ga', '--llm-settings', invalid_settings_str]):
            main_cli()

        mock_json_loads.assert_called_once_with(invalid_settings_str)
        mock_logger_error.assert_any_call(f"Invalid JSON for llm_settings: {mock_json_loads.side_effect}. Using default LLM settings.")
        mock_main_ga_loop.assert_called_once_with(
            task_desc=ANY, keywords=ANY, num_generations=ANY, population_size=ANY, elitism_count=ANY,
            execution_mode=ExecutionMode.TEST, initial_prompt_str=None, return_best=True
        )

    @patch('prompthelix.cli.main_ga_loop')
    @patch('os.path.isfile', return_value=False) # Mock for settings strings
    @patch('json.loads') # Mock json.loads for settings strings
    @patch('prompthelix.cli.logger.info') # To check logging of settings
    def test_command_run_ga_combined_args(self, mock_logger_info, mock_json_loads, mock_os_isfile, mock_main_ga_loop, mock_stdout, mock_stderr):
        custom_prompt = "Combined prompt"
        custom_task = "Combined task"
        custom_keywords = ["combo1", "combo2"]
        custom_gens = 7
        custom_pop = 17
        custom_elitism = 2
        custom_exec_mode = ExecutionMode.REAL
        output_filename = "combo_output.txt"

        agent_settings_str = '{"SomeAgent": {"p1": "v1"}}'
        llm_settings_str = '{"t": 0.77}'

        # Simulate json.loads returning different values for agent and llm settings
        parsed_agent_settings = {"SomeAgent": {"p1": "v1"}}
        parsed_llm_settings = {"t": 0.77}
        mock_json_loads.side_effect = [parsed_agent_settings, parsed_llm_settings]

        mock_chromosome = MagicMock()
        mock_chromosome.fitness_score = 0.88
        mock_chromosome.to_prompt_string = MagicMock(return_value="Combined prompt result")
        mock_main_ga_loop.return_value = mock_chromosome

        argv = [
            'prompthelix', 'run', 'ga',
            '--prompt', custom_prompt,
            '--task-description', custom_task,
            '--keywords'] + custom_keywords + [
            '--num-generations', str(custom_gens),
            '--population-size', str(custom_pop),
            '--elitism-count', str(custom_elitism),
            '--execution-mode', custom_exec_mode.name,
            '--output-file', output_filename,
            '--agent-settings', agent_settings_str,
            '--llm-settings', llm_settings_str
        ]

        with patch.object(sys, 'argv', argv), \
             patch('builtins.open', new_callable=unittest.mock.mock_open) as mock_file_open_combined:
            main_cli()

        mock_main_ga_loop.assert_called_once_with(
            task_desc=custom_task,
            keywords=custom_keywords,
            num_generations=custom_gens,
            population_size=custom_pop,
            elitism_count=custom_elitism,
            execution_mode=custom_exec_mode,
            initial_prompt_str=custom_prompt,
            agent_settings_override=parsed_agent_settings, # Check this
            llm_settings_override=parsed_llm_settings,   # Check this
            return_best=True
        )

        # Check settings processing
        # os.path.isfile would be called twice (once for agent_settings, once for llm_settings)
        assert mock_os_isfile.call_count == 2
        # json.loads would be called twice
        mock_json_loads.assert_any_call(agent_settings_str)
        mock_json_loads.assert_any_call(llm_settings_str)
        assert mock_json_loads.call_count == 2

        mock_logger_info.assert_any_call("Loaded agent settings override from JSON string.")
        mock_logger_info.assert_any_call("Loaded LLM settings override from JSON string.")

        # Check output file writing
        mock_file_open_combined.assert_called_once_with(output_filename, 'w')
        mock_file_open_combined().write.assert_called_once_with(mock_chromosome.to_prompt_string())


    # 4. `run ga` command when `main_ga_loop` raises an exception
    @patch('prompthelix.cli.main_ga_loop', side_effect=Exception("GA Error")) # Corrected patch path
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

    def test_run_command_help_message_includes_new_options(self, mock_stdout, mock_stderr):
        # Test that `prompthelix run --help` shows the new GA options
        with patch.object(sys, 'argv', ['prompthelix', 'run', '--help']), \
             patch.object(sys, 'exit') as mock_exit: # argparse help usually exits
            main_cli()

        mock_exit.assert_called_once_with(0) # Successful help display should exit with 0

        help_text = sys.stdout.getvalue()

        # Check for a representative set of the new arguments' help strings
        assert "--prompt PROMPT" in help_text
        assert "Custom prompt string for the GA." in help_text

        assert "--task-description TASK_DESCRIPTION" in help_text
        assert "Detailed task description for the GA." in help_text

        assert "--keywords KEYWORDS [KEYWORDS ...]" in help_text
        assert "Keywords for the GA." in help_text

        assert "--num-generations NUM_GENERATIONS" in help_text
        assert "Number of generations for the GA." in help_text

        assert "--population-size POPULATION_SIZE" in help_text
        assert "Population size for the GA." in help_text

        assert "--elitism-count ELITISM_COUNT" in help_text
        assert "Elitism count for the GA." in help_text

        assert "--output-file OUTPUT_FILE" in help_text
        assert "File path to save the best prompt." in help_text

        assert "--agent-settings AGENT_SETTINGS" in help_text
        assert "JSON string or file path to override agent configurations." in help_text

        assert "--llm-settings LLM_SETTINGS" in help_text
        assert "JSON string or file path to override LLM utility settings." in help_text

        assert "--execution-mode {TEST,REAL}" in help_text # Argparse choices format
        assert "Set the execution mode for the GA (TEST or REAL)." in help_text

        # Also check for the existing 'module' argument
        assert "module" in help_text # The positional argument itself
        assert "Module to run (e.g., 'ga')" in help_text
