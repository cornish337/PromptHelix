import unittest
import subprocess
import sys
import os

class TestCli(unittest.TestCase):

    def test_run_ga_command(self):
        """Test the 'run' command for the GA module."""
        # Construct the command to run the CLI module
        # Assumes the tests are run from the root of the project or a similar context
        # where 'python -m prompthelix.cli' is valid.
        command = [
            sys.executable,
            "-m",
            "prompthelix.cli",
            "run",
            "--parallel-workers",
            "1",  # Force serial execution for this test
            # Using default GA parameters for this test, e.g. num_generations=2, pop_size=5
            # Ensure execution mode is TEST to avoid real LLM calls if not mocked
            "--execution-mode",
            "TEST"
        ]

        try:
            # Execute the command
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True, # Will raise CalledProcessError if return code is non-zero
                timeout=30  # Add a timeout to prevent tests from hanging indefinitely
            )

            # Check for successful execution (though check=True handles this)
            self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}")

            # Check for expected output fragments
            self.assertIn("CLI: Running Genetic Algorithm with specified parameters...", result.stdout, "CLI start message not found.")
            self.assertIn("Generation", result.stdout, "Generation keyword not found in output.")
            self.assertIn("Overall Best Prompt Found", result.stdout, "Final best prompt message not found.") # Changed this line
            self.assertTrue(len(result.stdout.splitlines()) > 5, "Output seems too short.") # Basic check for some iteration

        except subprocess.CalledProcessError as e:
            self.fail(f"CLI command '{' '.join(command)}' failed with error: {e.stderr}")
        except subprocess.TimeoutExpired:
            self.fail(f"CLI command '{' '.join(command)}' timed out.")
        except FileNotFoundError:
            self.fail(f"CLI command failed. Ensure 'python -m prompthelix.cli' can be resolved. Is PYTHONPATH set up correctly or package installed?")

    def test_check_llm_command_mocked(self):
        """Test the 'check-llm' command with a mocked LLM call."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            sitecustomize_path = os.path.join(tmpdir, "sitecustomize.py")
            with open(sitecustomize_path, "w") as f:
                f.write(
                    "from unittest.mock import patch\n"
                    "patch('prompthelix.utils.llm_utils.call_llm_api', return_value='mocked-response').start()\n"
                )

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")

            command = [
                sys.executable,
                "-m",
                "prompthelix.cli",
                "check-llm",
                "--provider",
                "openai",
            ]

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("mocked-response", result.stdout)

    # Helper method to run CLI commands
    def _run_cli_command(self, command_args, timeout=30, env=None):
        base_command = [sys.executable, "-m", "prompthelix.cli"]
        full_command = base_command + command_args
        try:
            return subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                check=False, # We will check returncode manually to get more info from stderr
                timeout=timeout,
                env=(env or os.environ.copy()) # Pass provided environment
            )
        except subprocess.TimeoutExpired as e:
            self.fail(f"CLI command '{' '.join(full_command)}' timed out: {e}")
        except FileNotFoundError: # pragma: no cover
            self.fail(f"CLI command failed. Ensure '{sys.executable} -m prompthelix.cli' can be resolved.")


    def test_run_ga_with_custom_prompt_and_output_file(self):
        """Test 'run ga' with --prompt and --output-file."""
        import tempfile

        seed_prompt = "This is my integration test seed prompt."

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file_path = os.path.join(tmpdir, "test_output.txt")

            args = [
                "run", "ga",
                "--prompt", seed_prompt,
                "--output-file", output_file_path,
                "--num-generations", "1", # Keep it minimal (or even 0 if supported for just init and output)
                "--population-size", "1",  # Set to 1 to ensure only seed prompt is used
                "--elitism-count", "1", # Ensure the one prompt is kept
                "--execution-mode", "TEST" # Ensure no real LLM calls
            ]
            result = self._run_cli_command(args)

            self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}\nStdout: {result.stdout}")

            self.assertTrue(os.path.exists(output_file_path), "Output file was not created.")

            with open(output_file_path, 'r') as f:
                content = f.read()

            # In TEST mode, if initial_prompt_str is provided, PopulationManager seeds one chromosome
            # with it. If pop_size is small, and generations are few, this seeded prompt is likely
            # to be the "best" or directly related to it.
            # The PromptChromosome.to_prompt_string() joins genes with "\n".
            # Our PopulationManager currently creates the seeded chromosome with genes=[initial_prompt_str].
            self.assertEqual(content.strip(), seed_prompt.strip(),
                             f"Output file content mismatch. Expected '{seed_prompt}', got '{content}'")

    def test_run_ga_with_population_path(self):
        """Test 'run ga' with --population-path to ensure persistence file is created."""
        import tempfile
        import textwrap
        import types

        with tempfile.TemporaryDirectory() as tmpdir:
            pop_file_path = os.path.join(tmpdir, "ga_population.json")
            sitecustomize_path = os.path.join(tmpdir, "sitecustomize.py")

            sitecustomize_code = textwrap.dedent(
                """
                import types, sys, os

                def main_ga_loop(*args, **kwargs):
                    pop = kwargs.get('population_path')
                    if pop:
                        with open(pop, 'w') as f:
                            f.write('mock')
                    return None

                fake = types.ModuleType('prompthelix.orchestrator')
                fake.main_ga_loop = main_ga_loop
                sys.modules['prompthelix.orchestrator'] = fake
                """
            )

            with open(sitecustomize_path, "w") as f:
                f.write(sitecustomize_code)

            env = os.environ.copy()
            env["PYTHONPATH"] = tmpdir + os.pathsep + env.get("PYTHONPATH", "")

            args = [
                "run", "ga",
                "--population-path", pop_file_path,
                "--num-generations", "1",
                "--population-size", "2",
                "--execution-mode", "TEST"
            ]

            result = self._run_cli_command(args, timeout=30, env=env)

            self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}\nStdout: {result.stdout}")
            self.assertTrue(os.path.exists(pop_file_path), "Population file was not created.")

    def test_run_ga_with_ga_parameter_overrides(self):
        """Test 'run ga' with overridden GA parameters like num-generations."""
        args = [
            "run", "ga",
            "--num-generations", "1",
            "--population-size", "3",
            "--elitism-count", "1",
            "--execution-mode", "TEST"
        ]
        result = self._run_cli_command(args)
        self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}\nStdout: {result.stdout}")

        # Check logs for confirmation (relies on main_ga_loop logging these)
        # main_ga_loop in orchestrator.py logs:
        # logger.info(f"Num Generations: {num_generations}, Population Size: {population_size}, Elitism Count: {elitism_count}")
        # The CLI (cli.py) also logs:
        # logger.info(f"GA Parameters: Generations={num_generations}, Population Size={population_size}, Elitism Count={elitism_count}")
        # Since cli.py logs are INFO level and go to stdout by default in main_cli's logging config:
        self.assertIn("GA Parameters: Generations=1, Population Size=3, Elitism Count=1", result.stdout)


    def test_run_ga_with_agent_settings_json_string(self):
        """Test 'run ga' with --agent-settings as a JSON string."""
        agent_settings_json = '{"PromptArchitectAgent": {"default_llm_model": "integration-test-model-agent"}}'
        args = [
            "run", "ga",
            "--agent-settings", agent_settings_json,
            "--num-generations", "1",
            "--population-size", "2",
            "--execution-mode", "TEST"
        ]
        result = self._run_cli_command(args)
        self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}\nStdout: {result.stdout}")

        # Check logs from cli.py for settings loading
        self.assertIn("Loaded agent settings override from JSON string.", result.stdout)
        # Check logs from orchestrator.py (main_ga_loop) for receiving it
        # main_ga_loop logs: logger.info(f"Agent Settings Override provided: {agent_settings_override}")
        # Need to be careful about how the dict is stringified in the log.
        # A simpler check is that the log message "Agent Settings Override provided" is present.
        self.assertIn("Agent Settings Override provided", result.stdout)


    def test_run_ga_with_llm_settings_file(self):
        """Test 'run ga' with --llm-settings from a JSON file."""
        import tempfile
        import json

        llm_settings_data = {"default_timeout": 15, "default_model": "model-from-file-integration-test"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(llm_settings_data, tmp_file)
            tmp_file_path = tmp_file.name

        args = [
            "run", "ga",
            "--llm-settings", tmp_file_path,
            "--num-generations", "1",
            "--population-size", "2",
            "--execution-mode", "TEST"
        ]
        try:
            result = self._run_cli_command(args)
            self.assertEqual(result.returncode, 0, f"CLI command failed with stderr: {result.stderr}\nStdout: {result.stdout}")
            self.assertIn(f"Loaded LLM settings override from file: {tmp_file_path}", result.stdout)
            self.assertIn("LLM Settings Override provided", result.stdout)
        finally:
            os.remove(tmp_file_path)

    def test_run_ga_invalid_settings_json(self):
        """Test 'run ga' with invalid JSON for settings."""
        invalid_json_string = '{"this_is_bad_json": "value"' # Missing closing brace
        args = [
            "run", "ga",
            "--agent-settings", invalid_json_string,
            "--execution-mode", "TEST"
            # No need for num-generations etc. as it should fail before GA loop
        ]
        result = self._run_cli_command(args)

        # main_cli() in cli.py does not sys.exit(1) on JSON error, it logs an error and proceeds.
        # The GA then runs with default settings. So, returncode should be 0 if GA itself completes.
        self.assertEqual(result.returncode, 0, f"CLI command unexpectedly failed with stderr: {result.stderr}\nStdout: {result.stdout}")
        # Check for the error log message from cli.py
        self.assertIn("Invalid JSON for agent_settings", result.stdout) # Logs go to stdout by default
        self.assertIn("Using default agent settings.", result.stdout)


if __name__ == '__main__':
    unittest.main()
