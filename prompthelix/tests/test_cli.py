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
        command = [sys.executable, "-m", "prompthelix.cli", "run"]

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
            self.assertIn("CLI: Running Genetic Algorithm...", result.stdout, "CLI start message not found.")
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

if __name__ == '__main__':
    unittest.main()
