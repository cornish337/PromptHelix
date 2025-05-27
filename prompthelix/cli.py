"""
Command Line Interface for the PromptHelix application.

Provides commands for interacting with the PromptHelix system,
such as running tests, managing configurations, etc.
"""
import argparse
import subprocess
import sys
import os

def main_cli():
    """
    Main function for the PromptHelix CLI.
    Parses arguments and dispatches commands.
    """
    parser = argparse.ArgumentParser(description="PromptHelix CLI")
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Subcommand for "test"
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument(
        "--all", action="store_true", help="Run all tests"
    )
    # Add more arguments for test_parser as needed, e.g., specific test files or modules

    # Placeholder for a "run" command
    run_parser = subparsers.add_parser("run", help="Run the PromptHelix application or a specific module")
    run_parser.add_argument("module", nargs="?", default="ga", help="Module to run (e.g., 'ga')")

    args = parser.parse_args()

    if args.command == "test":
        if args.all:
            print("CLI: Running all tests (placeholder)...")
            # Placeholder: Here you would integrate with your test runner
            # For example:
            # import unittest
            # loader = unittest.TestLoader()
            # suite = loader.discover(start_dir='prompthelix/tests/unit') # or prompthelix/tests for all
            # runner = unittest.TextTestRunner()
            # runner.run(suite)
        else:
            print("CLI: Running specific tests (placeholder)...")
            # Placeholder: Logic to run specific tests
            print("CLI: Use 'prompthelix test --all' to run all tests for now.")
    elif args.command == "run":
        if args.module == "ga":
            print("CLI: Running Genetic Algorithm...")
            # Determine the correct path to run_ga.py
            # Assuming cli.py is in prompthelix/ and run_ga.py is in prompthelix/
            script_path = os.path.join(os.path.dirname(__file__), "run_ga.py")
            
            # Check if the script_path is correct by checking if the file exists
            if not os.path.exists(script_path):
                # Fallback or error if structure is different than expected
                # For example, if cli.py is in prompthelix/ and run_ga.py is in prompthelix/genetics
                # This part might need adjustment based on actual project structure if different
                alt_script_path = os.path.join(os.path.dirname(__file__), "genetics", "run_ga.py") # Common alternative
                if os.path.exists(alt_script_path):
                    script_path = alt_script_path
                else:
                    # If run_ga.py is at the root of the project (e.g. /app/run_ga.py)
                    # and cli.py is in /app/prompthelix/cli.py
                    # then script_path should be ../run_ga.py relative to cli.py
                    proj_root_script_path = os.path.join(os.path.dirname(__file__), "..", "run_ga.py")
                    if os.path.exists(proj_root_script_path):
                         script_path = proj_root_script_path
                    else:
                        print(f"Error: Could not find run_ga.py at {script_path} or alternative paths.", file=sys.stderr)
                        sys.exit(1)
            
            try:
                process = subprocess.run(
                    [sys.executable, script_path],
                    capture_output=True,
                    text=True,
                    check=True
                )
                print("CLI: Genetic Algorithm Output:")
                print(process.stdout)
            except subprocess.CalledProcessError as e:
                print("CLI: Error running Genetic Algorithm.", file=sys.stderr)
                print("Stderr:", file=sys.stderr)
                print(e.stderr, file=sys.stderr)
            except FileNotFoundError:
                print(f"Error: The script {script_path} was not found. Ensure Python is installed and the path is correct.", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Unknown module '{args.module}'. Currently, only 'ga' module is supported for the run command.", file=sys.stderr)
            sys.exit(1)
    # No specific action needed for --version as argparse handles it
    elif hasattr(args, 'version') and args.version:
        pass
    # If no command is given, print help
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()
