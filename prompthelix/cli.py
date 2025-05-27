"""
Command Line Interface for the PromptHelix application.

Provides commands for interacting with the PromptHelix system,
such as running tests, managing configurations, etc.
"""
import argparse
import subprocess
import sys
import os
import unittest

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
    test_parser = subparsers.add_parser("test", help="Run all tests")
    # The --all argument is removed, test command will now always run all tests.

    # "run" command
    run_parser = subparsers.add_parser("run", help="Run the PromptHelix application or a specific module")
    run_parser.add_argument("module", nargs="?", default="ga", help="Module to run (e.g., 'ga')")

    args = parser.parse_args()

    if args.command == "test":
        print("CLI: Running all tests...")
        loader = unittest.TestLoader()
        # Determine the correct start_dir relative to the project root
        # Assuming cli.py is in prompthelix/ and tests are in prompthelix/tests
        # For the discover method, the path should be relative to the project root if running `python -m prompthelix.cli`
        # or relative to the current working directory if running the script directly.
        # Let's try to make it robust by finding the project root or using a path relative to this file.
        
        # Path to the 'tests' directory relative to this cli.py file
        # __file__ is prompthelix/cli.py, so dirname(__file__) is prompthelix/
        # tests_dir_relative_to_cli = os.path.join(os.path.dirname(__file__), "tests")

        # However, TestLoader.discover usually works best with paths relative to the project root,
        # or python package paths.
        # If 'prompthelix' is in PYTHONPATH or installed, 'prompthelix.tests' should work.
        # If running as 'python -m prompthelix.cli', CWD is usually project root.
        
        # Let's assume the tests are within the 'prompthelix' package, under a 'tests' sub-package.
        # The pattern 'test*.py' is the default for discover.
        
        # Determine the project root path (parent of the 'prompthelix' directory)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        start_dir = os.path.join(project_root, 'prompthelix', 'tests')

        # A more common way if 'prompthelix' is a package and tests are inside it:
        # suite = loader.discover('prompthelix.tests', pattern='test_*.py', top_level_dir=project_root)
        # For simplicity and directness with file paths:
        
        if not os.path.isdir(start_dir):
            print(f"Error: Test directory not found at {start_dir}", file=sys.stderr)
            print(f"Project root detected as: {project_root}", file=sys.stderr)
            print(f"Current __file__ is: {__file__}", file=sys.stderr)
            # Fallback if the structure is different, e.g., /app/tests
            alt_start_dir = os.path.join(project_root, 'tests')
            if os.path.isdir(alt_start_dir):
                start_dir = alt_start_dir
            else:
                # If the cli.py is not where we think, this might also be problematic.
                # For `python -m prompthelix.cli`, CWD is usually the project root.
                # So 'prompthelix/tests' should be discoverable from there.
                # If running `python prompthelix/cli.py`, then start_dir needs to be '../prompthelix/tests' if CWD is `prompthelix`
                # or `prompthelix/tests` if CWD is project root.
                # The most robust way is to use package discovery if tests are part of the package.
                # For discover by path, it's relative to CWD or an absolute path.
                
                # Using 'prompthelix.tests' as the start_dir for package-based discovery
                # This requires that 'prompthelix' is in sys.path (e.g. installed or PYTHONPATH set)
                # and that tests directory is a package (has __init__.py)
                # And subdirectories (unit, integration) also have __init__.py
                try:
                    print(f"Attempting package discovery for 'prompthelix.tests' from project root '{project_root}'")
                    suite = loader.discover(start_dir='prompthelix.tests', top_level_dir=project_root)
                except ImportError:
                     print(f"Package discovery failed. Ensure 'prompthelix' is in PYTHONPATH or installed.", file=sys.stderr)
                     print(f"Attempting path discovery from CWD: 'prompthelix/tests'", file=sys.stderr)
                     # This assumes CWD is project root
                     suite = loader.discover(start_dir='prompthelix/tests')


        else: # start_dir (e.g. /app/prompthelix/tests) was found
            print(f"Using path discovery from: {start_dir}")
            suite = loader.discover(start_dir)

        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)

        if result.wasSuccessful():
            print("CLI: Tests completed successfully.")
            sys.exit(0)
        else:
            print("CLI: Some tests failed.")
            sys.exit(1)

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
