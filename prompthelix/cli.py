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

from prompthelix.enums import ExecutionMode


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
    # TODO: Future enhancement could allow specifying individual test files or modules.

    # "run" command
    run_parser = subparsers.add_parser("run", help="Run the PromptHelix application or a specific module")
    run_parser.add_argument("module", nargs="?", default="ga", help="Module to run (e.g., 'ga')")
    run_parser.add_argument(
        "--mode",
        choices=["TEST", "REAL"],
        default="TEST",
        help="Execution mode for the GA run (defaults to TEST)",
    )


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
            from prompthelix.orchestrator import main_ga_loop # Import directly
            from prompthelix.enums import ExecutionMode # Import ExecutionMode
            # Define default parameters for the CLI run
            default_task_desc = "Generate a creative story about a space explorer."
            default_keywords = ["space", "adventure", "discovery"]
            default_num_generations = 2 # Keep low for CLI example
            default_population_size = 5 # Keep low for CLI example
            default_elitism_count = 1   # Keep low for CLI example

            print(
                f"Using default parameters for GA: task='{default_task_desc}', "
                f"generations={default_num_generations}, population={default_population_size}, mode={args.mode}"
            )
            
            try:
                # Call main_ga_loop directly
                # main_ga_loop prints its own progress, including "Generation X of Y"
                best_chromosome = main_ga_loop(
                    task_desc=default_task_desc,
                    keywords=default_keywords,
                    num_generations=default_num_generations,
                    population_size=default_population_size,
                    elitism_count=default_elitism_count,
                    execution_mode=ExecutionMode.TEST, # Pass execution_mode

                    return_best=True  # Ensure it returns to potentially print results
                )
                if best_chromosome:
                    print("\nCLI: Genetic Algorithm completed.")
                    print(f"Best prompt fitness: {best_chromosome.fitness_score}")
                    # print(f"Best prompt content: {best_chromosome.to_prompt_string()}") # Could be very long
                else:
                    print("\nCLI: Genetic Algorithm completed, but no best prompt was found.")
            except Exception as e:
                print(f"CLI: Error running Genetic Algorithm: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
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
