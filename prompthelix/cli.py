"""
Command Line Interface for the PromptHelix application.

Provides commands for interacting with the PromptHelix system,
such as running tests, managing configurations, etc.
"""
import argparse

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
    # run_parser = subparsers.add_parser("run", help="Run the PromptHelix application or a specific module")
    # run_parser.add_argument("module", nargs="?", help="Optional module to run")

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
    # No specific action needed for --version as argparse handles it
    elif hasattr(args, 'version') and args.version:
        pass
    # If no command is given, print help
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()
