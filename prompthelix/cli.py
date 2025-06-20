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
import logging # Added for logging configuration
try:
    import openai  # Used for catching openai.RateLimitError during GA runs
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class _DummyRateLimitError(Exception):
        pass

    openai = type("openai", (), {"RateLimitError": _DummyRateLimitError})()
import json # For parsing settings overrides

logger = logging.getLogger(__name__)


def main_cli():
    """
    Main function for the PromptHelix CLI.
    Parses arguments and dispatches commands.
    """
    # Configure logging for CLI visibility
    # Set up basic configuration for the root logger.
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout  # Ensure logs go to stdout
    )
    # Control verbosity of noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="PromptHelix CLI")
    parser.add_argument(
        "--version", action="version", version="%(prog)s 0.1.0"
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Subcommand for "test"
    test_parser = subparsers.add_parser("test", help="Run tests")
    # Allow specifying a path to limit discovery to a subset of tests
    test_parser.add_argument(
        "--path",
        "-p",
        help="Optional directory or pattern to discover tests under",
    )
    test_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive tests located under prompthelix/tests/interactive",
    )

    # "run" command
    run_parser = subparsers.add_parser("run", help="Run the PromptHelix application or a specific module")
    run_parser.add_argument("module", nargs="?", default="ga", help="Module to run (e.g., 'ga')")
    run_parser.add_argument("--prompt", type=str, help="Custom prompt string for the GA.")
    run_parser.add_argument("--task-description", type=str, help="Detailed task description for the GA.")
    run_parser.add_argument("--keywords", type=str, nargs='+', help="Keywords for the GA.")
    run_parser.add_argument("--num-generations", type=int, help="Number of generations for the GA.")
    run_parser.add_argument("--parallel-workers", type=int, default=None, help="Number of parallel workers for fitness evaluation. 1 for serial execution. Default: None (uses os.cpu_count() or similar).")
    run_parser.add_argument("--population-size", type=int, help="Population size for the GA.")
    run_parser.add_argument("--elitism-count", type=int, help="Elitism count for the GA.")
    run_parser.add_argument("--population-path", type=str,
                            help="File path to load/save GA population state.")
    run_parser.add_argument(
        "--population-file",
        type=str,
        dest="population_path",
        help="Alias for --population-path. File to load/save GA population state.",
    )
    run_parser.add_argument("--output-file", type=str, help="File path to save the best prompt.")
    run_parser.add_argument("--agent-settings", type=str, help="JSON string or file path to override agent configurations.")
    run_parser.add_argument("--llm-settings", type=str, help="JSON string or file path to override LLM utility settings.")
    run_parser.add_argument("--execution-mode", type=str, choices=['TEST', 'REAL'], default='TEST', help="Set the execution mode for the GA (TEST or REAL).")

    # "check-llm" command for quick connectivity testing
    check_parser = subparsers.add_parser("check-llm", help="Test LLM provider connectivity")
    check_parser.add_argument("--provider", default="openai", help="LLM provider name")
    check_parser.add_argument("--model", help="Model name for the provider")


    args = parser.parse_args()

    if args.command == "test":
        loader = unittest.TestLoader()
        if args.path:
            print(f"CLI: Running tests from {args.path}...")
            suite = loader.discover(start_dir=args.path)
        else:
            if args.interactive:
                print("CLI: Running interactive tests...")
            else:
                print("CLI: Running all tests...")
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
            if args.interactive:
                start_dir = os.path.join(project_root, 'prompthelix', 'tests', 'interactive')
            else:
                start_dir = os.path.join(project_root, 'prompthelix', 'tests')

            # A more common way if 'prompthelix' is a package and tests are inside it:
            # suite = loader.discover('prompthelix.tests', pattern='test_*.py', top_level_dir=project_root)
            # For simplicity and directness with file paths:

            if not os.path.isdir(start_dir):
                print(f"Error: Test directory not found at {start_dir}", file=sys.stderr)
                print(f"Project root detected as: {project_root}", file=sys.stderr)
                print(f"Current __file__ is: {__file__}", file=sys.stderr)
                # Fallback if the structure is different, e.g., /app/tests
                alt_start_dir = os.path.join(project_root, 'tests', 'interactive') if args.interactive else os.path.join(project_root, 'tests')
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
                        print(
                            "Package discovery failed. Ensure 'prompthelix' is in PYTHONPATH or installed.",
                            file=sys.stderr,
                        )
                        print(
                            "Attempting path discovery from CWD: 'prompthelix/tests'",
                            file=sys.stderr,
                        )
                        # This assumes CWD is project root
                        package_dir = 'prompthelix/tests/interactive' if args.interactive else 'prompthelix/tests'
                        suite = loader.discover(start_dir=package_dir)

            else:  # start_dir (e.g. /app/prompthelix/tests) was found
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
            logger.info("CLI: Initializing Genetic Algorithm run...")
            from prompthelix.orchestrator import main_ga_loop # Import directly
            from prompthelix.enums import ExecutionMode # Import ExecutionMode

            # Prepare parameters for main_ga_loop, using defaults if not provided
            task_desc = args.task_description if args.task_description else "Generate a creative story about a space explorer."
            keywords = args.keywords if args.keywords else ["space", "adventure", "discovery"]
            num_generations = args.num_generations if args.num_generations is not None else 2
            population_size = args.population_size if args.population_size is not None else 5
            elitism_count = args.elitism_count if args.elitism_count is not None else 1
            initial_prompt_str = args.prompt # Can be None

            try:
                execution_mode = ExecutionMode[args.execution_mode.upper()]
            except KeyError:
                logger.error(f"Invalid execution mode: {args.execution_mode}. Defaulting to TEST.")
                execution_mode = ExecutionMode.TEST

            logger.info(f"GA Parameters: Task Description='{task_desc}'")
            logger.info(f"GA Parameters: Keywords={keywords}")
            logger.info(f"GA Parameters: Generations={num_generations}, Population Size={population_size}, Elitism Count={elitism_count}")
            logger.info(f"GA Parameters: Execution Mode='{execution_mode.name}'")
            if initial_prompt_str:
                logger.info(f"GA Parameters: Initial Prompt Provided.")
            
            agent_settings_override = None
            if args.agent_settings:
                try:
                    if os.path.isfile(args.agent_settings):
                        with open(args.agent_settings, 'r') as f:
                            agent_settings_override = json.load(f)
                        logger.info(f"Loaded agent settings override from file: {args.agent_settings}")
                    else:
                        agent_settings_override = json.loads(args.agent_settings)
                        logger.info("Loaded agent settings override from JSON string.")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON for agent_settings: {e}. Using default agent settings.")
                except Exception as e:
                    logger.error(f"Error processing agent_settings: {e}. Using default agent settings.")

            llm_settings_override = None
            if args.llm_settings:
                try:
                    if os.path.isfile(args.llm_settings):
                        with open(args.llm_settings, 'r') as f:
                            llm_settings_override = json.load(f)
                        logger.info(f"Loaded LLM settings override from file: {args.llm_settings}")
                    else:
                        llm_settings_override = json.loads(args.llm_settings)
                        logger.info("Loaded LLM settings override from JSON string.")
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON for llm_settings: {e}. Using default LLM settings.")
                except Exception as e:
                    logger.error(f"Error processing llm_settings: {e}. Using default LLM settings.")

            # TODO: Pass agent_settings_override and llm_settings_override to main_ga_loop or a config manager
            # For now, they are just logged. The subtask specifies loading here.

            print("CLI: Running Genetic Algorithm with specified parameters...")
            try:
                best_chromosome = main_ga_loop(
                    task_desc=task_desc,
                    keywords=keywords,
                    num_generations=num_generations,
                    population_size=population_size,
                    elitism_count=elitism_count,
                    execution_mode=execution_mode,
                    initial_prompt_str=initial_prompt_str,
                    agent_settings_override=agent_settings_override,
                    llm_settings_override=llm_settings_override,
                    population_path=args.population_path,
                    parallel_workers=args.parallel_workers, # Pass the new argument
                    return_best=True
                )

                if best_chromosome:
                    logger.info("CLI: Genetic Algorithm completed successfully.")
                    logger.info(f"Best prompt fitness: {best_chromosome.fitness_score}")
                    # logger.debug(f"Best prompt content: {best_chromosome.to_prompt_string()}") # Potentially very long

                    if args.output_file:
                        try:
                            with open(args.output_file, 'w') as f:
                                f.write(best_chromosome.to_prompt_string())
                            logger.info(f"Best prompt saved to: {args.output_file}")
                        except IOError as e:
                            logger.error(f"Error writing best prompt to file {args.output_file}: {e}")
                            print(f"Error: Could not write to output file: {args.output_file}", file=sys.stderr)
                    else:
                        # If no output file, print a concise version or just fitness
                        print(f"Best prompt fitness: {best_chromosome.fitness_score}")
                        # Optionally print the prompt if it's not too long or provide a way to view it
                        # print(f"Best prompt: {best_chromosome.to_prompt_string()[:200]}...")


                else:
                    logger.info("CLI: Genetic Algorithm completed, but no best prompt was found.")
                    print("\nCLI: Genetic Algorithm completed, but no best prompt was found.")

            except openai.RateLimitError as rle:
                print(f"CLI: CRITICAL ERROR - OpenAI Rate Limit Exceeded: {rle}", file=sys.stderr)
                print("Your OpenAI account has hit its usage quota or rate limits. The Genetic Algorithm cannot proceed with LLM evaluations.", file=sys.stderr)
                print("Please check your OpenAI plan and billing details. https://platform.openai.com/docs/guides/error-codes/api-errors", file=sys.stderr)
                sys.exit(1) # Exit with error
            except Exception as e:
                print(f"CLI: Error running Genetic Algorithm: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            print(f"Error: Unknown module '{args.module}'. Currently, only 'ga' module is supported for the run command.", file=sys.stderr)
            sys.exit(1)

    elif args.command == "check-llm":
        logging.debug(
            f"CLI: Checking LLM connectivity for provider {args.provider}, model {args.model}"
        )
        try:
            from prompthelix.utils import llm_utils

            response = llm_utils.call_llm_api(
                prompt="Hello from PromptHelix", provider=args.provider, model=args.model
            )
            print(f"LLM response from {args.provider}: {response}")
        except Exception as e:
            logging.exception("CLI: LLM connectivity check failed")
            print(f"CLI: Failed to contact {args.provider}: {e}", file=sys.stderr)
            sys.exit(1)

    # No specific action needed for --version as argparse handles it
    elif hasattr(args, 'version') and args.version:
        pass
    # If no command is given, print help
    else:
        parser.print_help()

if __name__ == "__main__":
    main_cli()
