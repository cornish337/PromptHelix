#!/usr/bin/env python3
"""Entry point for running the PromptHelix genetic algorithm."""
import os
import sys

# Ensure the package root is on the path when executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompthelix.orchestrator import main_ga_loop

if __name__ == "__main__":
    result = main_ga_loop()
    if result: # Ensure there's something to print
        import json # Import json for potentially prettier printing if desired, or just print directly
        # For structured output that matches API, printing JSON might be good.
        # However, direct print of dict is also fine for CLI.
        # Let's use json.dumps for a clean, consistent output.
        try:
            print(json.dumps(result, indent=4))
        except TypeError: # In case result is not JSON serializable for some unexpected reason
            print(result)
