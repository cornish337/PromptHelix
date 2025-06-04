#!/usr/bin/env python3
"""Entry point for running the PromptHelix genetic algorithm."""
import os
import sys

# Ensure the package root is on the path when executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompthelix.orchestrator import main_ga_loop

if __name__ == "__main__":
    main_ga_loop()
