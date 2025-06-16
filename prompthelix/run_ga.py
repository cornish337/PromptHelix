#!/usr/bin/env python3
"""Entry point for running the PromptHelix genetic algorithm."""
import os
import sys

# Ensure the package root is on the path when executed directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from prompthelix.orchestrator import main_ga_loop
from prompthelix.enums import ExecutionMode

if __name__ == "__main__":
    # Default to TEST mode for simple standalone runs
    main_ga_loop(
        task_desc="Quick start demo",
        keywords=["demo"],
        num_generations=1,
        population_size=2,
        elitism_count=1,
        execution_mode=ExecutionMode.TEST,
        return_best=True,
    )
