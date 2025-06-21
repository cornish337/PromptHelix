# prompthelix/globals.py
"""
Global shared instances for the PromptHelix application.
This module should be kept lightweight and free of complex imports
to avoid circular dependencies.
"""
from typing import Optional
from prometheus_client import Gauge
# Forward reference for GeneticAlgorithmRunner to avoid circular import issues
# as experiment_runners.ga_runner will import this file.
if False: # TYPE_CHECKING can also be used here if preferred
    from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner

from prompthelix.websocket_manager import ConnectionManager

# Global WebSocket connection manager instance
websocket_manager = ConnectionManager()

# Definition for the active Genetic Algorithm runner
active_ga_runner: Optional["GeneticAlgorithmRunner"] = None

# Fitness history collected during GA runs
ga_history: list[dict] = []

# Prometheus metrics gauges (initialized lazily by metrics_exporter)
generation_gauge: Gauge | None = None
best_fitness_gauge: Gauge | None = None

# You can add other global instances here if needed, e.g., a global MessageBus
# from prompthelix.database import SessionLocal
# from prompthelix.message_bus import MessageBus
# message_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
# Note: If adding MessageBus here, ensure SessionLocal import doesn't create cycles.
# For now, only websocket_manager is strictly needed to break the current cycle.

print("PromptHelix Config: GLOBALS - websocket_manager initialized.")
