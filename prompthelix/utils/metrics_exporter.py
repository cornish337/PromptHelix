"""Prometheus metrics exporter for PromptHelix."""

try:
    from prometheus_client import Gauge, start_http_server
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Gauge = None  # type: ignore
    def start_http_server(*args, **kwargs):
        return
from prompthelix import config
from prompthelix import globals as ph_globals

_started = False

def start_exporter_if_enabled():
    """Start Prometheus HTTP server if enabled in settings."""
    global _started
    if not config.settings.PROMETHEUS_METRICS_ENABLED or _started:
        return
    start_http_server(config.settings.PROMETHEUS_METRICS_PORT)
    ph_globals.generation_gauge = Gauge(
        "prompthelix_current_generation",
        "Current generation number of the GA",
    )
    ph_globals.best_fitness_gauge = Gauge(
        "prompthelix_best_fitness",
        "Best fitness score in the population",
    )
    _started = True


def update_generation(generation: int) -> None:
    if ph_globals.generation_gauge is not None:
        ph_globals.generation_gauge.set(generation)


def update_best_fitness(fitness: float | None) -> None:
    if ph_globals.best_fitness_gauge is not None and fitness is not None:
        ph_globals.best_fitness_gauge.set(fitness)
