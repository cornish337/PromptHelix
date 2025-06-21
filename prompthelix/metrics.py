from fastapi import APIRouter, Response
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST

from .wandb_logger import log_metrics

# Gauges for key GA metrics
GA_GENERATION = Gauge(
    "prompthelix_ga_generation",
    "Current generation of the running genetic algorithm",
)
GA_BEST_FITNESS = Gauge(
    "prompthelix_ga_best_fitness",
    "Best fitness score of the current generation",
)

router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


def update_ga_metrics(generation: int, best_fitness: float | None) -> None:
    """Update in-memory metrics and optionally log to Weights & Biases."""
    GA_GENERATION.set(generation)
    if best_fitness is not None:
        GA_BEST_FITNESS.set(best_fitness)
    log_metrics(
        {
            "prompthelix_ga_generation": generation,
            "prompthelix_ga_best_fitness": best_fitness,
        }
    )
