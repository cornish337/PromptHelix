from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session as DbSession
import logging # Added for logging

from prompthelix.models.evolution_models import (
    GAExperimentRun,
    GAChromosome,
    GAGenerationMetrics,
)
from prompthelix.genetics.engine import PromptChromosome

logger = logging.getLogger(__name__) # Added logger


def create_experiment_run(db: DbSession, parameters: Optional[Dict[str, Any]] = None) -> GAExperimentRun:
    run = GAExperimentRun(parameters=parameters or {})
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def complete_experiment_run(db: DbSession, run: GAExperimentRun, prompt_version_id: Optional[int] = None) -> GAExperimentRun:
    run.completed_at = datetime.utcnow()
    if prompt_version_id is not None:
        run.prompt_version_id = prompt_version_id
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def add_chromosome_record(db: DbSession, run: GAExperimentRun, generation_number: int, chromosome: PromptChromosome) -> GAChromosome:
    record = GAChromosome(
        id=str(chromosome.id),
        run_id=run.id,
        generation_number=generation_number,
        genes=chromosome.genes,
        fitness_score=chromosome.fitness_score,
        evaluation_details=getattr(chromosome, "evaluation_details", None),

        parent_ids=getattr(chromosome, "parents", []), # Changed from parent_ids to parents
        mutation_operator=getattr(chromosome, "mutation_operator", None), # changed from mutation_strategy to mutation_operator

    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def add_generation_metrics( # This function seems to be for bulk adding, maybe from a dict
    db: DbSession, run: GAExperimentRun, metrics: Dict[str, Any]
) -> GAGenerationMetrics:
    record = GAGenerationMetrics(
        run_id=run.id,
        generation_number=metrics.get("generation_number"),
        best_fitness=metrics.get("best_fitness"),
        avg_fitness=metrics.get("avg_fitness"),
        population_diversity=metrics.get("population_diversity", 0.0), # Ensure this matches model
        population_size=metrics.get("population_size"), # Ensure this matches model
        diversity=metrics.get("diversity"), # This is a JSON field in model, ensure metrics dict matches

    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_chromosomes_for_run(db: DbSession, run_id: int) -> List[GAChromosome]:
    return db.query(GAChromosome).filter(GAChromosome.run_id == run_id).all()


def get_experiment_runs(db: DbSession, skip: int = 0, limit: int = 100) -> List[GAExperimentRun]:
    """Retrieve a paginated list of GA experiment runs ordered by creation time."""
    return (
        db.query(GAExperimentRun)
        .order_by(GAExperimentRun.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_experiment_run(db: DbSession, run_id: int) -> Optional[GAExperimentRun]:
    """Return a single GA experiment run by ID, if it exists."""
    return db.query(GAExperimentRun).filter(GAExperimentRun.id == run_id).first()


def add_generation_metric( # This is for individual metric points
    db: DbSession,
    run: GAExperimentRun,
    generation_number: int,
    best_fitness: float,
    avg_fitness: float,
    population_diversity: float, # Matches model field
    population_size: int, # Matches model field
) -> GAGenerationMetrics:
    metric = GAGenerationMetrics(
        run_id=run.id,
        generation_number=generation_number,
        best_fitness=best_fitness,
        avg_fitness=avg_fitness,
        population_diversity=population_diversity,
        population_size=population_size,
        # diversity field (JSON) is not set here, which is fine if not always available/needed
    )
    db.add(metric)
    db.commit()
    db.refresh(metric)
    return metric


def get_latest_ga_run_id(db: DbSession) -> Optional[int]:
    """Retrieves the ID of the most recent GA experiment run."""
    latest_run = (
        db.query(GAExperimentRun.id)
        .order_by(GAExperimentRun.created_at.desc())
        .first()
    )
    return latest_run.id if latest_run else None


def get_generation_metrics_for_run(db: DbSession, run_id: Optional[int] = None) -> List[GAGenerationMetrics]:
    """
    Retrieves generation metrics for a specific GA run_id.
    If run_id is None, it attempts to fetch metrics for the latest run.
    """
    actual_run_id = run_id
    if actual_run_id is None:
        actual_run_id = get_latest_ga_run_id(db)
        if actual_run_id is None:
            logger.info("get_generation_metrics_for_run: No run_id provided and no past runs found.")
            return []

    return (
        db.query(GAGenerationMetrics)
        .filter(GAGenerationMetrics.run_id == actual_run_id)
        .order_by(GAGenerationMetrics.generation_number.asc())
        .all()
    )
