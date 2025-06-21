import pytest
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.services import (
    create_experiment_run,
    complete_experiment_run,
    add_chromosome_record,
    add_generation_metrics,
    get_chromosomes_for_run,
)
from prompthelix.models.evolution_models import GAExperimentRun, GAChromosome
from prompthelix.genetics.engine import PromptChromosome


def test_create_and_complete_run(db_session: SQLAlchemySession):
    run = create_experiment_run(db_session, parameters={"p": 1})
    assert run.id is not None
    assert run.parameters == {"p": 1}
    assert run.completed_at is None

    run = complete_experiment_run(db_session, run, prompt_version_id=None)
    assert run.completed_at is not None


def test_add_and_fetch_chromosome(db_session: SQLAlchemySession):
    run = create_experiment_run(db_session)
    chromo = PromptChromosome(genes=["a"], fitness_score=0.5)
    record = add_chromosome_record(db_session, run, generation_number=1, chromosome=chromo)
    assert record.id == str(chromo.id)
    assert record.generation_number == 1

    retrieved = get_chromosomes_for_run(db_session, run.id)
    assert len(retrieved) == 1
    assert retrieved[0].id == str(chromo.id)


def test_add_generation_metrics(db_session: SQLAlchemySession):
    run = create_experiment_run(db_session)
    metrics = {
        "generation_number": 1,
        "best_fitness": 0.9,
        "avg_fitness": 0.5,
        "population_size": 2,
        "diversity": {"unique_ratio": 1.0},
    }
    record = add_generation_metrics(db_session, run, metrics)
    assert record.id is not None
    assert record.generation_number == 1
    assert record.best_fitness == 0.9


