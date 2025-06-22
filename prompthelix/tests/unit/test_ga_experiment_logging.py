import pytest
from unittest.mock import MagicMock
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators, FitnessEvaluator
from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner
from prompthelix.services import create_experiment_run, complete_experiment_run, add_chromosome_record


class DummyEvaluator:
    def evaluate(self, chromo, task_description, success_criteria):
        return 0.5


class DummyArchitect:
    def process_request(self, request_data):
        return PromptChromosome(genes=["seed"])


@pytest.fixture
def simple_pm():
    gen_ops = MagicMock(spec=GeneticOperators)
    gen_ops.selection.side_effect = lambda pop: pop[0]
    gen_ops.crossover.return_value = (PromptChromosome(["c1"]), PromptChromosome(["c2"]))
    gen_ops.mutate.side_effect = lambda c, target_style=None: c
    pm = PopulationManager(
        genetic_operators=gen_ops,
        fitness_evaluator=DummyEvaluator(),
        prompt_architect_agent=DummyArchitect(),
        population_size=2,
        elitism_count=0,
        parallel_workers=1,
    )
    pm.population = [PromptChromosome(["a"]), PromptChromosome(["b"])]
    return pm


def test_population_manager_records(db_session: SQLAlchemySession, simple_pm: PopulationManager, monkeypatch):
    run = create_experiment_run(db_session)
    add_mock = MagicMock()
    monkeypatch.setattr("prompthelix.services.add_chromosome_record", add_mock)

    simple_pm.evolve_population(
        task_description="t",
        target_style=None,
        db_session=db_session,
        experiment_run=run,
    )

    assert add_mock.call_count == 2


def test_ga_runner_creates_and_completes_run(db_session: SQLAlchemySession, simple_pm: PopulationManager, monkeypatch):
    create_mock = MagicMock(side_effect=create_experiment_run)
    complete_mock = MagicMock(side_effect=complete_experiment_run)
    monkeypatch.setattr("prompthelix.experiment_runners.ga_runner.create_experiment_run", create_mock)
    monkeypatch.setattr("prompthelix.experiment_runners.ga_runner.complete_experiment_run", complete_mock)
    monkeypatch.setattr("prompthelix.experiment_runners.ga_runner.SessionLocal", lambda: db_session)

    runner = GeneticAlgorithmRunner(simple_pm, num_generations=1)
    runner.run(task_description="t")

    create_mock.assert_called_once()
    complete_mock.assert_called_once()
