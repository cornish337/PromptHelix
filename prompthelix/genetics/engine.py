import uuid
import json
import os
import random
from typing import List, Optional

class PromptChromosome:
    """Simple representation of a prompt chromosome used in tests."""

    def __init__(
        self,
        genes: Optional[List[str]] = None,
        fitness_score: float = 0.0,
        parent_ids: Optional[List[str]] = None,
        mutation_strategy: Optional[str] = None
    ):
        self.id = uuid.uuid4()
        self.genes = genes or []
        self.fitness_score = fitness_score
        self.parent_ids = parent_ids or []
        self.mutation_strategy = mutation_strategy

    def clone(self) -> "PromptChromosome":
        return PromptChromosome(
            genes=list(self.genes),
            fitness_score=self.fitness_score,
            parent_ids=list(self.parent_ids),
            mutation_strategy=self.mutation_strategy,
        )


class GeneticOperators:
    """Minimal genetic operators used by unit tests."""

    def __init__(self, style_optimizer_agent=None, mutation_strategies: Optional[List] = None, **_):
        self.style_optimizer_agent = style_optimizer_agent
        self.mutation_strategies = mutation_strategies or []

    def crossover(
        self,
        parent1: PromptChromosome,
        parent2: PromptChromosome,
        crossover_rate: float = 1.0,
        **_
    ) -> tuple[PromptChromosome, PromptChromosome]:
        child1 = parent1.clone()
        child2 = parent2.clone()
        if (
            random.random() <= crossover_rate
            and parent1.genes
            and parent2.genes
        ):
            pivot = len(parent1.genes) // 2
            child1.genes = parent1.genes[:pivot] + parent2.genes[pivot:]
            child2.genes = parent2.genes[:pivot] + parent1.genes[pivot:]
        child1.parent_ids = [str(parent1.id), str(parent2.id)]
        child2.parent_ids = [str(parent1.id), str(parent2.id)]
        return child1, child2

    def mutate(
        self,
        chromosome: PromptChromosome,
        mutation_rate: float = 1.0,
        gene_mutation_prob: float = 1.0,
        **_
    ) -> PromptChromosome:
        if not self.mutation_strategies or random.random() > mutation_rate:
            return chromosome.clone()
        strategy = self.mutation_strategies[0]
        mutated = strategy.mutate(chromosome.clone())
        mutated.parent_ids = [str(chromosome.id)]
        mutated.mutation_strategy = strategy.__class__.__name__
        return mutated


class FitnessEvaluator:
    """Very small evaluator wrapper used in tests."""

    def __init__(self, results_evaluator_agent, execution_mode, **_):
        self.results_evaluator_agent = results_evaluator_agent
        self.execution_mode = execution_mode


class PopulationManager:
    """Simplified population manager supporting persistence and broadcasts."""

    def __init__(
        self,
        genetic_operators: GeneticOperators,
        fitness_evaluator: FitnessEvaluator,
        prompt_architect_agent,
        population_size: int = 0,
        elitism_count: int = 0,
        parallel_workers: int = 1,
        message_bus=None
    ):
        self.genetic_operators = genetic_operators
        self.fitness_evaluator = fitness_evaluator
        self.prompt_architect_agent = prompt_architect_agent
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.parallel_workers = parallel_workers
        self.population: List[PromptChromosome] = []
        self.generation_number = 0
        self.message_bus = message_bus

    def save_population(self, file_path: str) -> None:
        data = {
            "generation_number": self.generation_number,
            "population": [
                {
                    "genes": c.genes,
                    "fitness_score": c.fitness_score,
                    "parents": c.parent_ids,
                    "mutation_strategy": c.mutation_strategy,
                }
                for c in self.population
            ],
        }
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    def load_population(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            self.population = []
            self.generation_number = 0
            return
        with open(file_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.generation_number = data.get("generation_number", 0)
        self.population = [
            PromptChromosome(
                genes=item.get("genes", []),
                fitness_score=item.get("fitness_score", 0.0),
                parent_ids=item.get("parents", []),
                mutation_strategy=item.get("mutation_strategy"),
            )
            for item in data.get("population", [])
        ]
        if self.population:
            self.population_size = len(self.population)

    def get_fittest_individual(self) -> Optional[PromptChromosome]:
        """Return the chromosome with the highest fitness or None."""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.fitness_score)

    def broadcast_ga_update(
        self,
        event_type: str,
        selected_parent_ids=None,
        additional_data=None
    ):
        if (
            self.message_bus
            and getattr(self.message_bus, "connection_manager", None)
        ):
            payload = {"type": event_type, "data": {"selected_parent_ids": selected_parent_ids}}
            if additional_data:
                payload["data"].update(additional_data)
            self.message_bus.connection_manager.broadcast_json(payload)

    def evolve_population(self, task_description: str, db_session=None, experiment_run=None):
        if not self.population:
            return
        best = max(self.population, key=lambda c: c.fitness_score)
        avg = sum(c.fitness_score for c in self.population) / len(self.population)
        try:
            from prompthelix.services import add_generation_metric
            add_generation_metric(
                db_session,
                experiment_run,
                self.generation_number,
                best.fitness_score,
                avg,
                0.0
            )
        except Exception:
            pass
