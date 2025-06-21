import uuid
import json
import os
import random
import copy # Added for deepcopy
import asyncio # Added for asyncio.gather
from typing import List, Optional, Dict # Added Dict

class PromptChromosome:
    """Simple representation of a prompt chromosome used in tests."""

    def __init__(
        self,
        genes: Optional[List[str]] = None,
        fitness_score: float = 0.0,
        parent_ids: Optional[List[str]] = None,
        mutation_strategy: Optional[str] = None # Retained this field
    ):
        self.id = uuid.uuid4()
        self.genes = genes if genes is not None else []
        self.fitness_score = fitness_score
        self.parent_ids = parent_ids if parent_ids is not None else []
        self.mutation_strategy = mutation_strategy # Retained this field

    def clone(self) -> "PromptChromosome":
        """Creates a deep copy of this chromosome with a new ID."""
        cloned = PromptChromosome(
            genes=copy.deepcopy(self.genes), # Use deepcopy for genes
            fitness_score=self.fitness_score, # Fitness is usually reset or recalculated later
            parent_ids=list(self.parent_ids), # Shallow copy of parent_ids list is fine
            mutation_strategy=self.mutation_strategy # Copy mutation strategy
        )
        # The new ID is already handled by PromptChromosome.__init__
        return cloned

    def to_prompt_string(self, separator: str = "\n") -> str:
        """Converts the chromosome's genes into a single string."""
        return separator.join(map(str, self.genes))

    def __str__(self) -> str:
        """Returns a human-readable string representation of the chromosome."""
        gene_summary = self.to_prompt_string(separator=" ")[:100] # Show a snippet
        if not self.genes:
            gene_summary = "(No genes)"
        return (
            f"PromptChromosome(ID: {self.id}, Fitness: {self.fitness_score:.4f}, "
            f"Genes: '{gene_summary}...', Parents: {self.parent_ids}, MutationOp: {self.mutation_strategy})"
        )

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the chromosome."""
        return (
            f"PromptChromosome(id='{self.id}', genes={self.genes!r}, "
            f"fitness_score={self.fitness_score!r}, parent_ids={self.parent_ids!r}, "
            f"mutation_strategy={self.mutation_strategy!r})"
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

    def selection(self, population: List[PromptChromosome], tournament_size: int = 2) -> PromptChromosome:
        """
        Selects a chromosome from the population using tournament selection.
        """
        if not population:
            raise ValueError("Population cannot be empty for selection.")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive.")

        actual_tournament_size = min(tournament_size, len(population))
        if actual_tournament_size == 0: # Should be caught by "not population" but defensive
             raise ValueError("Effective tournament size is zero, cannot select.")


        tournament_contestants = random.sample(population, actual_tournament_size)

        # Return the fittest individual from the tournament
        return max(tournament_contestants, key=lambda c: c.fitness_score)


from prompthelix.enums import ExecutionMode # Added for ExecutionMode.TEST comparison
from prompthelix.agents.base import BaseAgent # Added for type hint

class FitnessEvaluator:
    """Evaluator wrapper, potentially simplified for tests or base for complex evaluators."""

    def __init__(self, results_evaluator_agent, execution_mode, llm_settings: Optional[Dict] = None, **_): # Added llm_settings
        if not isinstance(results_evaluator_agent, BaseAgent): # Assuming BaseAgent or a more specific type
            raise TypeError("results_evaluator_agent must be an instance of a BaseAgent derivative.")
        self.results_evaluator_agent = results_evaluator_agent
        self.execution_mode = execution_mode
        self.llm_settings = llm_settings if llm_settings is not None else {}


    async def evaluate(self, chromosome: PromptChromosome, task_description: str, success_criteria: Optional[Dict] = None) -> float:
        """
        Evaluates a chromosome's fitness.
        In TEST mode, simulates LLM output. Otherwise, this basic evaluator might not be fully functional
        for REAL mode without actual LLM calls.
        """
        if not isinstance(chromosome, PromptChromosome):
            raise TypeError("chromosome must be an instance of PromptChromosome.")

        prompt_string = chromosome.to_prompt_string()
        llm_output = ""

        if self.execution_mode == ExecutionMode.TEST:
            # Simulate LLM output for TEST mode, similar to old tests
            keywords_snippet = ", ".join(str(g) for g in chromosome.genes[:2])
            random_num = random.randint(0, 100)
            llm_output = (
                f"Mock LLM output for: {prompt_string[:50]}. "
                f"Keywords found: {keywords_snippet}. "
                f"Random number: {random_num}"
            )
        else:
            # In REAL mode, this basic FitnessEvaluator would need to call an LLM
            # to get the llm_output based on prompt_string.
            # For this "small wrapper", we'll raise an error or return a default.
            # Or, it assumes llm_output is somehow already populated if not TEST mode.
            # For now, let's keep it simple and rely on ResultsEvaluatorAgent to handle it,
            # assuming llm_output must be generated *before* calling this basic evaluate.
            # This part is tricky: FitnessEvaluator might be expected to *get* the llm_output.
            # The ResultsEvaluatorAgent *evaluates* a given llm_output.
            # Let's assume for now that the "llm_output" must be obtained by the GA loop
            # and passed into a more complex FitnessEvaluator.
            # This current simple one will just pass an empty string if not TEST mode,
            # which ResultsEvaluatorAgent will then score poorly.
            # This is consistent with it being a "small wrapper for tests".
            # A real FitnessEvaluator would handle the llm_utils.call_llm_api itself.
            pass # llm_output remains "" if not TEST mode, to be evaluated by REA

        eval_request_data = {
            "prompt_chromosome": chromosome,
            "llm_output": llm_output, # This is the generated output for the prompt
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria is not None else {}
        }

        # ResultsEvaluatorAgent.process_request is now async
        evaluation_result = await self.results_evaluator_agent.process_request(eval_request_data)

        fitness_score = evaluation_result.get("fitness_score", 0.0)
        chromosome.fitness_score = fitness_score
        return fitness_score


class PopulationManager:
    """Simplified population manager supporting persistence and broadcasts."""

    def __init__(
        self,
        genetic_operators: GeneticOperators,
        fitness_evaluator: FitnessEvaluator,
        prompt_architect_agent,
        population_size: int = 0,
        elitism_count: int = 0,
        initial_prompt_str: Optional[str] = None, # Added initial_prompt_str
        parallel_workers: int = 1,
        message_bus=None
    ):
        self.genetic_operators = genetic_operators
        self.fitness_evaluator = fitness_evaluator
        self.prompt_architect_agent = prompt_architect_agent
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.initial_prompt_str = initial_prompt_str # Store it
        self.parallel_workers = parallel_workers
        self.population: List[PromptChromosome] = []
        self.generation_number = 0
        self.message_bus = message_bus
        self.status: str = "IDLE" # Possible statuses: IDLE, INITIALIZING, RUNNING, PAUSED, STOPPED, COMPLETED, ERROR
        self.is_paused: bool = False
        self.should_stop: bool = False

    async def initialize_population(self, initial_task_description: str, initial_keywords: List[str],
                                  constraints: Optional[Dict] = None,
                                  success_criteria: Optional[Dict] = None): # Removed initial_prompt_str, use self.initial_prompt_str
        self.status = "INITIALIZING"
        self.broadcast_ga_update(event_type="population_initialization_started")
        self.population = []
        self.generation_number = 0

        if self.initial_prompt_str: # Use self.initial_prompt_str
            # Create one chromosome from the initial prompt string
            # Genes could be the whole string, or parsed if it has structure. Assuming simple for now.
            genes = [self.initial_prompt_str] # Use self.initial_prompt_str
            chromosome = PromptChromosome(genes=genes)
            self.population.append(chromosome)
            # Fill remaining population if size is larger than 1
            # For simplicity, if initial_prompt_str is given, population_size might be implicitly 1
            # or architect generates pop_size-1 new ones. Let's assume architect generates others.

        # Use PromptArchitectAgent to generate individuals
        # process_request might be async if it involves LLM calls
        # Assuming PromptArchitectAgent.process_request is async for now
        num_to_generate = self.population_size - len(self.population)

        # Create a list of tasks for asyncio.gather
        arch_tasks = []
        for _ in range(num_to_generate):
            request_data = {
                "task_description": initial_task_description,
                "keywords": initial_keywords,
                "constraints": constraints if constraints is not None else {}
            }
            # Assuming process_request is async
            arch_tasks.append(self.prompt_architect_agent.process_request(request_data))

        if arch_tasks:
            new_chromosomes = await asyncio.gather(*arch_tasks)
            self.population.extend(new_chromosomes)

        # Evaluate the initial population
        eval_tasks = []
        for chromo in self.population:
            eval_tasks.append(self.fitness_evaluator.evaluate(chromo, initial_task_description, success_criteria))

        if eval_tasks:
            await asyncio.gather(*eval_tasks)

        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        self.status = "IDLE" # Or RUNNING if it immediately proceeds to evolve
        self.broadcast_ga_update(event_type="population_initialized", additional_data={"population_size": len(self.population)})


    def pause_evolution(self):
        self.is_paused = True
        self.status = "PAUSED"
        self.broadcast_ga_update(event_type="ga_run_paused")
        # logger.info(f"PopulationManager (ID: {id(self)}): Evolution paused.")

    def resume_evolution(self):
        self.is_paused = False
        self.status = "RUNNING"
        self.broadcast_ga_update(event_type="ga_run_resumed")
        # logger.info(f"PopulationManager (ID: {id(self)}): Evolution resumed.")

    def stop_evolution(self):
        self.should_stop = True
        self.status = "STOPPED" # Or "STOPPING" then orchestrator sets to "STOPPED"
        self.broadcast_ga_update(event_type="ga_run_stopped")
        # logger.info(f"PopulationManager (ID: {id(self)}): Evolution stop requested.")

    def get_ga_status(self) -> Dict:
        return {
            "status": self.status,
            "generation": self.generation_number,
            "population_size": len(self.population),
            "is_paused": self.is_paused,
            "should_stop": self.should_stop,
            "fittest_individual_id": str(self.get_fittest_individual().id) if self.population else None,
            "fittest_individual_score": self.get_fittest_individual().fitness_score if self.population else None,
        }

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

    async def evolve_population(self, task_description: str, success_criteria: Optional[Dict] = None, db_session=None, experiment_run=None): # Added success_criteria, made async
        if not self.population:
            return

        # Evaluate fitness for each chromosome
        # This part needs to be async if self.fitness_evaluator.evaluate is async
        for chromosome in self.population:
            await self.fitness_evaluator.evaluate(chromosome, task_description, success_criteria)

        # Sort population by fitness score in descending order
        self.population.sort(key=lambda c: c.fitness_score, reverse=True)

        best = self.population[0] # Best is now the first element after sorting
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
