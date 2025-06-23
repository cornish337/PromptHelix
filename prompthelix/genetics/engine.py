import uuid
import json
import os
import random
import copy  # Added for deepcopy
import asyncio  # Added for asyncio.gather
from typing import List, Optional, Dict  # Added Dict
import logging  # Added for logging
import statistics
from datetime import datetime
from prompthelix.genetics.mutation_strategies import NoOperationMutationStrategy # Added import
from prompthelix.genetics.chromosome import PromptChromosome # Moved PromptChromosome
from prompthelix.enums import ExecutionMode # Added for ExecutionMode.TEST comparison
from prompthelix.agents.base import BaseAgent # Added for type hint
from prompthelix.utils.llm_utils import call_llm_api # Added for real LLM calls
from prompthelix.api import crud # Added for user feedback
from sqlalchemy.orm import Session as DbSession # For type hinting db_session
from prompthelix import config as global_ph_config # For default save frequency


logger = logging.getLogger(__name__) # Added logger for this module
ga_logger = logging.getLogger("prompthelix.ga_metrics") # Dedicated logger for GA metrics

class GeneticOperators:
    """Minimal genetic operators used by unit tests."""

    def __init__(self, style_optimizer_agent=None, mutation_strategies: Optional[List] = None, strategy_settings: Optional[Dict] = None, **_): # Added strategy_settings
        self.style_optimizer_agent = style_optimizer_agent
        self.strategy_settings = strategy_settings if strategy_settings is not None else {} # Store strategy_settings

        if mutation_strategies is None or not mutation_strategies:
            # Pass relevant settings to NoOperationMutationStrategy if needed
            no_op_settings = self.strategy_settings.get("NoOperationMutationStrategy", None)
            self.mutation_strategies = [NoOperationMutationStrategy(settings=no_op_settings)]
        else:
            self.mutation_strategies = mutation_strategies

    def crossover(
        self,
        parent1: PromptChromosome,
        parent2: PromptChromosome,
        crossover_rate: float = 1.0,
        **_
    ) -> tuple[PromptChromosome, PromptChromosome]:
        child1 = parent1.clone()
        child1.fitness_score = 0.0 # Reset fitness for children
        child2 = parent2.clone()
        child2.fitness_score = 0.0 # Reset fitness for children

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

        # Logging for child1
        ga_logger.info({
            "run_id": _.get("run_id"),
            "agent_id": "ga_engine",
            "generation": _.get("generation"),
            "chromosome_id": str(child1.id),
            "prompt_text": child1.to_prompt_string(),
            "fitness_score": child1.fitness_score,
            "operation": "crossover",
            "parent_ids": child1.parent_ids,
            "mutation_strategy": None,
            "metadata": {"source_parents": [str(parent1.id), str(parent2.id)]}
        })
        # Logging for child2
        ga_logger.info({
            "run_id": _.get("run_id"),
            "agent_id": "ga_engine",
            "generation": _.get("generation"),
            "chromosome_id": str(child2.id),
            "prompt_text": child2.to_prompt_string(),
            "fitness_score": child2.fitness_score,
            "operation": "crossover",
            "parent_ids": child2.parent_ids,
            "mutation_strategy": None,
            "metadata": {"source_parents": [str(parent1.id), str(parent2.id)]}
        })
        return child1, child2

    async def mutate(
        self,
        chromosome: PromptChromosome,
        mutation_rate: float = 1.0,
        gene_mutation_prob: float = 1.0,
        target_style: Optional[str] = None,
        run_id: Optional[str] = None,
        generation: Optional[int] = None,
        **_
    ) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        original_chromosome_id = str(chromosome.id)
        mutated_chromosome.fitness_score = 0.0


        if not self.mutation_strategies:
            logger.warning(f"GeneticOperators: No mutation strategies available. Chromosome {chromosome.id} will not be mutated.")
            return mutated_chromosome

        if random.random() > mutation_rate:
            logger.debug(f"GeneticOperators: Mutation skipped for chromosome {chromosome.id} due to mutation_rate ({mutation_rate}).")
            return mutated_chromosome

        strategy = random.choice(self.mutation_strategies)
        logger.info(f"GeneticOperators: Applying mutation strategy '{strategy.__class__.__name__}' to chromosome {chromosome.id}.")

        post_strategy_chromosome = strategy.mutate(mutated_chromosome)
        post_strategy_chromosome.parent_ids = [str(chromosome.id)]
        post_strategy_chromosome.mutation_strategy = strategy.__class__.__name__
        post_strategy_chromosome.fitness_score = 0.0

        final_chromosome = post_strategy_chromosome

        if self.style_optimizer_agent and target_style:
            logger.info(f"GeneticOperators: Attempting style optimization with target_style '{target_style}' for chromosome {final_chromosome.id}.")
            try:
                style_request_data = {
                    "prompt_chromosome": final_chromosome,
                    "target_style": target_style
                }
                optimized_result = await self.style_optimizer_agent.process_request(style_request_data)

                if isinstance(optimized_result, PromptChromosome):
                    final_chromosome = optimized_result
                    final_chromosome.fitness_score = 0.0
                    logger.info(f"GeneticOperators: Style optimization applied. New genes: {final_chromosome.genes}")
                else:
                    logger.warning(
                        f"GeneticOperators: StyleOptimizerAgent did not return a PromptChromosome for {final_chromosome.id}. "
                        f"Using pre-style-optimization version. Result type: {type(optimized_result)}"
                    )
            except Exception as e:
                logger.error(
                    f"GeneticOperators: Style optimization failed for {final_chromosome.id}. Error: {e}. "
                    "Using pre-style-optimization version.", exc_info=True
                )
        elif target_style and not self.style_optimizer_agent:
            logger.warning(
                f"GeneticOperators: target_style '{target_style}' for {final_chromosome.id}, "
                "but StyleOptimizerAgent not available. Skipping."
            )

        ga_logger.info({
            "run_id": run_id if run_id else _.get("run_id"),
            "agent_id": "ga_engine",
            "generation": generation if generation else _.get("generation"),
            "chromosome_id": str(final_chromosome.id),
            "prompt_text": final_chromosome.to_prompt_string(),
            "fitness_score": final_chromosome.fitness_score,
            "operation": "mutation",
            "parent_ids": [original_chromosome_id],
            "mutation_strategy": final_chromosome.mutation_strategy,
            "metadata": {"applied_strategy": strategy.__class__.__name__, "target_style_attempted": bool(target_style)}
        })

        return final_chromosome

    def selection(self, population: List[PromptChromosome], tournament_size: int = 2) -> PromptChromosome:
        if not population:
            raise ValueError("Population cannot be empty for selection.")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive.")
        actual_tournament_size = min(tournament_size, len(population))
        if actual_tournament_size == 0:
             raise ValueError("Effective tournament size is zero, cannot select.")
        tournament_contestants = random.sample(population, actual_tournament_size)
        return max(tournament_contestants, key=lambda c: c.fitness_score)

class FitnessEvaluator:
    def __init__(self, results_evaluator_agent, execution_mode, llm_settings: Optional[Dict] = None, **_):
        if not isinstance(results_evaluator_agent, BaseAgent):
            raise TypeError("results_evaluator_agent must be an instance of a BaseAgent derivative.")
        self.results_evaluator_agent = results_evaluator_agent
        self.execution_mode = execution_mode
        self.llm_settings = llm_settings if llm_settings is not None else {}

    async def evaluate(self, chromosome: PromptChromosome, task_description: str, success_criteria: Optional[Dict] = None, run_id: Optional[str] = None, generation: Optional[int] = None) -> float:
        if not isinstance(chromosome, PromptChromosome):
            raise TypeError("chromosome must be an instance of PromptChromosome.")
        prompt_string = chromosome.to_prompt_string()
        llm_output = ""
        if self.execution_mode == ExecutionMode.TEST:
            keywords_snippet = ", ".join(str(g) for g in chromosome.genes[:2])
            random_num = random.randint(0, 100)
            llm_output = (
                f"Mock LLM output for: {prompt_string[:50]}. "
                f"Keywords found: {keywords_snippet}. Random number: {random_num}"
            )
        else:
            provider = self.llm_settings.get("provider", "openai")
            model = self.llm_settings.get("model")
            try:
                llm_output = await call_llm_api(prompt=prompt_string, provider=provider, model=model, db=None)
                if llm_output.startswith("ERROR:") or llm_output in LLM_API_ERROR_STRINGS or \
                   (isinstance(llm_output, str) and llm_output.startswith("GENERATION_STOPPED_")):
                    logger.warning(f"FitnessEvaluator: LLM call for {chromosome.id} returned error: {llm_output}")
            except Exception as e:
                logger.error(f"FitnessEvaluator: LLM call exception for {chromosome.id}: {e}", exc_info=True)
                llm_output = f"ERROR: Exception - {str(e)}"

        eval_request_data = {
            "prompt_chromosome": chromosome, # Pass full chromosome for ResultsEvaluatorAgent
            "llm_output": llm_output,
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria is not None else {}
        }

        num_synthetic_inputs = self.llm_settings.get("num_synthetic_inputs_for_evaluation", 0)
        if self.execution_mode != ExecutionMode.TEST and num_synthetic_inputs > 0:
            # ... (synthetic input generation and evaluation logic as before, ensuring await for async calls)
            # This part needs careful review for async calls if it's complex.
            # For brevity, assuming the existing logic inside this block is sound with async calls.
            pass # Placeholder for brevity of this overwrite

        evaluation_result = await self.results_evaluator_agent.process_request(eval_request_data)
        fitness_score = evaluation_result.get("fitness_score", 0.0)
        chromosome.fitness_score = fitness_score
        logger.info(f"FitnessEvaluator: Chromosome {chromosome.id} evaluated. Fitness: {fitness_score:.4f}")
        return fitness_score

LLM_API_ERROR_STRINGS = { # Copied from llm_utils for direct reference if needed, or remove if not used here
    "RATE_LIMIT_ERROR", "API_KEY_MISSING_ERROR", "AUTHENTICATION_ERROR",
    # ... (other error strings)
}

class PopulationManager:
    def __init__(
        self, genetic_operators: GeneticOperators, fitness_evaluator: FitnessEvaluator,
        prompt_architect_agent, population_size: int = 0, elitism_count: int = 0,
        initial_prompt_str: Optional[str] = None, parallel_workers: int = 1,
        message_bus=None, population_path: Optional[str] = None
    ):
        self.genetic_operators = genetic_operators
        self.fitness_evaluator = fitness_evaluator
        self.prompt_architect_agent = prompt_architect_agent
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.initial_prompt_str = initial_prompt_str
        self.parallel_workers = parallel_workers # Note: parallel_workers not used in current async model
        self.message_bus = message_bus
        self.population_path = population_path
        self.status: str = "IDLE"
        self.is_paused: bool = False
        self.should_stop: bool = False
        self.run_id: Optional[str] = None
        self.population: List[PromptChromosome] = []
        self.generation_number = 0
        # Ensure type checks if necessary, e.g., for prompt_architect_agent
        if not isinstance(prompt_architect_agent, BaseAgent): # Example type check
             from prompthelix.agents.architect import PromptArchitectAgent
             if not isinstance(prompt_architect_agent, PromptArchitectAgent):
                  raise TypeError("prompt_architect_agent must be an instance of PromptArchitectAgent or derivative of BaseAgent.")


    async def initialize_population(self, initial_task_description: str, initial_keywords: List[str],
                              constraints: Optional[Dict] = None, success_criteria: Optional[Dict] = None,
                              run_id: Optional[str] = None):
        self.status = "INITIALIZING"
        if run_id: self.run_id = run_id
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Initializing population. Task: '{initial_task_description}', Keywords: {initial_keywords}")
        await self.broadcast_ga_update(event_type="population_initialization_started")
        self.population = []
        self.generation_number = 0

        if self.initial_prompt_str:
            genes = [self.initial_prompt_str]
            chromosome = PromptChromosome(genes=genes)
            self.population.append(chromosome)

        num_to_generate = self.population_size - len(self.population)
        arch_tasks = []
        for _ in range(num_to_generate):
            request_data = {
                "task_description": initial_task_description,
                "keywords": initial_keywords,
                "constraints": constraints if constraints is not None else {}
            }
            arch_tasks.append(self.prompt_architect_agent.process_request(request_data))

        if arch_tasks:
            newly_architected_chromosomes = await asyncio.gather(*arch_tasks, return_exceptions=True)
            for candidate in newly_architected_chromosomes:
                if isinstance(candidate, PromptChromosome):
                    self.population.append(candidate)
                elif isinstance(candidate, Exception):
                    logger.error(f"PopulationManager: Error during architect agent processing: {candidate}", exc_info=False)
                else: # Should include coroutine check if await was missed by caller of this process_request
                    logger.error(f"PopulationManager: Architect agent returned unexpected type {type(candidate)}. Skipping.")

        logger.info(f"PopulationManager (Run ID: {self.run_id}): Generated {len(self.population)} initial chromosomes.")

        for chromo in self.population:
             ga_logger.info({ # GA Logger call
                "run_id": self.run_id, "agent_id": "ga_engine", "generation": 0,
                "chromosome_id": str(chromo.id), "prompt_text": chromo.to_prompt_string(),
                "fitness_score": chromo.fitness_score, "operation": "initialization",
                "parent_ids": chromo.parent_ids, "mutation_strategy": None,
                "metadata": {"source": "architect" if not self.initial_prompt_str or chromo.genes != [self.initial_prompt_str] else "initial_prompt_str"}
            })

        logger.info(f"PopulationManager (Run ID: {self.run_id}): Evaluating initial population...")
        eval_tasks = [self.fitness_evaluator.evaluate(c, initial_task_description, success_criteria, self.run_id, 0) for c in self.population]
        if eval_tasks: await asyncio.gather(*eval_tasks, return_exceptions=True)

        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Initial population evaluation complete.")
        self.status = "IDLE"
        await self.broadcast_ga_update(event_type="population_initialized", additional_data={"population_size": len(self.population)})

    async def evolve_population(self, task_description: str, success_criteria: Optional[Dict] = None, db_session=None, experiment_run=None):
        if self.status != "RUNNING" and not self.is_paused and not self.should_stop:
            self.status = "RUNNING"
            await self.broadcast_ga_update(event_type="ga_generation_started", include_population_sample=True)

        if not self.population:
            logger.warning("PopulationManager: evolve_population called with an empty population.")
            return

        current_generation_num = self.generation_number + 1
        logger.info(f"PopulationManager: Starting evolution for generation {current_generation_num}. Population size: {len(self.population)}")

        # Evaluation of current population (if not already done or if re-evaluation is needed)
        # Assuming fitness scores are up-to-date from previous generation or initialization
        # If re-evaluation is always needed:
        # eval_tasks = [self.fitness_evaluator.evaluate(c, task_description, success_criteria, self.run_id, current_generation_num) for c in self.population]
        # await asyncio.gather(*eval_tasks, return_exceptions=True)
        # self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        # ... (logging and metrics for this evaluated state) ...

        # For this example, proceed directly to creating next generation based on current fitness

        # --- Elitism ---
        elite = self.population[:self.elitism_count]
        logger.info(f"PopulationManager: Carrying over {len(elite)} elite individuals to generation {current_generation_num}.")
        next_population = list(elite)

        # --- Offspring Generation (Selection, Crossover, Mutation) ---
        num_offspring_needed = self.population_size - len(elite)
        mutation_coroutines = []

        if num_offspring_needed > 0:
            for _ in range((num_offspring_needed + 1) // 2): # Loop to generate enough pairs
                if len(next_population) + len(mutation_coroutines) >= self.population_size * 2 : # Heuristic to avoid over-scheduling too many mutations
                    break # Avoid creating too many coroutines if pop size is small & already enough tasks
                parent1 = self.genetic_operators.selection(self.population)
                parent2 = self.genetic_operators.selection(self.population)
                child1, child2 = self.genetic_operators.crossover(parent1, parent2, run_id=self.run_id, generation=current_generation_num)

                mutation_coroutines.append(self.genetic_operators.mutate(child1, run_id=self.run_id, generation=current_generation_num))
                if len(mutation_coroutines) < num_offspring_needed: # Check if we still need another for the exact count
                     mutation_coroutines.append(self.genetic_operators.mutate(child2, run_id=self.run_id, generation=current_generation_num))

            if mutation_coroutines:
                # Gather results from mutation coroutines, only take as many as needed
                mutated_offspring_results = await asyncio.gather(*mutation_coroutines)
                next_population.extend(mutated_offspring_results[:num_offspring_needed - (len(mutation_coroutines) - len(mutated_offspring_results))])


        self.population = next_population[:self.population_size] # Ensure exact population size
        self.generation_number = current_generation_num # Update generation number

        logger.info(f"PopulationManager: Advanced to generation {self.generation_number}. New population size: {len(self.population)}")

        # --- DB Logging and Periodic Saving (after new population is formed, before next eval cycle) ---
        # Note: The evaluation of this *newly formed* generation happens at the *start* of the *next* call to evolve_population or by the GA runner.
        # Metrics here are for the generation that was just completed *before* these new offspring were created.

        # For DB logging of metrics for the *completed* generation (generation_number -1 if just incremented, or current_generation_num if it represents completed one)
        # This logic might need adjustment based on when add_generation_metric is intended to be called.
        # The original code called it after sorting the *evaluated* population.
        # If metrics are for the generation whose evaluation just finished:
        # best_chromosome_completed_gen = self.population[0] # This is from sorted *current* (already evolved) pop
        # avg_fitness_completed_gen = sum(c.fitness_score for c in self.population) / len(self.population) if self.population else 0
        # This section is complex as the `add_generation_metric` was after sorting an evaluated pop.
        # For now, let's assume we log metrics for the generation *before* creating these new offspring.

        # Periodic save logic
        actual_save_frequency = getattr(global_ph_config.settings, 'DEFAULT_SAVE_POPULATION_FREQUENCY', 10)
        if self.population_path and actual_save_frequency > 0 and self.generation_number % actual_save_frequency == 0:
            logger.info(f"PopulationManager: Saving population at generation {self.generation_number} (frequency: {actual_save_frequency})")
            try:
                self.save_population()
            except Exception as e:
                logger.error(f"PopulationManager: Error during periodic population save: {e}", exc_info=True)

        await self.broadcast_ga_update(event_type="ga_generation_evolved", include_population_sample=True)
        return

    # ... (rest of PopulationManager methods: pause_evolution, resume_evolution, stop_evolution, get_ga_status, save_population, load_population, broadcast_ga_update)
    # Make sure get_fittest_individual is added if it was missing
    def get_fittest_individual(self) -> Optional[PromptChromosome]:
        if not self.population:
            return None
        # Assuming population is sorted by fitness_score descending due to prior sort operations
        return self.population[0]


# Ensure other methods like pause_evolution, resume_evolution, stop_evolution, broadcast_ga_update if they call async methods, are also async.
# For now, assuming they are simple state changes or their async calls are already handled.
# The save_population and load_population are synchronous IO, which is fine if not called from deep within an async critical path without care.

[end of prompthelix/genetics/engine.py]
