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


logger = logging.getLogger(__name__) # Added logger for this module

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
        return child1, child2

    def mutate(
        self,
        chromosome: PromptChromosome,
        mutation_rate: float = 1.0,
        gene_mutation_prob: float = 1.0, # Retained for compatibility
        target_style: Optional[str] = None,
        **_
    ) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.fitness_score = 0.0

        if not self.mutation_strategies:
            logger.warning(f"GeneticOperators: No mutation strategies available. Chromosome {chromosome.id} will not be mutated.")
            return mutated_chromosome

        if random.random() > mutation_rate:
            logger.debug(f"GeneticOperators: Mutation skipped for chromosome {chromosome.id} due to mutation_rate ({mutation_rate}).")
            return mutated_chromosome

        # Ensure there's at least one strategy (NoOperation by default in __init__)
        strategy = random.choice(self.mutation_strategies)

        logger.info(f"GeneticOperators: Applying mutation strategy '{strategy.__class__.__name__}' to chromosome {chromosome.id}.")

        # Strategy's mutate method should return a new (or modified in-place) chromosome.
        # We pass the clone `mutated_chromosome` to it.
        post_strategy_chromosome = strategy.mutate(mutated_chromosome)
        post_strategy_chromosome.parent_ids = [str(chromosome.id)]
        post_strategy_chromosome.mutation_strategy = strategy.__class__.__name__
        post_strategy_chromosome.fitness_score = 0.0 # Ensure fitness is reset

        final_chromosome = post_strategy_chromosome

        if self.style_optimizer_agent and target_style:
            logger.info(f"GeneticOperators: Attempting style optimization with target_style '{target_style}' for chromosome {final_chromosome.id}.")
            try:
                style_request_data = {
                    "prompt_chromosome": final_chromosome,
                    "target_style": target_style
                }
                optimized_result = self.style_optimizer_agent.process_request(style_request_data)

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

        return final_chromosome

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

class FitnessEvaluator:
    """Evaluator wrapper, potentially simplified for tests or base for complex evaluators."""

    def __init__(self, results_evaluator_agent, execution_mode, llm_settings: Optional[Dict] = None, **_): # Added llm_settings
        if not isinstance(results_evaluator_agent, BaseAgent): # Assuming BaseAgent or a more specific type
            raise TypeError("results_evaluator_agent must be an instance of a BaseAgent derivative.")
        self.results_evaluator_agent = results_evaluator_agent
        self.execution_mode = execution_mode
        self.llm_settings = llm_settings if llm_settings is not None else {}


    async def evaluate(self, chromosome: PromptChromosome, task_description: str, success_criteria: Optional[Dict] = None) -> float: # Changed to async
        """
        Evaluates a chromosome's fitness.
        In TEST mode, simulates LLM output. Otherwise, this basic evaluator might not be fully functional
        for REAL mode without actual LLM calls.
        """
        if not isinstance(chromosome, PromptChromosome):
            raise TypeError("chromosome must be an instance of PromptChromosome.")

        prompt_string = chromosome.to_prompt_string()
        logger.debug(f"FitnessEvaluator: Evaluating chromosome {chromosome.id} with prompt string: \"{prompt_string[:200]}...\"")
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
            logger.debug(f"FitnessEvaluator: ExecutionMode is not TEST. LLM output is empty for chromosome {chromosome.id}.")
            pass # llm_output remains "" if not TEST mode, to be evaluated by REA

        if self.execution_mode == ExecutionMode.TEST: # Added this line to ensure the log is conditional
            logger.debug(f"FitnessEvaluator: ExecutionMode is TEST. Simulated LLM output for chromosome {chromosome.id}: \"{llm_output[:100]}...\"")


        eval_request_data = {
            "prompt_chromosome": chromosome.id, # Logging ID instead of full object for brevity
            "llm_output_snippet": llm_output[:100] + "..." if llm_output else "N/A",
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria is not None else {}
        }
        logger.debug(f"FitnessEvaluator: Sending request to ResultsEvaluatorAgent for chromosome {chromosome.id}: {eval_request_data}")

        evaluation_result = await self.results_evaluator_agent.process_request( # Added await
            {
                "prompt_chromosome": chromosome, # Agent expects the full chromosome
                "llm_output": llm_output,
                "task_description": task_description,
                "success_criteria": success_criteria if success_criteria is not None else {}
            }
        )
        logger.debug(f"FitnessEvaluator: Received evaluation result for chromosome {chromosome.id}: {evaluation_result}")

        fitness_score = evaluation_result.get("fitness_score", 0.0)
        chromosome.fitness_score = fitness_score
        logger.info(f"FitnessEvaluator: Chromosome {chromosome.id} evaluated. Fitness: {fitness_score:.4f}")
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

        if not isinstance(genetic_operators, GeneticOperators):
            raise TypeError("genetic_operators must be an instance of GeneticOperators.")
        if not isinstance(fitness_evaluator, FitnessEvaluator): # Assuming FitnessEvaluator is the base or expected type
            raise TypeError("fitness_evaluator must be an instance of FitnessEvaluator.")
        if not isinstance(prompt_architect_agent, BaseAgent): # Check against BaseAgent or a more specific type if available
            # The test expects PromptArchitectAgent, but PromptArchitectAgent inherits BaseAgent
            from prompthelix.agents.architect import PromptArchitectAgent # Local import for isinstance check if needed
            if not isinstance(prompt_architect_agent, PromptArchitectAgent):
                 raise TypeError("prompt_architect_agent must be an instance of PromptArchitectAgent.")

        if not isinstance(population_size, int) or population_size <= 0:
            # Aligning with test test_init_invalid_population_size which expects positive.
            raise ValueError("Population size must be positive.")
        # No warning needed for population_size = 0 here as it's now an error.

        if not isinstance(elitism_count, int) or elitism_count < 0:
            raise ValueError("Elitism count must be non-negative.") # Simpler message
        if elitism_count > population_size : # Check this even if population_size is 0 (where elitism_count must also be 0)
            # This also covers cases where population_size might be 0 and elitism_count is > 0.
            raise ValueError("Elitism count cannot exceed population size.") # Simpler message


        self.population: List[PromptChromosome] = []
        self.generation_number = 0
        self.message_bus = message_bus
        self.status: str = "IDLE" # Possible statuses: IDLE, INITIALIZING, RUNNING, PAUSED, STOPPED, COMPLETED, ERROR
        self.is_paused: bool = False
        self.should_stop: bool = False

    async def initialize_population(self, initial_task_description: str, initial_keywords: List[str],
                              constraints: Optional[Dict] = None,
                              success_criteria: Optional[Dict] = None):
        self.status = "INITIALIZING"
        logger.info(f"PopulationManager: Initializing population. Task: '{initial_task_description}', Keywords: {initial_keywords}")
        await self.broadcast_ga_update(event_type="population_initialization_started")
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

        new_chromosomes = []
        for _ in range(num_to_generate):
            request_data = {
                "task_description": initial_task_description,
                "keywords": initial_keywords,
                "constraints": constraints if constraints is not None else {}
            }
            # Ensure that a valid PromptChromosome is returned
            chromosome_candidate = self.prompt_architect_agent.process_request(request_data)
            if isinstance(chromosome_candidate, PromptChromosome):
                new_chromosomes.append(chromosome_candidate)
            else:
                logger.error(f"PopulationManager: PromptArchitectAgent returned an invalid type ({type(chromosome_candidate)}) instead of a PromptChromosome. Skipping this candidate.")
                # Optionally, create a very basic fallback chromosome here or just skip
                # For now, skipping seems safer than adding a potentially malformed None.

        if new_chromosomes:
            self.population.extend(new_chromosomes)
        logger.info(f"PopulationManager: Generated {len(self.population)} initial chromosomes (valid only).")

        # Evaluate the initial population
        logger.info("PopulationManager: Evaluating initial population...")
        evaluation_tasks = []
        for i, chromo in enumerate(self.population):
            logger.debug(f"PopulationManager: Scheduling evaluation for initial chromosome {i+1}/{len(self.population)}, ID: {chromo.id}")
            evaluation_tasks.append(self.fitness_evaluator.evaluate(chromo, initial_task_description, success_criteria))

        await asyncio.gather(*evaluation_tasks) # Evaluate concurrently
            # FitnessEvaluator now logs the score, so no need to duplicate here unless for summary

        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        logger.info("PopulationManager: Initial population evaluation complete and sorted.")
        if self.population:
            for i in range(min(3, len(self.population))): # Log top 3
                chromo = self.population[i]
                logger.info(f"PopulationManager: Initial Top {i+1}: Chromosome ID {chromo.id}, Fitness {chromo.fitness_score:.4f}, Prompt: \"{chromo.to_prompt_string()[:150]}...\"")
        else:
            logger.warning("PopulationManager: Population is empty after initialization and evaluation.")

        self.status = "IDLE" # Or RUNNING if it immediately proceeds to evolve
        await self.broadcast_ga_update(event_type="population_initialized", additional_data={"population_size": len(self.population)})


    async def pause_evolution(self):
        self.is_paused = True
        self.status = "PAUSED"
        await self.broadcast_ga_update(event_type="ga_paused")
        # logger.info(f"PopulationManager (ID: {id(self)}): Evolution paused.")

    async def resume_evolution(self):
        self.is_paused = False
        self.status = "RUNNING"
        await self.broadcast_ga_update(event_type="ga_resumed")
        # logger.info(f"PopulationManager (ID: {id(self)}): Evolution resumed.")

    async def stop_evolution(self):
        self.should_stop = True
        self.is_paused = False # Clear pause state on stop
        self.status = "STOPPING"  # Set STOPPING before broadcasting
        await self.broadcast_ga_update(event_type="ga_stopping")
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
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            logger.info(f"PopulationManager: Population saved successfully to {file_path}")
        except (IOError, OSError) as e:
            logger.error(f"Error saving population to {file_path}: {e}", exc_info=True)

    def load_population(self, file_path: str) -> None:
        if not os.path.exists(file_path):
            logger.info(f"PopulationManager: No population file at {file_path}; starting fresh.")
            self.population = []
            self.generation_number = 0
            return

        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                content = fh.read()
                if not content.strip(): # Check for empty or whitespace-only content
                    logger.error(f"Error loading population from {file_path}: File is empty.")
                    self.population = []
                    self.generation_number = 0
                    return

                data = json.loads(content) # Use json.loads after reading

            if not isinstance(data, dict):
                logger.error(f"Error loading population from {file_path}: File content is not a JSON object (dictionary). Loaded type: {type(data)}")
                self.population = []
                self.generation_number = 0
                return

            self.generation_number = data.get("generation_number", 0)
            loaded_population_data = data.get("population", [])

            if not isinstance(loaded_population_data, list):
                logger.error(f"Error loading population from {file_path}: 'population' key is not a list. Found type: {type(loaded_population_data)}")
                self.population = []
            else:
                self.population = [
                    PromptChromosome(
                        genes=item.get("genes", []),
                        fitness_score=item.get("fitness_score", 0.0),
                        parent_ids=item.get("parents", []),
                        mutation_strategy=item.get("mutation_strategy"),
                    )
                    for item in loaded_population_data if isinstance(item, dict)
                ]

            if self.population:
                self.population_size = len(self.population)

        except json.JSONDecodeError as e:
            logger.error(f"Error loading population from {file_path} due to JSON decoding error: {e}")
            self.population = []
            self.generation_number = 0
        except Exception as e:
            logger.error(f"Unexpected error loading population from {file_path}: {e}", exc_info=True)
            self.population = []
            self.generation_number = 0

    def get_fittest_individual(self) -> Optional[PromptChromosome]:
        """Return the chromosome with the highest fitness or None."""
        if not self.population:
            return None
        return max(self.population, key=lambda c: c.fitness_score)

    async def broadcast_ga_update(
        self,
        event_type: str,
        selected_parent_ids=None,
        additional_data=None
    ):
        if (
            self.message_bus
            and getattr(self.message_bus, "connection_manager", None)
        ):
            fitness_scores = [c.fitness_score for c in self.population]

            if fitness_scores:
                best_fitness = max(fitness_scores)
                fitness_min = min(fitness_scores)
                fitness_max = max(fitness_scores)
                fitness_mean = statistics.mean(fitness_scores)
                fitness_median = statistics.median(fitness_scores)
                fitness_std_dev = (
                    statistics.stdev(fitness_scores)
                    if len(fitness_scores) > 1
                    else 0.0
                )
                fittest_chromosome_string = (
                    self.get_fittest_individual().to_prompt_string()
                )
            else:
                best_fitness = None
                fitness_min = None
                fitness_max = None
                fitness_mean = None
                fitness_median = None
                fitness_std_dev = None
                fittest_chromosome_string = None

            payload = {
                "type": event_type,
                "data": {
                    "status": self.status,
                    "generation": self.generation_number,
                    "population_size": len(self.population),
                    "best_fitness": best_fitness,
                    "fitness_min": fitness_min,
                    "fitness_max": fitness_max,
                    "fitness_mean": fitness_mean,
                    "fitness_median": fitness_median,
                    "fitness_std_dev": fitness_std_dev,
                    "fittest_chromosome_string": fittest_chromosome_string,
                    "is_paused": self.is_paused,
                    "should_stop": self.should_stop,
                    "selected_parent_ids": selected_parent_ids,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
            if additional_data:
                payload["data"].update(additional_data)

            try:
                await self.message_bus.connection_manager.broadcast_json(payload)
            except Exception as e:
                logger.error(f"Error during broadcast_json in broadcast_ga_update: {e}", exc_info=True)
                # Decide if this should raise or just log. For now, logging.


    async def evolve_population(self, task_description: str, success_criteria: Optional[Dict] = None, db_session=None, experiment_run=None): # Changed to async def
        if not self.population:
            logger.warning("PopulationManager: evolve_population called with an empty population.")
            return

        logger.info(f"PopulationManager: Starting evolution for generation {self.generation_number + 1}. Population size: {len(self.population)}")

        # Evaluate fitness for each chromosome
        logger.debug(f"PopulationManager: Evaluating fitness for {len(self.population)} chromosomes in generation {self.generation_number + 1}...")

        evaluation_tasks = []
        for i, chromosome in enumerate(self.population):
            logger.debug(f"PopulationManager: Scheduling evaluation for chromosome {i+1}/{len(self.population)}, ID: {chromosome.id} for generation {self.generation_number + 1}")
            evaluation_tasks.append(
                self.fitness_evaluator.evaluate(chromosome, task_description, success_criteria)
            )

        # Use return_exceptions=True to handle individual task failures
        evaluation_results_or_exceptions = await asyncio.gather(*evaluation_tasks, return_exceptions=True)

        for i, result_or_exc in enumerate(evaluation_results_or_exceptions):
            chromosome_for_this_task = self.population[i] # Assuming population order matches tasks
            if isinstance(result_or_exc, Exception):
                logger.error(f"PopulationManager: Error evaluating chromosome {chromosome_for_this_task.id} during evolution: {result_or_exc}. Assigning fitness 0.0.", exc_info=False) # Set exc_info=False as it's already an exception object
                chromosome_for_this_task.fitness_score = 0.0 # Assign default low fitness
            # else:
                # Fitness score is already set on the chromosome by the evaluate method if successful
                # No need to re-assign it here from result_or_exc unless evaluate returned something else
                pass


        # Sort population by fitness score in descending order
        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        logger.info(f"PopulationManager: Population sorted for generation {self.generation_number + 1}.")

        if not self.population: # Should not happen if we checked above, but defensive
            logger.error("PopulationManager: Population became empty after evaluation and sorting in evolve_population. This should not happen.")
            return

        best_chromosome = self.population[0] # Best is now the first element after sorting
        avg_fitness = sum(c.fitness_score for c in self.population) / len(self.population)

        logger.info(f"PopulationManager: Generation {self.generation_number + 1} evaluation complete. Best Fitness: {best_chromosome.fitness_score:.4f}, Avg Fitness: {avg_fitness:.4f}")
        logger.info(f"PopulationManager: Best chromosome ID {best_chromosome.id} in gen {self.generation_number + 1}, Prompt: \"{best_chromosome.to_prompt_string()[:150]}...\"")

        try:
            from prompthelix.services import add_generation_metric
            add_generation_metric( # Note: generation_number is 0 for initial, then increments
                db_session,
                experiment_run,
                self.generation_number, # This might be off by 1 if it's called before incrementing gen_number for the current evolution
                best_chromosome.fitness_score,
                avg_fitness,
                0.0, # Assuming min_fitness or other metric might go here for population_diversity
                len(self.population) # Pass current population size
            )
            logger.debug(f"PopulationManager: Generation {self.generation_number} metrics (best: {best_chromosome.fitness_score:.4f}, avg: {avg_fitness:.4f}, size: {len(self.population)}) sent to DB.")
        except Exception as e:
            logger.warning(f"PopulationManager: Failed to add generation metric to DB for generation {self.generation_number}. Error: {e}", exc_info=True)
            pass
