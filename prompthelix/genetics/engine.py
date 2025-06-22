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
            "run_id": _.get("run_id"), # Assuming run_id is passed in **_ or directly
            "agent_id": "ga_engine",
            "generation": _.get("generation"),
            "chromosome_id": str(child1.id),
            "prompt_text": child1.to_prompt_string(),
            "fitness_score": child1.fitness_score, # Will be 0.0 or unevaluated
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
            "fitness_score": child2.fitness_score, # Will be 0.0 or unevaluated
            "operation": "crossover",
            "parent_ids": child2.parent_ids,
            "mutation_strategy": None,
            "metadata": {"source_parents": [str(parent1.id), str(parent2.id)]}
        })
        return child1, child2

    def mutate(
        self,
        chromosome: PromptChromosome,
        mutation_rate: float = 1.0,
        gene_mutation_prob: float = 1.0, # Retained for compatibility
        target_style: Optional[str] = None,
        run_id: Optional[str] = None, # Added run_id
        generation: Optional[int] = None, # Added generation
        **_ # Captures run_id and generation if passed this way too
    ) -> PromptChromosome:
        mutated_chromosome = chromosome.clone() # Clone first to keep original intact for logging if needed
        original_chromosome_id = str(chromosome.id) # Save original ID for parent_ids field
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

        # Log the mutation event
        # The `final_chromosome` is the one that resulted from mutation (and possibly style optimization)
        # Its parent_ids should have been set to the original chromosome's ID during the process.
        # If style optimization occurred, final_chromosome might be a new object, ensure its parent_ids are correct.
        # The `mutate` method in PromptChromosome's strategy should set parent_ids.
        # Let's ensure `final_chromosome.parent_ids` is correctly set before logging.
        # If `strategy.mutate` returns a new chromosome, it must set `parent_ids`.
        # If it modifies in-place, `post_strategy_chromosome.parent_ids = [str(chromosome.id)]` handles it.
        # If style optimizer returns a *new* chromosome, its parentage should trace to `post_strategy_chromosome`.
        # For simplicity here, we log `final_chromosome` and assume its `parent_ids` correctly reflect its immediate predecessor.

        ga_logger.info({
            "run_id": run_id if run_id else _.get("run_id"), # Prioritize direct param
            "agent_id": "ga_engine",
            "generation": generation if generation else _.get("generation"), # Prioritize direct param
            "chromosome_id": str(final_chromosome.id),
            "prompt_text": final_chromosome.to_prompt_string(),
            "fitness_score": final_chromosome.fitness_score, # Should be 0.0 as it's reset
            "operation": "mutation",
            "parent_ids": [original_chromosome_id], # Parent is the chromosome before mutation
            "mutation_strategy": final_chromosome.mutation_strategy, # This is set by strategy.mutate
            "metadata": {"applied_strategy": strategy.__class__.__name__, "target_style_attempted": bool(target_style)}
        })

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


    async def evaluate(self, chromosome: PromptChromosome, task_description: str, success_criteria: Optional[Dict] = None, run_id: Optional[str] = None, generation: Optional[int] = None) -> float: # Changed to async
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
            logger.debug(f"FitnessEvaluator: ExecutionMode is TEST. Simulated LLM output for chromosome {chromosome.id}: \"{llm_output[:100]}...\"")
        else: # REAL mode or any other mode that is not TEST
            logger.info(f"FitnessEvaluator: ExecutionMode is {self.execution_mode.name}. Attempting real LLM call for chromosome {chromosome.id}.")
            provider = self.llm_settings.get("provider", "openai") # Default to openai
            model = self.llm_settings.get("model") # Let call_llm_api handle default model if None

            # Ensure llm_settings are passed correctly, call_llm_api might need them for specific configurations
            # However, call_llm_api primarily uses global settings or db for API keys.
            # For now, we assume provider and model are the main things from self.llm_settings here.
            # The `db` parameter for `call_llm_api` is not readily available here.
            # `call_llm_api` can fetch keys from settings if db is None.
            try:
                llm_output = await call_llm_api(
                    prompt=prompt_string,
                    provider=provider,
                    model=model,
                    db=None # No direct DB session here, API keys should be in settings
                )
                logger.info(f"FitnessEvaluator: Real LLM call successful for chromosome {chromosome.id}. Output snippet: \"{llm_output[:100]}...\"")
                if llm_output.startswith("ERROR:") or llm_output in [
                    "API_KEY_MISSING_ERROR", "RATE_LIMIT_ERROR", "AUTHENTICATION_ERROR",
                    "API_CONNECTION_ERROR", "INVALID_REQUEST_ERROR", "API_ERROR", "OPENAI_ERROR",
                    "UNEXPECTED_OPENAI_CALL_ERROR", "ANTHROPIC_ERROR", "UNEXPECTED_ANTHROPIC_CALL_ERROR",
                    "MALFORMED_CLAUDE_RESPONSE_CONTENT", "EMPTY_CLAUDE_RESPONSE", "BLOCKED_PROMPT_ERROR",
                    "EMPTY_GOOGLE_RESPONSE", "GOOGLE_SDK_ERROR", "UNEXPECTED_GOOGLE_CALL_ERROR",
                    "UNSUPPORTED_PROVIDER_ERROR", "UNEXPECTED_CALL_LLM_API_ERROR"
                ] or (isinstance(llm_output, str) and llm_output.startswith("GENERATION_STOPPED_")):
                    logger.warning(f"FitnessEvaluator: LLM call for chromosome {chromosome.id} returned an error status: {llm_output}")
            except Exception as e:
                logger.error(f"FitnessEvaluator: Exception during real LLM call for chromosome {chromosome.id}: {e}", exc_info=True)
                llm_output = f"ERROR: Exception during LLM call - {str(e)}"


        eval_request_data = {
            "prompt_chromosome": chromosome.id, # Logging ID instead of full object for brevity
            "llm_output_snippet": llm_output[:100] + "..." if llm_output else "N/A",
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria is not None else {}
        }
        logger.debug(f"FitnessEvaluator: Sending request to ResultsEvaluatorAgent for chromosome {chromosome.id}: {eval_request_data}")

        # --- Synthetic Test Generation and Evaluation Logic ---
        num_synthetic_inputs = self.llm_settings.get("num_synthetic_inputs_for_evaluation", 0) # Default to 0 (disabled)

        if self.execution_mode != ExecutionMode.TEST and num_synthetic_inputs > 0:
            logger.info(f"FitnessEvaluator: Generating {num_synthetic_inputs} synthetic inputs for chromosome {chromosome.id} based on task: '{task_description}'")

            synthetic_inputs_prompt = (
                f"Given the task description: '{task_description}', generate {num_synthetic_inputs} diverse and concise input scenarios "
                f"that a prompt designed for this task should be able_to handle. "
                f"Each scenario should be on a new line. Do not number them or add extra text."
            )

            # Use a general-purpose LLM for generating these inputs, can be configured via llm_settings
            generation_provider = self.llm_settings.get("synthetic_input_generation_provider", "openai")
            generation_model = self.llm_settings.get("synthetic_input_generation_model") # Default model handled by call_llm_api

            generated_inputs_str = await call_llm_api(
                prompt=synthetic_inputs_prompt,
                provider=generation_provider,
                model=generation_model,
                db=None
            )

            synthetic_inputs = []
            if not generated_inputs_str.startswith("ERROR:") and generated_inputs_str:
                synthetic_inputs = [line.strip() for line in generated_inputs_str.split('\n') if line.strip()]
                logger.info(f"FitnessEvaluator: Generated {len(synthetic_inputs)} synthetic inputs: {synthetic_inputs}")
            else:
                logger.warning(f"FitnessEvaluator: Failed to generate synthetic inputs or received error: {generated_inputs_str}")

            if synthetic_inputs:
                fitness_scores = []
                original_prompt_text = chromosome.to_prompt_string() # Cache original prompt

                for i, synthetic_input in enumerate(synthetic_inputs):
                    # Combine original prompt with synthetic input
                    # A simple concatenation might work, or a more structured approach if prompts have placeholders.
                    # Assuming simple concatenation for now: prompt + "\n\nInput: " + synthetic_input
                    combined_prompt_text = f"{original_prompt_text}\n\nInput Scenario: {synthetic_input}"
                    logger.debug(f"FitnessEvaluator: Evaluating synthetic test {i+1}/{len(synthetic_inputs)} for chromosome {chromosome.id} with input: '{synthetic_input}'")

                    # Get LLM output for the combined prompt
                    current_provider = self.llm_settings.get("provider", "openai")
                    current_model = self.llm_settings.get("model")

                    synthetic_llm_output = await call_llm_api(
                        prompt=combined_prompt_text,
                        provider=current_provider,
                        model=current_model,
                        db=None
                    )

                    if synthetic_llm_output.startswith("ERROR:"):
                        logger.warning(f"FitnessEvaluator: Error getting LLM output for synthetic test {i+1} (input: '{synthetic_input}'). Error: {synthetic_llm_output}")
                        fitness_scores.append(0.0) # Penalize errors heavily
                        continue

                    # Evaluate this specific output using ResultsEvaluatorAgent
                    # The chromosome object itself is passed, but the llm_output is specific to this synthetic test.
                    # Task description and success criteria remain the same.
                    synthetic_eval_result = await self.results_evaluator_agent.process_request({
                        "prompt_chromosome": chromosome, # Pass the original chromosome for context
                        "llm_output": synthetic_llm_output,
                        "task_description": task_description, # Original task
                        "success_criteria": success_criteria, # Original criteria
                        "synthetic_input_context": synthetic_input # Provide context to evaluator if it can use it
                    })
                    score = synthetic_eval_result.get("fitness_score", 0.0)
                    fitness_scores.append(score)
                    logger.debug(f"FitnessEvaluator: Synthetic test {i+1} for C:{chromosome.id} scored: {score:.4f}")

                if fitness_scores:
                    final_fitness = sum(fitness_scores) / len(fitness_scores)
                    logger.info(f"FitnessEvaluator: Chromosome {chromosome.id} final fitness (avg over {len(synthetic_inputs)} synthetic tests): {final_fitness:.4f}")
                else: # Should not happen if synthetic_inputs was non-empty, but as a fallback
                    logger.warning(f"FitnessEvaluator: No scores from synthetic tests for {chromosome.id}, though inputs were generated. Using 0.0 fitness.")
                    final_fitness = 0.0

                chromosome.fitness_score = final_fitness
                return final_fitness # Return the aggregated fitness

        # --- Original Evaluation Logic (if no synthetic tests or in TEST mode) ---
        evaluation_result = await self.results_evaluator_agent.process_request( # Added await
            {
                "prompt_chromosome": chromosome, # Agent expects the full chromosome
                "llm_output": llm_output, # This is mock output in TEST, or single real output if synthetic tests disabled
                "task_description": task_description,
                "success_criteria": success_criteria if success_criteria is not None else {}
            }
        )
        logger.debug(f"FitnessEvaluator: Received evaluation result for chromosome {chromosome.id}: {evaluation_result}")

        fitness_score = evaluation_result.get("fitness_score", 0.0)
        chromosome.fitness_score = fitness_score

        logger.info(f"FitnessEvaluator: Chromosome {chromosome.id} evaluated (single pass/mock). Fitness: {fitness_score:.4f}")

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
        message_bus=None,
        population_path: Optional[str] = None
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
        self.population_path = population_path
        self.status: str = "IDLE" # Possible statuses: IDLE, INITIALIZING, RUNNING, PAUSED, STOPPED, COMPLETED, ERROR
        self.is_paused: bool = False
        self.should_stop: bool = False
        self.run_id: Optional[str] = None # Added to store the experiment run ID

    async def initialize_population(self, initial_task_description: str, initial_keywords: List[str],
                              constraints: Optional[Dict] = None,
                              success_criteria: Optional[Dict] = None,
                              run_id: Optional[str] = None): # Added run_id parameter
        self.status = "INITIALIZING"
        if run_id: # Store run_id if provided, useful if initialization is called separately
            self.run_id = run_id
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Initializing population. Task: '{initial_task_description}', Keywords: {initial_keywords}")
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
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Generated {len(self.population)} initial chromosomes (valid only).")

        # Log "initialization" event for architected chromosomes before evaluation
        for chromo in self.population: # Assuming self.population contains only newly architected ones or ones from initial_prompt_str
            # If chromo was from initial_prompt_str, its fitness is 0.0.
            # If from architect, fitness is also 0.0 before explicit evaluation.
            ga_logger.info({
                "run_id": self.run_id,
                "agent_id": "ga_engine", # Or "prompt_architect_agent" if we want to be specific about source
                "generation": self.generation_number, # Initial population is generation 0
                "chromosome_id": str(chromo.id),
                "prompt_text": chromo.to_prompt_string(),
                "fitness_score": chromo.fitness_score, # Will be 0.0
                "operation": "initialization",
                "parent_ids": chromo.parent_ids, # Likely empty or None for initial
                "mutation_strategy": None,
                "metadata": {"source": "architect" if not self.initial_prompt_str or chromo.genes != [self.initial_prompt_str] else "initial_prompt_str"}
            })

        # Evaluate the initial population
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Evaluating initial population...")
        evaluation_tasks = []
        for i, chromo in enumerate(self.population):
            logger.debug(f"PopulationManager (Run ID: {self.run_id}): Scheduling evaluation for initial chromosome {i+1}/{len(self.population)}, ID: {chromo.id}")
            evaluation_tasks.append(self.fitness_evaluator.evaluate(
                chromosome=chromo,
                task_description=initial_task_description,
                success_criteria=success_criteria,
                run_id=self.run_id, # Pass run_id
                generation=self.generation_number # generation is 0 for initial population
            ))

        await asyncio.gather(*evaluation_tasks) # Evaluate concurrently
            # FitnessEvaluator now logs the score, so no need to duplicate here unless for summary

        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        logger.info(f"PopulationManager (Run ID: {self.run_id}): Initial population evaluation complete and sorted.")
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

    def save_population(self, file_path: Optional[str] = None) -> None:
        file_path = file_path or self.population_path
        if not file_path:
            raise ValueError("population_path must be provided to save_population")
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

def load_population(self, file_path: Optional[str] = None) -> None:
    file_path = file_path or self.population_path
    if not file_path or not os.path.exists(file_path):
        logger.info(f"PopulationManager: No population file at {file_path}; starting fresh.")
        self.population = []
        self.generation_number = 0
        return

    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            content = fh.read()
            if not content.strip():  # Check for empty or whitespace-only content
                logger.error(f"Error loading population from {file_path}: File is empty.")
                self.population = []
                self.generation_number = 0
                return

            data = json.loads(content)

        if not isinstance(data, dict):
            logger.error(
                f"Error loading population from {file_path}: File content is not a JSON object (dictionary). "
                f"Loaded type: {type(data)}"
            )
            self.population = []
            self.generation_number = 0
            return

        self.generation_number = data.get("generation_number", 0)
        loaded_population_data = data.get("population", [])

        if not isinstance(loaded_population_data, list):
            logger.error(
                f"Error loading population from {file_path}: 'population' key is not a list. "
                f"Found type: {type(loaded_population_data)}"
            )
            self.population = []
        else:
            self.population = [
                PromptChromosome(
                    genes=item.get("genes", []),
                    fitness_score=item.get("fitness_score", 0.0),
                    parent_ids=item.get("parents", []),
                    mutation_strategy=item.get("mutation_strategy"),
                )
                for item in loaded_population_data
                if isinstance(item, dict)
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
                self.fitness_evaluator.evaluate(
                    chromosome=chromosome,
                    task_description=task_description,
                    success_criteria=success_criteria,
                    run_id=self.run_id, # Pass run_id
                    generation=self.generation_number + 1 # Pass current generation number
                )
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

        # Apply user feedback if db_session is available
        if db_session:
            logger.info(f"PopulationManager: Applying user feedback for generation {self.generation_number + 1}.")
            current_run_id = experiment_run.id if experiment_run else None
            for chromosome in self.population:
                all_feedback_for_chromosome = []
                # Get feedback by chromosome ID (most specific)
                feedback_by_id = crud.get_user_feedback_for_chromosome(db_session, chromosome_id_str=str(chromosome.id))
                all_feedback_for_chromosome.extend(feedback_by_id)

                # Optional: Get feedback by prompt content snapshot (less specific, but can catch identical prompts)
                # This might be slow if there's a lot of feedback data and no indexing on prompt_content_snapshot.
                # feedback_by_content = crud.get_user_feedback_for_prompt_content(db_session, prompt_content_snapshot=chromosome.to_prompt_string())
                # all_feedback_for_chromosome.extend(feedback_by_content) # Be careful about duplicates if combining

                if all_feedback_for_chromosome:
                    # Deduplicate feedback if collected from multiple sources, e.g., by feedback ID
                    unique_feedback = {f.id: f for f in all_feedback_for_chromosome}.values()

                    # Prioritize feedback for the current run if available
                    run_specific_feedback = [f for f in unique_feedback if f.ga_run_id == current_run_id]

                    feedback_to_consider = run_specific_feedback if run_specific_feedback else unique_feedback

                    if feedback_to_consider:
                        # Simple strategy: use the average rating of the most recent N feedback items, or just latest.
                        # For now, let's use the average rating from all relevant feedback.
                        avg_rating = sum(f.rating for f in feedback_to_consider) / len(feedback_to_consider)
                        original_fitness = chromosome.fitness_score

                        # Define fitness adjustment scale
                        # Example: Max adjustment of +/- 0.2 to fitness score (assuming fitness is 0-1)
                        # Rating 5: +0.2; Rating 4: +0.1; Rating 3: 0; Rating 2: -0.1; Rating 1: -0.2
                        adjustment = 0.0
                        if avg_rating >= 4.5: # Effectively 5 stars
                            adjustment = 0.2
                        elif avg_rating >= 3.5: # 4 stars
                            adjustment = 0.1
                        elif avg_rating <= 1.5: # 1 star
                            adjustment = -0.2
                        elif avg_rating <= 2.5: # 2 stars
                            adjustment = -0.1

                        chromosome.fitness_score += adjustment
                        # Clamp fitness score to a valid range, e.g., [0, 1] or whatever your system uses.
                        # Assuming fitness_score is typically within [0,1]. Adjust if max can be higher.
                        chromosome.fitness_score = max(0.0, min(chromosome.fitness_score, 1.0))

                        if adjustment != 0:
                            logger.info(f"PopulationManager: Chromosome {chromosome.id} fitness adjusted by {adjustment:.2f} (avg rating: {avg_rating:.2f}). Original: {original_fitness:.4f}, New: {chromosome.fitness_score:.4f}.")
                        # Optionally, log suggested_improvement_text if present and consider for future mutation strategies.

        # Sort population by fitness score in descending order (now includes feedback adjustments)
        self.population.sort(key=lambda c: c.fitness_score, reverse=True)
        logger.info(f"PopulationManager: Population sorted for generation {self.generation_number + 1} (feedback considered if available).")

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
