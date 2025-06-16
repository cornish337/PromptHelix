from __future__ import annotations

import uuid
import copy
import random
from typing import TYPE_CHECKING
import openai
from openai import OpenAIError
from prompthelix.config import settings
import logging
from prompthelix.enums import ExecutionMode # Added import

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from prompthelix.agents.architect import PromptArchitectAgent
    from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
    from prompthelix.agents.style_optimizer import StyleOptimizerAgent


# PromptChromosome class remains unchanged
class PromptChromosome:
    """
    Represents an individual prompt in the genetic algorithm.

    Each chromosome consists of a list of 'genes' (which are typically strings
    or more structured objects representing parts of a prompt), a fitness score,
    and a unique identifier.
    """

    def __init__(self, genes: list | None = None, fitness_score: float = 0.0):
        """
        Initializes a PromptChromosome.

        Args:
            genes (list | None, optional): A list representing the components (genes)
                                           of the prompt. Defaults to an empty list if None.
            fitness_score (float, optional): The initial fitness score of the chromosome.
                                             Defaults to 0.0.
        """
        self.id = uuid.uuid4()
        self.genes: list = [] if genes is None else genes
        self.fitness_score: float = fitness_score

    def calculate_fitness(self) -> float:
        """
        Returns the current fitness score of the chromosome.

        Note: This method simply returns the stored fitness_score. The actual
        calculation and setting of this score are typically handled externally by
        a FitnessEvaluator or a similar mechanism within the genetic algorithm,
        which then updates self.fitness_score.

        Returns:
            float: The fitness score of the chromosome.
        """
        return self.fitness_score

    def to_prompt_string(self, separator: str = "\n") -> str:
        """
        Concatenates all gene strings into a single prompt string.

        This string is typically what would be sent to an LLM for execution.

        Args:
            separator (str, optional): The separator to use between genes.
                                       Defaults to a newline character.

        Returns:
            str: A single string representing the full prompt.
        """
        return separator.join(str(gene) for gene in self.genes)

    def clone(self) -> 'PromptChromosome':
        """
        Creates a deep copy of this chromosome with a new unique ID.

        The genes are deep-copied to ensure the clone is independent of the
        original. The fitness score is also copied.

        Returns:
            PromptChromosome: A new PromptChromosome instance that is a deep copy
                              of the current one, but with a new ID.
        """
        cloned_genes = copy.deepcopy(self.genes)
        cloned_chromosome = PromptChromosome(genes=cloned_genes, fitness_score=self.fitness_score)
        return cloned_chromosome

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the chromosome.

        Returns:
            str: A string detailing the chromosome's ID, fitness, and genes.
        """
        gene_representation = "\n".join([f"  - {str(gene)}" for gene in self.genes])
        if not self.genes:
            gene_representation = "  - (No genes)"
        return (
            f"Chromosome ID: {self.id}\n"
            f"Fitness: {self.fitness_score:.4f}\n"
            f"Genes:\n{gene_representation}"
        )

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the chromosome object.

        Returns:
            str: A string that could ideally be used to recreate the object.
        """
        return f"PromptChromosome(id='{self.id}', genes={self.genes!r}, fitness_score={self.fitness_score:.4f})"


# GeneticOperators class remains unchanged
class GeneticOperators:
    """
    Encapsulates genetic operators like selection, crossover, and mutation
    for PromptChromosome objects.
    """

    def __init__(self, style_optimizer_agent: 'StyleOptimizerAgent' | None = None):
        """Initializes the operator with an optional StyleOptimizerAgent."""
        self.style_optimizer_agent = style_optimizer_agent

    def selection(self, population: list[PromptChromosome], tournament_size: int = 3) -> PromptChromosome:
        """
        Selects an individual from the population using tournament selection.

        Args:
            population (list[PromptChromosome]): A list of PromptChromosome objects.
            tournament_size (int, optional): The number of individuals to select
                                             for the tournament. Defaults to 3.

        Returns:
            PromptChromosome: The individual with the highest fitness_score from
                              the tournament. 

        Raises:
            ValueError: If population is empty or tournament_size is not positive.
        """
        if not population:
            raise ValueError("Population cannot be empty for selection.")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive.")

        actual_tournament_size = min(len(population), tournament_size)
        tournament_contenders = random.sample(population, actual_tournament_size)
        
        winner = tournament_contenders[0]
        for contender in tournament_contenders[1:]:
            if contender.fitness_score > winner.fitness_score:
                winner = contender
        return winner

    def crossover(self, parent1: PromptChromosome, parent2: PromptChromosome, 
                  crossover_rate: float = 0.7) -> tuple[PromptChromosome, PromptChromosome]:
        """
        Performs single-point crossover between two parent chromosomes.

        If random.random() < crossover_rate, crossover occurs. Otherwise, children
        are clones of the parents.


        Args:
            parent1 (PromptChromosome): The first parent chromosome.
            parent2 (PromptChromosome): The second parent chromosome.
            crossover_rate (float, optional): The probability of crossover occurring.
                                             Defaults to 0.7.

        Returns:
            tuple[PromptChromosome, PromptChromosome]: Two new child chromosomes.
        """
        child1_genes = []
        child2_genes = []

        if random.random() < crossover_rate:
            len1 = len(parent1.genes)
            len2 = len(parent2.genes)
            
            if len1 == 0 and len2 == 0:
                child1_genes, child2_genes = [], []
            elif len1 == 0:
                child1_genes, child2_genes = copy.deepcopy(parent2.genes), []
            elif len2 == 0:
                child1_genes, child2_genes = [], copy.deepcopy(parent1.genes)
            else:
                shorter_parent_len = min(len1, len2)
                crossover_point = random.randint(0, shorter_parent_len)
                child1_genes.extend(parent1.genes[:crossover_point])
                child1_genes.extend(parent2.genes[crossover_point:])
                child2_genes.extend(parent2.genes[:crossover_point])
                child2_genes.extend(parent1.genes[crossover_point:])
            
            child1 = PromptChromosome(genes=child1_genes, fitness_score=0.0)
            child2 = PromptChromosome(genes=child2_genes, fitness_score=0.0)
        else:
            child1 = parent1.clone()
            child2 = parent2.clone()
            child1.fitness_score = 0.0 
            child2.fitness_score = 0.0
        return child1, child2

    def mutate(self, chromosome: PromptChromosome, mutation_rate: float = 0.1,
               gene_mutation_prob: float = 0.2,
               target_style: str | None = None) -> PromptChromosome:
        """
        Mutates a chromosome based on mutation_rate and gene_mutation_prob.

        Args:
            chromosome (PromptChromosome): The chromosome to mutate.
            mutation_rate (float, optional): The overall probability that any mutation
                                             will occur on the chromosome. Defaults to 0.1.
            gene_mutation_prob (float, optional): The probability that an individual
                                                  gene will be mutated, if the chromosome
                                                  is selected for mutation. Defaults to 0.2.
        Returns:
            PromptChromosome: A new, potentially mutated, PromptChromosome instance.
        """
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.fitness_score = 0.0

        mutation_applied = False

        if random.random() < mutation_rate:
            genes_modified = False
            for i in range(len(mutated_chromosome.genes)):
                if random.random() < gene_mutation_prob:
                    genes_modified = True
                    mutation_applied = True
                    original_gene_str = str(mutated_chromosome.genes[i])
                    # Added "style_optimization_placeholder" to the list of choices
                    mutation_type = random.choice(["append_char", "reverse_slice", "placeholder_replace", "style_optimization_placeholder"])
                    if mutation_type == "append_char":
                        mutated_chromosome.genes[i] = original_gene_str + random.choice("!*?_")
                    elif mutation_type == "reverse_slice" and len(original_gene_str) > 2:
                        slice_len = random.randint(1, max(2, len(original_gene_str) // 2))
                        start_index = random.randint(0, len(original_gene_str) - slice_len)
                        segment_to_reverse = original_gene_str[start_index : start_index + slice_len]
                        mutated_chromosome.genes[i] = (
                            original_gene_str[:start_index] + 
                            segment_to_reverse[::-1] + 
                            original_gene_str[start_index + slice_len:]
                        )
                    elif mutation_type == "style_optimization_placeholder":
                        print(f"GeneticOperators.mutate: Conceptual call to StyleOptimizerAgent for gene: {original_gene_str}")
                        mutated_chromosome.genes[i] = original_gene_str + " [StyleOptimized_Placeholder]"
                        # In a real scenario:
                        # style_optimizer_payload = {"prompt_chromosome": PromptChromosome(genes=[original_gene_str]), "target_style": "concise"} # or another dynamic style
                        # Assuming a StyleOptimizerAgent instance was available (e.g., passed to __init__ or a global registry):
                        # optimized_segment_chromosome = style_optimizer_agent.process_request(style_optimizer_payload)
                        # mutated_chromosome.genes[i] = optimized_segment_chromosome.genes[0]
                    else: # Fallback for placeholder_replace or short strings (if placeholder_replace was chosen or other conditions failed)
                        mutated_chromosome.genes[i] = "[MUTATED_GENE_SEGMENT]"
            if not genes_modified and len(mutated_chromosome.genes) > 0:
                mutation_applied = True
                gene_to_mutate_idx = random.randrange(len(mutated_chromosome.genes))
                mutated_chromosome.genes[gene_to_mutate_idx] = str(mutated_chromosome.genes[gene_to_mutate_idx]) + "*"

        if mutation_applied and target_style and self.style_optimizer_agent:
            try:
                request = {"prompt_chromosome": mutated_chromosome, "target_style": target_style}
                optimized = self.style_optimizer_agent.process_request(request)
                if isinstance(optimized, PromptChromosome):
                    mutated_chromosome = optimized
            except Exception as e:  # pragma: no cover - logging/exception path
                logger.error(f"Style optimization failed during mutation: {e}")

        return mutated_chromosome


# FitnessEvaluator class remains unchanged
class FitnessEvaluator:
    """
    Evaluates the fitness of PromptChromosome instances.
    This class simulates interaction with an LLM and uses a ResultsEvaluatorAgent
    to determine the fitness score based on the LLM's output.
    """
    def __init__(self, results_evaluator_agent: 'ResultsEvaluatorAgent', execution_mode: ExecutionMode):
        """
        Initializes the FitnessEvaluator.
        Args:
            results_evaluator_agent (ResultsEvaluatorAgent): An instance of
                ResultsEvaluatorAgent that will be used to assess the quality
                of LLM outputs.
            execution_mode (ExecutionMode): The mode of execution (TEST or REAL).
        """
        from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
        if not isinstance(results_evaluator_agent, ResultsEvaluatorAgent):
            raise TypeError("results_evaluator_agent must be an instance of ResultsEvaluatorAgent.")
        self.results_evaluator_agent = results_evaluator_agent
        self.execution_mode = execution_mode # Stored instance attribute
        self.openai_client = None  # Initialize to None

        if not settings.OPENAI_API_KEY and self.execution_mode == ExecutionMode.REAL: # Check only if in REAL mode
            logger.error("OPENAI_API_KEY not found in settings. FitnessEvaluator will not be able to make LLM calls in REAL mode.")
            # The print warning can be removed or kept for immediate console feedback if desired.
            # print("Warning: OPENAI_API_KEY not found in settings. FitnessEvaluator may not function correctly.")
        elif self.execution_mode == ExecutionMode.REAL: # Only init client if REAL mode and key might be present
            try:
                self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("OpenAI client initialized successfully for REAL mode.")
            except Exception as e: # Catch a broader exception during client initialization
                logger.error(f"Error initializing OpenAI client for REAL mode: {e}", exc_info=True)
                # Depending on the desired behavior, either raise an error or ensure client remains None
                # and is handled in _call_llm_api
                # For robustness, we'll let it remain None and _call_llm_api will handle it.
                # raise RuntimeError(f"Failed to initialize OpenAI client: {e}") # Or re-raise
        else: # TEST mode
            logger.info("FitnessEvaluator initialized in TEST mode. LLM calls will be skipped.")

    def _call_llm_api(self, prompt_string: str, model_name: str = "gpt-3.5-turbo") -> str:
        """
        Calls the LLM API with the given prompt string.
        Args:
            prompt_string (str): The prompt to send to the LLM.
            model_name (str, optional): The model to use. Defaults to "gpt-3.5-turbo".
        Returns:
            str: The LLM's response content, or an error message string if the call fails.
        """
        if self.execution_mode == ExecutionMode.TEST:
            logger.info(f"Executing in TEST mode. Returning dummy LLM output for prompt: {prompt_string[:100]}...")
            return "This is a test output from dummy LLM in TEST mode."

        if not self.openai_client: # Check if client was initialized (relevant for REAL mode)
            logger.error("OpenAI client is not initialized. Cannot call LLM API in REAL mode.")
            return "Error: LLM client not initialized for REAL mode."

        logger.info(f"Calling OpenAI API model {model_name} for prompt (first 100 chars): {prompt_string[:100]}...")
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt_string}
                ]
            )
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                content = response.choices[0].message.content.strip()
                logger.info(f"OpenAI API call successful. Response (first 100 chars): {content[:100]}...")
                return content
            else:
                logger.warning(f"OpenAI API call for prompt '{prompt_string[:50]}...' returned no content or unexpected response structure.")
                return "Error: No content from LLM."
        except OpenAIError as e:
            logger.error(f"OpenAI API error for prompt '{prompt_string[:50]}...': {e}", exc_info=True)
            return f"Error: LLM API call failed. Details: {str(e)}"
        except Exception as e: # Catch any other unexpected errors
            logger.critical(f"Unexpected error during OpenAI API call for prompt '{prompt_string[:50]}...': {e}", exc_info=True)
            return f"Error: Unexpected issue during LLM API call. Details: {str(e)}"

    def evaluate(self, chromosome: PromptChromosome, task_description: str,
                 success_criteria: dict | None = None) -> float:
        """
        Evaluates the fitness of a given chromosome.
        This involves converting the chromosome to a prompt string, simulating
        an LLM call with this prompt, and then using the ResultsEvaluatorAgent
        to score the LLM's output. The chromosome's fitness_score attribute
        is updated with the result.
        Args:
            chromosome (PromptChromosome): The chromosome to evaluate.
            task_description (str): A description of the task the prompt is for.
            success_criteria (dict | None, optional): Criteria for evaluating the
                success of the LLM output. Defaults to None.
        Returns:
            float: The calculated fitness score for the chromosome.
        """
        if not isinstance(chromosome, PromptChromosome):
            raise TypeError("chromosome must be an instance of PromptChromosome.")
        prompt_string = chromosome.to_prompt_string()

        # The call to _call_llm_api is now logged within that method.
        llm_output = self._call_llm_api(prompt_string)

        if llm_output.startswith("Error:"):
            logger.warning(f"LLM call for prompt ID {chromosome.id} (text: {prompt_string[:50]}...) failed. Output: {llm_output}")
        # else:
            # Successful LLM output is logged in _call_llm_api.
            # If further logging of the output snippet is desired here, it can be added.
            # logger.debug(f"FitnessEvaluator: LLM Output for prompt ID {chromosome.id}: {llm_output[:150]}...")


        request_data = {
            "prompt_chromosome": chromosome,
            "llm_output": llm_output, # Pass the actual or error string from LLM
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria else {}
        }
        eval_result = self.results_evaluator_agent.process_request(request_data)
        fitness_score = eval_result.get("fitness_score", 0.0)
        chromosome.fitness_score = fitness_score
        print(f"FitnessEvaluator: Evaluated chromosome {chromosome.id}, Fitness: {fitness_score:.4f}")
        return fitness_score

# Updated PopulationManager class
class PopulationManager:
    """
    Manages the population of prompts, including initialization and evolution
    through generations using genetic operators and fitness evaluation.
    """

    def __init__(self, genetic_operators: GeneticOperators, 
                 fitness_evaluator: FitnessEvaluator, 
                 prompt_architect_agent: 'PromptArchitectAgent',
                 population_size: int = 50, 
                 elitism_count: int = 2):
        """
        Initializes the PopulationManager.

        Args:
            genetic_operators (GeneticOperators): Instance of GeneticOperators.
            fitness_evaluator (FitnessEvaluator): Instance of FitnessEvaluator.
            prompt_architect_agent (PromptArchitectAgent): Instance for creating initial prompts.
            population_size (int, optional): The desired size of the population. Defaults to 50.
            elitism_count (int, optional): The number of top individuals to carry over
                                           to the next generation without modification. Defaults to 2.
        """
        if not isinstance(genetic_operators, GeneticOperators):
            raise TypeError("genetic_operators must be an instance of GeneticOperators.")
        if not isinstance(fitness_evaluator, FitnessEvaluator):
            raise TypeError("fitness_evaluator must be an instance of FitnessEvaluator.")
        from prompthelix.agents.architect import PromptArchitectAgent
        if not isinstance(prompt_architect_agent, PromptArchitectAgent):
            raise TypeError("prompt_architect_agent must be an instance of PromptArchitectAgent.")
        if population_size <= 0:
            raise ValueError("Population size must be positive.")
        if elitism_count < 0 or elitism_count > population_size:
            raise ValueError("Elitism count must be non-negative and not exceed population size.")

        self.genetic_operators = genetic_operators
        self.fitness_evaluator = fitness_evaluator
        self.prompt_architect_agent = prompt_architect_agent
        self.population_size = population_size
        self.elitism_count = elitism_count
        

        self.population: list[PromptChromosome] = []
        self.generation_number: int = 0

    def initialize_population(self, initial_task_description: str, 
                              initial_keywords: list | None = None, 
                              initial_constraints: dict | None = None):
        """
        Initializes the population with new prompts created by the PromptArchitectAgent.

        Args:
            initial_task_description (str): The task description for the initial prompts.
            initial_keywords (list | None, optional): Keywords for initial prompts. Defaults to None.
            initial_constraints (dict | None, optional): Constraints for initial prompts. Defaults to None.
        """
        print(f"PopulationManager: Initializing population of size {self.population_size} for task: '{initial_task_description}'")
        self.population = []
        
        actual_initial_keywords = initial_keywords if initial_keywords is not None else []
        actual_initial_constraints = initial_constraints if initial_constraints is not None else {}

        for i in range(self.population_size):
            request_data = {
                "task_description": initial_task_description,
                "keywords": copy.deepcopy(actual_initial_keywords), # Use copy to avoid modification issues if architect changes them
                "constraints": copy.deepcopy(actual_initial_constraints)
            }
            # Add some variability for the architect if needed, or assume it handles diversity
            # request_data["keywords"].append(f"variant_{i}") 
            
            chromosome = self.prompt_architect_agent.process_request(request_data)
            if not isinstance(chromosome, PromptChromosome):
                print(f"PopulationManager: Warning - PromptArchitectAgent did not return a PromptChromosome. Got: {type(chromosome)}. Skipping.")
                continue

            self.population.append(chromosome)
        
        if len(self.population) != self.population_size:
            print(f"PopulationManager: Warning - Initialized population size {len(self.population)} does not match target {self.population_size}.")

        self.generation_number = 0
        print(f"PopulationManager: Population initialized. Generation: {self.generation_number}")

    def evolve_population(self, task_description: str, success_criteria: dict | None = None,
                          target_style: str | None = None):
        """
        Orchestrates one generation of evolution: evaluation, selection, crossover, and mutation.

        Args:
            task_description (str): The task description for fitness evaluation.
            success_criteria (dict | None, optional): Success criteria for fitness evaluation. Defaults to None.
            target_style (str | None, optional): Desired style used during mutation when
                StyleOptimizerAgent is available. Defaults to None.
        """
        if not self.population:
            print("PopulationManager: Cannot evolve an empty population. Please initialize first.")
            return

        print(f"PopulationManager: Evolving population for generation {self.generation_number + 1}. Evaluating current population...")
        
        # 1. Evaluate Population
        futures = []
        # Ensure ProcessPoolExecutor is imported
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            for chromosome in self.population:
                # Submit evaluation task to the executor
                future = executor.submit(self.fitness_evaluator.evaluate, chromosome, task_description, success_criteria)
                futures.append((future, chromosome)) # Store future and chromosome

            # Retrieve results and update fitness scores
            for future, chromosome in futures:
                try:
                    fitness_score = future.result()  # This will block until the future is complete
                    chromosome.fitness_score = fitness_score
                except Exception as e:
                    logger.error(f"Error evaluating chromosome {chromosome.id} in parallel: {e}", exc_info=True)
                    chromosome.fitness_score = 0.0 # Assign a default low fitness on error


        # 2. Sort Population by fitness (descending) without mutating original list reference
        # Population is now updated with fitness scores from parallel execution
        sorted_population = sorted(self.population, key=lambda c: c.fitness_score, reverse=True)
        
        print(
            f"PopulationManager: Fittest individual in current generation ({self.generation_number}):"
            f" {sorted_population[0].id if sorted_population else 'N/A'} with fitness"
            f" {sorted_population[0].fitness_score if sorted_population else 'N/A'}"
        )

        new_population: list[PromptChromosome] = []

        # 3. Elitism: Carry over the best individuals
        if self.elitism_count > 0:
            print(
                f"PopulationManager: Applying elitism for top {self.elitism_count} individuals."
            )
            new_population.extend(sorted_population[: self.elitism_count])

        # 4. Generate Offspring
        print(f"PopulationManager: Generating offspring through selection, crossover, and mutation.")
        num_offspring_needed = self.population_size - len(new_population)
        
        generated_offspring_count = 0
        while generated_offspring_count < num_offspring_needed:
            # Selection
            parent1 = self.genetic_operators.selection(sorted_population)
            parent2 = self.genetic_operators.selection(sorted_population)
            
            # Crossover
            child1, child2 = self.genetic_operators.crossover(parent1, parent2)
            
            # Mutation
            mutated_child1 = self.genetic_operators.mutate(child1, target_style=target_style)
            mutated_child2 = self.genetic_operators.mutate(child2, target_style=target_style)
            
            new_population.append(mutated_child1)
            generated_offspring_count += 1
            if generated_offspring_count < num_offspring_needed:
                new_population.append(mutated_child2)
                generated_offspring_count += 1
        
        self.population = new_population[: self.population_size]

        self.generation_number += 1
        print(f"PopulationManager: Evolution complete. New generation: {self.generation_number}. Population size: {len(self.population)}")


    def get_fittest_individual(self) -> PromptChromosome | None:
        """
        Returns the fittest individual from the current population.

        The population is sorted by fitness score in descending order after evaluation,
        so the fittest individual is the first one if the population is not empty.

        Returns:
            PromptChromosome | None: The fittest chromosome, or None if the
                                      population is empty.
        """
        if not self.population:
            return None
        # Ensure population is sorted by fitness in descending order
        self.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        return self.population[0]
