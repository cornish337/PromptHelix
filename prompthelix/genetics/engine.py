import uuid
import copy
import random
# Imports for PopulationManager
from prompthelix.agents.architect import PromptArchitectAgent
# ResultsEvaluatorAgent is already imported for FitnessEvaluator
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent


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
               gene_mutation_prob: float = 0.2) -> PromptChromosome:
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

        if random.random() < mutation_rate:
            genes_modified = False
            for i in range(len(mutated_chromosome.genes)):
                if random.random() < gene_mutation_prob:
                    genes_modified = True
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
                gene_to_mutate_idx = random.randrange(len(mutated_chromosome.genes))
                mutated_chromosome.genes[gene_to_mutate_idx] = str(mutated_chromosome.genes[gene_to_mutate_idx]) + "*"
        return mutated_chromosome


# FitnessEvaluator class remains unchanged
class FitnessEvaluator:
    """
    Evaluates the fitness of PromptChromosome instances.
    This class simulates interaction with an LLM and uses a ResultsEvaluatorAgent
    to determine the fitness score based on the LLM's output.
    """
    def __init__(self, results_evaluator_agent: ResultsEvaluatorAgent):
        """
        Initializes the FitnessEvaluator.
        Args:
            results_evaluator_agent (ResultsEvaluatorAgent): An instance of
                ResultsEvaluatorAgent that will be used to assess the quality
                of LLM outputs.
        """
        if not isinstance(results_evaluator_agent, ResultsEvaluatorAgent):
            raise TypeError("results_evaluator_agent must be an instance of ResultsEvaluatorAgent.")
        self.results_evaluator_agent = results_evaluator_agent

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
        print(f"FitnessEvaluator: Simulating LLM call for prompt: {prompt_string[:100]}...")
        mock_llm_output = (
            f"Mock LLM output for: {prompt_string[:50]}. "
            f"Keywords found: {', '.join(str(g) for g in chromosome.genes[:2]) if chromosome.genes else 'N/A'}. "
            f"Random number: {random.randint(0, 100)}"
        )
        print(f"FitnessEvaluator: Mock LLM Output: {mock_llm_output[:150]}...")
        request_data = {
            "prompt_chromosome": chromosome,
            "llm_output": mock_llm_output,
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
                 prompt_architect_agent: PromptArchitectAgent, 
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

    def evolve_population(self, task_description: str, success_criteria: dict | None = None):
        """
        Orchestrates one generation of evolution: evaluation, selection, crossover, and mutation.

        Args:
            task_description (str): The task description for fitness evaluation.
            success_criteria (dict | None, optional): Success criteria for fitness evaluation. Defaults to None.
        """
        if not self.population:
            print("PopulationManager: Cannot evolve an empty population. Please initialize first.")
            return

        print(f"PopulationManager: Evolving population for generation {self.generation_number + 1}. Evaluating current population...")
        
        # 1. Evaluate Population
        for chromosome in self.population:
            self.fitness_evaluator.evaluate(chromosome, task_description, success_criteria)

        # 2. Sort Population by fitness (descending)
        self.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        
        print(f"PopulationManager: Fittest individual in current generation ({self.generation_number}): {self.population[0].id if self.population else 'N/A'} with fitness {self.population[0].fitness_score if self.population else 'N/A'}")

        new_population: list[PromptChromosome] = []

        # 3. Elitism: Carry over the best individuals
        if self.elitism_count > 0:
            print(f"PopulationManager: Applying elitism for top {self.elitism_count} individuals.")
            new_population.extend(self.population[:self.elitism_count])

        # 4. Generate Offspring
        print(f"PopulationManager: Generating offspring through selection, crossover, and mutation.")
        num_offspring_needed = self.population_size - len(new_population)
        
        generated_offspring_count = 0
        while generated_offspring_count < num_offspring_needed:
            # Selection
            parent1 = self.genetic_operators.selection(self.population) # Using current (sorted) population
            parent2 = self.genetic_operators.selection(self.population)
            
            # Crossover
            child1, child2 = self.genetic_operators.crossover(parent1, parent2)
            
            # Mutation
            mutated_child1 = self.genetic_operators.mutate(child1)
            mutated_child2 = self.genetic_operators.mutate(child2)
            
            new_population.append(mutated_child1)
            generated_offspring_count += 1
            if generated_offspring_count < num_offspring_needed:
                new_population.append(mutated_child2)
                generated_offspring_count += 1
        
        self.population = new_population[:self.population_size] # Ensure population size is maintained
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
        # Assuming population is sorted by evolve_population or needs sorting if called ad-hoc
        # For safety, re-sort if evolve_population isn't guaranteed to have just run
        # self.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        return self.population[0]
