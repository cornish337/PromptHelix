import unittest

import uuid
import copy
from prompthelix.genetics.engine import PromptChromosome

class TestPromptChromosome(unittest.TestCase):
    """Comprehensive test suite for the PromptChromosome class."""

    def setUp(self):
        """Set up common test data if needed, though most tests create their own."""
        self.genes1 = ["Instruction: Summarize.", "Context: Long text...", "Output: Short summary."]
        self.fitness1 = 0.75

    def test_init_with_genes_and_fitness(self):
        """Test __init__ with specified genes and fitness score."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        self.assertIsInstance(chromosome.id, uuid.UUID, "ID should be a UUID instance.")
        self.assertEqual(chromosome.genes, self.genes1, "Genes not initialized correctly.")
        self.assertEqual(chromosome.fitness_score, self.fitness1, "Fitness score not initialized correctly.")

    def test_init_default_fitness(self):
        """Test __init__ with default fitness score."""
        chromosome = PromptChromosome(genes=self.genes1)
        self.assertEqual(chromosome.fitness_score, 0.0, "Fitness score should default to 0.0.")

    def test_init_genes_none(self):
        """Test __init__ with genes=None, expecting an empty list."""
        chromosome = PromptChromosome(genes=None)
        self.assertEqual(chromosome.genes, [], "Genes should default to an empty list if None is provided.")
        self.assertIsInstance(chromosome.id, uuid.UUID)

    def test_init_default_genes(self):
        """Test __init__ with no genes argument, expecting an empty list."""
        chromosome = PromptChromosome()
        self.assertEqual(chromosome.genes, [], "Genes should default to an empty list if no argument is provided.")
        self.assertIsInstance(chromosome.id, uuid.UUID)


    def test_calculate_fitness(self):
        """Test calculate_fitness method."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        self.assertEqual(chromosome.calculate_fitness(), self.fitness1, "calculate_fitness should return self.fitness_score.")
        
        chromosome.fitness_score = 0.9
        self.assertEqual(chromosome.calculate_fitness(), 0.9, "calculate_fitness should reflect updated self.fitness_score.")

    def test_to_prompt_string_default_separator(self):
        """Test to_prompt_string with the default newline separator."""
        chromosome = PromptChromosome(genes=self.genes1)
        expected_string = "Instruction: Summarize.\nContext: Long text...\nOutput: Short summary."
        self.assertEqual(chromosome.to_prompt_string(), expected_string)

    def test_to_prompt_string_custom_separator(self):
        """Test to_prompt_string with a custom separator."""
        chromosome = PromptChromosome(genes=self.genes1)
        expected_string = "Instruction: Summarize. Context: Long text... Output: Short summary."
        self.assertEqual(chromosome.to_prompt_string(separator=" "), expected_string)

    def test_to_prompt_string_empty_genes(self):
        """Test to_prompt_string with an empty gene list."""
        chromosome = PromptChromosome(genes=[])
        self.assertEqual(chromosome.to_prompt_string(), "", "Should return an empty string for empty genes.")

    def test_to_prompt_string_non_string_genes(self):
        """Test to_prompt_string with non-string elements in genes list."""
        chromosome = PromptChromosome(genes=["Gene 1", 123, {"type": "Instruction"}])
        expected_string = "Gene 1\n123\n{'type': 'Instruction'}"
        self.assertEqual(chromosome.to_prompt_string(), expected_string)

    def test_clone(self):
        """Test the clone method for deep copying and new ID generation."""
        original_genes = [["MutableGene1"], "Gene2"]
        original_chromosome = PromptChromosome(genes=original_genes, fitness_score=self.fitness1)
        
        cloned_chromosome = original_chromosome.clone()

        # Test for new instance and different ID
        self.assertIsNot(cloned_chromosome, original_chromosome, "Clone should be a new instance.")
        self.assertNotEqual(cloned_chromosome.id, original_chromosome.id, "Clone should have a new ID.")

        # Test fitness score is copied
        self.assertEqual(cloned_chromosome.fitness_score, original_chromosome.fitness_score, "Fitness score should be copied to clone.")

        # Test genes are deep-copied
        self.assertEqual(cloned_chromosome.genes, original_chromosome.genes, "Genes content should be equal after cloning.")
        self.assertIsNot(cloned_chromosome.genes[0], original_chromosome.genes[0], "Mutable gene elements should be different instances (deep copy).")
        
        # Modify original's mutable gene element and check clone is unaffected
        original_chromosome.genes[0].append("ModifiedInOriginal")
        self.assertNotEqual(cloned_chromosome.genes[0], original_chromosome.genes[0], "Modifying original's mutable gene should not affect clone's.")
        self.assertEqual(cloned_chromosome.genes[0], ["MutableGene1"], "Clone's mutable gene element changed unexpectedly.")

        # Modify clone's genes and check original is unaffected
        cloned_chromosome.genes.append("AddedToClone")
        self.assertNotEqual(cloned_chromosome.genes, original_chromosome.genes, "Modifying clone's genes should not affect original's.")
        
        cloned_chromosome.genes[0].append("ModifiedInClone")
        self.assertNotEqual(cloned_chromosome.genes[0], original_chromosome.genes[0], "Modifying clone's mutable gene should not affect original's.")


    def test_str_representation(self):
        """Test the __str__ method for human-readable output."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        chromosome_str = str(chromosome)

        self.assertIn(str(chromosome.id), chromosome_str, "__str__ should include ID.")
        self.assertIn(f"Fitness: {self.fitness1:.4f}", chromosome_str, "__str__ should include formatted fitness.")
        self.assertIn("Genes:", chromosome_str, "__str__ should include 'Genes:' label.")
        for gene in self.genes1:
            self.assertIn(f"  - {gene}", chromosome_str, f"Gene '{gene}' not found in __str__ output.")

    def test_str_representation_empty_genes(self):
        """Test __str__ with an empty gene list."""
        chromosome = PromptChromosome(genes=[])
        chromosome_str = str(chromosome)
        self.assertIn("  - (No genes)", chromosome_str, "__str__ should indicate no genes for an empty list.")

    def test_repr_representation(self):
        """Test the __repr__ method for unambiguous output."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        chromosome_repr = repr(chromosome)
        
        # Expected format: PromptChromosome(id='...', genes=[...], fitness_score=...)
        self.assertTrue(chromosome_repr.startswith("PromptChromosome(id="), "__repr__ should start with class name and id.")
        self.assertIn(f"id='{chromosome.id}'", chromosome_repr)
        self.assertIn(f"genes={self.genes1!r}", chromosome_repr, "__repr__ should include repr of genes.")
        self.assertIn(f"fitness_score={self.fitness1:.4f}", chromosome_repr, "__repr__ should include formatted fitness.")
        self.assertTrue(chromosome_repr.endswith(")"), "__repr__ should end with a parenthesis.")


from unittest.mock import MagicMock, patch
from prompthelix.genetics.engine import PopulationManager, GeneticOperators, FitnessEvaluator
from prompthelix.agents.architect import PromptArchitectAgent
# Import ProcessPoolExecutor to allow patching it for max_workers control in tests if necessary,
# though current engine.py doesn't allow evolve_population to change max_workers.
from concurrent.futures import ProcessPoolExecutor

from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent # For type hint and mock
from prompthelix.genetics.engine import FitnessEvaluator # For MagicMock spec

# Global variable to control which chromosome fails during testing.
FAILING_CHROMOSOME_ID_FOR_TEST = None

# This class is designed to be simple and picklable for ProcessPoolExecutor tests.
# It does NOT inherit from FitnessEvaluator to avoid pickling issues with base class.
class SimplePicklableEvaluator:
    def __init__(self):
        self.evaluation_mode = "normal"

    def set_mode_fail_specific(self):
        self.evaluation_mode = "fail_specific"

    def set_mode_normal(self):
        self.evaluation_mode = "normal"

    def evaluate(self, chromosome, task_desc, success_crit):
        global FAILING_CHROMOSOME_ID_FOR_TEST
        # print(f"SimplePicklableEvaluator ({id(self)}): Evaluating {chromosome.id}, Mode: {self.evaluation_mode}, Failing ID: {FAILING_CHROMOSOME_ID_FOR_TEST}")
        if self.evaluation_mode == "fail_specific":
            if FAILING_CHROMOSOME_ID_FOR_TEST and chromosome.id == FAILING_CHROMOSOME_ID_FOR_TEST:
                # print(f"SimplePicklableEvaluator: Simulating failure for {chromosome.id}")
                raise ValueError(f"Simulated evaluation error for {chromosome.id}")
        return len(chromosome.genes) * 0.1


class TestPopulationManagerParallelEvaluation(unittest.TestCase):
    def setUp(self):
        global FAILING_CHROMOSOME_ID_FOR_TEST
        FAILING_CHROMOSOME_ID_FOR_TEST = None

        self.mock_genetic_operators = MagicMock(spec=GeneticOperators)
        self.mock_prompt_architect_agent = MagicMock(spec=PromptArchitectAgent)

        # This mock is just to satisfy PopulationManager's __init__ type check.
        # It won't be the one used by ProcessPoolExecutor in the tests.
        init_time_mock_fitness_evaluator = MagicMock(spec=FitnessEvaluator)

        # Sample chromosomes for testing
        self.chromosome1_genes = ["gene1a", "gene1b"]
        self.chromosome2_genes = ["gene2a", "gene2b", "gene2c"]
        self.chromosome3_genes = ["gene3a"] # For exception testing

        self.chromosome1 = PromptChromosome(genes=self.chromosome1_genes, fitness_score=0)
        self.chromosome1.id = uuid.uuid4()
        self.chromosome2 = PromptChromosome(genes=self.chromosome2_genes, fitness_score=0)
        self.chromosome2.id = uuid.uuid4()
        self.chromosome3 = PromptChromosome(genes=self.chromosome3_genes, fitness_score=0)
        self.chromosome3.id = uuid.uuid4()

        self.initial_population = [self.chromosome1, self.chromosome2, self.chromosome3]

        # Configure mock genetic operators
        self.mock_genetic_operators.selection.side_effect = lambda pop: pop[0].clone() if pop else PromptChromosome()
        def mock_crossover(p1, p2):
            # Ensure returned children have unique IDs if they might be added to population directly
            child1 = PromptChromosome(genes=["child1_gene"])
            child1.id = uuid.uuid4()
            child2 = PromptChromosome(genes=["child2_gene"])
            child2.id = uuid.uuid4()
            return child1, child2
        self.mock_genetic_operators.crossover.side_effect = mock_crossover
        self.mock_genetic_operators.mutate.side_effect = lambda chromo: chromo.clone()


        self.mock_prompt_architect_agent.process_request.side_effect = [
            c.clone() for c in self.initial_population
        ]

        self.population_manager = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=init_time_mock_fitness_evaluator, # Passes __init__ type check
            prompt_architect_agent=self.mock_prompt_architect_agent,
            population_size=len(self.initial_population),
            elitism_count=0
        )

        # This is the actual evaluator instance that will be used by ProcessPoolExecutor.
        # We replace the one in population_manager after its __init__ has passed.
        self.picklable_evaluator = SimplePicklableEvaluator()
        self.population_manager.fitness_evaluator = self.picklable_evaluator

        # These are the actual chromosome objects that will be processed by evolve_population
        self.processed_chromosomes = [c.clone() for c in self.initial_population]
        self.population_manager.population = self.processed_chromosomes # Assign the list of clones

        # Task description and success criteria for evolve_population
        self.task_description = "Test Task"
        self.success_criteria = {"type": "accuracy"}

    def _get_expected_fitnesses(self, population_for_ids_and_genes, failing_chromosome_id_val=None):
        """
        Helper to calculate expected fitness scores.
        The keys in the returned dictionary are the IDs from `population_for_ids_and_genes`.
        The values are calculated based on the gene content of each chromosome in that same population.
        """
        expected_fitnesses = {}
        for chromo in population_for_ids_and_genes:
            if failing_chromosome_id_val and chromo.id == failing_chromosome_id_val:
                expected_fitnesses[chromo.id] = 0.0
            else:
                expected_fitnesses[chromo.id] = len(chromo.genes) * 0.1
        return expected_fitnesses

    def test_evolve_population_single_and_multiple_workers_match_expected(self):
        """
        Tests that evolve_population correctly evaluates fitness scores.
        This implicitly covers single/multi-worker as ProcessPoolExecutor is used.
        """
        self.picklable_evaluator.set_mode_normal() # Ensure normal evaluation
        # Calculate expected fitness based on the processed_chromosomes themselves
        expected_fitnesses_by_id = self._get_expected_fitnesses(self.processed_chromosomes)

        print("Running evolve_population for test_evolve_population_single_and_multiple_workers_match_expected...")
        self.population_manager.evolve_population(self.task_description, self.success_criteria)

        # Assertions are now made on self.processed_chromosomes
        for processed_chromo in self.processed_chromosomes:
            expected_fitness = expected_fitnesses_by_id.get(processed_chromo.id)
            self.assertIsNotNone(expected_fitness, f"Chromosome ID {processed_chromo.id} not found in expected fitnesses.")
            self.assertAlmostEqual(processed_chromo.fitness_score, expected_fitness, places=5,
                                 msg=f"Fitness for {processed_chromo.id} ({processed_chromo.fitness_score}) did not match expected ({expected_fitness}).")




    def test_evolve_population_handles_evaluation_exception(self):
        """
        Tests that if fitness_evaluator.evaluate raises an exception for a chromosome,
        it's handled, and that chromosome gets a default fitness score (0.0).
        """
        global FAILING_CHROMOSOME_ID_FOR_TEST
        failing_chromosome_obj = self.processed_chromosomes[1]
        FAILING_CHROMOSOME_ID_FOR_TEST = failing_chromosome_obj.id

        self.picklable_evaluator.set_mode_fail_specific()

        # Calculate expected fitness based on the processed_chromosomes themselves
        expected_fitnesses_by_id = self._get_expected_fitnesses(self.processed_chromosomes,
                                                                failing_chromosome_id_val=FAILING_CHROMOSOME_ID_FOR_TEST)

        print("Running evolve_population for test_evolve_population_handles_evaluation_exception...")
        self.population_manager.evolve_population(self.task_description, self.success_criteria)

        # Assertions are now made on self.processed_chromosomes
        for processed_chromo in self.processed_chromosomes:
            expected_fitness = expected_fitnesses_by_id.get(processed_chromo.id)
            self.assertIsNotNone(expected_fitness, f"Chromosome ID {processed_chromo.id} not found in expected fitnesses.")

            if processed_chromo.id == FAILING_CHROMOSOME_ID_FOR_TEST:
                self.assertEqual(processed_chromo.fitness_score, 0.0,
                                 f"Failing chromosome {processed_chromo.id} should have 0.0 fitness, got {processed_chromo.fitness_score}.")
            else:
                self.assertAlmostEqual(processed_chromo.fitness_score, expected_fitness, places=5,
                                     msg=f"Fitness for {processed_chromo.id} ({processed_chromo.fitness_score}) did not match expected ({expected_fitness}).")

        FAILING_CHROMOSOME_ID_FOR_TEST = None # Clean up global


if __name__ == '__main__':
    unittest.main()
