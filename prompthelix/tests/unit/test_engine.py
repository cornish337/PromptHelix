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


    # def test_calculate_fitness(self):
    #     """Test calculate_fitness method."""
    #     chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
    #     self.assertEqual(chromosome.calculate_fitness(), self.fitness1, "calculate_fitness should return self.fitness_score.")
        
    #     chromosome.fitness_score = 0.9
    #     self.assertEqual(chromosome.calculate_fitness(), 0.9, "calculate_fitness should reflect updated self.fitness_score.")

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

    async def evaluate(self, chromosome, task_desc, success_crit, *args, **kwargs): # Changed to async def, added *args, **kwargs
        global FAILING_CHROMOSOME_ID_FOR_TEST
        # print(f"SimplePicklableEvaluator ({id(self)}): Evaluating {chromosome.id}, Mode: {self.evaluation_mode}, Failing ID: {FAILING_CHROMOSOME_ID_FOR_TEST}")
        # logger.info(f"SimplePicklableEvaluator evaluating {chromosome.id} with task_desc: {task_desc}, success_crit: {success_crit}, args: {args}, kwargs: {kwargs}")
        if self.evaluation_mode == "fail_specific":
            if FAILING_CHROMOSOME_ID_FOR_TEST and chromosome.id == FAILING_CHROMOSOME_ID_FOR_TEST:
                # print(f"SimplePicklableEvaluator: Simulating failure for {chromosome.id}")
                chromosome.fitness_score = 0.0 # Ensure it's set before raising for consistency if test checked it
                raise ValueError(f"Simulated evaluation error for {chromosome.id}")

        score = len(chromosome.genes) * 0.1
        chromosome.fitness_score = score # Set the score on the chromosome
        return score


class TestPopulationManagerParallelEvaluation(unittest.IsolatedAsyncioTestCase): # Changed to IsolatedAsyncioTestCase
    async def asyncSetUp(self): # Changed to asyncSetUp
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
        def mock_crossover(p1, p2, run_id=None, generation=None, **kwargs): # Added run_id, generation, and **kwargs
            # Ensure returned children have unique IDs if they might be added to population directly
            child1 = PromptChromosome(genes=["child1_gene"])
            child1.id = uuid.uuid4() # Assign a new UUID
            child2 = PromptChromosome(genes=["child2_gene"])
            child2.id = uuid.uuid4() # Assign a new UUID
            return child1, child2
        self.mock_genetic_operators.crossover.side_effect = mock_crossover
        # Make the mock accept any arguments
        self.mock_genetic_operators.mutate.side_effect = lambda chromo, *args, **kwargs: chromo.clone()


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

    async def test_evolve_population_single_and_multiple_workers_match_expected(self): # Changed to async def
        """
        Tests that evolve_population correctly evaluates fitness scores.
        This implicitly covers single/multi-worker as ProcessPoolExecutor is used.
        """
        self.picklable_evaluator.set_mode_normal() # Ensure normal evaluation
        # Calculate expected fitness based on the processed_chromosomes themselves
        expected_fitnesses_by_id = self._get_expected_fitnesses(self.processed_chromosomes)

        print("Running evolve_population for test_evolve_population_single_and_multiple_workers_match_expected...")

        await self.population_manager.evolve_population(self.task_description, self.success_criteria) # Added await

        # Assertions are made on the final population members
        # Due to elitism=0 and mocked crossover/mutation, all members are new objects.
        # Their genes will be ["child1_gene"] or ["child2_gene"] based on mock_crossover.
        # Expected fitness for such genes is 0.1.
        self.assertEqual(len(self.population_manager.population), self.population_manager.population_size)
        for evolved_chromo in self.population_manager.population:
            self.assertTrue(evolved_chromo.genes == ["child1_gene"] or evolved_chromo.genes == ["child2_gene"],
                            f"Evolved chromosome genes are unexpected: {evolved_chromo.genes}")
            expected_fitness = 0.1 # len(["childX_gene"]) * 0.1
            self.assertAlmostEqual(evolved_chromo.fitness_score, expected_fitness, places=5,
                                 msg=f"Fitness for {evolved_chromo.id} ({evolved_chromo.fitness_score}) with genes {evolved_chromo.genes} did not match expected ({expected_fitness}).")


    async def test_evolve_population_handles_evaluation_exception(self): # Changed to async def
        """
        Tests that if fitness_evaluator.evaluate raises an exception for a chromosome,
        it's handled, and that chromosome gets a default fitness score (0.0).
        """
        global FAILING_CHROMOSOME_ID_FOR_TEST
        # We can't use the parent's ID directly. Instead, we'll make the evaluator fail for specific gene content
        # that we know will be produced by our mocked crossover.
        # Let's make it fail for genes ["child1_gene"].
        # FAILING_CHROMOSOME_ID_FOR_TEST is not used by SimplePicklableEvaluator anymore for this test.
        # Instead, SimplePicklableEvaluator will be modified or a new mode added to fail on specific gene content.

        # Modify SimplePicklableEvaluator to fail for specific genes for this test
        original_evaluate = self.picklable_evaluator.evaluate
        async def evaluate_fail_on_genes(chromosome, task_desc, success_crit, *args, **kwargs):
            if chromosome.genes == ["child1_gene"]:
                chromosome.fitness_score = 0.0 # Set before raising
                raise ValueError("Simulated evaluation error for child1_gene")
            return await original_evaluate(chromosome, task_desc, success_crit, *args, **kwargs)

        self.picklable_evaluator.evaluate = evaluate_fail_on_genes
        # No need for set_mode_fail_specific if we override 'evaluate' directly for the test

        print("Running evolve_population for test_evolve_population_handles_evaluation_exception...")

        with patch('prompthelix.genetics.engine.logger.error') as mock_log_error:
            await self.population_manager.evolve_population(self.task_description, self.success_criteria)

        chromosomes_with_child1_gene = 0
        chromosomes_with_child2_gene = 0

        for evolved_chromo in self.population_manager.population:
            if evolved_chromo.genes == ["child1_gene"]:
                chromosomes_with_child1_gene += 1
                self.assertEqual(evolved_chromo.fitness_score, 0.0,
                                 f"Chromosome {evolved_chromo.id} with genes ['child1_gene'] should have 0.0 fitness due to simulated error.")
            elif evolved_chromo.genes == ["child2_gene"]:
                chromosomes_with_child2_gene += 1
                expected_fitness_child2 = 0.1 # len(["child2_gene"]) * 0.1
                self.assertAlmostEqual(evolved_chromo.fitness_score, expected_fitness_child2, places=5,
                                     msg=f"Fitness for {evolved_chromo.id} ({evolved_chromo.fitness_score}) with genes ['child2_gene'] did not match expected ({expected_fitness_child2}).")

        # Ensure we have both types of children to validate both paths
        self.assertTrue(chromosomes_with_child1_gene > 0, "Test setup error: No child with ['child1_gene'] was produced/found.")
        self.assertTrue(chromosomes_with_child2_gene > 0, "Test setup error: No child with ['child2_gene'] was produced/found.")


        # Check if the error was logged. The log message will contain the chromosome ID of the *offspring*.
        found_log_for_failing_gene_type = False
        for call_args in mock_log_error.call_args_list:
            log_message = str(call_args[0][0]) # First positional argument to logger.error
            # Example: "PopulationManager: Error evaluating a chromosome in new generation: ValueError('Simulated evaluation error for child1_gene')"
            if "Error evaluating a chromosome" in log_message and "Simulated evaluation error for child1_gene" in log_message:
                found_log_for_failing_gene_type = True
                break
        self.assertTrue(found_log_for_failing_gene_type,
                        "Expected error log for failing gene type ['child1_gene'] not found.")

        # Restore original evaluate method if it was patched on instance
        self.picklable_evaluator.evaluate = original_evaluate
        FAILING_CHROMOSOME_ID_FOR_TEST = None # Clean up global


if __name__ == '__main__':
    unittest.main()
