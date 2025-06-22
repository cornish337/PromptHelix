import unittest
import json
import os
import tempfile
import logging
from unittest.mock import MagicMock, patch

# Assuming prompthelix.genetics.engine and other necessary modules are in PYTHONPATH
from prompthelix.genetics.engine import PopulationManager, GeneticOperators, FitnessEvaluator
from prompthelix.genetics.chromosome import PromptChromosome
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent # Import the base class
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.strategy_base import BaseMutationStrategy # Changed import
from prompthelix.genetics.mutation_strategies import NoOperationMutationStrategy


# Configure a logger to capture log messages for verification if needed
# For simplicity in this context, we might directly inspect logs or rely on try/except
# For more robust log testing, a custom handler or a library like 'testfixtures' could be used.
# Here, we'll ensure errors are logged by checking for logger calls via patching if necessary.

# Minimal mock for StyleOptimizerAgent if GeneticOperators requires it
class MockStyleOptimizerAgent:
    def process_request(self, request):
        return request.get("prompt_chromosome")

# Mock ResultsEvaluatorAgent that IS an instance of the base class
class MockResultsEvaluatorAgent(ResultsEvaluatorAgent): # Inherit from actual ResultsEvaluatorAgent
    def __init__(self, message_bus=None, settings=None, knowledge_file_path=None):
        # Call super().__init__ if ResultsEvaluatorAgent has its own __init__ that needs to be called
        # Assuming ResultsEvaluatorAgent's __init__ is compatible or can be handled.
        # If ResultsEvaluatorAgent.__init__ requires specific arguments not available here,
        # or if it performs actions not desired in mock (like DB init), this might need adjustment.
        # For now, let's assume a simple super call or that its __init__ is benign.
        # Based on FitnessEvaluator.__setstate__ it seems ResultsEvaluatorAgent can be initialized with these.
        super().__init__(message_bus=message_bus, settings=settings, knowledge_file_path=knowledge_file_path)
        self.settings = settings or {} # Overwrite if super() sets them differently and we need mock values
        self.knowledge_file_path = knowledge_file_path

    def process_request(self, request_data: dict) -> dict: # Ensure signature matches
        # This mock implementation should align with what ResultsEvaluatorAgent.process_request returns
        return {'fitness_score': 0.5, 'detailed_metrics': {}, 'llm_analysis_status': 'success', 'llm_assessment_feedback': 'mocked feedback'}

class MockPromptArchitectAgent(PromptArchitectAgent):
    def __init__(self, message_bus=None, settings=None):
        super().__init__(message_bus, settings)

    def process_request(self, request_data: dict) -> PromptChromosome:
        return PromptChromosome(genes=[f"architected_gene_{random.randint(1,1000)}"], fitness_score=0.1)

class MockMutationStrategy(BaseMutationStrategy): # Changed inheritance
    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.genes.append("mutated_gene")
        return mutated_chromosome

class TestPopulationManagerPersistence(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_file_path = os.path.join(self.test_dir.name, "population_data.json")

        # Mock dependencies for PopulationManager
        # Using actual PromptArchitectAgent if it's simple enough, or a mock
        self.mock_architect_agent = MockPromptArchitectAgent()

        # Mock GeneticOperators
        mock_mutation_strategies = [NoOperationMutationStrategy()]
        self.mock_genetic_operators = GeneticOperators(
            style_optimizer_agent=None,
            mutation_strategies=mock_mutation_strategies
        )

        # Mock FitnessEvaluator
        # FitnessEvaluator needs a ResultsEvaluatorAgent
        # Ensure settings for MockResultsEvaluatorAgent are sufficient for its __init__
        mock_rea_settings = {"knowledge_file_path": "dummy_path.json"}
        mock_results_eval_agent = MockResultsEvaluatorAgent(settings=mock_rea_settings)

        self.mock_fitness_evaluator = FitnessEvaluator(
            results_evaluator_agent=mock_results_eval_agent,
            execution_mode=ExecutionMode.TEST
        )

        # Suppress logging output during tests unless specifically testing for it
        # Or use patch('logging.Logger.error') etc. to check calls
        logging.disable(logging.CRITICAL)


    def tearDown(self):
        self.test_dir.cleanup()
        logging.disable(logging.NOTSET) # Re-enable logging

    def _create_pm_with_sample_population(self, population_size=3, initial_prompt_str=None, p_workers=None, eval_timeout=60) -> PopulationManager:
        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=population_size,
            initial_prompt_str=initial_prompt_str,
            parallel_workers=p_workers
        )
        # Manually add some chromosomes for testing save/load if initialize_population is too complex here
        pm.population = [
            PromptChromosome(genes=["gene1", "gene2"], fitness_score=0.75),
            PromptChromosome(genes=["geneA"], fitness_score=0.90),
            PromptChromosome(genes=["test"], fitness_score=0.5)
        ]
        pm.generation_number = 5
        return pm

    def test_save_and_load_population_successfully(self):
        pm_to_save = self._create_pm_with_sample_population()
        pm_to_save.save_population(self.temp_file_path)

        self.assertTrue(os.path.exists(self.temp_file_path), "Population file was not created.")

        with open(self.temp_file_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["generation_number"], 5)
        self.assertEqual(len(saved_data["population"]), 3)
        self.assertEqual(saved_data["population"][0]["genes"], ["gene1", "gene2"])
        self.assertEqual(saved_data["population"][0]["fitness_score"], 0.75)
        self.assertEqual(saved_data["population"][1]["genes"], ["geneA"])
        self.assertEqual(saved_data["population"][1]["fitness_score"], 0.90)

        # Now test loading
        pm_to_load = PopulationManager( # Create a fresh instance
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=10 # different initial size to see if load overrides
        )
        pm_to_load.load_population(self.temp_file_path)

        self.assertEqual(pm_to_load.generation_number, 5)
        self.assertEqual(len(pm_to_load.population), 3)
        # The population_size of manager should be updated by load_population
        self.assertEqual(pm_to_load.population_size, 3, "PopulationManager's population_size should be updated after load.")


        self.assertIsInstance(pm_to_load.population[0], PromptChromosome)
        self.assertEqual(pm_to_load.population[0].genes, ["gene1", "gene2"])
        self.assertEqual(pm_to_load.population[0].fitness_score, 0.75)
        self.assertEqual(pm_to_load.population[1].genes, ["geneA"])
        self.assertEqual(pm_to_load.population[1].fitness_score, 0.90)

    def test_load_population_non_existent_file(self):
        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5
        )

        non_existent_path = os.path.join(self.test_dir.name, "does_not_exist.json")

        with patch('logging.Logger.info') as mock_log_info:
            pm.load_population(non_existent_path)

        self.assertEqual(len(pm.population), 0, "Population should be empty after trying to load non-existent file.")
        self.assertEqual(pm.generation_number, 0, "Generation number should be default after trying to load non-existent file.")
        mock_log_info.assert_any_call(f"PopulationManager: No population file at {non_existent_path}; starting fresh.")


    def test_load_population_malformed_json(self):
        malformed_file_path = os.path.join(self.test_dir.name, "malformed.json")
        with open(malformed_file_path, "w") as f:
            f.write("{'genes': ['bad_json, not_quite_right") # Invalid JSON

        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5
        )

        with patch('logging.Logger.error') as mock_log_error:
            pm.load_population(malformed_file_path)

        self.assertEqual(len(pm.population), 0, "Population should be empty after trying to load malformed file.")
        self.assertEqual(pm.generation_number, 0, "Generation number should be default after trying to load malformed file.")

        # Check that a relevant error was logged
        args_list = [call_args[0][0] for call_args in mock_log_error.call_args_list]
        self.assertTrue(any(f"Error loading population from {malformed_file_path}" in arg for arg in args_list))


    def test_load_population_empty_file(self):
        empty_file_path = os.path.join(self.test_dir.name, "empty.json")
        with open(empty_file_path, "w") as f:
            pass # Create an empty file

        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5 # Initial size
        )

        with patch('logging.Logger.error') as mock_log_error:
            pm.load_population(empty_file_path)

        self.assertEqual(len(pm.population), 0)
        # If an empty file is loaded, population_size might be set to 0 by `len(self.population) or self.population_size`
        # The logic `self.population_size = len(self.population) or self.population_size` might lead to pop size 0
        # Let's check the updated logic from previous step:
        # `if not self.population and self.population_size > 0: ... else: self.population_size = len(self.population) or self.population_size`
        # So, if initial population_size was >0, it should be retained.
        self.assertEqual(pm.population_size, 5, "Population size should be retained if loaded file is empty and initial size was > 0.")
        args_list = [call_args[0][0] for call_args in mock_log_error.call_args_list]
        self.assertTrue(any(f"Error loading population from {empty_file_path}" in arg for arg in args_list))


    def test_load_population_json_not_dict(self):
        invalid_json_path = os.path.join(self.test_dir.name, "not_dict.json")
        with open(invalid_json_path, "w") as f:
            json.dump(["list", "not", "dict"], f) # Valid JSON, but not a dictionary

        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5
        )

        with patch('logging.Logger.error') as mock_log_error:
            pm.load_population(invalid_json_path)

        self.assertEqual(len(pm.population), 0)
        self.assertEqual(pm.population_size, 5)
        args_list = [call_args[0][0] for call_args in mock_log_error.call_args_list]
        self.assertTrue(any(f"Error loading population from {invalid_json_path}" in arg for arg in args_list))


    def test_load_population_missing_keys_in_json_structure(self):
        # File with valid JSON, but "population" or "generation_number" key is missing
        # Test 1: Missing "population"
        missing_pop_path = os.path.join(self.test_dir.name, "missing_pop.json")
        with open(missing_pop_path, "w") as f:
            json.dump({"generation_number": 3}, f)

        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5
        )
        pm.load_population(missing_pop_path)
        self.assertEqual(len(pm.population), 0) # individuals will be data.get("population", []) -> []
        self.assertEqual(pm.generation_number, 3) # generation_number should be loaded
        self.assertEqual(pm.population_size, 5) # Retained because loaded population is empty


        # Test 2: Missing "generation_number"
        missing_gen_path = os.path.join(self.test_dir.name, "missing_gen.json")
        sample_chromosomes_data = [{"genes": ["g1"], "fitness_score": 0.1}]
        with open(missing_gen_path, "w") as f:
            json.dump({"population": sample_chromosomes_data}, f)

        pm = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5
        )
        pm.load_population(missing_gen_path)
        self.assertEqual(len(pm.population), 1)
        self.assertEqual(pm.population[0].genes, ["g1"])
        self.assertEqual(pm.generation_number, 0) # Defaults to 0 if key is missing
        self.assertEqual(pm.population_size, 1) # Updated to loaded population size


    def test_save_population_io_error(self):
        pm = self._create_pm_with_sample_population()

        # Path to a directory that doesn't exist (or a read-only location if possible)
        # Forcing an IOError reliably across platforms is tricky.
        # Patching 'open' is a common way to simulate this.

        with patch('builtins.open', side_effect=IOError("Simulated save error")) as mock_open, \
             patch('logging.Logger.error') as mock_log_error:

            # Note: The path passed here won't actually be used by the mocked 'open',
            # but it's good practice to pass the intended path.
            pm.save_population("/path_that_will_trigger_mocked_open_error/population.json")

        mock_open.assert_called_once() # Check that we attempted to open the file

        # Check that a relevant error was logged
        found_error_log = False
        for call_args_tuple in mock_log_error.call_args_list:
            log_message = call_args_tuple[0][0] # The first positional argument to the logger
            if "Error saving population to" in log_message and "Simulated save error" in log_message:
                found_error_log = True
                break
        self.assertTrue(found_error_log, "The specific IOError was not logged as expected.")

if __name__ == '__main__':
    unittest.main()
import random
