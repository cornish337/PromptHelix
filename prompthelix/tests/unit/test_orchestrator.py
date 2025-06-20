import unittest
from unittest.mock import patch, MagicMock, call

from prompthelix.orchestrator import main_ga_loop
from prompthelix import config as global_config
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.engine import PromptChromosome # Needed for mocking population

# Basic set of arguments for main_ga_loop to avoid repetition in tests
BASE_ARGS = {
    "task_desc": "test task",
    "keywords": ["test"],
    "num_generations": 1,
    "population_size": 1,
    "elitism_count": 0,
    "execution_mode": ExecutionMode.TEST,
    "return_best": False # Simplify mocking runner.run()
}

class TestOrchestratorConfigPropagation(unittest.TestCase):

    # Patch all external dependencies of main_ga_loop
    @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')
    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('prompthelix.orchestrator.FitnessEvaluator')
    @patch('prompthelix.orchestrator.GeneticOperators')
    @patch('prompthelix.orchestrator.StyleOptimizerAgent')
    @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')
    @patch('prompthelix.orchestrator.PromptArchitectAgent')
    @patch('prompthelix.orchestrator.MessageBus')
    @patch('prompthelix.orchestrator.logger')
    @patch('prompthelix.orchestrator.settings') # Mock the imported settings object
    def test_main_ga_loop_uses_default_persistence_settings(
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        # 1. Configure mock_settings
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "default/path/from/settings.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 30

        # 2. Mock instances returned by constructors to allow loop to run
        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.fitness_score = 0.88
        mock_chromosome.id = "default-test-id"
        mock_chromosome.to_prompt_string.return_value = "default test prompt"

        # Simulate a population that allows the orchestrator to proceed
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock() # Mock save_population
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock()
        mock_ga_runner_instance.run.return_value = mock_chromosome
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        # 3. Call main_ga_loop with None for overrides
        args_for_loop = {**BASE_ARGS, "population_path": None, "save_frequency_override": None}
        main_ga_loop(**args_for_loop)

        # 4. Assertions
        # Check PopulationManager instantiation
        mock_pop_manager_cls.assert_called_once()
        _, pop_manager_kwargs = mock_pop_manager_cls.call_args
        self.assertEqual(pop_manager_kwargs.get('population_path'), "default/path/from/settings.json")

        # Check GeneticAlgorithmRunner instantiation
        mock_ga_runner_cls.assert_called_once()
        _, ga_runner_kwargs = mock_ga_runner_cls.call_args
        self.assertEqual(ga_runner_kwargs.get('save_frequency'), 30)

        # Check logger calls for effective path and frequency
        mock_logger.info.assert_any_call("Effective Population Persistence Path: default/path/from/settings.json")
        mock_logger.info.assert_any_call("Effective Save Population Frequency: Every 30 generations (0 means periodic saving disabled)")

        # Check that the final save is called on the PopulationManager instance
        mock_pop_manager_instance.save_population.assert_called_with("default/path/from/settings.json")

    @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')
    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('prompthelix.orchestrator.FitnessEvaluator')
    @patch('prompthelix.orchestrator.GeneticOperators')
    @patch('prompthelix.orchestrator.StyleOptimizerAgent')
    @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')
    @patch('prompthelix.orchestrator.PromptArchitectAgent')
    @patch('prompthelix.orchestrator.MessageBus')
    @patch('prompthelix.orchestrator.logger')
    @patch('prompthelix.orchestrator.settings') # Mock the imported settings object
    def test_main_ga_loop_uses_override_persistence_settings(
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        # 1. Define override values
        override_pop_path = "override/path/custom.json"
        override_save_freq = 5

        # mock_settings should not be used if overrides are provided, but set them anyway to ensure overrides work
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "default/path/ignored.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 50

        # 2. Mock instances
        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.fitness_score = 0.77
        mock_chromosome.id = "override-test-id"
        mock_chromosome.to_prompt_string.return_value = "override test prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock()
        mock_ga_runner_instance.run.return_value = mock_chromosome
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        # 3. Call main_ga_loop with override values
        args_for_loop = {**BASE_ARGS, "population_path": override_pop_path, "save_frequency_override": override_save_freq}
        main_ga_loop(**args_for_loop)

        # 4. Assertions
        mock_pop_manager_cls.assert_called_once()
        _, pop_manager_kwargs = mock_pop_manager_cls.call_args
        self.assertEqual(pop_manager_kwargs.get('population_path'), override_pop_path)

        mock_ga_runner_cls.assert_called_once()
        _, ga_runner_kwargs = mock_ga_runner_cls.call_args
        self.assertEqual(ga_runner_kwargs.get('save_frequency'), override_save_freq)

        mock_logger.info.assert_any_call(f"Effective Population Persistence Path: {override_pop_path}")
        mock_logger.info.assert_any_call(f"Effective Save Population Frequency: Every {override_save_freq} generations (0 means periodic saving disabled)")

        mock_pop_manager_instance.save_population.assert_called_with(override_pop_path)

    @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')
    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('prompthelix.orchestrator.FitnessEvaluator')
    @patch('prompthelix.orchestrator.GeneticOperators')
    @patch('prompthelix.orchestrator.StyleOptimizerAgent')
    @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')
    @patch('prompthelix.orchestrator.PromptArchitectAgent')
    @patch('prompthelix.orchestrator.MessageBus')
    @patch('prompthelix.orchestrator.logger')
    @patch('prompthelix.orchestrator.settings')
    def test_main_ga_loop_no_final_save_if_no_path_configured(
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        # Configure mock_settings so that default path is None or empty
        # For this test, we need DEFAULT_POPULATION_PERSISTENCE_PATH to evaluate to None or empty.
        # The actual_population_path logic is: population_path if population_path is not None else settings.DEFAULT_POPULATION_PERSISTENCE_PATH
        # So, if population_path override is None, AND settings.DEFAULT_POPULATION_PERSISTENCE_PATH is effectively None for the test:
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = None # Explicitly None
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 10 # Irrelevant for this specific assertion but needs a value

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.fitness_score = 0.66
        mock_chromosome.id = "no-save-path-id"
        mock_chromosome.to_prompt_string.return_value = "no save path test prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock() # This should NOT be called
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock()
        mock_ga_runner_instance.run.return_value = mock_chromosome
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        args_for_loop = {**BASE_ARGS, "population_path": None, "save_frequency_override": None} # No override path
        main_ga_loop(**args_for_loop)

        # Assert that PopulationManager was still called, but with population_path=None
        mock_pop_manager_cls.assert_called_once()
        _, pop_manager_kwargs = mock_pop_manager_cls.call_args
        self.assertIsNone(pop_manager_kwargs.get('population_path'))

        # Assert logger shows path is None (or similar, depending on how None path is logged)
        # The logging statement is: f"Effective Population Persistence Path: {actual_population_path}"
        mock_logger.info.assert_any_call("Effective Population Persistence Path: None")

        # Crucially, assert that save_population on the instance was NOT called
        mock_pop_manager_instance.save_population.assert_not_called()

    @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')
    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('prompthelix.orchestrator.FitnessEvaluator')
    @patch('prompthelix.orchestrator.GeneticOperators')
    @patch('prompthelix.orchestrator.StyleOptimizerAgent')
    @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')
    @patch('prompthelix.orchestrator.PromptArchitectAgent')
    @patch('prompthelix.orchestrator.MessageBus')
    @patch('prompthelix.orchestrator.logger')
    @patch('prompthelix.orchestrator.settings')
    def test_main_ga_loop_forwards_prompt_and_llm_settings(
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "def_path.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 10

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.fitness_score = 0.5
        mock_chromosome.to_prompt_string.return_value = "prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock()
        mock_ga_runner_instance.run.return_value = mock_chromosome
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        initial_prompt = "Seed prompt"
        override_llm = {"api_key": "test-key"}
        args_for_loop = {
            **BASE_ARGS,
            "population_path": None,
            "save_frequency_override": None,
            "initial_prompt_str": initial_prompt,
            "llm_settings_override": override_llm,
        }

        main_ga_loop(**args_for_loop)

        # Check FitnessEvaluator received merged llm settings
        mock_fitness_eval_cls.assert_called_once()
        _, fe_kwargs = mock_fitness_eval_cls.call_args
        expected_llm = global_config.LLM_UTILS_SETTINGS.copy()
        expected_llm.update(override_llm)
        self.assertEqual(fe_kwargs.get("llm_settings"), expected_llm)

        # Check PopulationManager received the initial prompt string
        mock_pop_manager_cls.assert_called_once()
        _, pm_kwargs = mock_pop_manager_cls.call_args
        self.assertEqual(pm_kwargs.get("initial_prompt_str"), initial_prompt)

if __name__ == '__main__':
    unittest.main()
