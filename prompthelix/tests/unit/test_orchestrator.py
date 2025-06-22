import unittest
from unittest.mock import patch, MagicMock, call, AsyncMock
import asyncio # Added asyncio
import importlib # Added for the test_main_ga_loop_forwards_prompt_and_llm_settings

from prompthelix.orchestrator import main_ga_loop
from prompthelix import config as global_config
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.engine import PromptChromosome, PopulationManager, GeneticOperators, FitnessEvaluator
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.message_bus import MessageBus
from prompthelix.experiment_runners import GeneticAlgorithmRunner

BASE_ARGS = {
    "task_desc": "test task",
    "keywords": ["test"],
    "num_generations": 1,
    "population_size": 1,
    "elitism_count": 0,
    "execution_mode": ExecutionMode.TEST,
    "return_best": False
}

class TestOrchestratorConfigPropagation(unittest.IsolatedAsyncioTestCase): # Changed to IsolatedAsyncioTestCase

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
    async def test_main_ga_loop_uses_default_persistence_settings( # Changed to async
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "default/path/from/settings.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 30

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.genes = ["mock_gene1", "mock_gene2"]
        mock_chromosome.fitness_score = 0.88
        mock_chromosome.id = "default-test-id"
        mock_chromosome.to_prompt_string.return_value = "default test prompt"

        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_instance.status = "COMPLETED"
        mock_pop_manager_instance.initialize_population = AsyncMock() # Make it async
        mock_pop_manager_instance.broadcast_ga_update = AsyncMock() # Make it async
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock(spec=GeneticAlgorithmRunner) # Use spec
        mock_ga_runner_instance.run = AsyncMock(return_value=mock_chromosome) # run is now async
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        mock_message_bus_instance = mock_message_bus_cls.return_value
        mock_message_bus_instance.broadcast_message = AsyncMock()

        args_for_loop = {**BASE_ARGS, "population_path": None, "save_frequency_override": None}
        await main_ga_loop(**args_for_loop) # await

        mock_pop_manager_cls.assert_called_once()
        mock_ga_runner_cls.assert_called_once()
        _, ga_runner_kwargs = mock_ga_runner_cls.call_args
        self.assertEqual(ga_runner_kwargs.get('population_persistence_path'), "default/path/from/settings.json")
        self.assertEqual(ga_runner_kwargs.get('save_frequency'), 30)
        mock_logger.info.assert_any_call("Effective Population Persistence Path: default/path/from/settings.json")
        mock_logger.info.assert_any_call("Effective Save Population Frequency: Every 30 generations (0 means periodic saving disabled)")
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
    @patch('prompthelix.orchestrator.settings')
    async def test_main_ga_loop_uses_override_persistence_settings( # Changed to async
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        override_pop_path = "override/path/custom.json"
        override_save_freq = 5
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "default/path/ignored.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 50

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.genes = ["override_gene1"]
        mock_chromosome.fitness_score = 0.77
        mock_chromosome.id = "override-test-id"
        mock_chromosome.to_prompt_string.return_value = "override test prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_instance.status = "COMPLETED"
        mock_pop_manager_instance.initialize_population = AsyncMock()
        mock_pop_manager_instance.broadcast_ga_update = AsyncMock()
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock(spec=GeneticAlgorithmRunner)
        mock_ga_runner_instance.run = AsyncMock(return_value=mock_chromosome)
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        mock_message_bus_instance = mock_message_bus_cls.return_value
        mock_message_bus_instance.broadcast_message = AsyncMock()

        args_for_loop = {**BASE_ARGS, "population_path": override_pop_path, "save_frequency_override": override_save_freq}
        await main_ga_loop(**args_for_loop) # await

        mock_pop_manager_cls.assert_called_once()
        mock_ga_runner_cls.assert_called_once()
        _, ga_runner_kwargs = mock_ga_runner_cls.call_args
        self.assertEqual(ga_runner_kwargs.get('population_persistence_path'), override_pop_path)
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
    async def test_main_ga_loop_no_final_save_if_no_path_configured( # Changed to async
        self, mock_settings, mock_logger, mock_message_bus_cls,
        mock_architect_cls, mock_results_eval_cls, mock_style_opt_cls,
        mock_gen_ops_cls, mock_fitness_eval_cls, mock_pop_manager_cls, mock_ga_runner_cls
    ):
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = None
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 10

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.genes = ["no_save_gene"]
        mock_chromosome.fitness_score = 0.66
        mock_chromosome.id = "no-save-path-id"
        mock_chromosome.to_prompt_string.return_value = "no save path test prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_instance.status = "COMPLETED"
        mock_pop_manager_instance.initialize_population = AsyncMock()
        mock_pop_manager_instance.broadcast_ga_update = AsyncMock()
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock(spec=GeneticAlgorithmRunner)
        mock_ga_runner_instance.run = AsyncMock(return_value=mock_chromosome)
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        mock_message_bus_instance = mock_message_bus_cls.return_value
        mock_message_bus_instance.broadcast_message = AsyncMock()

        args_for_loop = {**BASE_ARGS, "population_path": None, "save_frequency_override": None}
        await main_ga_loop(**args_for_loop) # await

        mock_pop_manager_cls.assert_called_once()
        mock_ga_runner_cls.assert_called_once()
        _, ga_runner_kwargs = mock_ga_runner_cls.call_args
        self.assertIsNone(ga_runner_kwargs.get('population_persistence_path'))
        mock_logger.info.assert_any_call("Effective Population Persistence Path: None")
        mock_pop_manager_instance.save_population.assert_not_called()

    @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')             # -> mock_ga_runner_cls
    @patch('prompthelix.orchestrator.PopulationManager')                 # -> mock_pop_manager_cls
    @patch('prompthelix.orchestrator.GeneticOperators')                  # -> mock_gen_ops_cls
    @patch('prompthelix.orchestrator.StyleOptimizerAgent')               # -> mock_style_opt_cls
    @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')             # -> mock_results_eval_cls
    @patch('prompthelix.orchestrator.PromptArchitectAgent')              # -> mock_architect_cls
    @patch('prompthelix.orchestrator.MessageBus')                        # -> mock_message_bus_cls
    @patch('prompthelix.orchestrator.logger')                            # -> mock_logger
    @patch('prompthelix.orchestrator.importlib.import_module')           # -> mock_import_module
    @patch('prompthelix.orchestrator.settings')                          # -> mock_settings
    async def test_main_ga_loop_forwards_prompt_and_llm_settings(
        self,
        mock_settings,              # Corresponds to @patch('prompthelix.orchestrator.settings')
        mock_import_module,         # Corresponds to @patch('prompthelix.orchestrator.importlib.import_module')
        mock_logger,                # Corresponds to @patch('prompthelix.orchestrator.logger')
        mock_message_bus_cls,       # Corresponds to @patch('prompthelix.orchestrator.MessageBus')
        mock_architect_cls,         # Corresponds to @patch('prompthelix.orchestrator.PromptArchitectAgent')
        mock_results_eval_cls,      # Corresponds to @patch('prompthelix.orchestrator.ResultsEvaluatorAgent')
        mock_style_opt_cls,         # Corresponds to @patch('prompthelix.orchestrator.StyleOptimizerAgent')
        mock_gen_ops_cls,           # Corresponds to @patch('prompthelix.orchestrator.GeneticOperators')
        mock_pop_manager_cls,       # Corresponds to @patch('prompthelix.orchestrator.PopulationManager')
        mock_ga_runner_cls          # Corresponds to @patch('prompthelix.orchestrator.GeneticAlgorithmRunner')
    ):
        # --- Setup mock_settings (used by orchestrator for some defaults) ---
        mock_settings.DEFAULT_POPULATION_PERSISTENCE_PATH = "def_path.json"
        mock_settings.DEFAULT_SAVE_POPULATION_FREQUENCY = 10

        # The FitnessEvaluator is dynamically imported. We mock import_module to control this.
        MockFitnessEvaluatorClass = MagicMock(spec=FitnessEvaluator)
        mock_fitness_eval_instance = MagicMock(spec=FitnessEvaluator)
        mock_fitness_eval_instance.evaluate = AsyncMock(return_value=0.75)
        MockFitnessEvaluatorClass.return_value = mock_fitness_eval_instance

        # We need to store the original importlib.import_module to call it for non-target imports
        # However, directly accessing importlib.import_module here would use the *real* one,
        # not the one from the orchestrator's context if it were different (it's not here, but good practice).
        # The mock_import_module itself will be called by the orchestrator.
        original_import_module = importlib.import_module

        def import_module_side_effect(name):
            # Use global_config.settings from the main app, not the mocked 'mock_settings' for this path check
            if name == global_config.settings.FITNESS_EVALUATOR_CLASS.rsplit('.', 1)[0] or \
               name == global_config.settings.FITNESS_EVALUATOR_CLASS: # check full path or module path

                # Ensure the dynamically imported module has the expected class attribute
                mock_fe_module = MagicMock()
                class_name_to_set = global_config.settings.FITNESS_EVALUATOR_CLASS.split('.')[-1]
                setattr(mock_fe_module, class_name_to_set, MockFitnessEvaluatorClass)
                return mock_fe_module

            # Allow actual import for strategies and other non-critical dynamic imports
            if name.startswith("prompthelix.genetics.mutation_strategies") or \
               name.startswith("prompthelix.genetics.selection_strategies") or \
               name.startswith("prompthelix.genetics.crossover_strategies") or \
               name.startswith("prompthelix.database"): # Allow database import
                return original_import_module(name)

            # Fallback for any other dynamic import that we haven't specifically mocked
            # print(f"Warning: Unmocked dynamic import in test: {name} - returning generic mock")
            # return MagicMock()
            # More robust: if it's not one we specifically handle, let the original call attempt it
            # This requires careful consideration of what 'name' might be.
            # For this test, we are primarily concerned with the FitnessEvaluator.
            # Using unittest.mock.DEFAULT will fall back to the original function if not handled.
            # However, since we are patching 'prompthelix.orchestrator.importlib.import_module',
            # we need to be careful not to create a loop if the original is needed.
            # The safest is to explicitly call the real one for non-target paths.
            # If the path isn't one of the above, it's an unexpected dynamic import for this test.
            # Raise an error or return a generic mock. For now, let's assume other dynamic imports are not critical.
            # print(f"TestOrchestrator: Passing through unhandled import: {name}")
            return original_import_module(name)


        mock_import_module.side_effect = import_module_side_effect

        # --- Configure mock classes and their instances ---
        mock_architect_instance = MagicMock(spec=PromptArchitectAgent)
        mock_architect_cls.return_value = mock_architect_instance

        mock_results_eval_instance = MagicMock(spec=ResultsEvaluatorAgent)
        mock_results_eval_instance.process_request = AsyncMock(return_value={"fitness_score": 0.5, "detailed_metrics": {}})
        mock_results_eval_cls.return_value = mock_results_eval_instance

        mock_style_opt_instance = MagicMock(spec=StyleOptimizerAgent)
        mock_style_opt_cls.return_value = mock_style_opt_instance

        mock_genetic_ops_instance = MagicMock(spec=GeneticOperators)
        mock_gen_ops_cls.return_value = mock_genetic_ops_instance

        # mock_fitness_eval_instance is already configured from MockFitnessEvaluatorClass

        mock_pop_manager_instance = MagicMock(spec=PopulationManager)
        mock_chromosome = MagicMock(spec=PromptChromosome)
        mock_chromosome.genes = ["forward_gene"]
        mock_chromosome.fitness_score = 0.5
        mock_chromosome.id = "mock-chromosome-id"
        mock_chromosome.to_prompt_string.return_value = "prompt"
        mock_pop_manager_instance.population = [mock_chromosome]
        mock_pop_manager_instance.get_fittest_individual.return_value = mock_chromosome
        mock_pop_manager_instance.save_population = MagicMock()
        mock_pop_manager_instance.status = "COMPLETED"
        mock_pop_manager_instance.initialize_population = AsyncMock()
        mock_pop_manager_instance.broadcast_ga_update = AsyncMock()
        mock_pop_manager_cls.return_value = mock_pop_manager_instance

        mock_ga_runner_instance = MagicMock(spec=GeneticAlgorithmRunner)
        mock_ga_runner_instance.run = AsyncMock(return_value=mock_chromosome)
        mock_ga_runner_cls.return_value = mock_ga_runner_instance

        mock_message_bus_instance = mock_message_bus_cls.return_value
        mock_message_bus_instance.broadcast_message = AsyncMock()

        initial_prompt = "Seed prompt"
        override_llm = {"api_key": "test-key", "default_model": "test-override-model"}
        args_for_loop = {
            **BASE_ARGS,
            "population_path": None,
            "save_frequency_override": None,
            "initial_prompt_str": initial_prompt,
            "llm_settings_override": override_llm,
        }

        # Patch global_config.settings.FITNESS_EVALUATOR_CLASS for the duration of this call
        # This ensures that when orchestrator.py reads this value, it gets our test path.
        original_fe_path = global_config.settings.FITNESS_EVALUATOR_CLASS
        # We need a path that, when .rsplit('.', 1) is called, gives a module and class name
        # that our mock_import_module can handle to return MockFitnessEvaluatorClass.
        # Example: "prompthelix.tests.mock_fitness_evaluator.MockFitnessEvaluator"
        test_fe_class_path = "prompthelix.genetics.engine.FitnessEvaluator" # Default path
        # If AGENT_SETTINGS has a different path, use that, otherwise default.
        # This ensures consistency with how orchestrator determines the class path.
        # No, orchestrator uses global_settings_obj.FITNESS_EVALUATOR_CLASS
        # So we set global_config.settings.FITNESS_EVALUATOR_CLASS

        # The side_effect for mock_import_module already uses global_config.settings.FITNESS_EVALUATOR_CLASS
        # to determine when to return the MockFitnessEvaluatorClass. So, we don't need to patch
        # global_config.settings.FITNESS_EVALUATOR_CLASS here if the default value in config.py
        # is what our side_effect is expecting, or if the side_effect is robust enough.

        # Let's assume global_config.settings.FITNESS_EVALUATOR_CLASS is 'prompthelix.genetics.engine.FitnessEvaluator'
        # Our side_effect is set to return MockFitnessEvaluatorClass when this path is imported.

        await main_ga_loop(**args_for_loop)

        mock_architect_cls.assert_called_once()
        mock_results_eval_cls.assert_called_once()
        mock_style_opt_cls.assert_called_once()
        mock_gen_ops_cls.assert_called_once()

        # Assert FitnessEvaluator (which is MockFitnessEvaluatorClass) was called
        MockFitnessEvaluatorClass.assert_called_once()

        fe_args, fe_kwargs = MockFitnessEvaluatorClass.call_args
        self.assertEqual(fe_kwargs.get("results_evaluator_agent"), mock_results_eval_instance)
        self.assertEqual(fe_kwargs.get("execution_mode"), args_for_loop['execution_mode'])

        # Construct expected LLM settings based on global defaults and overrides
        # The orchestrator uses global_ph_config.LLM_UTILS_SETTINGS as base
        expected_llm_settings = global_config.LLM_UTILS_SETTINGS.copy() # Start with global defaults
        expected_llm_settings = global_config.utils.config_utils.update_settings(expected_llm_settings, override_llm)

        self.assertEqual(fe_kwargs.get("llm_settings"), expected_llm_settings)
        # The 'settings' dict passed to FitnessEvaluator should also contain these llm_settings
        self.assertIn("settings", fe_kwargs)
        self.assertEqual(fe_kwargs["settings"].get("llm_settings"), expected_llm_settings)

        mock_pop_manager_cls.assert_called_once()
        _, pm_kwargs = mock_pop_manager_cls.call_args
        self.assertEqual(pm_kwargs.get("initial_prompt_str"), initial_prompt)


if __name__ == '__main__':
    unittest.main()
