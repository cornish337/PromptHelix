import unittest
from unittest.mock import MagicMock, patch, ANY, AsyncMock
import time # For time.sleep

# Adjust imports based on your project structure
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators, FitnessEvaluator
from prompthelix.agents.architect import PromptArchitectAgent # Assuming a concrete or mockable agent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.message_bus import MessageBus
from prompthelix.enums import ExecutionMode
from prompthelix.config import AGENT_SETTINGS, LLM_UTILS_SETTINGS # For default settings

# If websocket_manager and SessionLocal are needed for MessageBus instantiation
from prompthelix.main import websocket_manager # Assuming this can be imported
from prompthelix.database import SessionLocal # Assuming this can be imported

class TestOrchestratorGAControl(unittest.TestCase):

    def setUp(self):
        # Mock agents and GA components to control their behavior and avoid side effects
        self.mock_architect_agent = MagicMock(spec=PromptArchitectAgent)
        self.mock_results_eval_agent = MagicMock(spec=ResultsEvaluatorAgent)
        self.mock_style_optimizer_agent = MagicMock(spec=StyleOptimizerAgent)

        # Configure mock agents to return valid objects where needed
        self.mock_architect_agent.process_request.return_value = PromptChromosome(genes=["initial"])
        self.mock_architect_agent.agent_id = "MockArchitectAgent" # agent_id is accessed by MessageBus if registered
        self.mock_results_eval_agent.process_request.return_value = {"fitness_score": 0.5, "detailed_metrics": {}} # Ensure detailed_metrics
        self.mock_results_eval_agent.agent_id = "MockResultsEvaluatorAgent"
        self.mock_style_optimizer_agent.process_request.return_value = PromptChromosome(genes=["styled"])
        self.mock_style_optimizer_agent.agent_id = "MockStyleOptimizerAgent"


        # Mock GA components
        self.mock_genetic_ops = MagicMock(spec=GeneticOperators)
        # Configure crossover and mutate to return new chromosome instances
        dummy_child_chromo = PromptChromosome(genes=["child"])
        self.mock_genetic_ops.crossover.return_value = (dummy_child_chromo, dummy_child_chromo.clone())
        self.mock_genetic_ops.mutate.return_value = dummy_child_chromo.clone()
        self.mock_genetic_ops.selection.return_value = PromptChromosome(genes=["selected"])


        # Real MessageBus, but with a mocked ConnectionManager for broadcasting checks
        # Ensure ConnectionManager is AsyncMock if its methods are async
        self.mock_connection_manager_for_bus = AsyncMock()
        self.mock_connection_manager_for_bus.broadcast_json = AsyncMock()

        # Use a real SessionLocal if possible, or mock it if DB interactions are problematic for this test scope
        # For this integration test, focusing on GA control, direct DB interaction isn't primary.
        # However, MessageBus instantiation requires it.
        mock_db_session_factory = MagicMock(return_value=MagicMock())


        self.message_bus = MessageBus(
            db_session_factory=mock_db_session_factory,
            connection_manager=self.mock_connection_manager_for_bus # Pass the AsyncMock
        )

        # FitnessEvaluator needs to be more functional for the loop to proceed
        self.fitness_evaluator = FitnessEvaluator(
            results_evaluator_agent=self.mock_results_eval_agent,
            execution_mode=ExecutionMode.TEST,
            llm_settings=LLM_UTILS_SETTINGS.get('openai', {})
        )
        # To ensure FitnessEvaluator doesn't try actual LLM calls in TEST mode if not fully mocked for it
        self.fitness_evaluator._call_llm_api = MagicMock(return_value="mocked llm output for fitness test")


        # This is the PopulationManager instance that will be returned by the patched constructor
        self.controlled_pop_manager = PopulationManager(
            genetic_operators=self.mock_genetic_ops,
            fitness_evaluator=self.fitness_evaluator,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=5,
            elitism_count=1,
            message_bus=self.message_bus,
            parallel_workers=None
        )
        # Pre-seed population
        self.controlled_pop_manager.population = [PromptChromosome(genes=[f"chromo{i}"]) for i in range(5)]
        for chromo in self.controlled_pop_manager.population:
            chromo.fitness_score = 0.1
        self.controlled_pop_manager.generation_number = 0


    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('time.sleep', return_value=None)
    def test_main_ga_loop_pause_and_resume(self, mock_time_sleep, MockOrchestratorPopulationManager):
        # Configure the patched PopulationManager in orchestrator.py to return our controlled instance
        MockOrchestratorPopulationManager.return_value = self.controlled_pop_manager

        # Start the controlled_pop_manager in a paused state
        self.controlled_pop_manager.pause_evolution()
        self.mock_connection_manager_for_bus.broadcast_json.reset_mock()

        # Define side effect for evolve_population on our controlled instance
        original_evolve_population = self.controlled_pop_manager.evolve_population

        # Use a simple counter for calls, as MagicMock might be reset if instance is re-patched
        evolve_call_count = 0

        def side_effect_evolve(*args, **kwargs):
            nonlocal evolve_call_count
            evolve_call_count += 1

            # Simulate resuming the GA externally after the first pause check in main_ga_loop
            if evolve_call_count == 1: # This means first attempt to evolve
                print("Test: Simulating resume after first pause check in orchestrator loop.")
                self.controlled_pop_manager.resume_evolution()

            # Simulate what evolve_population does: increments generation number
            # The actual evolution logic is mocked away by not calling original_evolve_population
            # or by having its dependencies (like fitness_evaluator) fully controlled.
            # Here, we directly control generation number for simplicity in test.
            self.controlled_pop_manager.generation_number += 1
            # Ensure status reflects running if not paused/stopped by test logic
            if not self.controlled_pop_manager.is_paused and not self.controlled_pop_manager.should_stop:
                 self.controlled_pop_manager.status = "RUNNING"

        # Replace the method on the instance main_ga_loop will use
        self.controlled_pop_manager.evolve_population = MagicMock(side_effect=side_effect_evolve)

        num_generations = 3
        main_ga_loop(
            task_desc="test task", keywords=["test"],
            num_generations=num_generations, population_size=5, elitism_count=1,
            execution_mode=ExecutionMode.TEST,
            agent_settings_override=None, llm_settings_override=None, parallel_workers=None
        )

        self.controlled_pop_manager.evolve_population = original_evolve_population # Restore

        mock_time_sleep.assert_called() # Check orchestrator loop paused
        self.assertEqual(evolve_call_count, num_generations) # Check it ran all generations

        resume_broadcast_found = False
        for call_args in self.mock_connection_manager_for_bus.broadcast_json.call_args_list:
            if call_args[0][0]['type'] == 'ga_resumed':
                resume_broadcast_found = True
                break
        self.assertTrue(resume_broadcast_found, "ga_resumed broadcast was not found")
        self.assertFalse(self.controlled_pop_manager.is_paused)


    @patch('prompthelix.orchestrator.PopulationManager')
    @patch('time.sleep', return_value=None) # Mock sleep even if not directly testing pause
    def test_main_ga_loop_stop(self, mock_time_sleep, MockOrchestratorPopulationManager):
        MockOrchestratorPopulationManager.return_value = self.controlled_pop_manager
        self.mock_connection_manager_for_bus.broadcast_json.reset_mock()

        original_evolve_population = self.controlled_pop_manager.evolve_population
        evolve_call_count = 0

        def side_effect_evolve(*args, **kwargs):
            nonlocal evolve_call_count
            evolve_call_count += 1

            # Simulate stopping the GA externally after the first generation
            if evolve_call_count == 1:
                print("Test: Simulating stop request after first generation.")
                self.controlled_pop_manager.stop_evolution()

            self.controlled_pop_manager.generation_number += 1
            if not self.controlled_pop_manager.is_paused and not self.controlled_pop_manager.should_stop:
                 self.controlled_pop_manager.status = "RUNNING"


        self.controlled_pop_manager.evolve_population = MagicMock(side_effect=side_effect_evolve)

        num_generations = 5
        main_ga_loop(
            task_desc="test task", keywords=["test"],
            num_generations=num_generations, population_size=5, elitism_count=1,
            execution_mode=ExecutionMode.TEST,
            agent_settings_override=None, llm_settings_override=None, parallel_workers=None
        )

        self.controlled_pop_manager.evolve_population = original_evolve_population # Restore

        # evolve_population on the instance should only be called once
        self.assertEqual(evolve_call_count, 1)
        self.assertTrue(self.controlled_pop_manager.should_stop)
        self.assertEqual(self.controlled_pop_manager.status, "STOPPED")

        stop_broadcast_found = False
        # Check for 'ga_stopping' from stop_evolution() or 'ga_run_stopped' from orchestrator loop
        for call_args in self.mock_connection_manager_for_bus.broadcast_json.call_args_list:
            if call_args[0][0]['type'] in ['ga_stopping', 'ga_run_stopped', 'ga_run_final_status']:
                 if call_args[0][0]['data']['status'] == 'STOPPED':
                    stop_broadcast_found = True
                    break
        self.assertTrue(stop_broadcast_found, "GA stop broadcast with status STOPPED was not found")


if __name__ == '__main__':
    unittest.main()
