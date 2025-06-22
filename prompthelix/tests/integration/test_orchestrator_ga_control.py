import unittest
from unittest.mock import MagicMock, patch, ANY, AsyncMock
import time # For time.sleep

# Adjust imports based on your project structure
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators, FitnessEvaluator # PopulationManager might be mostly mocked now
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
        # It's now a MagicMock to allow direct control and assertion of its methods.
        self.controlled_pop_manager = MagicMock(spec=PopulationManager)

        # Initialize attributes that PopulationManager instance would have,
        # and that GeneticAlgorithmRunner or main_ga_loop might interact with.
        self.controlled_pop_manager.population = [PromptChromosome(genes=[f"chromo{i}"]) for i in range(5)]
        for chromo in self.controlled_pop_manager.population:
            chromo.fitness_score = 0.1 # Default fitness
        self.controlled_pop_manager.generation_number = 0
        self.controlled_pop_manager.is_paused = False
        self.controlled_pop_manager.should_stop = False
        self.controlled_pop_manager.status = "IDLE"
        # These are accessed by GeneticAlgorithmRunner when creating experiment_parameters
        self.controlled_pop_manager.population_size = 5 # Mocked value
        self.controlled_pop_manager.elitism_count = 1   # Mocked value
        # self.controlled_pop_manager.population_path = "mock/integration_population.json" # Not needed on PM mock
        self.controlled_pop_manager.message_bus = self.message_bus # Runner might use this via PM
        # Ensure methods that return values are configured if the runner uses them directly
        self.controlled_pop_manager.get_fittest_individual.return_value = self.controlled_pop_manager.population[0]


    @patch('prompthelix.orchestrator.PopulationManager', new_callable=MagicMock)
    # No longer patching time.sleep for the orchestrator's pause loop here
    async def test_main_ga_loop_pause_and_resume(self, MockPmConstructorInOrchestrator): # Changed to async def
        # Configure the patched PopulationManager constructor in orchestrator.py
        # to return our MagicMock instance.
        MockPmConstructorInOrchestrator.return_value = self.controlled_pop_manager

        # Reset relevant mock states for this test
        self.controlled_pop_manager.reset_mock() # Resets call counts, etc. on the mock itself
        # Re-initialize attributes on the mock PM for this test run
        self.controlled_pop_manager.generation_number = 0
        self.controlled_pop_manager.is_paused = False
        self.controlled_pop_manager.should_stop = False
        self.controlled_pop_manager.status = "IDLE"
        # Ensure evolve_population is a fresh mock for this test's side effect
        self.controlled_pop_manager.evolve_population = MagicMock()
        # Ensure broadcast_ga_update is also part of the main mock or a fresh sub-mock
        self.controlled_pop_manager.broadcast_ga_update = MagicMock()
        # Ensure control methods are fresh mocks if we assert calls on them
        # Make sure these mock methods also call broadcast_ga_update as the real ones would
        self.controlled_pop_manager.pause_evolution = MagicMock(
            side_effect=lambda: (
                setattr(self.controlled_pop_manager, 'is_paused', True),
                setattr(self.controlled_pop_manager, 'status', "PAUSED"),
                self.controlled_pop_manager.broadcast_ga_update(event_type='ga_paused')
            )
        )
        self.controlled_pop_manager.resume_evolution = MagicMock(
            side_effect=lambda: (
                setattr(self.controlled_pop_manager, 'is_paused', False),
                setattr(self.controlled_pop_manager, 'status', "RUNNING"),
                self.controlled_pop_manager.broadcast_ga_update(event_type='ga_resumed')
            )
        )


        num_generations = 3

        # Simulate external system starting the PopulationManager in a paused state
        # BEFORE the runner's `run` method is called.
        # The runner's `run` method starts by setting PM status to "RUNNING".
        # To test pause *during* run, pause needs to be triggered by a side effect of evolve_population.
        # For this test, let's make the PM start as if it was already running then paused by external agent.

        # Initial state for PM before runner.run() is called by main_ga_loop
        # Runner will set it to "RUNNING" then "PAUSED" if pause_evolution is called by side_effect
        self.controlled_pop_manager.status = "RUNNING" # Runner will set this at start of its run

        evolve_call_attempts = 0

        def custom_evolve_side_effect(*args, **kwargs):
            nonlocal evolve_call_attempts
            evolve_call_attempts += 1

            was_paused_at_call_start = self.controlled_pop_manager.is_paused

            # Simulate external pause AFTER the first successful evolution
            if evolve_call_attempts == 1 and not was_paused_at_call_start:
                self.controlled_pop_manager.pause_evolution()

            # Simulate external resume if this is the attempt immediately following the pause
            if evolve_call_attempts == 2 and was_paused_at_call_start:
                self.controlled_pop_manager.resume_evolution()

            if was_paused_at_call_start: # If it *was* paused when called, it shouldn't evolve this round
                self.controlled_pop_manager.broadcast_ga_update(event_type="ga_status_heartbeat_mocked_evolve")
                return

            # --- Normal evolution logic if not paused ---
            if self.controlled_pop_manager.should_stop:
                self.controlled_pop_manager.status = "STOPPED"
                return

            self.controlled_pop_manager.generation_number += 1
            self.controlled_pop_manager.status = "RUNNING"

        self.controlled_pop_manager.evolve_population.side_effect = custom_evolve_side_effect

        await main_ga_loop( # Added await
            task_desc="test task", keywords=["test"],
            num_generations=num_generations, population_size=5, elitism_count=1,
            execution_mode=ExecutionMode.TEST,
            agent_settings_override=None, llm_settings_override=None, parallel_workers=None
        )

        # Runner calls evolve_population `num_generations` (3) times.
        # 1st call: successful, then pause_evolution() is called by side_effect. PM is_paused = True. Gen = 1.
        # 2nd call: PM is_paused. evolve_population mock returns early (heartbeat). resume_evolution() called by side_effect. PM is_paused = False. Gen = 1.
        # 3rd call: PM not paused. Successful. Gen = 2.
        self.assertEqual(self.controlled_pop_manager.evolve_population.call_count, num_generations)
        self.assertEqual(self.controlled_pop_manager.generation_number, num_generations - 1) # Two successful evolutions

        self.assertFalse(self.controlled_pop_manager.is_paused, "PopulationManager should not be paused at the end.")
        self.controlled_pop_manager.pause_evolution.assert_called_once()
        self.controlled_pop_manager.resume_evolution.assert_called_once()

        # Check broadcasts from the mock PM's control methods
        self.controlled_pop_manager.broadcast_ga_update.assert_any_call(event_type='ga_paused')
        self.controlled_pop_manager.broadcast_ga_update.assert_any_call(event_type='ga_resumed')
        # Check heartbeat broadcast during the paused evolution attempt
        self.controlled_pop_manager.broadcast_ga_update.assert_any_call(event_type='ga_status_heartbeat_mocked_evolve')


    @patch('prompthelix.orchestrator.PopulationManager', new_callable=MagicMock)
    async def test_main_ga_loop_stop(self, MockPmConstructorInOrchestrator): # Changed to async def
        MockPmConstructorInOrchestrator.return_value = self.controlled_pop_manager

        self.controlled_pop_manager.reset_mock()
        self.controlled_pop_manager.generation_number = 0
        self.controlled_pop_manager.is_paused = False
        self.controlled_pop_manager.should_stop = False
        self.controlled_pop_manager.status = "IDLE"
        self.controlled_pop_manager.evolve_population = MagicMock()
        self.controlled_pop_manager.broadcast_ga_update = MagicMock()
        # Ensure stop_evolution is a mock to assert calls and control side effects
        self.controlled_pop_manager.stop_evolution = MagicMock(
            side_effect=lambda: (
                setattr(self.controlled_pop_manager, 'should_stop', True),
                setattr(self.controlled_pop_manager, 'status', "STOPPING"), # Status PM sets
                self.controlled_pop_manager.broadcast_ga_update(event_type='ga_stopping') # Add this call
            )
        )


        num_generations_target = 5 # Orchestrator/Runner will try to run this many

        evolve_attempts_count = 0

        def side_effect_evolve_for_stop(*args, **kwargs):
            nonlocal evolve_attempts_count
            evolve_attempts_count += 1

            # This mock for PM's evolve_population needs to respect should_stop
            if self.controlled_pop_manager.should_stop:
                # print(f"Test evolve_for_stop: Detected should_stop. Attempt: {evolve_attempts_count}")
                # If PopulationManager.evolve_population itself sets status to STOPPED
                # self.controlled_pop_manager.status = "STOPPED"
                return

            # Simulate normal evolution if not stopped
            self.controlled_pop_manager.generation_number += 1
            self.controlled_pop_manager.status = "RUNNING" # After successful evolution part
            # print(f"Test evolve_for_stop: Evolved gen {self.controlled_pop_manager.generation_number}. Attempt: {evolve_attempts_count}")

            # Test logic: Stop after the first successful evolution
            if evolve_attempts_count == 1:
                # print("Test evolve_for_stop: Simulating external STOP request after 1st evolution.")
                self.controlled_pop_manager.stop_evolution() # Sets should_stop=True, status="STOPPING"

        self.controlled_pop_manager.evolve_population.side_effect = side_effect_evolve_for_stop

        await main_ga_loop( # Added await
            task_desc="test task", keywords=["test"],
            num_generations=num_generations_target, population_size=5, elitism_count=1,
            execution_mode=ExecutionMode.TEST,
            agent_settings_override=None, llm_settings_override=None, parallel_workers=None
        )

        # Runner's loop:
        # Gen 1: evolve_population called. side_effect runs, increments gen_number to 1. stop_evolution() called. should_stop = True.
        # Gen 2: Runner's loop checks should_stop *before* calling evolve_population. It's true. Loop breaks.
        # So, evolve_population (the mock) should be called only once.
        self.assertEqual(self.controlled_pop_manager.evolve_population.call_count, 1)
        self.assertTrue(self.controlled_pop_manager.should_stop)

        # The runner's loop detects should_stop and sets PM status to "STOPPED" and broadcasts.
        # The self.controlled_pop_manager.status might be "STOPPING" from its own stop_evolution,
        # but the runner updates it to "STOPPED".
        self.assertEqual(self.controlled_pop_manager.status, "STOPPED")
        self.controlled_pop_manager.stop_evolution.assert_called_once()

        # Check broadcasts
        # stop_evolution() on PM mock broadcasts "ga_stopping".
        # Runner broadcasts "ga_run_stopped_runner_signal" or similar when its loop breaks due to stop.
        self.controlled_pop_manager.broadcast_ga_update.assert_any_call(event_type='ga_stopping')
        self.controlled_pop_manager.broadcast_ga_update.assert_any_call(event_type='ga_run_stopped_runner_signal')


if __name__ == '__main__':
    unittest.main()
