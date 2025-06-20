import unittest
from unittest.mock import MagicMock, patch, call, AsyncMock
import logging
import asyncio # For asyncio.create_task patching

# Ensure imports for classes being tested or mocked
from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner
from prompthelix.genetics.engine import PopulationManager, PromptChromosome
from prompthelix.message_bus import MessageBus # For mocking if PopulationManager needs it
from prompthelix.websocket_manager import ConnectionManager # Added import

# Disable most logging for unit tests unless specifically testing log output
logging.disable(logging.CRITICAL) # Global disable

class TestGeneticAlgorithmRunner(unittest.TestCase):

    def setUp(self):
        # ... (existing setUp code) ...
        self.mock_pop_manager = MagicMock(spec=PopulationManager)
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        # Mock the broadcast method that GeneticAlgorithmRunner might call on pop_manager
        self.mock_pop_manager.broadcast_ga_update = MagicMock()
        # Mock get_fittest_individual to return a consistent mock chromosome
        self.mock_fittest_chromosome = MagicMock(spec=PromptChromosome)
        self.mock_fittest_chromosome.fitness_score = 0.9
        self.mock_fittest_chromosome.id = "mock_chromosome_id_123" # Add id attribute
        self.mock_pop_manager.get_fittest_individual.return_value = self.mock_fittest_chromosome

        # Mock evolve_population to simulate generation progression
        self.shared_evolve_side_effect_details = {'generations_run_in_test': 0}

        def default_evolve_side_effect(*args, **kwargs):
            if not self.mock_pop_manager.should_stop:
                self.mock_pop_manager.generation_number += 1
                self.shared_evolve_side_effect_details['generations_run_in_test'] = self.mock_pop_manager.generation_number
                self.mock_pop_manager.status = "RUNNING" # Runner should determine COMPLETED
            else:
                self.mock_pop_manager.status = "STOPPED"

        self.default_evolve_side_effect = default_evolve_side_effect
        self.mock_pop_manager.evolve_population.side_effect = self.default_evolve_side_effect
        self.num_generations_for_test = 3

        self.ga_runner_logger = logging.getLogger('prompthelix.experiment_runners.ga_runner')
        self.original_ga_runner_logger_handlers = list(self.ga_runner_logger.handlers)
        self.original_ga_runner_logger_level = self.ga_runner_logger.level
        self.original_ga_runner_logger_disabled_status = self.ga_runner_logger.disabled

        self.ga_runner_logger.setLevel(logging.INFO)
        self.ga_runner_logger.disabled = False

        # This test class might add its own handler for specific tests,
        # so original handlers are stored to be restored.

    def tearDown(self):
        self.ga_runner_logger.setLevel(self.original_ga_runner_logger_level)
        self.ga_runner_logger.disabled = self.original_ga_runner_logger_disabled_status
        # Restore original handlers, removing any added during tests
        self.ga_runner_logger.handlers = self.original_ga_runner_logger_handlers

    def test_init_successful(self):
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 5)
        self.assertIsInstance(runner, GeneticAlgorithmRunner)
        self.assertEqual(runner.population_manager, self.mock_pop_manager)
        self.assertEqual(runner.num_generations, 5)

    def test_init_invalid_pop_manager_type(self):
        with self.assertRaisesRegex(TypeError, "population_manager must be an instance of PopulationManager."):
            GeneticAlgorithmRunner("not_a_pop_manager", 5)

    def test_init_invalid_num_generations(self):
        with self.assertRaisesRegex(ValueError, "num_generations must be a positive integer."):
            GeneticAlgorithmRunner(self.mock_pop_manager, 0)
        with self.assertRaisesRegex(ValueError, "num_generations must be a positive integer."):
            GeneticAlgorithmRunner(self.mock_pop_manager, -1)

    def test_run_completes_all_generations(self):
        self.num_generations_for_test = 3 # Specific to this test, used by the default side_effect
        # Reset generation_number and other relevant states for this specific test run
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE" # Initial status before run
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0
        # Ensure the default side effect is used for this test
        self.mock_pop_manager.evolve_population.side_effect = self.default_evolve_side_effect


        runner = GeneticAlgorithmRunner(self.mock_pop_manager, self.num_generations_for_test)

        run_kwargs = {"task_description": "test task", "success_criteria": {}, "target_style": "formal"}
        best_chromosome = runner.run(**run_kwargs)

        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, self.num_generations_for_test)
        # runner.current_generation is updated: self.current_generation = self.population_manager.generation_number + 1
        # After 3 successful evolutions, pop_manager.generation_number will be 3.
        # In the loop, runner.current_generation will be 1, then 2, then 3.
        self.assertEqual(runner.current_generation, self.num_generations_for_test)
        self.assertEqual(self.mock_pop_manager.status, "COMPLETED") # Set by runner
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(event_type="ga_run_completed_runner")
        self.assertEqual(best_chromosome, self.mock_fittest_chromosome)

        expected_call = call(
            task_description="test task",
            success_criteria={},
            target_style="formal"
        )
        self.mock_pop_manager.evolve_population.assert_has_calls([expected_call] * self.num_generations_for_test)


    def test_run_stops_early_if_should_stop_is_true(self):
        num_target_generations_for_runner = 5 # Runner aims for 5
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0

        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_target_generations_for_runner)

        generations_evolved_in_test = 0
        def evolve_side_effect_with_early_stop(*args, **kwargs):
            nonlocal generations_evolved_in_test
            # This side effect is specific to this test
            if self.mock_pop_manager.should_stop:
                self.mock_pop_manager.status = "STOPPED" # PM might set this if it detects stop early
                return

            self.mock_pop_manager.generation_number += 1
            generations_evolved_in_test += 1

            if generations_evolved_in_test == 2: # After 2 successful evolutions
                self.mock_pop_manager.should_stop = True # Signal stop *before* the next generation check in runner
                self.mock_pop_manager.status = "RUNNING" # Status after current evolution completes
            else: # Should not be called if should_stop is true earlier
                self.mock_pop_manager.status = "RUNNING"

        self.mock_pop_manager.evolve_population.side_effect = evolve_side_effect_with_early_stop

        run_kwargs = {"task_description": "test task"}
        runner.run(**run_kwargs)

        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, 2) # Evolved twice
        # current_generation is set at the start of the loop iteration.
        # Gen 1: current_generation = 1, evolve runs.
        # Gen 2: current_generation = 2, evolve runs, should_stop becomes true.
        # Gen 3: current_generation = 3, loop starts, should_stop is true, breaks.
        # So runner.current_generation will be 3 when it breaks.
        self.assertEqual(runner.current_generation, 3)
        self.assertEqual(self.mock_pop_manager.status, "STOPPED") # Runner sets this
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(event_type="ga_run_stopped_runner_signal")


    def test_run_handles_exception_in_evolve_population(self):
        num_target_generations = 3
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0

        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_target_generations)

        evolve_call_count = 0
        def evolve_exception_side_effect(*args, **kwargs):
            nonlocal evolve_call_count
            evolve_call_count += 1
            if evolve_call_count == 1: # Raise exception on the first call
                # PM's generation_number would not have incremented yet by this mock
                raise Exception("Evolve failed!")
            # Fallback (should not be reached in this test if exception is handled)
            self.mock_pop_manager.generation_number +=1
            self.mock_pop_manager.status="RUNNING"

        self.mock_pop_manager.evolve_population.side_effect = evolve_exception_side_effect

        run_kwargs = {"task_description": "test task"}
        with self.assertRaisesRegex(Exception, "Evolve failed!"):
            runner.run(**run_kwargs)

        self.assertEqual(self.mock_pop_manager.status, "ERROR") # Set by runner's except block
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(
            event_type="ga_run_error_runner",
            additional_data={"error": "Evolve failed!"}
        )
        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, 1)
        # runner.current_generation would be 1 when the exception occurred.
        self.assertEqual(runner.current_generation, 1)


    def test_pause_calls_pop_manager_pause(self):
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3)
        runner.pause()
        self.mock_pop_manager.pause_evolution.assert_called_once()

    def test_resume_calls_pop_manager_resume(self):
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3)
        runner.resume()
        self.mock_pop_manager.resume_evolution.assert_called_once()

    def test_stop_calls_pop_manager_stop(self):
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3)
        runner.stop()
        self.mock_pop_manager.stop_evolution.assert_called_once()

    def test_get_status_combines_statuses(self):
        runner_target_generations = 5
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, runner_target_generations)

        # Simulate runner being partway through, about to start generation 2
        runner.current_generation = 2

        # Simulate PopulationManager's actual state: completed 1 generation
        self.mock_pop_manager.generation_number = 1
        self.mock_pop_manager.status = "RUNNING" # PM status

        mock_pm_status_payload = {
            "status": "RUNNING",
            "generation": self.mock_pop_manager.generation_number,
            "population_size": 10,
            "best_fitness": 0.8
        }
        self.mock_pop_manager.get_ga_status.return_value = mock_pm_status_payload

        expected_pm_id = id(self.mock_pop_manager)

        status_report = runner.get_status()

        self.mock_pop_manager.get_ga_status.assert_called_once()
        expected_combined_status = {
            "status": "RUNNING",
            "generation": 1,
            "population_size": 10,
            "best_fitness": 0.8,
            "runner_current_generation": 2,
            "runner_target_generations": runner_target_generations,
            "runner_population_manager_id": expected_pm_id
        }
        self.assertEqual(status_report, expected_combined_status)

    @patch('prompthelix.globals.websocket_manager.broadcast_json', new_callable=AsyncMock) # Reverted to AsyncMock
    @patch('prompthelix.logging_handlers.asyncio.create_task') # Corrected patch target for create_task
    def test_ga_runner_init_logs_are_sent_via_websocket_handler(self, mock_create_task, mock_broadcast_json_patched_on_globals): # Name back to original
        # This test verifies the integration of ga_runner's logging with WebSocketLogHandler.

        # The WebSocketLogHandler in ga_runner.py is added at module import time and
        # captures the original websocket_manager.broadcast_json.
        # Patching prompthelix.globals.websocket_manager.broadcast_json directly in the test
        # won't affect the already configured handler instance.
        # Instead, we need to replace the handler or its connection_manager's broadcast_json.

        ga_logger = logging.getLogger('prompthelix.experiment_runners.ga_runner')
        original_handlers = list(ga_logger.handlers) # Store original handlers

        # Find the existing WebSocketLogHandler and temporarily replace its connection_manager's method
        # OR, more robustly, remove it and add a new one with a mock connection manager.

        # Create a new mock connection manager for this test
        test_mock_connection_manager = MagicMock(spec=ConnectionManager)
        test_mock_connection_manager.broadcast_json = mock_broadcast_json_patched_on_globals # Use original name

        # Import WebSocketLogHandler here to avoid issues if it's not yet fully imported elsewhere
        from prompthelix.logging_handlers import WebSocketLogHandler
        test_ws_log_handler = WebSocketLogHandler(connection_manager=test_mock_connection_manager)
        test_ws_log_handler.setLevel(logging.INFO) # Ensure it processes INFO logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        test_ws_log_handler.setFormatter(formatter)

        # Remove existing WebSocketLogHandlers and add our test-specific one
        ga_logger.handlers = [h for h in original_handlers if not isinstance(h, WebSocketLogHandler)]
        ga_logger.addHandler(test_ws_log_handler)

        # Ensure logger is enabled (it's handled in setUp/tearDown generally, but double check for this specific logger)
        ga_logger.disabled = False
        ga_logger.setLevel(logging.INFO)

        # Temporarily change global logging disable level for this test
        original_global_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET) # Allow INFO logs globally


        def side_effect_for_create_task(coro):
            try:
                coro.send(None)
            except StopIteration:
                pass
            except Exception:
                coro.close()
            return MagicMock()

        mock_create_task.side_effect = side_effect_for_create_task

        # Wrap the test handler's emit method to check if it's called
        original_emit = test_ws_log_handler.emit
        test_ws_log_handler.emit = MagicMock(wraps=original_emit)

        num_generations = 5
        # Instantiating GeneticAlgorithmRunner should trigger its __init__ log
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_generations)

        # Debug: Check if the test handler's emit was called
        self.assertTrue(test_ws_log_handler.emit.called, "Test WebSocketLogHandler's emit method was not called.")

        # Assert that our mock (indirectly via test_mock_connection_manager) was called
        self.assertTrue(mock_broadcast_json_patched_on_globals.called, # Use original name
                        "Patched broadcast_json (via test_mock_connection_manager) was not called.")

        found_init_log = False
        # The WebSocketLogHandler in ga_runner.py is configured with a specific formatter:
        # '%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
        # The 'message' part of the log_payload['message'] will be the raw message given to logger.info()
        # The other fields like module, funcName, etc., are separate in log_payload.

        for call_args in mock_broadcast_json_patched_on_globals.call_args_list: # Corrected variable name
            log_data_sent = call_args[0][0]
            if log_data_sent.get('type') == 'debug_log':
                log_payload = log_data_sent.get('data', {})

                # Check for the specific message from GeneticAlgorithmRunner.__init__
                # The logged message is:
                # f"GeneticAlgorithmRunner initialized for {num_generations} generations "
                # f"with PopulationManager (ID: {id(population_manager)})"
                # We check for a substring.
                if ("GeneticAlgorithmRunner initialized" in log_payload.get('message', "") and
                    log_payload.get('module') == 'ga_runner' and
                    log_payload.get('funcName') == '__init__'):
                    found_init_log = True
                    self.assertEqual(log_payload.get('level'), 'INFO')
                    self.assertIn(f"for {num_generations} generations", log_payload.get('message', ""))
                    self.assertIn(f"with PopulationManager (ID: {id(self.mock_pop_manager)})", log_payload.get('message', ""))
                    break

        self.assertTrue(found_init_log,
                        f"The specific GA Runner __init__ log message was not found. Calls: {mock_broadcast_json_patched_on_globals.call_args_list}") # Use original name

        # Restore original handlers after the test
        ga_logger.handlers = original_handlers
        # Restore global logging disable level
        logging.disable(original_global_disable_level)


if __name__ == '__main__':
    unittest.main()
