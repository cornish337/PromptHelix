import unittest
from unittest.mock import MagicMock, patch, call, AsyncMock, ANY
import logging
import asyncio

from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner
from prompthelix.genetics.engine import PopulationManager, PromptChromosome
from prompthelix.message_bus import MessageBus
from prompthelix.websocket_manager import ConnectionManager

logging.disable(logging.CRITICAL)

class TestGeneticAlgorithmRunner(unittest.TestCase):

    def setUp(self):
        self.mock_pop_manager = MagicMock(spec=PopulationManager)
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.mock_pop_manager.population_size = 10
        self.mock_pop_manager.elitism_count = 1

        # Make PopulationManager's async methods AsyncMocks
        self.mock_pop_manager.broadcast_ga_update = AsyncMock()
        self.mock_pop_manager.pause_evolution = AsyncMock()
        self.mock_pop_manager.resume_evolution = AsyncMock()
        self.mock_pop_manager.stop_evolution = AsyncMock()
        self.mock_pop_manager.evolve_population = AsyncMock() # Now async

        self.mock_fittest_chromosome = MagicMock(spec=PromptChromosome)
        self.mock_fittest_chromosome.fitness_score = 0.9
        self.mock_fittest_chromosome.id = "mock_chromosome_id_123"
        self.mock_pop_manager.get_fittest_individual.return_value = self.mock_fittest_chromosome

        self.shared_evolve_side_effect_details = {'generations_run_in_test': 0}

        async def default_async_evolve_side_effect(*args, **kwargs): # Made async
            if not self.mock_pop_manager.should_stop:
                self.mock_pop_manager.generation_number += 1
                self.shared_evolve_side_effect_details['generations_run_in_test'] = self.mock_pop_manager.generation_number
                self.mock_pop_manager.status = "RUNNING"
            else:
                self.mock_pop_manager.status = "STOPPED"

        self.default_evolve_side_effect = default_async_evolve_side_effect # Keep name for ref if needed
        self.mock_pop_manager.evolve_population.side_effect = self.default_evolve_side_effect
        self.num_generations_for_test = 3

        self.ga_runner_logger = logging.getLogger('prompthelix.experiment_runners.ga_runner')
        self.original_ga_runner_logger_handlers = list(self.ga_runner_logger.handlers)
        self.original_ga_runner_logger_level = self.ga_runner_logger.level
        self.original_ga_runner_logger_disabled_status = self.ga_runner_logger.disabled

        self.ga_runner_logger.setLevel(logging.INFO)
        self.ga_runner_logger.disabled = False

    def tearDown(self):
        self.ga_runner_logger.setLevel(self.original_ga_runner_logger_level)
        self.ga_runner_logger.disabled = self.original_ga_runner_logger_disabled_status
        self.ga_runner_logger.handlers = self.original_ga_runner_logger_handlers

    def test_init_successful(self):
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 5, population_persistence_path="mock/path.json")
        self.assertIsInstance(runner, GeneticAlgorithmRunner)
        self.assertEqual(runner.population_manager, self.mock_pop_manager)
        self.assertEqual(runner.num_generations, 5)

    def test_init_invalid_pop_manager_type(self):
        with self.assertRaisesRegex(TypeError, "population_manager must be an instance of PopulationManager."):
            GeneticAlgorithmRunner("not_a_pop_manager", 5, population_persistence_path=None)

    def test_init_invalid_num_generations(self):
        with self.assertRaisesRegex(ValueError, "num_generations must be a positive integer."):
            GeneticAlgorithmRunner(self.mock_pop_manager, 0, population_persistence_path=None)
        with self.assertRaisesRegex(ValueError, "num_generations must be a positive integer."):
            GeneticAlgorithmRunner(self.mock_pop_manager, -1, population_persistence_path=None)

    async def test_run_completes_all_generations(self): # Changed to async
        self.num_generations_for_test = 3
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0
        self.mock_pop_manager.evolve_population.side_effect = self.default_evolve_side_effect

        runner = GeneticAlgorithmRunner(self.mock_pop_manager, self.num_generations_for_test, population_persistence_path=None)

        run_kwargs = {"task_description": "test task", "success_criteria": {}, "target_style": "formal"}
        best_chromosome = await runner.run(**run_kwargs) # await

        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, self.num_generations_for_test)
        self.assertEqual(runner.current_generation, self.num_generations_for_test)
        self.assertEqual(self.mock_pop_manager.status, "COMPLETED")
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(event_type="ga_run_completed_runner")
        self.assertEqual(best_chromosome, self.mock_fittest_chromosome)

        expected_call = call(
            task_description="test task",
            success_criteria={},
            db_session=ANY,
            experiment_run=ANY
        )
        self.mock_pop_manager.evolve_population.assert_has_calls([expected_call] * self.num_generations_for_test)


    async def test_run_stops_early_if_should_stop_is_true(self): # Changed to async
        num_target_generations_for_runner = 5
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0

        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_target_generations_for_runner, population_persistence_path=None)

        generations_evolved_in_test = 0
        async def evolve_side_effect_with_early_stop(*args, **kwargs): # Made async
            nonlocal generations_evolved_in_test
            if self.mock_pop_manager.should_stop:
                self.mock_pop_manager.status = "STOPPED"
                return

            self.mock_pop_manager.generation_number += 1
            generations_evolved_in_test += 1

            if generations_evolved_in_test == 2:
                self.mock_pop_manager.should_stop = True
                self.mock_pop_manager.status = "RUNNING"
            else:
                self.mock_pop_manager.status = "RUNNING"

        self.mock_pop_manager.evolve_population.side_effect = evolve_side_effect_with_early_stop

        run_kwargs = {"task_description": "test task"}
        await runner.run(**run_kwargs) # await

        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, 2)
        self.assertEqual(runner.current_generation, 3)
        self.assertEqual(self.mock_pop_manager.status, "STOPPED")
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(event_type="ga_run_stopped_runner_signal")


    async def test_run_handles_exception_in_evolve_population(self): # Changed to async
        num_target_generations = 3
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        self.shared_evolve_side_effect_details['generations_run_in_test'] = 0

        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_target_generations, population_persistence_path=None)

        evolve_call_count = 0
        async def evolve_exception_side_effect(*args, **kwargs): # Made async
            nonlocal evolve_call_count
            evolve_call_count += 1
            if evolve_call_count == 1:
                raise Exception("Evolve failed!")
            self.mock_pop_manager.generation_number +=1
            self.mock_pop_manager.status="RUNNING"

        self.mock_pop_manager.evolve_population.side_effect = evolve_exception_side_effect

        run_kwargs = {"task_description": "test task"}
        with self.assertRaisesRegex(Exception, "Evolve failed!"):
            await runner.run(**run_kwargs) # await

        self.assertEqual(self.mock_pop_manager.status, "ERROR")
        self.mock_pop_manager.broadcast_ga_update.assert_any_call(
            event_type="ga_run_error_runner",
            additional_data={"error": "Evolve failed!"}
        )
        self.assertEqual(self.mock_pop_manager.evolve_population.call_count, 1)
        self.assertEqual(runner.current_generation, 1)


    async def test_pause_calls_pop_manager_pause(self): # Changed to async
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3, population_persistence_path=None)
        await runner.pause() # await
        self.mock_pop_manager.pause_evolution.assert_called_once()

    async def test_resume_calls_pop_manager_resume(self): # Changed to async
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3, population_persistence_path=None)
        await runner.resume() # await
        self.mock_pop_manager.resume_evolution.assert_called_once()

    async def test_stop_calls_pop_manager_stop(self): # Changed to async
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, 3, population_persistence_path=None)
        await runner.stop() # await
        self.mock_pop_manager.stop_evolution.assert_called_once()

    def test_get_status_combines_statuses(self):
        runner_target_generations = 5
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, runner_target_generations, population_persistence_path=None)
        runner.current_generation = 2
        self.mock_pop_manager.generation_number = 1
        self.mock_pop_manager.status = "RUNNING"

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

    @patch('prompthelix.globals.websocket_manager.broadcast_json', new_callable=AsyncMock)
    @patch('prompthelix.logging_handlers.asyncio.create_task')
    def test_ga_runner_init_logs_are_sent_via_websocket_handler(self, mock_create_task, mock_broadcast_json_patched_on_globals):
        ga_logger = logging.getLogger('prompthelix.experiment_runners.ga_runner')
        original_handlers = list(ga_logger.handlers)
        test_mock_connection_manager = MagicMock(spec=ConnectionManager)
        test_mock_connection_manager.broadcast_json = mock_broadcast_json_patched_on_globals
        from prompthelix.logging_handlers import WebSocketLogHandler
        test_ws_log_handler = WebSocketLogHandler(connection_manager=test_mock_connection_manager)
        test_ws_log_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
        test_ws_log_handler.setFormatter(formatter)
        ga_logger.handlers = [h for h in original_handlers if not isinstance(h, WebSocketLogHandler)]
        ga_logger.addHandler(test_ws_log_handler)
        ga_logger.disabled = False
        ga_logger.setLevel(logging.INFO)
        original_global_disable_level = logging.root.manager.disable
        logging.disable(logging.NOTSET)

        def side_effect_for_create_task(coro):
            try:
                coro.send(None)
            except StopIteration:
                pass
            except Exception:
                coro.close()
            return MagicMock()

        mock_create_task.side_effect = side_effect_for_create_task
        original_emit = test_ws_log_handler.emit
        test_ws_log_handler.emit = MagicMock(wraps=original_emit)
        num_generations = 5
        runner = GeneticAlgorithmRunner(self.mock_pop_manager, num_generations, population_persistence_path="mock/ws_test_path.json")
        self.assertTrue(test_ws_log_handler.emit.called, "Test WebSocketLogHandler's emit method was not called.")
        self.assertTrue(mock_broadcast_json_patched_on_globals.called,
                        "Patched broadcast_json (via test_mock_connection_manager) was not called.")
        found_init_log = False
        for call_args in mock_broadcast_json_patched_on_globals.call_args_list:
            log_data_sent = call_args[0][0]
            if log_data_sent.get('type') == 'debug_log':
                log_payload = log_data_sent.get('data', {})
                if ("GeneticAlgorithmRunner initialized" in log_payload.get('message', "") and
                    log_payload.get('module') == 'ga_runner' and
                    log_payload.get('funcName') == '__init__'):
                    found_init_log = True
                    self.assertEqual(log_payload.get('level'), 'INFO')
                    self.assertIn(f"for {num_generations} generations", log_payload.get('message', ""))
                    self.assertIn(f"with PopulationManager (ID: {id(self.mock_pop_manager)})", log_payload.get('message', ""))
                    break
        self.assertTrue(found_init_log,
                        f"The specific GA Runner __init__ log message was not found. Calls: {mock_broadcast_json_patched_on_globals.call_args_list}")
        ga_logger.handlers = original_handlers
        logging.disable(original_global_disable_level)

    @patch('prompthelix.experiment_runners.ga_runner.logger')
    async def test_periodic_save_triggered_correctly(self, mock_logger): # Changed to async
        self.mock_pop_manager.population_path = "test/path/pop.json"
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "RUNNING"
        self.mock_pop_manager.generation_number = 0
        num_generations_to_run = 5
        save_freq = 2
        current_generation_in_test = 0
        async def fake_evolve_population(*args, **kwargs): # Made async
            nonlocal current_generation_in_test
            current_generation_in_test += 1
            self.mock_pop_manager.generation_number = current_generation_in_test
            if current_generation_in_test >= num_generations_to_run:
                self.mock_pop_manager.should_stop = True
        self.mock_pop_manager.evolve_population.side_effect = fake_evolve_population
        runner = GeneticAlgorithmRunner(
            population_manager=self.mock_pop_manager,
            num_generations=num_generations_to_run,
            save_frequency=save_freq,
            population_persistence_path=self.mock_pop_manager.population_path
        )
        await runner.run(task_description="test task for periodic save") # await
        self.assertEqual(self.mock_pop_manager.save_population.call_count, 2)
        self.mock_pop_manager.save_population.assert_any_call("test/path/pop.json")
        self.assertTrue(any("Periodically saved population at generation 2" in str(c[0][0]) for c in mock_logger.info.call_args_list))
        self.assertTrue(any("Periodically saved population at generation 4" in str(c[0][0]) for c in mock_logger.info.call_args_list))

    @patch('prompthelix.experiment_runners.ga_runner.logger')
    async def test_periodic_save_not_triggered_when_frequency_is_zero(self, mock_logger): # Changed to async
        self.mock_pop_manager.population_path = "test/path/pop.json"
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "RUNNING"
        self.mock_pop_manager.generation_number = 0
        num_generations_to_run = 3
        save_freq = 0
        current_generation_in_test = 0
        async def fake_evolve_population(*args, **kwargs): # Made async
            nonlocal current_generation_in_test
            current_generation_in_test += 1
            self.mock_pop_manager.generation_number = current_generation_in_test
            if current_generation_in_test >= num_generations_to_run:
                self.mock_pop_manager.should_stop = True
        self.mock_pop_manager.evolve_population.side_effect = fake_evolve_population
        runner = GeneticAlgorithmRunner(
            population_manager=self.mock_pop_manager,
            num_generations=num_generations_to_run,
            save_frequency=save_freq,
            population_persistence_path=self.mock_pop_manager.population_path
        )
        await runner.run(task_description="test task no save freq 0") # await
        self.mock_pop_manager.save_population.assert_not_called()

    @patch('prompthelix.experiment_runners.ga_runner.logger')
    async def test_periodic_save_not_triggered_when_no_population_path(self, mock_logger): # Changed to async
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "RUNNING"
        self.mock_pop_manager.generation_number = 0
        num_generations_to_run = 3
        save_freq = 1
        current_generation_in_test = 0
        async def fake_evolve_population(*args, **kwargs): # Made async
            nonlocal current_generation_in_test
            current_generation_in_test += 1
            self.mock_pop_manager.generation_number = current_generation_in_test
            if current_generation_in_test >= num_generations_to_run:
                self.mock_pop_manager.should_stop = True
        self.mock_pop_manager.evolve_population.side_effect = fake_evolve_population
        runner = GeneticAlgorithmRunner(
            population_manager=self.mock_pop_manager,
            num_generations=num_generations_to_run,
            save_frequency=save_freq,
            population_persistence_path=None
        )
        await runner.run(task_description="test task no save path") # await
        self.mock_pop_manager.save_population.assert_not_called()

    @patch('prompthelix.experiment_runners.ga_runner.logger')
    async def test_periodic_save_not_triggered_at_generation_zero(self, mock_logger): # Changed to async
        self.mock_pop_manager.population_path = "test/path/pop.json"
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "RUNNING"
        self.mock_pop_manager.generation_number = 0
        num_generations_to_run = 1
        save_freq = 1
        current_generation_in_test = 0
        async def fake_evolve_population(*args, **kwargs): # Made async
            nonlocal current_generation_in_test
            current_generation_in_test += 1
            self.mock_pop_manager.generation_number = current_generation_in_test
            if current_generation_in_test >= num_generations_to_run:
                 self.mock_pop_manager.should_stop = True
        self.mock_pop_manager.evolve_population.side_effect = fake_evolve_population
        runner = GeneticAlgorithmRunner(
            population_manager=self.mock_pop_manager,
            num_generations=num_generations_to_run,
            save_frequency=save_freq,
            population_persistence_path=self.mock_pop_manager.population_path
        )
        await runner.run(task_description="test task gen zero") # await
        self.assertEqual(self.mock_pop_manager.save_population.call_count, 1)
        self.mock_pop_manager.save_population.assert_called_with("test/path/pop.json")
        self.assertTrue(any("Periodically saved population at generation 1" in str(c[0][0]) for c in mock_logger.info.call_args_list))

    @patch('prompthelix.experiment_runners.ga_runner.logger')
    async def test_periodic_save_handles_exception_during_save_operation(self, mock_logger): # Changed to async
        self.mock_pop_manager.population_path = "test/path/pop.json"
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "RUNNING"
        self.mock_pop_manager.generation_number = 0
        num_generations_to_run = 2
        save_freq = 1
        save_call_count = 0
        def fake_save_population_with_error(*args, **kwargs):
            nonlocal save_call_count
            save_call_count += 1
            if save_call_count == 1:
                raise IOError("Disk full!")
        self.mock_pop_manager.save_population.side_effect = fake_save_population_with_error
        current_generation_in_test = 0
        async def fake_evolve_population(*args, **kwargs): # Made async
            nonlocal current_generation_in_test
            current_generation_in_test += 1
            self.mock_pop_manager.generation_number = current_generation_in_test
            if current_generation_in_test >= num_generations_to_run:
                self.mock_pop_manager.should_stop = True
        self.mock_pop_manager.evolve_population.side_effect = fake_evolve_population
        runner = GeneticAlgorithmRunner(
            population_manager=self.mock_pop_manager,
            num_generations=num_generations_to_run,
            save_frequency=save_freq,
            population_persistence_path=self.mock_pop_manager.population_path
        )
        await runner.run(task_description="test task save exception") # await
        self.assertEqual(self.mock_pop_manager.save_population.call_count, 2)
        self.assertTrue(any("Error during periodic save of population at generation 1" in str(c[0][0]) for c in mock_logger.error.call_args_list))
        self.assertTrue(any("Disk full!" in str(c[0][0]) for c in mock_logger.error.call_args_list))
        self.assertTrue(any("Periodically saved population at generation 2" in str(c[0][0]) for c in mock_logger.info.call_args_list))

if __name__ == '__main__':
    unittest.main()
