import unittest
from unittest.mock import MagicMock, patch, call
import logging

# Ensure imports for classes being tested or mocked
from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner
from prompthelix.genetics.engine import PopulationManager, PromptChromosome
from prompthelix.message_bus import MessageBus # For mocking if PopulationManager needs it

# Disable most logging for unit tests unless specifically testing log output
logging.disable(logging.CRITICAL)

class TestGeneticAlgorithmRunner(unittest.TestCase):

    def setUp(self):
        self.mock_pop_manager = MagicMock(spec=PopulationManager)
        self.mock_pop_manager.generation_number = 0
        self.mock_pop_manager.should_stop = False
        self.mock_pop_manager.status = "IDLE"
        # Mock the broadcast method that GeneticAlgorithmRunner might call on pop_manager
        self.mock_pop_manager.broadcast_ga_update = MagicMock()
        # Mock get_fittest_individual to return a consistent mock chromosome
        self.mock_fittest_chromosome = MagicMock(spec=PromptChromosome)
        self.mock_fittest_chromosome.fitness_score = 0.9
        self.mock_pop_manager.get_fittest_individual.return_value = self.mock_fittest_chromosome

        # Mock evolve_population to simulate generation progression
        # This will be a shared reference, so tests should be careful if they modify it
        # or set it directly in the test for specific behaviors.
        self.shared_evolve_side_effect_details = {'generations_run_in_test': 0}

        # This default side_effect is for simple cases or as a fallback.
        # Tests with complex run logic will define their own side_effect.
        def default_evolve_side_effect(*args, **kwargs):
            # Only increment generation_number if not stopping
            if not self.mock_pop_manager.should_stop:
                self.mock_pop_manager.generation_number += 1
                self.shared_evolve_side_effect_details['generations_run_in_test'] = self.mock_pop_manager.generation_number

            # Use self.num_generations_for_test which should be set by each test method that relies on this default side_effect
            # This is used to determine when to set status to "COMPLETED" by the mock PM
            num_generations_target_for_pm_completion = getattr(self, 'num_generations_for_test', 3)

            if self.mock_pop_manager.should_stop:
                 self.mock_pop_manager.status = "STOPPED"
            elif self.mock_pop_manager.generation_number >= num_generations_target_for_pm_completion:
                 self.mock_pop_manager.status = "COMPLETED"
            else:
                 self.mock_pop_manager.status = "RUNNING"

        self.mock_pop_manager.evolve_population.side_effect = default_evolve_side_effect
        self.num_generations_for_test = 3 # Default for setUp, can be overridden by test methods

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
        self.mock_pop_manager.evolve_population.side_effect = self.setUp.__defaults__[0] if hasattr(self.setUp, '__defaults__') and self.setUp.__defaults__ else type(self).setUp.__dict__['evolve_side_effect']


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

if __name__ == '__main__':
    unittest.main()
