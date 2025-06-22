import asyncio
import unittest
from unittest.mock import MagicMock, AsyncMock, patch, ANY

# Adjust imports based on your project structure
from prompthelix.genetics.engine import PopulationManager
from prompthelix.genetics.chromosome import PromptChromosome
from prompthelix.message_bus import MessageBus
# ConnectionManager might not be directly imported by PopulationManager,
# but MessageBus uses it. We mock the relevant parts.

class TestPopulationManagerControls(unittest.TestCase):

    def setUp(self):
        self.mock_genetic_operators = MagicMock()
        self.mock_fitness_evaluator = MagicMock()
        self.mock_prompt_architect_agent = MagicMock()

        self.mock_message_bus = MagicMock(spec=MessageBus)
        # Mock the connection_manager attribute that PopulationManager will access via message_bus
        self.mock_message_bus.connection_manager = AsyncMock()
        self.mock_message_bus.connection_manager.broadcast_json = AsyncMock()

        self.pop_manager = PopulationManager(
            genetic_operators=self.mock_genetic_operators,
            fitness_evaluator=self.mock_fitness_evaluator,
            prompt_architect_agent=self.mock_prompt_architect_agent,
            population_size=10,
            elitism_count=1,
            message_bus=self.mock_message_bus # Pass the mocked message bus
        )

        # Ensure a minimal population for get_fittest_individual to work if called by broadcast_ga_update
        self.fittest_chromo = PromptChromosome(genes=["fittest"], fitness_score=0.9)
        self.other_chromo = PromptChromosome(genes=["other"], fitness_score=0.5)
        # PopulationManager's __init__ calls broadcast_ga_update.
        # If population is empty, get_fittest_individual is None.
        # We set it here for subsequent tests, after __init__ has run.
        self.pop_manager.population = [self.fittest_chromo, self.other_chromo]


    @patch('asyncio.create_task')
    def test_pause_evolution(self, mock_create_task):
        # Reset mock from __init__'s broadcast or other setup
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.pop_manager.is_paused = False # Ensure starting state for this test
        self.pop_manager.status = "RUNNING" # Ensure starting state

        self.pop_manager.pause_evolution()

        self.assertTrue(self.pop_manager.is_paused)
        self.assertEqual(self.pop_manager.status, "PAUSED")

        # Check that broadcast_json (which is called by create_task in broadcast_ga_update) was called
        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_paused')
        self.assertEqual(args[0]['data']['status'], 'PAUSED')


    @patch('asyncio.create_task')
    def test_resume_evolution(self, mock_create_task):
        self.pop_manager.is_paused = True # Set to paused state first
        self.pop_manager.status = "PAUSED"

        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.pop_manager.resume_evolution()

        self.assertFalse(self.pop_manager.is_paused)
        self.assertEqual(self.pop_manager.status, "RUNNING")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_resumed')
        self.assertEqual(args[0]['data']['status'], 'RUNNING')

    @patch('asyncio.create_task')
    def test_stop_evolution(self, mock_create_task):
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.pop_manager.stop_evolution()

        self.assertTrue(self.pop_manager.should_stop)
        self.assertFalse(self.pop_manager.is_paused) # Should be unpaused
        self.assertEqual(self.pop_manager.status, "STOPPING")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_stopping')
        self.assertEqual(args[0]['data']['status'], 'STOPPING')

    @patch('asyncio.create_task')
    def test_broadcast_ga_update_no_message_bus(self, mock_create_task):
        # Store original bus, set to None, then restore
        original_bus = self.pop_manager.message_bus
        self.pop_manager.message_bus = None

        self.pop_manager.broadcast_ga_update(event_type="test_event")
        mock_create_task.assert_not_called()

        self.pop_manager.message_bus = original_bus # Restore

    @patch('asyncio.create_task')
    def test_broadcast_ga_update_no_connection_manager(self, mock_create_task):
        original_cm = self.mock_message_bus.connection_manager
        self.mock_message_bus.connection_manager = None

        self.pop_manager.broadcast_ga_update(event_type="test_event")
        mock_create_task.assert_not_called()

        self.mock_message_bus.connection_manager = original_cm # Restore


    @patch('asyncio.create_task')
    def test_broadcast_ga_update_payload_structure(self, mock_create_task):
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.pop_manager.generation_number = 5
        self.pop_manager.status = "TESTING_STATUS"
        # is_paused and should_stop are at their defaults (False) from setUp

        # Population is set in setUp

        additional_test_data = {"extra_key": "extra_value"}
        self.pop_manager.broadcast_ga_update(
            event_type="custom_ga_event",
            additional_data=additional_test_data
        )

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args

        broadcasted_message = args[0]
        self.assertEqual(broadcasted_message['type'], 'custom_ga_event')

        data_payload = broadcasted_message['data']
        self.assertEqual(data_payload['status'], "TESTING_STATUS")
        self.assertEqual(data_payload['generation'], 5)
        self.assertEqual(data_payload['population_size'], 2) # From setUp
        self.assertEqual(data_payload['best_fitness'], self.fittest_chromo.fitness_score)
        self.assertFalse(data_payload['is_paused']) # Default from setup
        self.assertFalse(data_payload['should_stop']) # Default from setup
        self.assertEqual(data_payload['extra_key'], "extra_value")
        self.assertIn('timestamp', data_payload) # timestamp is added by broadcast_ga_update in MessageBus
                                                # Correction: broadcast_ga_update in PopulationManager does not add timestamp
                                                # The payload in broadcast_ga_update includes status, gen, pop_size, best_fitness, is_paused, should_stop
                                                # The timestamp is expected in agent_metric_update and new_conversation_log.
                                                # For GA updates, the dashboard JS doesn't use a timestamp from the payload directly.
                                                # Let's remove this timestamp check as it's not added by PopulationManager's broadcast method.
        # self.assertIn('timestamp', data_payload) # Removed as per above comment


    @patch('asyncio.create_task')
    def test_initialize_population_broadcasts(self, mock_create_task):
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.mock_prompt_architect_agent.process_request.return_value = PromptChromosome(genes=["test"])
        # Temporarily empty population for initialize_population to run fully
        self.pop_manager.population = []

        self.pop_manager.initialize_population("task desc")

        self.assertEqual(self.mock_message_bus.connection_manager.broadcast_json.call_count, 2)

        args_started, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args_list[0]
        self.assertEqual(args_started[0]['type'], 'ga_initialization_started')
        self.assertEqual(args_started[0]['data']['status'], 'INITIALIZING')

        args_complete, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args_list[1]
        self.assertEqual(args_complete[0]['type'], 'ga_initialization_complete')
        self.assertEqual(args_complete[0]['data']['status'], 'IDLE')

        # Restore population for other tests if needed, though setUp handles it per test
        self.pop_manager.population = [self.fittest_chromo, self.other_chromo]


    @patch('asyncio.create_task')
    @patch.object(PopulationManager, 'get_fittest_individual')
    def test_evolve_population_broadcasts(self, mock_get_fittest, mock_create_task):
        mock_get_fittest.return_value = self.fittest_chromo
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        self.mock_fitness_evaluator.evaluate = MagicMock(return_value=0.8)

        mock_child_chromo = PromptChromosome(genes=["child"], fitness_score=0.0)
        self.mock_genetic_operators.selection.return_value = self.fittest_chromo
        self.mock_genetic_operators.crossover.return_value = (mock_child_chromo, mock_child_chromo.clone())
        self.mock_genetic_operators.mutate.return_value = mock_child_chromo.clone()

        # Ensure population is set for evolve_population to run
        self.pop_manager.population = [self.fittest_chromo, self.other_chromo]


        self.pop_manager.evolve_population("task desc")

        self.assertGreaterEqual(self.mock_message_bus.connection_manager.broadcast_json.call_count, 3)

        broadcast_types_called = [
            call_args[0][0]['type'] for call_args in self.mock_message_bus.connection_manager.broadcast_json.call_args_list
        ]
        self.assertIn('ga_generation_started', broadcast_types_called)
        self.assertIn('ga_evaluation_complete', broadcast_types_called)
        self.assertIn('ga_generation_complete', broadcast_types_called)


if __name__ == '__main__':
    unittest.main()
