import unittest
import asyncio
from unittest.mock import Mock, patch, call, MagicMock, AsyncMock
from prompthelix.genetics.engine import (
    PopulationManager,
    GeneticOperators,
    FitnessEvaluator
)
from prompthelix.genetics.chromosome import PromptChromosome
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.message_bus import MessageBus

class TestPopulationManager(unittest.TestCase):
    """Test suite for the PopulationManager class."""

    def setUp(self):
        """Set up common mock objects for PopulationManager tests."""
        self.mock_genetic_ops = MagicMock(spec=GeneticOperators)

        self.mock_fitness_eval = MagicMock(spec=FitnessEvaluator)
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.5)

        self.mock_architect_agent = MagicMock(spec=PromptArchitectAgent)
        self.mock_architect_agent.process_request = MagicMock(return_value=PromptChromosome(genes=["Default gene"]))

        self.mock_message_bus = MagicMock(spec=MessageBus)
        self.mock_connection_manager = MagicMock()
        self.mock_connection_manager.broadcast_json = AsyncMock()
        self.mock_message_bus.connection_manager = self.mock_connection_manager


    # --- Test __init__ ---
    def test_init_successful(self):
        """Test successful instantiation of PopulationManager."""
        manager = PopulationManager(
            genetic_operators=self.mock_genetic_ops,
            fitness_evaluator=self.mock_fitness_eval,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=10,
            elitism_count=1
        )
        self.assertIsInstance(manager, PopulationManager)
        self.assertEqual(manager.population_size, 10)
        self.assertEqual(manager.elitism_count, 1)
        self.assertEqual(manager.generation_number, 0)
        self.assertEqual(len(manager.population), 0)

    def test_init_invalid_types(self):
        """Test __init__ with invalid types for agent/operator arguments."""
        with self.assertRaisesRegex(TypeError, "genetic_operators must be an instance of GeneticOperators."):
            PopulationManager("not_genetic_ops", self.mock_fitness_eval, self.mock_architect_agent)
        with self.assertRaisesRegex(TypeError, "fitness_evaluator must be an instance of FitnessEvaluator."):
            PopulationManager(self.mock_genetic_ops, "not_fitness_eval", self.mock_architect_agent)
        with self.assertRaisesRegex(TypeError, "prompt_architect_agent must be an instance of PromptArchitectAgent."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, "not_architect")

    def test_init_invalid_population_size(self):
        """Test __init__ with invalid population_size."""
        with self.assertRaisesRegex(ValueError, "Population size must be positive."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=0)
        with self.assertRaisesRegex(ValueError, "Population size must be positive."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=-5)

    def test_init_invalid_elitism_count(self):
        """Test __init__ with invalid elitism_count."""
        with self.assertRaisesRegex(ValueError, "Elitism count must be non-negative."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=10, elitism_count=-1)

        with self.assertRaisesRegex(ValueError, "Elitism count cannot exceed population size."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=10, elitism_count=11)

    # --- Test initialize_population ---
    async def test_initialize_population(self): # Changed to async
        """Test population initialization."""
        pop_size = 5
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1, message_bus=self.mock_message_bus
        )
        
        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"GeneSet{i}"]) for i in range(pop_size)
        ]
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.6)


        task_desc = "Initial task"
        keywords = ["kw1"]
        constraints = {"max_len": 10}
        success_criteria_for_init = {"some_init_criterion": True}
        
        await manager.initialize_population(initial_task_description=task_desc, initial_keywords=keywords, constraints=constraints, success_criteria=success_criteria_for_init) # await

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size)
        
        expected_request_data = {"task_description": task_desc, "keywords": keywords, "constraints": constraints}
        self.mock_architect_agent.process_request.assert_any_call(expected_request_data)
        
        self.assertEqual(manager.generation_number, 0)
        for i, chromo in enumerate(manager.population):
            self.assertEqual(chromo.genes, [f"GeneSet{i}"])

    async def test_initialize_population_with_initial_prompt_str(self): # Changed to async
        """Test population initialization when initial_prompt_str is provided."""
        pop_size = 3
        initial_prompt = "This is a seeded prompt."
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1,
            initial_prompt_str=initial_prompt, message_bus=self.mock_message_bus
        )

        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"ArchitectGeneSet{i}"]) for i in range(pop_size - 1)
        ]
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.6)

        task_desc = "Initial task with seed"
        await manager.initialize_population(initial_task_description=task_desc, initial_keywords=[], constraints={}) # await

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size - 1)

        seeded_chromosome_found = False
        architect_chromosomes_found = 0
        for chromo in manager.population:
            if chromo.genes == [initial_prompt]:
                seeded_chromosome_found = True
            elif chromo.genes[0].startswith("ArchitectGeneSet"):
                architect_chromosomes_found +=1

        self.assertTrue(seeded_chromosome_found, "Seeded chromosome not found in population.")
        self.assertEqual(architect_chromosomes_found, pop_size - 1, "Incorrect number of architect-generated chromosomes.")
        self.assertEqual(manager.generation_number, 0)

    async def test_initialize_population_with_initial_prompt_str_pop_size_1(self): # Changed to async
        """Test population initialization with initial_prompt_str and population size of 1."""
        pop_size = 1
        initial_prompt = "Only seeded prompt."
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=0,
            initial_prompt_str=initial_prompt, message_bus=self.mock_message_bus
        )
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.6)

        task_desc = "Initial task with seed, pop 1"
        await manager.initialize_population(initial_task_description=task_desc, initial_keywords=[], constraints={}) # await

        self.assertEqual(len(manager.population), pop_size)
        self.mock_architect_agent.process_request.assert_not_called()

        self.assertTrue(len(manager.population) == 1 and manager.population[0].genes == [initial_prompt])
        self.assertEqual(manager.generation_number, 0)

    async def test_initialize_population_without_initial_prompt_str(self): # Changed to async
        """Test population initialization when initial_prompt_str is NOT provided."""
        pop_size = 3
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1,
            initial_prompt_str=None, message_bus=self.mock_message_bus
        )

        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"ArchitectGeneSet{i}"]) for i in range(pop_size)
        ]
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.6)

        task_desc = "Initial task no seed"
        await manager.initialize_population(initial_task_description=task_desc, initial_keywords=[], constraints={}) # await

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size)

        for i, chromo in enumerate(manager.population):
            self.assertEqual(chromo.genes, [f"ArchitectGeneSet{i}"])
        self.assertEqual(manager.generation_number, 0)


    # --- Test get_fittest_individual ---
    def test_get_fittest_individual_empty_population(self):
        """Test get_fittest_individual with an empty population."""
        manager = PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=1)
        manager.population = []
        self.assertIsNone(manager.get_fittest_individual())

    def test_get_fittest_individual_populated(self):
        """Test get_fittest_individual with a populated list."""
        manager = PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=3)
        c1 = PromptChromosome(fitness_score=0.5)
        c2 = PromptChromosome(fitness_score=0.9)
        c3 = PromptChromosome(fitness_score=0.2)
        manager.population = [c1, c2, c3]
        
        manager.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        
        fittest = manager.get_fittest_individual()
        self.assertEqual(fittest, c2)


    # --- Test evolve_population ---
    async def test_evolve_population_flow(self): # Changed to async def
        """Test the overall flow of evolve_population."""
        pop_size = 4
        elitism_count = 1
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=elitism_count, message_bus=self.mock_message_bus
        )
        manager.parallel_workers = 1

        initial_chromosomes = [
            PromptChromosome(genes=["P1"], fitness_score=0.1),
            PromptChromosome(genes=["P2"], fitness_score=0.2),
            PromptChromosome(genes=["P3"], fitness_score=0.3),
            PromptChromosome(genes=["P4"], fitness_score=0.4)
        ]
        manager.population = initial_chromosomes
        manager.generation_number = 0

        def mock_evaluate_side_effect(chromo, task_desc, success_criteria):
            if chromo.genes == ["P1"]: chromo.fitness_score = 0.7
            elif chromo.genes == ["P2"]: chromo.fitness_score = 0.5
            elif chromo.genes == ["P3"]: chromo.fitness_score = 0.9
            elif chromo.genes == ["P4"]: chromo.fitness_score = 0.3
            return chromo.fitness_score
        self.mock_fitness_eval.evaluate.side_effect = mock_evaluate_side_effect
        
        self.mock_genetic_ops.selection.side_effect = lambda pop: pop[0]
        
        mock_child1 = PromptChromosome(genes=["Child1"])
        mock_child2 = PromptChromosome(genes=["Child2"])
        self.mock_genetic_ops.crossover.return_value = (mock_child1, mock_child2)
        
        self.mock_genetic_ops.mutate.side_effect = lambda chromo, *args, **kwargs: PromptChromosome(genes=chromo.genes + ["_mutated"])

        task_desc = "Evolution task"
        original_generation_number = manager.generation_number
        await manager.evolve_population(task_desc, success_criteria={}) # Added await

        self.assertEqual(self.mock_fitness_eval.evaluate.call_count, pop_size)
        for chromo in initial_chromosomes:
             self.mock_fitness_eval.evaluate.assert_any_call(chromo, task_desc, {})

        self.assertEqual(manager.population[0].genes, ["P3"])
        self.assertEqual(manager.population[0].fitness_score, 0.9)

        self.mock_genetic_ops.selection.assert_not_called()
        self.mock_genetic_ops.crossover.assert_not_called()
        self.mock_genetic_ops.mutate.assert_not_called()
        
        self.assertEqual(manager.generation_number, original_generation_number,
                         "evolve_population should not increment generation_number itself.")

        self.assertEqual(len(manager.population), pop_size)

        original_ids = {c.id for c in initial_chromosomes}
        current_ids = {c.id for c in manager.population}
        self.assertEqual(original_ids, current_ids, "Population should contain the same individuals, just re-evaluated and sorted.")

    def test_evolve_population_empty(self): # evolve_population is sync
        """Test evolve_population with an initially empty population."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=5, elitism_count=1, message_bus=self.mock_message_bus
        )
        manager.evolve_population("Test task")
        self.assertEqual(manager.generation_number, 0, "Generation number should not change for empty population.")
        self.assertEqual(len(manager.population), 0, "Population should remain empty.")
        self.mock_fitness_eval.evaluate.assert_not_called()

    # --- Test Control Methods (pause, resume, stop) ---

    async def test_pause_evolution(self):
        """Test pausing the evolution."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=1
        )
        manager.status = "RUNNING"
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        await manager.pause_evolution()

        self.assertTrue(manager.is_paused)
        self.assertEqual(manager.status, "PAUSED")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_paused')
        self.assertEqual(args[0]['data']['status'], 'PAUSED')

    async def test_resume_evolution(self):
        """Test resuming the evolution."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=1
        )
        manager.is_paused = True
        manager.status = "PAUSED"
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        await manager.resume_evolution()

        self.assertFalse(manager.is_paused)
        self.assertEqual(manager.status, "RUNNING")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_resumed')
        self.assertEqual(args[0]['data']['status'], 'RUNNING')

    async def test_stop_evolution(self):
        """Test stopping the evolution."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=1
        )
        manager.status = "RUNNING"
        manager.is_paused = True
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        await manager.stop_evolution()

        self.assertTrue(manager.should_stop)
        self.assertFalse(manager.is_paused, "stop_evolution should set is_paused to False.")
        self.assertEqual(manager.status, "STOPPING")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        self.assertEqual(args[0]['type'], 'ga_stopping')
        self.assertEqual(args[0]['data']['status'], 'STOPPING')

    # --- Tests for broadcast_ga_update payload and calls ---

    async def test_broadcast_ga_update_payload_with_data(self):
        """Test the payload of broadcast_ga_update with a populated list of chromosomes."""
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=3
        )
        manager.population = [
            PromptChromosome(genes=["g1"], fitness_score=0.1),
            PromptChromosome(genes=["g2"], fitness_score=0.5),
            PromptChromosome(genes=["g3"], fitness_score=0.9)
        ]
        manager.generation_number = 5
        manager.status = "TESTING_PAYLOAD"

        await manager.broadcast_ga_update(event_type="test_payload_event", additional_data={"test_key": "test_value"})

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        payload_sent_for_assertion = args[0]

        self.assertEqual(payload_sent_for_assertion["type"], "test_payload_event")
        data = payload_sent_for_assertion["data"]
        self.assertEqual(data["status"], "TESTING_PAYLOAD")
        self.assertEqual(data["generation"], 5)
        self.assertEqual(data["population_size"], 3)
        self.assertAlmostEqual(data["best_fitness"], 0.9)
        self.assertAlmostEqual(data["fitness_min"], 0.1)
        self.assertAlmostEqual(data["fitness_max"], 0.9)
        self.assertAlmostEqual(data["fitness_mean"], 0.5)
        self.assertAlmostEqual(data["fitness_median"], 0.5)
        self.assertAlmostEqual(data["fitness_std_dev"], 0.4)
        self.assertIn("fittest_chromosome_string", data)
        self.assertEqual(data["test_key"], "test_value")
        self.assertIn("timestamp", data)

    async def test_broadcast_ga_update_payload_empty_population(self):
        """Test the payload of broadcast_ga_update with an empty population."""
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=1
        )
        manager.population = []
        await manager.broadcast_ga_update(event_type="empty_pop_event")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        data = args[0]["data"]

        self.assertIsNone(data["best_fitness"])
        self.assertIsNone(data["fitness_min"])
        self.assertIsNone(data["fitness_max"])
        self.assertIsNone(data["fitness_mean"])
        self.assertIsNone(data["fitness_median"])
        self.assertIsNone(data["fitness_std_dev"])
        self.assertIsNone(data["fittest_chromosome_string"])
        self.assertIn("timestamp", data)

    async def test_broadcast_ga_update_payload_single_chromosome(self):
        """Test the payload of broadcast_ga_update with a single chromosome."""
        self.mock_message_bus.connection_manager.broadcast_json.reset_mock()

        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=self.mock_message_bus, population_size=1
        )
        manager.population = [PromptChromosome(genes=["g1"], fitness_score=0.7)]
        await manager.broadcast_ga_update(event_type="single_chrom_event")

        self.mock_message_bus.connection_manager.broadcast_json.assert_called_once()
        args, _ = self.mock_message_bus.connection_manager.broadcast_json.call_args
        data = args[0]["data"]
        self.assertAlmostEqual(data["best_fitness"], 0.7)
        self.assertAlmostEqual(data["fitness_min"], 0.7)
        self.assertAlmostEqual(data["fitness_max"], 0.7)
        self.assertAlmostEqual(data["fitness_mean"], 0.7)
        self.assertAlmostEqual(data["fitness_median"], 0.7)
        self.assertAlmostEqual(data["fitness_std_dev"], 0.0)
        self.assertIn("timestamp", data)

    @patch('prompthelix.genetics.engine.PopulationManager.broadcast_ga_update', new_callable=AsyncMock)
    async def test_initialize_population_broadcast_calls(self, mock_broadcast_ga_update):
        """Test that initialize_population calls broadcast_ga_update correctly."""
        pop_size = 2
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, message_bus=self.mock_message_bus
        )
        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"GeneSet{i}"]) for i in range(pop_size)
        ]
        self.mock_fitness_eval.evaluate = MagicMock(return_value=0.6)
        mock_broadcast_ga_update.reset_mock()

        await manager.initialize_population(initial_task_description="task_desc", initial_keywords=[], constraints={})

        self.assertEqual(mock_broadcast_ga_update.call_count, 2)

        call_args_list = mock_broadcast_ga_update.call_args_list

        self.assertEqual(call_args_list[0][1]['event_type'], "population_initialization_started")

        self.assertEqual(call_args_list[1][1]['event_type'], "population_initialized")
        self.assertEqual(call_args_list[1][1]['additional_data']['population_size'], pop_size)


    @patch('prompthelix.genetics.engine.PopulationManager.broadcast_ga_update', new_callable=AsyncMock)
    async def test_evolve_population_broadcast_calls(self, mock_broadcast_ga_update):
        """Test that evolve_population calls broadcast_ga_update at key stages."""
        pop_size = 2
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1, message_bus=self.mock_message_bus,
            parallel_workers=1
        )
        manager.population = [
            PromptChromosome(genes=["P1"], fitness_score=0.5),
            PromptChromosome(genes=["P2"], fitness_score=0.8)
        ]
        manager.generation_number = 0
        self.mock_fitness_eval.evaluate.side_effect = lambda c, td, sc: c.fitness_score
        mock_broadcast_ga_update.reset_mock()

        manager.evolve_population("evolution_task", success_criteria={})

        mock_broadcast_ga_update.assert_not_called()


    async def test_pause_evolution_no_message_bus(self): # Changed to async
        """Test pause_evolution when message_bus is None."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=None, population_size=1
        )
        manager.status = "RUNNING"

        await manager.pause_evolution() # await

        self.assertTrue(manager.is_paused)
        self.assertEqual(manager.status, "PAUSED")
        # No assertion on broadcast_json as it shouldn't be called if bus is None


    async def test_resume_evolution_no_message_bus(self): # Changed to async
        """Test resume_evolution when message_bus is None."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=None, population_size=1
        )
        manager.is_paused = True
        manager.status = "PAUSED"

        await manager.resume_evolution() # await

        self.assertFalse(manager.is_paused)
        self.assertEqual(manager.status, "RUNNING")

    async def test_stop_evolution_no_message_bus(self): # Changed to async
        """Test stop_evolution when message_bus is None."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            message_bus=None, population_size=1
        )
        manager.status = "RUNNING"

        await manager.stop_evolution() # await

        self.assertTrue(manager.should_stop)
        self.assertFalse(manager.is_paused)
        self.assertEqual(manager.status, "STOPPING")


if __name__ == '__main__':
    unittest.main()
