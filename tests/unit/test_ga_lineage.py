import unittest
from unittest.mock import MagicMock, AsyncMock, patch

from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators
from prompthelix.genetics.mutation_strategies import AppendCharStrategy
from prompthelix.services import create_experiment_run

class DummyEvaluator:
    def evaluate(self, chromo, task_description, success_criteria):
        return 0.5

class DummyArchitect:
    def process_request(self, request_data):
        return PromptChromosome(genes=["seed"])

class TestGALineageAndMetrics(unittest.TestCase):
    def setUp(self):
        self.gen_ops = GeneticOperators(mutation_strategies=[AppendCharStrategy()])
        self.dummy_eval = DummyEvaluator()
        self.dummy_architect = DummyArchitect()
        self.mock_message_bus = MagicMock()
        self.mock_message_bus.connection_manager = AsyncMock()
        self.mock_message_bus.connection_manager.broadcast_json = AsyncMock()

        self.manager = PopulationManager(
            genetic_operators=self.gen_ops,
            fitness_evaluator=self.dummy_eval,
            prompt_architect_agent=self.dummy_architect,
            population_size=2,
            elitism_count=0,
            parallel_workers=1,
            message_bus=self.mock_message_bus,
        )
        self.manager.population = [PromptChromosome(["a"]), PromptChromosome(["b"])]

    def test_lineage_attributes_after_crossover_and_mutation(self):
        parent1, parent2 = self.manager.population
        with patch('random.random', return_value=0.0):
            child1, child2 = self.gen_ops.crossover(parent1, parent2, crossover_rate=1.0)
        self.assertEqual(getattr(child1, 'parent_ids', None), [str(parent1.id), str(parent2.id)])
        self.assertEqual(getattr(child2, 'parent_ids', None), [str(parent1.id), str(parent2.id)])

        with patch('random.random', return_value=0.0):
            mutated = self.gen_ops.mutate(child1, mutation_rate=1.0, gene_mutation_prob=1.0)
        self.assertEqual(getattr(mutated, 'parent_ids', None), [str(child1.id)])
        self.assertEqual(getattr(mutated, 'mutation_strategy', None), 'AppendCharStrategy')

    @patch('prompthelix.services.add_generation_metric')
    def test_generation_metric_created(self, mock_metric, db_session):
        run = create_experiment_run(db_session)
        self.manager.evolve_population(
            task_description='t',
            db_session=db_session,
            experiment_run=run,
        )
        self.assertTrue(mock_metric.called)

    def test_websocket_payload_includes_lineage(self):
        self.manager.broadcast_ga_update(
            event_type='ga_generation_complete',
            selected_parent_ids=['id1', 'id2']
        )
        args, _ = self.manager.message_bus.connection_manager.broadcast_json.call_args
        payload = args[0]
        self.assertEqual(payload['data']['selected_parent_ids'], ['id1', 'id2'])

if __name__ == '__main__':
    unittest.main()
