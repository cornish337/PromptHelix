import os
import json
import tempfile
import unittest
from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators, FitnessEvaluator
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.mutation_strategies import NoOperationMutationStrategy

class DummyResultsEvaluator(ResultsEvaluatorAgent):
    def __init__(self, **kwargs):
        super().__init__(settings={'knowledge_file_path': 'dummy.json'}, **kwargs)
    def process_request(self, request_data: dict) -> dict:
        return {'fitness_score': 0.5, 'detailed_metrics': {}, 'llm_analysis_status': 'ok', 'llm_assessment_feedback': 'ok'}

class DummyPromptArchitect(PromptArchitectAgent):
    def __init__(self):
        super().__init__(settings={'knowledge_file_path': 'dummy.json'})
    def process_request(self, request_data: dict) -> PromptChromosome:
        return PromptChromosome(genes=['dummy'])

class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmpdir.name, 'pop.json')
        gen_ops = GeneticOperators(style_optimizer_agent=None, mutation_strategies=[NoOperationMutationStrategy()])
        evaluator = DummyResultsEvaluator()
        self.fitness = FitnessEvaluator(results_evaluator_agent=evaluator, execution_mode=ExecutionMode.TEST)
        self.architect = DummyPromptArchitect()
        self.gen_ops = gen_ops
    def tearDown(self):
        self.tmpdir.cleanup()

    def test_save_and_load_roundtrip(self):
        pm = PopulationManager(self.gen_ops, self.fitness, self.architect, population_size=2)
        pm.population = [PromptChromosome(genes=['a'], fitness_score=0.1), PromptChromosome(genes=['b'], fitness_score=0.2)]
        pm.generation_number = 3
        pm.save_population(self.path)
        self.assertTrue(os.path.exists(self.path))
        pm2 = PopulationManager(self.gen_ops, self.fitness, self.architect, population_size=10)
        pm2.load_population(self.path)
        self.assertEqual(pm2.generation_number, 3)
        self.assertEqual(len(pm2.population), 2)
        self.assertEqual(pm2.population_size, 2)
        self.assertEqual(pm2.population[0].genes, ['a'])

    def test_load_nonexistent_file(self):
        pm = PopulationManager(self.gen_ops, self.fitness, self.architect, population_size=2)
        missing = os.path.join(self.tmpdir.name, 'missing.json')
        pm.load_population(missing)
        self.assertEqual(pm.generation_number, 0)
        self.assertEqual(pm.population, [])

if __name__ == '__main__':
    unittest.main()
