import os
import tempfile
import unittest
from prompthelix.genetics.engine import PopulationManager, PromptChromosome, GeneticOperators, FitnessEvaluator
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.mutation_strategies import NoOperationMutationStrategy
from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner

class DummyResultsEvaluator(ResultsEvaluatorAgent):
    def __init__(self, **kwargs):
        super().__init__(settings={'knowledge_file_path': 'dummy.json'}, **kwargs)
    def process_request(self, request_data: dict) -> dict:
        return {'fitness_score': 1.0, 'detailed_metrics': {}, 'llm_analysis_status': 'ok', 'llm_assessment_feedback': 'ok'}

class DummyPromptArchitect(PromptArchitectAgent):
    def __init__(self):
        super().__init__(settings={'knowledge_file_path': 'dummy.json'})
    def process_request(self, request_data: dict) -> PromptChromosome:
        return PromptChromosome(genes=['dummy'])

class GARunnerPersistenceIntegration(unittest.TestCase):
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

    def test_runner_saves_and_loads_population(self):
        pm = PopulationManager(self.gen_ops, self.fitness, self.architect, population_size=1, population_path=self.path)
        pm.population = [PromptChromosome(genes=['start'], fitness_score=0.5)]
        pm.generation_number = 0
        pm.status = "RUNNING"
        pm.should_stop = False

        def fake_evolve(*args, **kwargs):
            pm.generation_number += 1
            if pm.generation_number >= 2:
                pm.should_stop = True
        pm.evolve_population = fake_evolve

        runner = GeneticAlgorithmRunner(pm, num_generations=2, save_frequency=1)
        runner.run(task_description="test")

        self.assertTrue(os.path.exists(self.path))

        pm_loaded = PopulationManager(self.gen_ops, self.fitness, self.architect, population_size=1)
        pm_loaded.load_population(self.path)
        self.assertEqual(pm_loaded.generation_number, 2)
        self.assertEqual(len(pm_loaded.population), 1)

if __name__ == '__main__':
    unittest.main()
