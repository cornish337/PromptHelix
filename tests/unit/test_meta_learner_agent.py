import os
import tempfile
import unittest

from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.genetics.engine import PromptChromosome

class TestMetaLearnerAgent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kfile = os.path.join(self.tmpdir.name, "meta.json")
        self.agent = MetaLearnerAgent(knowledge_file_path=self.kfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_creates_file(self):
        self.assertTrue(os.path.exists(self.kfile))
        self.assertTrue(self.agent.knowledge_base)

    def test_save_and_load(self):
        self.agent.knowledge_base["legacy_successful_patterns"].append("p")
        self.agent.save_knowledge()
        other = MetaLearnerAgent(knowledge_file_path=self.kfile)
        self.assertIn("p", other.knowledge_base["legacy_successful_patterns"])

    def test_process_request_success(self):
        data = {"data_type": "evaluation_result", "data": {"prompt_chromosome": PromptChromosome(genes=["g1"]), "fitness_score": 0.8}}
        result = self.agent.process_request(data)
        self.assertEqual(result["status"], "Data processed successfully.")
        self.assertIsInstance(result["recommendations"], list)

    def test_process_request_missing(self):
        result = self.agent.process_request({})
        self.assertEqual(result["status"], "Error: Missing data_type or data.")

if __name__ == "__main__":
    unittest.main()
