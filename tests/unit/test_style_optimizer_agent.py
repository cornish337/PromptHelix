import os
import tempfile
import unittest
from unittest.mock import patch

from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.genetics.engine import PromptChromosome

class TestStyleOptimizerAgent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kfile = os.path.join(self.tmpdir.name, "style.json")
        self.agent = StyleOptimizerAgent(knowledge_file_path=self.kfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_creates_file(self):
        self.assertTrue(os.path.exists(self.kfile))
        self.assertTrue(self.agent.style_rules)

    def test_save_and_load(self):
        self.agent.style_rules["test"] = {"replace": {"a": "b"}}
        self.agent.save_knowledge()
        other = StyleOptimizerAgent(knowledge_file_path=self.kfile)
        self.assertIn("test", other.style_rules)

    def test_process_request_success(self):
        original = PromptChromosome(genes=["g1", "g2"])
        result = self.agent.process_request({"prompt_chromosome": original, "target_style": "formal"})
        self.assertIsInstance(result, PromptChromosome)
        self.assertTrue(len(result.genes) > 0)

    def test_process_request_fallback(self):
        with patch("prompthelix.agents.style_optimizer.call_llm_api", side_effect=Exception("fail")):
            original = PromptChromosome(genes=["don't do stuff", "Context"])
            result = self.agent.process_request({"prompt_chromosome": original, "target_style": "formal"})
        self.assertIsInstance(result, PromptChromosome)
        self.assertTrue(len(result.genes) > 0)

if __name__ == "__main__":
    unittest.main()
