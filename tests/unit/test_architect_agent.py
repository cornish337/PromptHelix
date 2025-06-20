import os
import tempfile
import unittest
from unittest.mock import patch

from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.genetics.engine import PromptChromosome

class TestPromptArchitectAgent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kfile = os.path.join(self.tmpdir.name, "arch.json")
        self.agent = PromptArchitectAgent(knowledge_file_path=self.kfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_creates_file(self):
        self.assertTrue(os.path.exists(self.kfile))
        self.assertTrue(self.agent.templates)

    def test_save_and_load(self):
        self.agent.templates["new"] = {"instruction": "i", "context_placeholder": "c", "output_format": "o"}
        self.agent.save_knowledge()
        other = PromptArchitectAgent(knowledge_file_path=self.kfile)
        self.assertIn("new", other.templates)

    @patch("prompthelix.agents.architect.call_llm_api")
    def test_process_request_success(self, mock_call):
        mock_call.side_effect = [
            "parsed",  # _parse_requirements
            "generic_v1",  # _select_template
            "Gene1\nGene2"  # _populate_genes
        ]
        req = {"task_description": "do something", "keywords": ["x"], "constraints": {}}
        chromo = self.agent.process_request(req)
        self.assertIsInstance(chromo, PromptChromosome)
        self.assertEqual(chromo.genes, ["Gene1", "Gene2"])

    @patch("prompthelix.agents.architect.call_llm_api", side_effect=Exception("fail"))
    def test_process_request_fallback(self, mock_call):
        req = {"task_description": "summarize text", "keywords": ["x"], "constraints": {}}
        chromo = self.agent.process_request(req)
        self.assertIsInstance(chromo, PromptChromosome)
        self.assertTrue(len(chromo.genes) > 0)

if __name__ == "__main__":
    unittest.main()
