import os
import tempfile
import unittest
from unittest.mock import patch

from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.engine import PromptChromosome

class TestResultsEvaluatorAgent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kfile = os.path.join(self.tmpdir.name, "eval.json")
        self.agent = ResultsEvaluatorAgent(knowledge_file_path=self.kfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_creates_file(self):
        self.assertTrue(os.path.exists(self.kfile))
        self.assertTrue(self.agent.evaluation_metrics_config)

    def test_save_and_load(self):
        self.agent.evaluation_metrics_config["new"] = ["a"]
        self.agent.save_knowledge()
        other = ResultsEvaluatorAgent(knowledge_file_path=self.kfile)
        self.assertIn("new", other.evaluation_metrics_config)

    def test_process_request_success(self):
        chromo = PromptChromosome(genes=["g1"])
        result = self.agent.process_request({"prompt_chromosome": chromo, "llm_output": "output", "task_description": "task"})
        self.assertEqual(result["detailed_metrics"].get("llm_analysis_status"), "fallback_due_to_error")

    def test_process_request_fallback(self):
        with patch("prompthelix.agents.results_evaluator.call_llm_api", return_value="UNSUPPORTED_PROVIDER_ERROR"):
            chromo = PromptChromosome(genes=["g1"])
            result = self.agent.process_request({"prompt_chromosome": chromo, "llm_output": "output", "task_description": "task"})
        self.assertEqual(result["detailed_metrics"].get("llm_analysis_status"), "fallback_due_to_error")

if __name__ == "__main__":
    unittest.main()
