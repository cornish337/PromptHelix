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

    @patch("prompthelix.agents.results_evaluator.call_llm_api")
    def test_process_request_success(self, mock_call):
        mock_call.return_value = '{"relevance_score":0.8,"coherence_score":0.9,"completeness_score":0.7,"accuracy_assessment":"ok","safety_score":1.0,"overall_quality_score":0.9,"feedback_text":"good"}'
        chromo = PromptChromosome(genes=["g1"])
        result = self.agent.process_request({"prompt_chromosome": chromo, "llm_output": "output", "task_description": "task"})
        self.assertIsInstance(result["fitness_score"], float)
        self.assertEqual(result["detailed_metrics"].get("llm_analysis_status"), "success")

    @patch("prompthelix.agents.results_evaluator.call_llm_api", side_effect=Exception("fail"))
    def test_process_request_fallback(self, mock_call):
        chromo = PromptChromosome(genes=["g1"])
        result = self.agent.process_request({"prompt_chromosome": chromo, "llm_output": "output", "task_description": "task"})
        self.assertEqual(result["detailed_metrics"].get("llm_analysis_status"), "fallback_due_to_error")

if __name__ == "__main__":
    unittest.main()
