import unittest
import random # random is used by the agent itself, not directly in tests for mocking here
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.engine import PromptChromosome

class TestResultsEvaluatorAgent(unittest.TestCase):
    """Test suite for the ResultsEvaluatorAgent."""

    def setUp(self):
        """Instantiate the ResultsEvaluatorAgent for each test."""
        self.evaluator = ResultsEvaluatorAgent()
        self.test_prompt = PromptChromosome(genes=["Instruction: Test prompt"])

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.agent_id, "ResultsEvaluator")
        self.assertTrue(self.evaluator.evaluation_metrics_config, "Evaluation metrics config should be loaded.")

    def test_load_metrics_config(self):
        """Test if evaluation metrics configurations are loaded correctly."""
        self.assertIsInstance(self.evaluator.evaluation_metrics_config, dict)
        self.assertIn("default_metrics", self.evaluator.evaluation_metrics_config)
        self.assertIn("task_specific", self.evaluator.evaluation_metrics_config)
        self.assertIsInstance(self.evaluator.evaluation_metrics_config["default_metrics"], list)
        self.assertIsInstance(self.evaluator.evaluation_metrics_config["task_specific"], dict)

    def test_process_request_basic_evaluation(self):
        """Test process_request with a basic evaluation scenario."""
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": "This is a good output based on the general task.",
            "task_description": "General task for basic evaluation"
        }
        result = self.evaluator.process_request(request_data)

        self.assertIsInstance(result, dict)
        self.assertIn("fitness_score", result)
        self.assertIsInstance(result["fitness_score"], float)
        self.assertGreaterEqual(result["fitness_score"], 0.0)
        self.assertLessEqual(result["fitness_score"], 1.0)
        
        self.assertIn("detailed_metrics", result)
        self.assertIsInstance(result["detailed_metrics"], dict)
        self.assertIn("relevance_placeholder", result["detailed_metrics"])
        self.assertIn("coherence_placeholder", result["detailed_metrics"])
        
        self.assertIn("error_analysis", result)
        self.assertIsInstance(result["error_analysis"], list)

    def test_process_request_constraints_met(self):
        """Test process_request where all specified constraints are met."""
        success_criteria = {"max_length": 100, "must_include_keywords": ["test"]}
        llm_output = "This output includes the test keyword and is short."
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": llm_output,
            "task_description": "Constraint test - met",
            "success_criteria": success_criteria
        }
        result = self.evaluator.process_request(request_data)

        self.assertEqual(len(result["error_analysis"]), 0, f"Error analysis should be empty if constraints are met. Got: {result['error_analysis']}")
        self.assertEqual(result["detailed_metrics"].get("constraint_adherence_placeholder"), 1.0)
        # Fitness score should be relatively high, exact value depends on other random metrics
        # but constraint adherence portion should contribute positively.
        # A perfect constraint score (1.0 * 0.25 weight) + base (0.3) + other randoms.
        # A rough check:
        self.assertGreater(result["fitness_score"], 0.5, "Fitness score should be relatively high when constraints are met.")

    def test_process_request_constraints_violated(self):
        """Test process_request where specified constraints are violated."""
        success_criteria = {"max_length": 20, "must_include_keywords": ["required"], "min_length": 5}
        llm_output = "This output is too long and misses the keyword." # Length 49, misses "required"
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": llm_output,
            "task_description": "Constraint violation test",
            "success_criteria": success_criteria
        }
        result = self.evaluator.process_request(request_data)

        self.assertTrue(len(result["error_analysis"]) > 0, "Error analysis should not be empty for violated constraints.")
        # Expected errors: max_length, must_include_keywords
        self.assertTrue(any("exceeds max_length" in err for err in result["error_analysis"]))
        self.assertTrue(any("missing required keywords" in err for err in result["error_analysis"]))
        
        # With 2 out of 3 constraints violated (max_length, must_include_keywords; min_length is met)
        # constraint_adherence_placeholder = 1/3 = 0.33
        self.assertAlmostEqual(result["detailed_metrics"].get("constraint_adherence_placeholder"), 1/3, places=2)
        
        # Fitness score should be relatively low.
        # (0.33 * 0.25 weight) + base (0.3) + other randoms - 2 errors * 0.1
        # Max possible for constraint part is 0.0825. Max for others is ~0.35. So ~0.73 - 0.2 = ~0.53
        # This is a rough check, as other metrics are random.
        self.assertLess(result["fitness_score"], 0.6, "Fitness score should be relatively low when constraints are violated.")

    def test_process_request_invalid_input(self):
        """Test process_request with invalid 'prompt_chromosome' input."""
        request_data = {
            "prompt_chromosome": "not a chromosome object",
            "llm_output": "Some output.",
            "task_description": "Invalid input test"
        }
        result = self.evaluator.process_request(request_data)

        self.assertEqual(result["fitness_score"], 0.0)
        self.assertTrue(len(result["error_analysis"]) > 0)
        self.assertIn("Error: Invalid prompt_chromosome.", result["error_analysis"])

        request_data_none = {
            "prompt_chromosome": None,
            "llm_output": "Some output.",
            "task_description": "Invalid input test with None"
        }
        result_none = self.evaluator.process_request(request_data_none)
        self.assertEqual(result_none["fitness_score"], 0.0)
        self.assertTrue(len(result_none["error_analysis"]) > 0)
        self.assertIn("Error: Invalid prompt_chromosome.", result_none["error_analysis"])


if __name__ == '__main__':
    unittest.main()
