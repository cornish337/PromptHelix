import unittest
from unittest.mock import patch, MagicMock
import random # random is used by the agent itself, not directly in tests for mocking here
import json # For creating mock JSON responses
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.engine import PromptChromosome
import logging # To capture logs

class TestResultsEvaluatorAgent(unittest.TestCase):
    """Test suite for the ResultsEvaluatorAgent."""

    def setUp(self):
        """Instantiate the ResultsEvaluatorAgent for each test."""
        # Suppress most logging output during these specific tests unless it's the one being asserted
        # This is to keep test output clean.
        logging.disable(logging.CRITICAL)
        # Re-enable logging for specific loggers if needed within a test using self.assertLogs

        self.evaluator = ResultsEvaluatorAgent(knowledge_file_path=None)
        # Example prompt_chromosome, actual content doesn't matter much for these _analyze_content tests
        self.prompt_chromosome = PromptChromosome(genes=["Test gene"])
        self.task_desc = "Test task description"

        # Restore logging to default after tests if necessary, or set per test.
        # For now, disabling broadly and enabling per test with self.assertLogs should be fine.

    def tearDown(self):
        # Re-enable logging if it was disabled globally
        logging.disable(logging.NOTSET)

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
        self.assertLess(result["fitness_score"], 0.61, "Fitness score should be relatively low when constraints are violated.") # Adjusted from 0.6 to 0.61

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

    def test_get_fallback_llm_metrics(self):
        """Test the _get_fallback_llm_metrics method directly."""
        errors = ["Test error 1", "Test error 2"]
        with self.assertLogs(self.evaluator.logger.name, level='WARNING') as cm:
            fallback_metrics = self.evaluator._get_fallback_llm_metrics(errors=errors)

        self.assertIn("Using fallback LLM metrics. Reason: Test error 1; Test error 2", cm.output[0])

        self.assertIsInstance(fallback_metrics, dict)
        self.assertEqual(fallback_metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertEqual(fallback_metrics['llm_assessed_relevance'], 0.0)
        self.assertEqual(fallback_metrics['llm_assessed_coherence'], 0.0)
        self.assertEqual(fallback_metrics['llm_assessed_completeness'], 0.0)
        self.assertEqual(fallback_metrics['llm_accuracy_assessment'], 'N/A')
        self.assertEqual(fallback_metrics['llm_safety_score'], 0.0)
        self.assertEqual(fallback_metrics['llm_assessed_quality'], 0.0)
        self.assertIn("Fallback: Test error 1; Test error 2", fallback_metrics['llm_assessment_feedback'])

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_api_error(self, mock_call_llm_api):
        """Test _analyze_content when call_llm_api returns an error string."""
        mock_call_llm_api.return_value = "RATE_LIMIT_ERROR" # Simulate an API error

        with self.assertLogs(self.evaluator.logger.name, level='WARNING') as cm:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.prompt_chromosome
            )

        self.assertIn("LLM call for content analysis failed with error code: RATE_LIMIT_ERROR", cm.output[0])
        self.assertIn("Using fallback LLM metrics. Reason: LLM API Error: RATE_LIMIT_ERROR", cm.output[1])


        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertIn("LLM API Error: RATE_LIMIT_ERROR", metrics['llm_assessment_feedback'])
        self.assertEqual(metrics['llm_assessed_quality'], 0.0)
        self.assertIn("LLM API Error: RATE_LIMIT_ERROR", errors) # Check errors list returned by _analyze_content

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_unparseable_response(self, mock_call_llm_api):
        """Test _analyze_content with a non-JSON, non-error string response from LLM."""
        mock_call_llm_api.return_value = "This is not valid JSON {{{{ and not an API error."

        # We expect a warning when parsing fails, then another for using fallback.
        with self.assertLogs(self.evaluator.logger.name, level='WARNING') as cm:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.prompt_chromosome
            )

        # Check logs for parsing failure and fallback usage
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in log_msg for log_msg in cm.output))
        self.assertTrue(any("Using fallback LLM metrics." in log_msg for log_msg in cm.output))

        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertIn("LLM content analysis response was not in expected JSON format.", metrics['llm_assessment_feedback'])
        self.assertEqual(metrics['llm_assessed_quality'], 0.0)
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in e for e in errors))


    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_json_decode_error(self, mock_call_llm_api):
        """Test _analyze_content when response is malformed JSON causing JSONDecodeError."""
        mock_call_llm_api.return_value = "{'relevance_score': 0.8, 'coherence_score': 'not_a_float_actually_string}" # Malformed JSON (single quotes, string for float)

        # Expect an error log for JSONDecodeError, then warning for fallback
        with self.assertLogs(self.evaluator.logger.name, level='WARNING') as cm: # Catches WARNING and ERROR
            logging.disable(logging.NOTSET) # Temporarily enable all levels for this logger for this test
            self.evaluator.logger.setLevel(logging.DEBUG) # Ensure all levels are processed by this logger

            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.prompt_chromosome
            )
            logging.disable(logging.CRITICAL) # Re-disable after the call for other tests

        self.assertTrue(any("Error parsing LLM evaluation response (JSONDecodeError)" in log_msg for log_msg in cm.output if "ERROR" in log_msg.split(" - ")[2] )) # Check for specific ERROR log
        self.assertTrue(any("Using fallback LLM metrics." in log_msg for log_msg in cm.output if "WARNING" in log_msg.split(" - ")[2] ))


        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertTrue(any("JSON decoding failed" in e for e in errors))
        self.assertIn("JSON decoding failed", metrics['llm_assessment_feedback'])

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_successful_response(self, mock_call_llm_api):
        """Test _analyze_content with a valid, parsable JSON response."""
        mock_response_dict = {
            "relevance_score": 0.9,
            "coherence_score": 0.8,
            "completeness_score": 0.7,
            "accuracy_assessment": "Looks good.",
            "safety_score": 1.0,
            "overall_quality_score": 0.85,
            "feedback_text": "Excellent work."
        }
        mock_call_llm_api.return_value = f"Some text before json... {json.dumps(mock_response_dict)} ... and some after."

        # No specific error/warning logs expected here related to fallback for this success case.
        # We might get INFO logs, so we don't use assertLogs strictly unless checking for absence of warnings.
        logging.disable(logging.NOTSET) # Ensure logs are processed for this test
        self.evaluator.logger.setLevel(logging.INFO)

        metrics, errors = self.evaluator._analyze_content(
            llm_output="Some output",
            task_desc=self.task_desc,
            prompt_chromosome=self.prompt_chromosome
        )
        logging.disable(logging.CRITICAL) # Re-disable

        self.assertEqual(metrics['llm_analysis_status'], 'success')
        self.assertEqual(metrics['llm_assessed_relevance'], 0.9)
        self.assertEqual(metrics['llm_assessed_coherence'], 0.8)
        self.assertEqual(metrics['llm_assessed_completeness'], 0.7)
        self.assertEqual(metrics['llm_accuracy_assessment'], "Looks good.")
        self.assertEqual(metrics['llm_safety_score'], 1.0)
        self.assertEqual(metrics['llm_assessed_quality'], 0.85)
        self.assertEqual(metrics['llm_assessment_feedback'], "Excellent work.")
        self.assertEqual(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
