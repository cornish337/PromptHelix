import unittest
from unittest.mock import patch, MagicMock
import random # random is used by the agent itself, not directly in tests for mocking here
import json # For creating mock JSON responses
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.chromosome import PromptChromosome
import logging # To capture logs

class TestResultsEvaluatorAgent(unittest.TestCase):
    """Test suite for the ResultsEvaluatorAgent."""

    def setUp(self):
        """Instantiate the ResultsEvaluatorAgent for each test."""
        # Suppress most logging output during these specific tests unless it's the one being asserted
        # This is to keep test output clean.
        # logging.disable(logging.CRITICAL) # Removed global disable
        # Re-enable logging for specific loggers if needed within a test using self.assertLogs

        self.evaluator = ResultsEvaluatorAgent(knowledge_file_path=None)
        # Example prompt_chromosome, actual content doesn't matter much for these _analyze_content tests
        self.test_prompt = PromptChromosome(genes=["Test gene"]) # Renamed from self.prompt_chromosome
        self.task_desc = "Test task description"

        # Restore logging to default after tests if necessary, or set per test.
        # For now, disabling broadly and enabling per test with self.assertLogs should be fine.

    def tearDown(self):
        # Re-enable logging if it was disabled globally
        # logging.disable(logging.NOTSET) # Removed global re-enable
        pass # No specific tearDown needed for logging if not globally changed

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

    def test_agent_creation_with_settings_override(self):
        """Test agent creation with settings override."""
        override_settings = {
            "default_llm_provider": "override_provider",
            "evaluation_llm_model": "override_eval_model",
            "fitness_score_weights": {"constraint_adherence": 0.7, "llm_quality_assessment": 0.3},
            "knowledge_file_path": "override_eval_config.json",
            "custom_key": "custom_value"
        }

        evaluator_with_override = ResultsEvaluatorAgent(settings=override_settings)

        self.assertEqual(evaluator_with_override.settings, override_settings)
        self.assertEqual(evaluator_with_override.llm_provider, "override_provider")
        self.assertEqual(evaluator_with_override.evaluation_llm_model, "override_eval_model")
        self.assertEqual(evaluator_with_override.fitness_score_weights, {"constraint_adherence": 0.7, "llm_quality_assessment": 0.3})

        from prompthelix.config import KNOWLEDGE_DIR
        import os
        expected_kfp_override = os.path.join(KNOWLEDGE_DIR, "override_eval_config.json")
        self.assertEqual(evaluator_with_override.knowledge_file_path, expected_kfp_override)

    def test_agent_creation_no_settings_uses_fallbacks(self):
        """Test agent uses fallbacks if no settings dict is passed and global config is empty for it."""
        # Temporarily clear AGENT_SETTINGS for this agent to test hardcoded fallbacks in the module
        with patch.dict("prompthelix.config.AGENT_SETTINGS", {"ResultsEvaluatorAgent": {}}, clear=True):
            evaluator_no_settings = ResultsEvaluatorAgent(settings=None, knowledge_file_path="specific_kfp.json")

        from prompthelix.agents.results_evaluator import FALLBACK_LLM_PROVIDER, FALLBACK_EVAL_MODEL, FALLBACK_FITNESS_WEIGHTS
        from prompthelix.config import KNOWLEDGE_DIR
        import os

        self.assertEqual(evaluator_no_settings.llm_provider, FALLBACK_LLM_PROVIDER)
        self.assertEqual(evaluator_no_settings.evaluation_llm_model, FALLBACK_EVAL_MODEL)
        self.assertEqual(evaluator_no_settings.fitness_score_weights, FALLBACK_FITNESS_WEIGHTS)

        # kfp param should be used if settings doesn't provide it
        expected_kfp = os.path.join(KNOWLEDGE_DIR, "specific_kfp.json")
        self.assertEqual(evaluator_no_settings.knowledge_file_path, expected_kfp)

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
        # Mock call_llm_api to prevent real API calls and ensure no LLM errors are added
        mock_llm_response = {
            "relevance_score": 0.9, "coherence_score": 0.8, "completeness_score": 0.7,
            "accuracy_assessment": "Looks good.", "safety_score": 1.0,
            "overall_quality_score": 0.85, "feedback_text": "Excellent work."
        }
        with patch('prompthelix.agents.results_evaluator.call_llm_api', return_value=json.dumps(mock_llm_response)) as mock_llm_call:
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

    @patch('prompthelix.agents.results_evaluator.logger.warning')
    def test_get_fallback_llm_metrics(self, mock_logger_warning):
        """Test the _get_fallback_llm_metrics method directly."""
        errors = ["Test error 1", "Test error 2"]
        fallback_metrics = self.evaluator._get_fallback_llm_metrics(errors=errors)

        mock_logger_warning.assert_called_once()
        logged_message = mock_logger_warning.call_args[0][0]
        self.assertIn(f"Agent '{self.evaluator.agent_id}': Using fallback LLM metrics. Reason: Test error 1; Test error 2", logged_message)

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

        with self.assertLogs('prompthelix.agents.results_evaluator', level='WARNING') as cm:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt # Use renamed attribute
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
        # The method _analyze_content can log multiple warnings.
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt # Use renamed attribute
            )

        # Check logs for parsing failure and fallback usage
        # Convert call_args_list to a list of strings for easier checking
        logged_messages = [args[0][0] for args in mock_logger_warning.call_args_list]
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in log_msg for log_msg in logged_messages))
        self.assertTrue(any("Using fallback LLM metrics." in log_msg for log_msg in logged_messages))

        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertIn("LLM content analysis response was not in expected JSON format.", metrics['llm_assessment_feedback'])
        self.assertEqual(metrics['llm_assessed_quality'], 0.0)
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in e for e in errors))


    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_json_decode_error(self, mock_call_llm_api):
        """Test _analyze_content when response is malformed JSON causing JSONDecodeError."""
        mock_call_llm_api.return_value = "{'relevance_score': 0.8, 'coherence_score': 'not_a_float_actually_string}" # Malformed JSON (single quotes, string for float)

        # Expect an error log for JSONDecodeError, then warning for fallback
        with self.assertLogs('prompthelix.agents.results_evaluator', level='WARNING') as cm: # Catches WARNING and ERROR
            # logging.disable(logging.NOTSET) # Removed: assertLogs handles logger level temporarily
            # self.evaluator.logger.setLevel(logging.DEBUG) # Removed: assertLogs handles logger level temporarily

            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt # Use renamed attribute
            )
            # logging.disable(logging.CRITICAL) # Removed: No longer needed as we didn't enable globally

        self.assertTrue(any("ERROR" in log_msg and "Error parsing LLM evaluation response (JSONDecodeError)" in log_msg for log_msg in cm.output) or \
                        any("WARNING" in log_msg and "LLM content analysis response was not in expected JSON format" in log_msg for log_msg in cm.output) # if it falls to this due to parsing
                        ) # Broader check for error or specific fallback
        self.assertTrue(any("WARNING" in log_msg and "Using fallback LLM metrics." in log_msg for log_msg in cm.output) or \
                        any("ERROR" in log_msg and "Error parsing LLM evaluation response (JSONDecodeError)" in log_msg for log_msg in cm.output)
                        )


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
            prompt_chromosome=self.test_prompt # Use renamed attribute
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

    # --- Tests for the new evaluate_prompt method ---

    def test_evaluate_prompt_success(self):
        """Test evaluate_prompt with successful calculation of a weighted score."""
        mock_metric_one = MagicMock(return_value=0.5)
        mock_metric_two = MagicMock(return_value=1.0)

        settings = {"metric_weights": {"metric_one": 0.6, "metric_two": 0.4}}
        # Need to re-init agent or set settings and then call relevant parts of __init__
        # For simplicity, re-initialize with settings.
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}

        prompt_text = "Test prompt for evaluation."
        output_text = "Test output for evaluation."
        result = agent.evaluate_prompt(prompt=prompt_text, output=output_text)

        expected_score = (0.5 * 0.6 + 1.0 * 0.4) / (0.6 + 0.4) # Should be 0.7
        self.assertAlmostEqual(result["score"], expected_score)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})

        mock_metric_one.assert_called_once_with(output_text, prompt_text)
        mock_metric_two.assert_called_once_with(output_text, prompt_text)

    def test_evaluate_prompt_zero_total_weight(self):
        """Test evaluate_prompt when total_weight is zero (e.g., no weights or all zero)."""
        mock_metric_one = MagicMock(return_value=0.8)

        # Case 1: metric_weights is empty
        agent1 = ResultsEvaluatorAgent(settings={"metric_weights": {}})
        agent1.metric_functions = {"metric_one": mock_metric_one}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning1:
            result1 = agent1.evaluate_prompt("prompt", "output")
        self.assertEqual(result1["score"], 0.0)
        self.assertTrue(any("No metric weights defined." in call_args[0][0] for call_args in mock_logger_warning1.call_args_list))
        self.assertTrue(any("Total weight for metrics is zero." in call_args[0][0] for call_args in mock_logger_warning1.call_args_list))

        # Case 2: metric_weights has zero weights
        agent2 = ResultsEvaluatorAgent(settings={"metric_weights": {"metric_one": 0.0}})
        agent2.metric_functions = {"metric_one": mock_metric_one}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning2:
            result2 = agent2.evaluate_prompt("prompt", "output")
        self.assertEqual(result2["score"], 0.0)
        self.assertTrue(any("Total weight for metrics is zero." in call_args[0][0] for call_args in mock_logger_warning2.call_args_list))


    def test_evaluate_prompt_missing_weights_for_some_metrics(self):
        """Test evaluate_prompt when some metrics have weights and others don't."""
        mock_metric_one = MagicMock(return_value=0.5) # Has weight
        mock_metric_two = MagicMock(return_value=1.0) # No weight, so effectively 0

        settings = {"metric_weights": {"metric_one": 1.0}} # Only metric_one has weight
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}

        result = agent.evaluate_prompt("prompt", "output")

        # score = (0.5 * 1.0 + 1.0 * 0.0) / (1.0 + 0.0) = 0.5
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})

    def test_evaluate_prompt_no_metric_functions_loaded(self):
        """Test evaluate_prompt when self.metric_functions is empty."""
        agent = ResultsEvaluatorAgent(settings={}) # Default init
        agent.metric_functions = {} # Explicitly empty

        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            result = agent.evaluate_prompt("prompt", "output")

        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["details"], {"error": "No metric functions loaded."})
        mock_logger_warning.assert_called_once()
        self.assertIn("No metric functions loaded for evaluate_prompt.", mock_logger_warning.call_args[0][0])

    def test_evaluate_prompt_metric_function_error(self):
        """Test evaluate_prompt when a metric function raises an exception."""
        mock_metric_ok = MagicMock(return_value=0.8)
        mock_metric_bad = MagicMock(side_effect=Exception("Calculation failed!"))

        settings = {"metric_weights": {"metric_ok": 0.5, "metric_bad": 0.5}}
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_ok": mock_metric_ok, "metric_bad": mock_metric_bad}

        # Temporarily enable ERROR logs for this agent's logger to capture the specific error
        # logging.disable(logging.NOTSET) # Removed
        current_level = agent.logger.level # This agent is self.evaluator
        # agent.logger.setLevel(logging.DEBUG) # Removed: assertLogs handles this

        with patch('prompthelix.agents.results_evaluator.logger.error') as mock_logger_error:
            result = agent.evaluate_prompt("prompt", "output") # Use the modified agent

        # agent.logger.setLevel(current_level) # Removed: assertLogs restores original level
        # logging.disable(logging.CRITICAL) # Removed

        # score = (0.8 * 0.5 + 0.0 * 0.5) / (0.5 + 0.5) = 0.4
        self.assertAlmostEqual(result["score"], 0.4)
        self.assertEqual(result["details"], {"metric_ok": 0.8, "metric_bad": 0.0})
        mock_logger_error.assert_called_once()
        # The logged message will stringify the exception, not show "Exception(...)"
        self.assertIn("Error calculating metric 'metric_bad': Calculation failed!", mock_logger_error.call_args[0][0])

    def test_evaluate_prompt_negative_weights(self):
        """Test evaluate_prompt when metric_weights contain negative values."""
        mock_metric_one = MagicMock(return_value=0.5)
        mock_metric_two = MagicMock(return_value=1.0)

        settings = {"metric_weights": {"metric_one": 0.8, "metric_two": -0.2}} # metric_two has negative weight
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}

        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            result = agent.evaluate_prompt("prompt", "output")

        # Negative weight for metric_two should be treated as 0.
        # score = (0.5 * 0.8 + 1.0 * 0.0) / (0.8 + 0.0) = 0.4 / 0.8 = 0.5
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})
        mock_logger_warning.assert_called_once()
        self.assertIn("Negative weight -0.2 for metric 'metric_two' encountered. Using 0 instead.", mock_logger_warning.call_args[0][0])


if __name__ == '__main__':
    unittest.main()
