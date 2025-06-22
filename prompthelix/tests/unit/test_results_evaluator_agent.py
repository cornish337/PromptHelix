import unittest
from unittest.mock import patch, MagicMock, AsyncMock # Added AsyncMock
import random
import json
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.chromosome import PromptChromosome
import logging
import asyncio # Added asyncio

class TestResultsEvaluatorAgent(unittest.TestCase):
    """Test suite for the ResultsEvaluatorAgent."""

    def setUp(self):
        self.evaluator = ResultsEvaluatorAgent(knowledge_file_path=None)
        self.test_prompt = PromptChromosome(genes=["Test gene"])
        self.task_desc = "Test task description"

        # Mock message_bus and its connection_manager for tests involving process_request
        self.mock_message_bus = MagicMock()
        self.mock_connection_manager = MagicMock()
        self.mock_connection_manager.broadcast_json = AsyncMock() # For broadcast_ga_update if called by MessageBus
        self.mock_message_bus.broadcast_message = AsyncMock() # For direct calls from process_request
        self.evaluator.message_bus = self.mock_message_bus


    def tearDown(self):
        pass

    def test_agent_creation(self):
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.agent_id, "ResultsEvaluator")
        self.assertTrue(self.evaluator.evaluation_metrics_config, "Evaluation metrics config should be loaded.")

    def test_load_metrics_config(self):
        self.assertIsInstance(self.evaluator.evaluation_metrics_config, dict)
        self.assertIn("default_metrics", self.evaluator.evaluation_metrics_config)
        self.assertIn("task_specific", self.evaluator.evaluation_metrics_config)
        self.assertIsInstance(self.evaluator.evaluation_metrics_config["default_metrics"], list)
        self.assertIsInstance(self.evaluator.evaluation_metrics_config["task_specific"], dict)

    def test_agent_creation_with_settings_override(self):
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
        EXPECTED_FALLBACK_LLM_PROVIDER = "openai"
        EXPECTED_FALLBACK_EVAL_MODEL = "gpt-4"
        EXPECTED_FALLBACK_FITNESS_WEIGHTS = {
            "constraint_adherence": 0.5,
            "llm_quality_assessment": 0.5,
        }
        with patch.dict("prompthelix.config.AGENT_SETTINGS", {"ResultsEvaluatorAgent": {}}, clear=True), \
             patch("prompthelix.agents.results_evaluator.FALLBACK_FITNESS_WEIGHTS", EXPECTED_FALLBACK_FITNESS_WEIGHTS):
            import importlib
            import prompthelix.agents.results_evaluator
            importlib.reload(prompthelix.agents.results_evaluator)
            from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent as ReloadedResultsEvaluatorAgent
            evaluator_no_settings = ReloadedResultsEvaluatorAgent(settings=None, knowledge_file_path="specific_kfp.json")

        from prompthelix.config import KNOWLEDGE_DIR
        import os
        self.assertEqual(evaluator_no_settings.llm_provider, EXPECTED_FALLBACK_LLM_PROVIDER)
        self.assertEqual(evaluator_no_settings.evaluation_llm_model, EXPECTED_FALLBACK_EVAL_MODEL)
        self.assertEqual(evaluator_no_settings.fitness_score_weights, EXPECTED_FALLBACK_FITNESS_WEIGHTS)
        expected_kfp = os.path.join(KNOWLEDGE_DIR, "specific_kfp.json")
        self.assertEqual(evaluator_no_settings.knowledge_file_path, expected_kfp)

    async def test_process_request_basic_evaluation(self): # Changed to async
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": "This is a good output based on the general task.",
            "task_description": "General task for basic evaluation"
        }
        result = await self.evaluator.process_request(request_data) # await

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

    async def test_process_request_constraints_met(self): # Changed to async
        success_criteria = {"max_length": 100, "must_include_keywords": ["test"]}
        llm_output = "This output includes the test keyword and is short."
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": llm_output,
            "task_description": "Constraint test - met",
            "success_criteria": success_criteria
        }
        mock_llm_response = {
            "relevance_score": 0.9, "coherence_score": 0.8, "completeness_score": 0.7,
            "accuracy_assessment": "Looks good.", "safety_score": 1.0,
            "overall_quality_score": 0.85, "feedback_text": "Excellent work."
        }
        with patch('prompthelix.agents.results_evaluator.call_llm_api', return_value=json.dumps(mock_llm_response)) as mock_llm_call:
            result = await self.evaluator.process_request(request_data) # await

        self.assertEqual(len(result["error_analysis"]), 0, f"Error analysis should be empty if constraints are met. Got: {result['error_analysis']}")
        self.assertEqual(result["detailed_metrics"].get("constraint_adherence_placeholder"), 1.0)
        self.assertGreater(result["fitness_score"], 0.5, "Fitness score should be relatively high when constraints are met.")

    async def test_process_request_constraints_violated(self): # Changed to async
        success_criteria = {"max_length": 20, "must_include_keywords": ["required"], "min_length": 5}
        llm_output = "This output is too long and misses the keyword."
        request_data = {
            "prompt_chromosome": self.test_prompt,
            "llm_output": llm_output,
            "task_description": "Constraint violation test",
            "success_criteria": success_criteria
        }
        # Mock call_llm_api to prevent real API calls for this specific test of constraints
        mock_llm_response = {
            "relevance_score": 0.5, "coherence_score": 0.5, "completeness_score": 0.5,
            "accuracy_assessment": "N/A", "safety_score": 1.0,
            "overall_quality_score": 0.5, "feedback_text": "Constraint issues noted."
        }
        with patch('prompthelix.agents.results_evaluator.call_llm_api', return_value=json.dumps(mock_llm_response)):
            result = await self.evaluator.process_request(request_data) # await

        self.assertTrue(len(result["error_analysis"]) > 0, "Error analysis should not be empty for violated constraints.")
        self.assertTrue(any("exceeds max_length" in err for err in result["error_analysis"]))
        self.assertTrue(any("missing required keywords" in err for err in result["error_analysis"]))
        self.assertAlmostEqual(result["detailed_metrics"].get("constraint_adherence_placeholder"), 1/3, places=2)
        self.assertLess(result["fitness_score"], 0.61, "Fitness score should be relatively low when constraints are violated.")

    async def test_process_request_invalid_input(self): # Changed to async
        request_data = {
            "prompt_chromosome": "not a chromosome object",
            "llm_output": "Some output.",
            "task_description": "Invalid input test"
        }
        result = await self.evaluator.process_request(request_data) # await

        self.assertEqual(result["fitness_score"], 0.0)
        self.assertTrue(len(result["error_analysis"]) > 0)
        self.assertIn("Error: Invalid prompt_chromosome.", result["error_analysis"])

        request_data_none = {
            "prompt_chromosome": None,
            "llm_output": "Some output.",
            "task_description": "Invalid input test with None"
        }
        result_none = await self.evaluator.process_request(request_data_none) # await
        self.assertEqual(result_none["fitness_score"], 0.0)
        self.assertTrue(len(result_none["error_analysis"]) > 0)
        self.assertIn("Error: Invalid prompt_chromosome.", result_none["error_analysis"])

    @patch('prompthelix.agents.results_evaluator.logger.warning')
    def test_get_fallback_llm_metrics(self, mock_logger_warning):
        errors = ["Test error 1", "Test error 2"]
        fallback_metrics = self.evaluator._get_fallback_llm_metrics(errors=errors)
        mock_logger_warning.assert_called_once()
        logged_message = mock_logger_warning.call_args[0][0]
        self.assertIn(f"Agent '{self.evaluator.agent_id}': Using fallback LLM metrics. Reason: Test error 1; Test error 2", logged_message)
        self.assertIsInstance(fallback_metrics, dict)
        self.assertEqual(fallback_metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertEqual(fallback_metrics['llm_assessed_relevance'], 0.0)
        self.assertIn("Fallback: Test error 1; Test error 2", fallback_metrics['llm_assessment_feedback'])

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_api_error(self, mock_call_llm_api):
        mock_call_llm_api.return_value = "RATE_LIMIT_ERROR"
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_log_warning:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt
            )
        self.assertEqual(mock_log_warning.call_count, 2)
        log_call_1_args = mock_log_warning.call_args_list[0][0]
        self.assertIn(f"Agent '{self.evaluator.agent_id}': LLM call for content analysis failed with error code: RATE_LIMIT_ERROR", log_call_1_args[0])
        log_call_2_args = mock_log_warning.call_args_list[1][0]
        self.assertIn(f"Agent '{self.evaluator.agent_id}': Using fallback LLM metrics. Reason: LLM API Error: RATE_LIMIT_ERROR", log_call_2_args[0])
        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertIn("LLM API Error: RATE_LIMIT_ERROR", metrics['llm_assessment_feedback'])
        self.assertEqual(metrics['llm_assessed_quality'], 0.0)
        self.assertIn("LLM API Error: RATE_LIMIT_ERROR", errors)

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_unparseable_response(self, mock_call_llm_api):
        mock_call_llm_api.return_value = "This is not valid JSON {{{{ and not an API error."
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt
            )
        logged_messages = [args[0][0] for args in mock_logger_warning.call_args_list]
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in log_msg for log_msg in logged_messages))
        self.assertTrue(any("Using fallback LLM metrics." in log_msg for log_msg in logged_messages))
        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertIn("LLM content analysis response was not in expected JSON format.", metrics['llm_assessment_feedback'])
        self.assertEqual(metrics['llm_assessed_quality'], 0.0)
        self.assertTrue(any("LLM content analysis response was not in expected JSON format." in e for e in errors))

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_json_decode_error(self, mock_call_llm_api):
        mock_call_llm_api.return_value = "{'relevance_score': 0.8, 'coherence_score': 'not_a_float_actually_string}"
        with patch('prompthelix.agents.results_evaluator.logger.error') as mock_log_error, \
             patch('prompthelix.agents.results_evaluator.logger.warning') as mock_log_warning:
            metrics, errors = self.evaluator._analyze_content(
                llm_output="Some output",
                task_desc=self.task_desc,
                prompt_chromosome=self.test_prompt
            )
        found_error_log = False
        for call_args in mock_log_error.call_args_list:
            if "Error parsing LLM evaluation response (JSONDecodeError)" in call_args[0][0]:
                found_error_log = True
                break
        self.assertTrue(found_error_log, "Expected ERROR log for JSONDecodeError not found.")
        found_warning_log = False
        for call_args in mock_log_warning.call_args_list:
            if "Using fallback LLM metrics." in call_args[0][0] and "JSON decoding failed" in call_args[0][0]:
                found_warning_log = True
                break
        self.assertTrue(found_warning_log, "Expected WARNING log for using fallback metrics due to JSON error not found.")
        self.assertEqual(metrics['llm_analysis_status'], 'fallback_due_to_error')
        self.assertTrue(any("JSON decoding failed" in e for e in errors))
        self.assertIn("JSON decoding failed", metrics['llm_assessment_feedback'])

    @patch('prompthelix.agents.results_evaluator.call_llm_api')
    def test_analyze_content_successful_response(self, mock_call_llm_api):
        mock_response_dict = {
            "relevance_score": 0.9, "coherence_score": 0.8, "completeness_score": 0.7,
            "accuracy_assessment": "Looks good.", "safety_score": 1.0,
            "overall_quality_score": 0.85, "feedback_text": "Excellent work."
        }
        mock_call_llm_api.return_value = f"Some text before json... {json.dumps(mock_response_dict)} ... and some after."
        logging.disable(logging.NOTSET)
        self.evaluator.logger.setLevel(logging.INFO)
        metrics, errors = self.evaluator._analyze_content(
            llm_output="Some output",
            task_desc=self.task_desc,
            prompt_chromosome=self.test_prompt
        )
        logging.disable(logging.CRITICAL)
        self.assertEqual(metrics['llm_analysis_status'], 'success')
        self.assertEqual(metrics['llm_assessed_relevance'], 0.9)
        self.assertEqual(len(errors), 0)

    def test_evaluate_prompt_success(self):
        mock_metric_one = MagicMock(return_value=0.5)
        mock_metric_two = MagicMock(return_value=1.0)
        settings = {"metric_weights": {"metric_one": 0.6, "metric_two": 0.4}}
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}
        prompt_text = "Test prompt for evaluation."
        output_text = "Test output for evaluation."
        result = agent.evaluate_prompt(prompt=prompt_text, output=output_text)
        expected_score = (0.5 * 0.6 + 1.0 * 0.4) / (0.6 + 0.4)
        self.assertAlmostEqual(result["score"], expected_score)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})
        mock_metric_one.assert_called_once_with(output_text, prompt_text)
        mock_metric_two.assert_called_once_with(output_text, prompt_text)

    def test_evaluate_prompt_zero_total_weight(self):
        mock_metric_one = MagicMock(return_value=0.8)
        agent1 = ResultsEvaluatorAgent(settings={"metric_weights": {}})
        agent1.metric_functions = {"metric_one": mock_metric_one}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning1:
            result1 = agent1.evaluate_prompt("prompt", "output")
        self.assertEqual(result1["score"], 0.0)
        self.assertTrue(any("No metric weights defined." in call_args[0][0] for call_args in mock_logger_warning1.call_args_list))
        self.assertTrue(any("Total weight for metrics is zero." in call_args[0][0] for call_args in mock_logger_warning1.call_args_list))
        agent2 = ResultsEvaluatorAgent(settings={"metric_weights": {"metric_one": 0.0}})
        agent2.metric_functions = {"metric_one": mock_metric_one}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning2:
            result2 = agent2.evaluate_prompt("prompt", "output")
        self.assertEqual(result2["score"], 0.0)
        self.assertTrue(any("Total weight for metrics is zero." in call_args[0][0] for call_args in mock_logger_warning2.call_args_list))

    def test_evaluate_prompt_missing_weights_for_some_metrics(self):
        mock_metric_one = MagicMock(return_value=0.5)
        mock_metric_two = MagicMock(return_value=1.0)
        settings = {"metric_weights": {"metric_one": 1.0}}
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}
        result = agent.evaluate_prompt("prompt", "output")
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})

    def test_evaluate_prompt_no_metric_functions_loaded(self):
        agent = ResultsEvaluatorAgent(settings={})
        agent.metric_functions = {}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            result = agent.evaluate_prompt("prompt", "output")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["details"], {"error": "No metric functions loaded."})
        mock_logger_warning.assert_called_once()
        self.assertIn("No metric functions loaded for evaluate_prompt.", mock_logger_warning.call_args[0][0])

    def test_evaluate_prompt_metric_function_error(self):
        mock_metric_ok = MagicMock(return_value=0.8)
        mock_metric_bad = MagicMock(side_effect=Exception("Calculation failed!"))
        settings = {"metric_weights": {"metric_ok": 0.5, "metric_bad": 0.5}}
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_ok": mock_metric_ok, "metric_bad": mock_metric_bad}
        with patch('prompthelix.agents.results_evaluator.logger.error') as mock_logger_error:
            result = agent.evaluate_prompt("prompt", "output")
        self.assertAlmostEqual(result["score"], 0.4)
        self.assertEqual(result["details"], {"metric_ok": 0.8, "metric_bad": 0.0})
        mock_logger_error.assert_called_once()
        self.assertIn("Error calculating metric 'metric_bad': Calculation failed!", mock_logger_error.call_args[0][0])

    def test_evaluate_prompt_negative_weights(self):
        mock_metric_one = MagicMock(return_value=0.5)
        mock_metric_two = MagicMock(return_value=1.0)
        settings = {"metric_weights": {"metric_one": 0.8, "metric_two": -0.2}}
        agent = ResultsEvaluatorAgent(settings=settings)
        agent.metric_functions = {"metric_one": mock_metric_one, "metric_two": mock_metric_two}
        with patch('prompthelix.agents.results_evaluator.logger.warning') as mock_logger_warning:
            result = agent.evaluate_prompt("prompt", "output")
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["details"], {"metric_one": 0.5, "metric_two": 1.0})
        mock_logger_warning.assert_called_once()
        self.assertIn("Negative weight -0.2 for metric 'metric_two' encountered. Using 0 instead.", mock_logger_warning.call_args[0][0])

if __name__ == '__main__':
    unittest.main()
