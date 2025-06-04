# prompthelix/tests/unit/test_evaluation.py
import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open

# Modules to be tested
from prompthelix.evaluation.evaluator import Evaluator
from prompthelix.evaluation import metrics as evaluation_metrics
from prompthelix.config import settings # To check for API key for optional tests

# Helper to create a dummy JSON file for testing data loading
@pytest.fixture
def temp_eval_data_file(tmp_path):
    data = [
        {"prompt": "Prompt 1", "expected_output": "Expected 1"},
        {"prompt": "Prompt 2", "expected_output": "Expected 2"},
        {"prompt": "No expected output here", "expected_output": None}
    ]
    file_path = tmp_path / "test_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def temp_malformed_eval_data_file(tmp_path):
    file_path = tmp_path / "malformed_data.json"
    with open(file_path, 'w') as f:
        f.write("This is not JSON")
    return file_path

@pytest.fixture
def temp_incomplete_eval_data_file(tmp_path):
    data = [{"prompt": "Only prompt here"}] # Missing expected_output
    file_path = tmp_path / "incomplete_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

@pytest.fixture
def temp_wrong_structure_eval_data_file(tmp_path):
    data = ["item1", "item2"] # List of strings, not dicts
    file_path = tmp_path / "wrong_structure_data.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path


# --- Test Metric Functions ---
class TestEvaluationMetrics:
    def test_calculate_exact_match(self):
        assert evaluation_metrics.calculate_exact_match("hello", "hello") == 1.0
        assert evaluation_metrics.calculate_exact_match("hello", "world") == 0.0
        assert evaluation_metrics.calculate_exact_match("", "") == 1.0
        assert evaluation_metrics.calculate_exact_match("hello", None) == 0.0
        assert evaluation_metrics.calculate_exact_match(None, "world") == 0.0
        assert evaluation_metrics.calculate_exact_match(None, None) == 0.0 # Both None -> 0.0

    def test_calculate_keyword_overlap(self):
        assert evaluation_metrics.calculate_keyword_overlap("cat dog", "dog fish") == pytest.approx(0.333333) # {dog} / {cat, dog, fish}
        assert evaluation_metrics.calculate_keyword_overlap("apple", "apple") == 1.0
        assert evaluation_metrics.calculate_keyword_overlap("apple pie", "banana cake") == 0.0
        assert evaluation_metrics.calculate_keyword_overlap("", "") == 1.0 # Both empty, perfect match
        assert evaluation_metrics.calculate_keyword_overlap("text", "") == 0.0 # Expected is empty
        assert evaluation_metrics.calculate_keyword_overlap("", "text") == 0.0 # Generated is empty
        assert evaluation_metrics.calculate_keyword_overlap(None, "text") == 0.0
        assert evaluation_metrics.calculate_keyword_overlap("text", None) == 0.0
        assert evaluation_metrics.calculate_keyword_overlap(None, None) == 0.0

        keywords = ["apple", "banana", "cherry"]
        assert evaluation_metrics.calculate_keyword_overlap("apple is sweet", "banana is fruit", keywords=keywords) == pytest.approx(0.333333) # {apple}, {banana} -> {apple} int {banana} = 0 / {apple, banana} -> should be 0.0 as intersection is empty
        # Corrected: apple in first, banana in second. Intersection is empty. Union is {apple, banana}. 0/2=0.
        # Let's re-evaluate the keyword logic:
        # generated_words = {k for k in keywords if k in generated_output}
        # expected_words = {k for k in keywords if k in expected_output}
        # For "apple is sweet", gen_words = {"apple"}
        # For "banana is fruit", exp_words = {"banana"}
        # Intersection = {} -> 0. Union = {"apple", "banana"} -> 2. Result = 0.0
        assert evaluation_metrics.calculate_keyword_overlap("apple is sweet", "banana is fruit", keywords=keywords) == 0.0
        assert evaluation_metrics.calculate_keyword_overlap("apple and banana", "apple only", keywords=keywords) == pytest.approx(0.5) # gen={"apple", "banana"}, exp={"apple"} -> int={"apple"}(1), union={"apple", "banana"}(2) => 0.5

    def test_calculate_output_length(self):
        assert evaluation_metrics.calculate_output_length("hello") == 5
        assert evaluation_metrics.calculate_output_length("") == 0
        assert evaluation_metrics.calculate_output_length(None) == 0
        # expected_output is not used by this metric, so no need to test variations of it

    def test_calculate_bleu_score_placeholder(self):
        # This is a placeholder, so tests are based on its simplified logic
        assert evaluation_metrics.calculate_bleu_score("the cat sat", "the cat sat") == 1.0
        assert evaluation_metrics.calculate_bleu_score("the cat sat", "a cat sat") == pytest.approx(0.666666) # cat, sat common / a, cat, sat
        assert evaluation_metrics.calculate_bleu_score("a b c", "d e f") == 0.0
        assert evaluation_metrics.calculate_bleu_score("", "") == 1.0 # Both empty
        assert evaluation_metrics.calculate_bleu_score("text", "") == 0.0 # Expected empty
        assert evaluation_metrics.calculate_bleu_score("", "text") == 0.0 # Generated empty
        assert evaluation_metrics.calculate_bleu_score(None, "text") == 0.0
        assert evaluation_metrics.calculate_bleu_score("text", None) == 0.0
        assert evaluation_metrics.calculate_bleu_score(None, None) == 0.0


# --- Test Evaluator Class ---
class TestEvaluator:
    def test_evaluator_initialization_default_metrics(self):
        evaluator = Evaluator()
        assert len(evaluator.metric_functions) > 0 # Has some default metrics
        assert evaluator.openai_client is not None if settings.OPENAI_API_KEY else evaluator.openai_client is None

    def test_evaluator_initialization_custom_metrics(self):
        def mock_metric(o, e): return 1.0
        evaluator = Evaluator(metric_functions=[mock_metric])
        assert len(evaluator.metric_functions) == 1
        assert evaluator.metric_functions[0] == mock_metric

    def test_add_metric(self):
        evaluator = Evaluator(metric_functions=[])
        def mock_metric_1(o, e): return 1.0
        def mock_metric_2(o, e): return 0.5

        evaluator.add_metric(mock_metric_1)
        assert len(evaluator.metric_functions) == 1
        evaluator.add_metric(mock_metric_2)
        assert len(evaluator.metric_functions) == 2
        evaluator.add_metric(mock_metric_1) # Adding same metric again
        assert len(evaluator.metric_functions) == 2 # Should not add duplicates

    def test_add_metric_invalid(self):
        evaluator = Evaluator(metric_functions=[])
        with pytest.raises(ValueError, match="Metric must be a callable function."):
            evaluator.add_metric("not_a_function")

    def test_load_evaluation_data_success(self, temp_eval_data_file):
        evaluator = Evaluator()
        evaluator.load_evaluation_data(str(temp_eval_data_file))
        assert len(evaluator.evaluation_data) == 3
        assert evaluator.evaluation_data[0]["prompt"] == "Prompt 1"
        assert evaluator.evaluation_data[2]["expected_output"] is None


    def test_load_evaluation_data_file_not_found(self):
        evaluator = Evaluator()
        with pytest.raises(FileNotFoundError):
            evaluator.load_evaluation_data("non_existent_file.json")

    def test_load_evaluation_data_malformed_json(self, temp_malformed_eval_data_file):
        evaluator = Evaluator()
        with pytest.raises(json.JSONDecodeError):
            evaluator.load_evaluation_data(str(temp_malformed_eval_data_file))

    def test_load_evaluation_data_incomplete_keys(self, temp_incomplete_eval_data_file):
        evaluator = Evaluator()
        # This should load but might log errors or fail at run_evaluation depending on strictness
        # The current implementation raises ValueError during load if keys are missing.
        with pytest.raises(ValueError, match="Data items must contain 'prompt' and 'expected_output' keys."):
            evaluator.load_evaluation_data(str(temp_incomplete_eval_data_file))

    def test_load_evaluation_data_wrong_structure(self, temp_wrong_structure_eval_data_file):
        evaluator = Evaluator()
        with pytest.raises(ValueError, match="Data items should be dictionaries."):
            evaluator.load_evaluation_data(str(temp_wrong_structure_eval_data_file))


    @patch('prompthelix.evaluation.evaluator.openai.OpenAI')
    def test_run_evaluation_success(self, mock_openai_class, temp_eval_data_file):
        # Mock the OpenAI client and its response
        mock_client_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = "Mocked LLM Output"
        mock_client_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client_instance

        # Dummy metric for simplicity in this test
        mock_metric_func = MagicMock(return_value=0.75)
        mock_metric_func.__name__ = "mock_metric"

        evaluator = Evaluator(metric_functions=[mock_metric_func])
        evaluator.load_evaluation_data(str(temp_eval_data_file))

        results = evaluator.run_evaluation()

        assert len(results) == 3 # 3 items in dummy data
        assert "item_0" in results
        assert results["item_0"]["actual_output"] == "Mocked LLM Output"
        assert results["item_0"]["scores"]["mock_metric"] == 0.75

        # Check if _call_llm_api was called for each item
        assert mock_client_instance.chat.completions.create.call_count == 3
        mock_client_instance.chat.completions.create.assert_any_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Prompt 1"}]
        )

    @patch('prompthelix.evaluation.evaluator.openai.OpenAI')
    def test_run_evaluation_llm_call_error(self, mock_openai_class, temp_eval_data_file):
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="Error: LLM API call failed. Details: some error"))]) # Simulate error message format
        # Or simulate an exception being raised by the LLM call
        # mock_client_instance.chat.completions.create.side_effect = OpenAIError("Simulated API Error")
        # For this test, let's assume the _call_llm_api returns the error string
        mock_client_instance.chat.completions.create.return_value.choices[0].message.content = "Error: LLM API call failed. Details: Test Error"

        mock_openai_class.return_value = mock_client_instance

        mock_metric_func = MagicMock(return_value=0.0) # Metric will receive the error string
        mock_metric_func.__name__ = "mock_metric_on_error"

        evaluator = Evaluator(metric_functions=[mock_metric_func])
        evaluator.load_evaluation_data(str(temp_eval_data_file))
        results = evaluator.run_evaluation()

        assert results["item_0"]["actual_output"].startswith("Error: LLM API call failed.")
        assert "mock_metric_on_error" in results["item_0"]["scores"]
        # The metric would have processed the error string as actual_output
        mock_metric_func.assert_called_with("Error: LLM API call failed. Details: Test Error", "Expected 1")


    @patch('prompthelix.evaluation.evaluator.openai.OpenAI')
    def test_run_evaluation_metric_calculation_error(self, mock_openai_class, temp_eval_data_file):
        mock_client_instance = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = "Valid LLM Output"
        mock_client_instance.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_client_instance

        error_metric = MagicMock(side_effect=Exception("Metric Error!"))
        error_metric.__name__ = "error_metric"

        evaluator = Evaluator(metric_functions=[error_metric])
        evaluator.load_evaluation_data(str(temp_eval_data_file))
        results = evaluator.run_evaluation()

        assert results["item_0"]["scores"]["error_metric"] is None
        assert "Metric error_metric calculation error: Metric Error!" in results["item_0"]["errors"]


    def test_run_evaluation_no_data_loaded(self):
        evaluator = Evaluator()
        with pytest.raises(ValueError, match="Evaluation data not loaded."):
            evaluator.run_evaluation()

    def test_run_evaluation_no_metrics_defined(self, temp_eval_data_file):
        evaluator = Evaluator(metric_functions=[])
        evaluator.load_evaluation_data(str(temp_eval_data_file))
        with pytest.raises(ValueError, match="No metric functions defined for evaluation."):
            evaluator.run_evaluation()

    @patch('prompthelix.evaluation.evaluator.openai.OpenAI')
    def test_evaluator_init_no_api_key(self, mock_openai_class_constructor):
        # Temporarily mock settings to simulate no API key
        with patch('prompthelix.evaluation.evaluator.settings.OPENAI_API_KEY', None):
            evaluator = Evaluator()
            assert evaluator.openai_client is None
            # Ensure constructor for OpenAI client was not called
            mock_openai_class_constructor.assert_not_called()

    @patch('prompthelix.evaluation.evaluator.openai.OpenAI')
    @patch('prompthelix.evaluation.evaluator.logger') # Mock logger
    def test_call_llm_api_client_not_initialized(self, mock_logger, mock_openai_class):
        # Ensure client is None
        with patch('prompthelix.evaluation.evaluator.settings.OPENAI_API_KEY', None):
            evaluator = Evaluator() # Client will be None
            assert evaluator.openai_client is None

            result = evaluator._call_llm_api("Test prompt")
            assert result == "Error: LLM client not initialized."
            mock_logger.error.assert_called_with("OpenAI client is not initialized in Evaluator. Cannot call LLM API.")
