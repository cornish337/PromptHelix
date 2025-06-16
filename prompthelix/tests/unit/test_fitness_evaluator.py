import pytest
from unittest.mock import patch, Mock, MagicMock

from prompthelix.genetics.engine import FitnessEvaluator, PromptChromosome
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent  # May need to mock this
from prompthelix.enums import ExecutionMode  # Added import
from prompthelix.config import settings
import openai  # Necessary to mock openai.OpenAI and specific errors like OpenAIError
import logging  # For capturing log messages

# Configure basic logging for tests if caplog is used and output is desired during runs
# logging.basicConfig(level=logging.INFO)  # Or DEBUG for more verbose output

@pytest.fixture
def mock_results_evaluator_agent():
    """Fixture for a mocked ResultsEvaluatorAgent."""
    agent = Mock(spec=ResultsEvaluatorAgent)
    agent.process_request.return_value = {"fitness_score": 0.75, "detailed_metrics": {}, "error_analysis": []}
    return agent

@pytest.fixture
def sample_chromosome():
    """Fixture for a sample PromptChromosome."""
    return PromptChromosome(genes=["Evaluate this prompt for effectiveness."])

@pytest.fixture
def mock_openai_client():
    """Fixture for a mocked OpenAI client instance."""
    client = MagicMock(spec=openai.OpenAI)  # Using MagicMock to handle attribute access like .chat.completions

    # Mocking the response structure for a successful call
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Mocked LLM response content."
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_completion
    return client

# Test for successful LLM call
@patch('openai.OpenAI')  # Mocks the OpenAI class constructor
def test_evaluate_successful_llm_call(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    sample_chromosome,
    mock_openai_client,
    caplog
):
    """Test FitnessEvaluator.evaluate for a successful LLM call."""
    caplog.set_level(logging.INFO)
    mock_openai_constructor.return_value = mock_openai_client # mock_openai_client is already a MagicMock instance

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'): # Ensure global SDK settings has a key
        # Test with default llm_settings (None)
        evaluator = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.REAL,
            llm_settings=None
        )

    # Check that the client was initialized (mock_openai_constructor was called by __init__)
    mock_openai_constructor.assert_called_once()
    # Verify API key used by the client (assuming default behavior of openai client mock)
    # This part is tricky as the actual client is mocked. We trust __init__ logic for now.

    fitness_score = evaluator.evaluate(sample_chromosome, "Test task", {"criteria": "none"})

    # Check that create was called on the instance returned by the constructor
    mock_openai_client.chat.completions.create.assert_called_once()

    # Verify parameters passed to create (using default model from FitnessEvaluator._call_llm_api)
    args, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs['model'] == 'gpt-3.5-turbo' # Default model in _call_llm_api if not overridden

    mock_results_evaluator_agent.process_request.assert_called_once_with({
        "prompt_chromosome": sample_chromosome,
        "llm_output": "Mocked LLM response content.",
        "task_description": "Test task",
        "success_criteria": {"criteria": "none"}
    })
    assert sample_chromosome.fitness_score == 0.75
    assert fitness_score == 0.75

    assert "Calling OpenAI API model gpt-3.5-turbo" in caplog.text
    assert "OpenAI API call successful." in caplog.text

# Test for LLM API error
@patch('openai.OpenAI')
def test_evaluate_llm_api_error(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    sample_chromosome,
    mock_openai_client,
    caplog
):
    """Test FitnessEvaluator.evaluate when LLM API call raises an OpenAIError."""
    caplog.set_level(logging.ERROR)
    mock_openai_constructor.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.side_effect = openai.APIError("Test API Error", request=Mock(), body=None)

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'):
        evaluator = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.REAL,
            llm_settings=None
        )

    evaluator.evaluate(sample_chromosome, "Test task", {})

    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert "Error: LLM API call failed. Details: Test API Error" in args[0]["llm_output"]

    assert "OpenAI API error for prompt" in caplog.text
    assert "Test API Error" in caplog.text

# Test for LLM returning no content
@patch('openai.OpenAI')
def test_evaluate_llm_no_content_response(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    sample_chromosome,
    mock_openai_client,
    caplog
):
    """Test FitnessEvaluator.evaluate when LLM returns no content in the response."""
    caplog.set_level(logging.WARNING)
    mock_openai_constructor.return_value = mock_openai_client

    mock_completion_no_content = Mock()
    mock_completion_no_content.choices = []
    mock_openai_client.chat.completions.create.return_value = mock_completion_no_content

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'):
        evaluator = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.REAL,
            llm_settings=None
        )

    evaluator.evaluate(sample_chromosome, "Test task", {})

    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert args[0]["llm_output"] == "Error: No content from LLM."

    assert "returned no content or unexpected response structure" in caplog.text

# Test for FitnessEvaluator initialization when API key is missing
@patch('openai.OpenAI')  # Still mock OpenAI constructor as it might be called if logic changes
def test_fitness_evaluator_init_no_api_key(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    sample_chromosome,
    caplog
):
    """Test FitnessEvaluator initialization and evaluate call when OPENAI_API_KEY is missing."""
    caplog.set_level(logging.ERROR)

    # Test REAL mode with no API key (either in settings or global_sdk_settings)
    with patch.object(settings, 'OPENAI_API_KEY', None): # Ensure global SDK key is None
        evaluator_real_no_key = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.REAL,
            llm_settings={'api_key': None} # Ensure llm_settings also has no key
        )

    assert "OpenAI API Key not found in settings or global config" in caplog.text # Updated message from __init__
    # OpenAI constructor might still be called if api_key_to_use is None but other client_params are present.
    # The important check is that the client is None or unusable.
    assert evaluator_real_no_key.openai_client is None
    mock_openai_constructor.assert_not_called() # If api_key_to_use is None, constructor shouldn't be called.


    caplog.clear()
    # Test TEST mode with no API key (should not log error about key)
    with patch.object(settings, 'OPENAI_API_KEY', None):
        evaluator_test_no_key = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.TEST,
            llm_settings=None
        )
    assert "OpenAI API Key not found" not in caplog.text # Error should not be logged for TEST mode
    # In TEST mode, client is not initialized.
    mock_openai_constructor.call_count = 0 # Reset call count if it was called in previous step by mistake

    caplog.clear()
    caplog.set_level(logging.WARNING) # Evaluate will try to call _call_llm_api which logs errors

    # Re-create evaluator_real_no_key to ensure its state before evaluate call
    with patch.object(settings, 'OPENAI_API_KEY', None):
         evaluator_real_no_key_for_eval = FitnessEvaluator(
            results_evaluator_agent=mock_results_evaluator_agent,
            execution_mode=ExecutionMode.REAL,
            llm_settings={'api_key': None}
        )

    evaluator_real_no_key_for_eval.evaluate(sample_chromosome, "Test task", {})
    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert args[0]["llm_output"] == "Error: LLM client not initialized."

    assert "LLM client is not initialized. Cannot call LLM API in REAL mode." in caplog.text
    assert f"LLM call for prompt ID {sample_chromosome.id}" in caplog.text
    assert "failed. Output: Error: LLM client not initialized for REAL mode." in caplog.text

# New tests for ExecutionMode

def test_fitness_evaluator_call_llm_api_test_mode(mock_results_evaluator_agent, caplog):
    """Test _call_llm_api in TEST mode."""
    caplog.set_level(logging.INFO)
    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
    )

    response = evaluator._call_llm_api("test prompt in test mode")
    assert response == "This is a test output from dummy LLM in TEST mode."
    assert "Executing in TEST mode." in caplog.text

@patch('prompthelix.genetics.engine.openai.OpenAI')
def test_fitness_evaluator_call_llm_api_real_mode(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    mock_openai_client, # This is the MagicMock instance of the client
    caplog
):
    """Test _call_llm_api in REAL mode with a mocked OpenAI client and specific llm_settings."""
    caplog.set_level(logging.INFO)

    # Configure the mock constructor to return our specific client mock instance
    mock_openai_constructor.return_value = mock_openai_client

    test_llm_settings = {
        'api_key': 'fake_key_for_real_mode', # This will be used by __init__
        'default_model': 'gpt-4-test',
        'default_timeout': 120,
        'max_tokens': 500,
        'temperature': 0.5
    }

    # No need to patch global settings.OPENAI_API_KEY if llm_settings provides the key
    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.REAL,
        llm_settings=test_llm_settings
    )

    # Ensure the constructor was called and client was set up
    mock_openai_constructor.assert_called_with(api_key='fake_key_for_real_mode', timeout=120)


    response = evaluator._call_llm_api("test prompt for real mode")

    mock_openai_client.chat.completions.create.assert_called_once()
    args, kwargs = mock_openai_client.chat.completions.create.call_args

    assert kwargs['model'] == 'gpt-4-test'
    assert kwargs['timeout'] == 120
    assert kwargs['max_tokens'] == 500
    assert kwargs['temperature'] == 0.5

    assert response == "Mocked LLM response content."
    assert "Calling OpenAI API model gpt-4-test" in caplog.text # Check for overridden model
    assert "OpenAI API call successful." in caplog.text


@patch('prompthelix.genetics.engine.openai.OpenAI') # Mock constructor
def test_fitness_evaluator_call_llm_api_real_mode_model_override_in_call(
    mock_openai_constructor, # Mock for constructor
    mock_results_evaluator_agent,
    caplog
):
    """Test _call_llm_api in REAL mode where model is specified in the call."""
    caplog.set_level(logging.INFO)

    # Create a new mock client instance for this test's specific assertions
    specific_client_mock = MagicMock(spec=openai.OpenAI)
    mock_completion = Mock()
    mock_choice = Mock()
    mock_message = Mock()
    mock_message.content = "Response from specific_call_model."
    mock_choice.message = mock_message
    mock_completion.choices = [mock_choice]
    specific_client_mock.chat.completions.create.return_value = mock_completion

    mock_openai_constructor.return_value = specific_client_mock

    test_llm_settings = {
        'api_key': 'fake_key_for_real_mode_override',
        'default_model': 'gpt-3.5-settings-default', # This should be overridden by call parameter
    }

    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.REAL,
        llm_settings=test_llm_settings
    )

    response = evaluator._call_llm_api("test prompt", model_name="specific_call_model")

    specific_client_mock.chat.completions.create.assert_called_once()
    args, kwargs = specific_client_mock.chat.completions.create.call_args
    assert kwargs['model'] == 'specific_call_model' # Model from call_llm_api param
    assert response == "Response from specific_call_model."
    assert "Calling OpenAI API model specific_call_model" in caplog.text



@patch('prompthelix.genetics.engine.openai.OpenAI')
def test_fitness_evaluator_evaluate_test_mode(
    mock_openai_constructor,
    mock_results_evaluator_agent,
    sample_chromosome,
    caplog
):
    """Test FitnessEvaluator.evaluate in TEST mode."""
    caplog.set_level(logging.INFO)

    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
    )

    fitness_score = evaluator.evaluate(sample_chromosome, "Test task for evaluate", {"criteria": "test_mode"})

    mock_results_evaluator_agent.process_request.assert_called_once_with({
        "prompt_chromosome": sample_chromosome,
        "llm_output": "This is a test output from dummy LLM in TEST mode.",
        "task_description": "Test task for evaluate",
        "success_criteria": {"criteria": "test_mode"}
    })
    assert sample_chromosome.fitness_score == 0.75
    assert fitness_score == 0.75

    assert "Executing in TEST mode." in caplog.text
    mock_openai_constructor.assert_not_called()
    if evaluator.openai_client:
        evaluator.openai_client.chat.completions.create.assert_not_called()

# --- Tests for logging based on llm_analysis_status ---

def test_evaluate_logs_fallback_status(mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test that evaluate logs when fallback LLM metrics were used."""
    caplog.set_level(logging.INFO)

    eval_details_with_fallback = {
        "content_metrics": {
            "llm_analysis_status": "fallback_due_to_error",
            "llm_assessment_feedback": "Simulated API error caused fallback."
        }
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.33,
        "evaluation_details": eval_details_with_fallback,
        "error_analysis": []
    }

    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
        )
    evaluator.evaluate(sample_chromosome, "Test task for fallback logging", {})

    assert sample_chromosome.fitness_score == 0.33
    assert sample_chromosome.evaluation_details == eval_details_with_fallback

    expected_log_part1 = f"FitnessEvaluator: Chromosome {sample_chromosome.id} evaluated using fallback LLM metrics."
    expected_log_part2 = "Status: 'fallback_due_to_error'."
    expected_log_part3 = "Feedback: Simulated API error caused fallback."

    found_log = False
    for record in caplog.records:
        if record.levelname == 'INFO' and \
           expected_log_part1 in record.message and \
           expected_log_part2 in record.message and \
           expected_log_part3 in record.message:
            found_log = True
            break
    assert found_log, f"Expected fallback log message not found in {caplog.text}"

def test_evaluate_logs_success_status(mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test that evaluate logs normally for a successful LLM analysis status."""
    caplog.set_level(logging.INFO)

    eval_details_success = {
        "content_metrics": {
            "llm_analysis_status": "success",
            "llm_assessment_feedback": "LLM analysis successful."
        }
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.88,
        "evaluation_details": eval_details_success,
        "error_analysis": []
    }

    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
    )
    evaluator.evaluate(sample_chromosome, "Test task for success logging", {})

    assert sample_chromosome.fitness_score == 0.88

    fallback_log_part = "evaluated using fallback LLM metrics"
    assert fallback_log_part not in caplog.text

    final_eval_log = f"FitnessEvaluator: Evaluated chromosome {sample_chromosome.id}, Assigned Fitness: {0.88:.4f}, LLM Analysis Status: success"
    assert final_eval_log in caplog.text

def test_evaluate_logs_missing_status(mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test that evaluate logs a warning if llm_analysis_status is missing."""
    caplog.set_level(logging.WARNING)

    eval_details_missing_status_key = {
        "content_metrics": {
            "llm_assessment_feedback": "Status key missing."
        }
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.44,
        "evaluation_details": eval_details_missing_status_key,
        "error_analysis": []
    }

    evaluator = FitnessEvaluator(
        results_evaluator_agent=mock_results_evaluator_agent,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
    )
    evaluator.evaluate(sample_chromosome, "Test task for missing status key", {})

    expected_warning_log = f"FitnessEvaluator: 'llm_analysis_status' not found in evaluation_details.content_metrics for Chromosome {sample_chromosome.id}."
    assert expected_warning_log in caplog.text

    caplog.clear()

    eval_details_missing_content_metrics = {
        "other_metrics": {}
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.22,
        "evaluation_details": eval_details_missing_content_metrics,
        "error_analysis": []
    }
    evaluator.evaluate(sample_chromosome, "Test task for missing content_metrics", {})
    assert expected_warning_log in caplog.text

# --- Test for FitnessEvaluator Picklability ---
import pickle
import json
import os

DUMMY_KNOWLEDGE_FILE_PATH = "dummy_rea_config_for_pickle_test.json"

@pytest.fixture(scope="module")
def real_results_evaluator_agent_for_pickle_test():
    # Create a dummy knowledge file for ResultsEvaluatorAgent
    dummy_config_content = {
        "default_metrics": ["relevance_placeholder"],
        "task_specific": {},
        "llm_assisted_metrics_prompt_template": "Evaluate: {output}",
        "metrics_config": {"weights": {"constraint_adherence": 0.5, "llm_quality_assessment": 0.5}}
    }
    with open(DUMMY_KNOWLEDGE_FILE_PATH, 'w') as f:
        json.dump(dummy_config_content, f)

    agent = ResultsEvaluatorAgent(message_bus=None, knowledge_file_path=DUMMY_KNOWLEDGE_FILE_PATH)
    yield agent

    if os.path.exists(DUMMY_KNOWLEDGE_FILE_PATH):
        os.remove(DUMMY_KNOWLEDGE_FILE_PATH)

@patch.object(settings, 'OPENAI_API_KEY', None)
def test_fitness_evaluator_picklable_test_mode(real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in TEST mode."""
    caplog.set_level(logging.INFO)
    rea = real_results_evaluator_agent_for_pickle_test

    fe_test_mode = FitnessEvaluator(
        results_evaluator_agent=rea,
        execution_mode=ExecutionMode.TEST,
        llm_settings=None
    )

    pickled_fe = pickle.dumps(fe_test_mode)
    unpickled_fe_test_mode = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_test_mode, FitnessEvaluator)
    assert unpickled_fe_test_mode.execution_mode == ExecutionMode.TEST
    assert isinstance(unpickled_fe_test_mode.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_test_mode.results_evaluator_agent.knowledge_file_path == DUMMY_KNOWLEDGE_FILE_PATH
    assert unpickled_fe_test_mode.openai_client is None

@patch.object(settings, 'OPENAI_API_KEY', "mock_openai_key_for_pickle_test")
def test_fitness_evaluator_picklable_real_mode_with_key(real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in REAL mode with an API key."""
    caplog.set_level(logging.INFO)
    rea = real_results_evaluator_agent_for_pickle_test

    fe_real_mode = FitnessEvaluator(
        results_evaluator_agent=rea,
        execution_mode=ExecutionMode.REAL,
        llm_settings={'api_key': "mock_openai_key_for_pickle_test"} # Provide key via settings
    )
    assert fe_real_mode.openai_client is not None
    assert fe_real_mode.openai_client.api_key == "mock_openai_key_for_pickle_test"
    # Check that global settings.OPENAI_API_KEY is not changed by this
    assert settings.OPENAI_API_KEY == "mock_openai_key_for_pickle_test" # This is from the @patch.object for this test

    pickled_fe = pickle.dumps(fe_real_mode)
    unpickled_fe_real_mode = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_real_mode, FitnessEvaluator)
    assert unpickled_fe_real_mode.execution_mode == ExecutionMode.REAL
    assert isinstance(unpickled_fe_real_mode.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_real_mode.results_evaluator_agent.knowledge_file_path == DUMMY_KNOWLEDGE_FILE_PATH
    assert unpickled_fe_real_mode.openai_client is not None
    assert unpickled_fe_real_mode.openai_client.api_key == "mock_openai_key_for_pickle_test"

@patch.object(settings, 'OPENAI_API_KEY', None)
def test_fitness_evaluator_picklable_real_mode_no_key(real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in REAL mode without an API key."""
    caplog.set_level(logging.INFO)
    rea = real_results_evaluator_agent_for_pickle_test

    # Ensure settings.OPENAI_API_KEY is None for this test case due to the @patch.object
    fe_real_mode_no_key = FitnessEvaluator(
        results_evaluator_agent=rea,
        execution_mode=ExecutionMode.REAL,
        llm_settings=None # No settings override, so it relies on global SDK settings
    )
    assert fe_real_mode_no_key.openai_client is None
    assert "OpenAI API Key not found in settings or global config" in caplog.text
    caplog.clear()

    pickled_fe = pickle.dumps(fe_real_mode_no_key)
    unpickled_fe_real_mode_no_key = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_real_mode_no_key, FitnessEvaluator)
    assert unpickled_fe_real_mode_no_key.execution_mode == ExecutionMode.REAL
    assert isinstance(unpickled_fe_real_mode_no_key.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_real_mode_no_key.openai_client is None
