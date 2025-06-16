import pytest
from unittest.mock import patch, Mock, MagicMock

from prompthelix.genetics.engine import FitnessEvaluator, PromptChromosome
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent # May need to mock this
from prompthelix.enums import ExecutionMode # Added import
from prompthelix.config import settings
import openai # Necessary to mock openai.OpenAI and specific errors like OpenAIError
import logging # For capturing log messages

# Configure basic logging for tests if caplog is used and output is desired during runs
# logging.basicConfig(level=logging.INFO) # Or DEBUG for more verbose output

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
    client = MagicMock(spec=openai.OpenAI) # Using MagicMock to handle attribute access like .chat.completions

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
@patch('openai.OpenAI') # Mocks the OpenAI class constructor
def test_evaluate_successful_llm_call(mock_openai_constructor, mock_results_evaluator_agent, sample_chromosome, mock_openai_client, caplog):
    """Test FitnessEvaluator.evaluate for a successful LLM call."""
    caplog.set_level(logging.INFO)
    mock_openai_constructor.return_value = mock_openai_client # The constructor now returns our mock client

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'):
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.REAL)

    fitness_score = evaluator.evaluate(sample_chromosome, "Test task", {"criteria": "none"})

    mock_openai_client.chat.completions.create.assert_called_once()
    mock_results_evaluator_agent.process_request.assert_called_once_with({
        "prompt_chromosome": sample_chromosome,
        "llm_output": "Mocked LLM response content.",
        "task_description": "Test task",
        "success_criteria": {"criteria": "none"}
    })
    assert sample_chromosome.fitness_score == 0.75 # From mock_results_evaluator_agent
    assert fitness_score == 0.75

    assert "Calling OpenAI API model gpt-3.5-turbo" in caplog.text
    assert "OpenAI API call successful." in caplog.text

# Test for LLM API error
@patch('openai.OpenAI')
def test_evaluate_llm_api_error(mock_openai_constructor, mock_results_evaluator_agent, sample_chromosome, mock_openai_client, caplog):
    """Test FitnessEvaluator.evaluate when LLM API call raises an OpenAIError."""
    caplog.set_level(logging.ERROR)
    mock_openai_constructor.return_value = mock_openai_client
    mock_openai_client.chat.completions.create.side_effect = openai.APIError("Test API Error", request=Mock(), body=None)

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'):
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.REAL)

    evaluator.evaluate(sample_chromosome, "Test task", {})

    mock_results_evaluator_agent.process_request.assert_called_once()
    # Check that the error string is passed to results_evaluator
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert "Error: LLM API call failed. Details: Test API Error" in args[0]["llm_output"]

    assert "OpenAI API error for prompt" in caplog.text
    assert "Test API Error" in caplog.text

# Test for LLM returning no content
@patch('openai.OpenAI')
def test_evaluate_llm_no_content_response(mock_openai_constructor, mock_results_evaluator_agent, sample_chromosome, mock_openai_client, caplog):
    """Test FitnessEvaluator.evaluate when LLM returns no content in the response."""
    caplog.set_level(logging.WARNING)
    mock_openai_constructor.return_value = mock_openai_client

    # Configure mock to simulate no content
    mock_completion_no_content = Mock()
    mock_completion_no_content.choices = [] # No choices
    mock_openai_client.chat.completions.create.return_value = mock_completion_no_content

    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_present'):
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.REAL)

    evaluator.evaluate(sample_chromosome, "Test task", {})

    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert args[0]["llm_output"] == "Error: No content from LLM."

    assert "returned no content or unexpected response structure" in caplog.text

# Test for FitnessEvaluator initialization when API key is missing
@patch('openai.OpenAI') # Still mock OpenAI constructor as it might be called if logic changes
def test_fitness_evaluator_init_no_api_key(mock_openai_constructor, mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test FitnessEvaluator initialization and evaluate call when OPENAI_API_KEY is missing."""
    caplog.set_level(logging.ERROR) # For init error

    # Test REAL mode init without API key
    with patch.object(settings, 'OPENAI_API_KEY', None): # Simulate missing API key
        evaluator_real_no_key = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.REAL)

    assert "OPENAI_API_KEY not found in settings" in caplog.text # Error logged by __init__
    mock_openai_constructor.assert_not_called() # OpenAI client should not be initialized in REAL mode if key is missing

    # Test TEST mode init without API key (should not log error about key)
    caplog.clear()
    with patch.object(settings, 'OPENAI_API_KEY', None): # Simulate missing API key
        evaluator_test_no_key = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)
    assert "OPENAI_API_KEY not found in settings" not in caplog.text # No error for TEST mode
    mock_openai_constructor.assert_not_called() # Client not initialized in TEST mode

    # Now test calling evaluate with no client (using the evaluator_real_no_key instance)
    caplog.clear() # Clear previous log messages
    caplog.set_level(logging.WARNING) # For the evaluate warning

    evaluator_real_no_key.evaluate(sample_chromosome, "Test task", {})

    # Now test calling evaluate with no client
    caplog.clear() # Clear previous log messages
    caplog.set_level(logging.WARNING) # For the evaluate warning

    evaluator.evaluate(sample_chromosome, "Test task", {})

    # Check that the results evaluator still gets called, but with an error message
    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert args[0]["llm_output"] == "Error: LLM client not initialized."

    # Check log for the specific failure within evaluate (or _call_llm_api)
    assert "LLM client is not initialized. Cannot call LLM API in REAL mode." in caplog.text # This is from _call_llm_api
    assert f"LLM call for prompt ID {sample_chromosome.id}" in caplog.text # This is from evaluate
    assert "failed. Output: Error: LLM client not initialized for REAL mode." in caplog.text # This is from evaluate


# New tests for ExecutionMode
def test_fitness_evaluator_call_llm_api_test_mode(mock_results_evaluator_agent, caplog):
    """Test _call_llm_api in TEST mode."""
    caplog.set_level(logging.INFO)
    evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)

    response = evaluator._call_llm_api("test prompt in test mode")
    assert response == "This is a test output from dummy LLM in TEST mode."
    assert "Executing in TEST mode." in caplog.text
    # Ensure no actual LLM call was attempted (e.g. by checking if openai_client.chat.completions.create was called, if client was mocked)
    # In this case, evaluator.openai_client is None in TEST mode, so no need to mock create.

@patch('prompthelix.genetics.engine.openai.OpenAI')
def test_fitness_evaluator_call_llm_api_real_mode(mock_openai_constructor, mock_results_evaluator_agent, mock_openai_client, caplog):
    """Test _call_llm_api in REAL mode with a mocked OpenAI client."""
    caplog.set_level(logging.INFO)
    mock_openai_constructor.return_value = mock_openai_client

    # Patch settings to ensure OPENAI_API_KEY is present for REAL mode client initialization
    with patch.object(settings, 'OPENAI_API_KEY', 'fake_key_for_real_mode'):
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.REAL)

    response = evaluator._call_llm_api("test prompt for real mode")

    mock_openai_client.chat.completions.create.assert_called_once()
    assert response == "Mocked LLM response content."
    assert "Calling OpenAI API model gpt-3.5-turbo" in caplog.text
    assert "OpenAI API call successful." in caplog.text

@patch('prompthelix.genetics.engine.openai.OpenAI')
def test_fitness_evaluator_evaluate_test_mode(mock_openai_constructor, mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test FitnessEvaluator.evaluate in TEST mode."""
    caplog.set_level(logging.INFO)
    # No need to mock_openai_constructor return value as it shouldn't be used for client.create in TEST mode

    evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)

    fitness_score = evaluator.evaluate(sample_chromosome, "Test task for evaluate", {"criteria": "test_mode"})

    # Check that _call_llm_api returned the dummy response
    # And results_evaluator_agent was called with this dummy response
    mock_results_evaluator_agent.process_request.assert_called_once_with({
        "prompt_chromosome": sample_chromosome,
        "llm_output": "This is a test output from dummy LLM in TEST mode.", # Expected dummy output
        "task_description": "Test task for evaluate",
        "success_criteria": {"criteria": "test_mode"}
    })
    assert sample_chromosome.fitness_score == 0.75 # As per mock_results_evaluator_agent
    assert fitness_score == 0.75

    assert "Executing in TEST mode." in caplog.text # Log from _call_llm_api
    mock_openai_constructor.assert_not_called() # Ensure OpenAI client was not even attempted to be constructed if FitnessEvaluator skips it in TEST mode
                                                # (Current __init__ logic does try to init if key is there, but _call_llm_api short-circuits)
                                                # A stricter test would be mock_openai_client.chat.completions.create.assert_not_called()
                                                # if mock_openai_client was passed to the evaluator or set on it.
                                                # Since the client is an instance var, we check its methods weren't called.
    # If evaluator.openai_client could exist, check its methods:
    if evaluator.openai_client: # Should be None in TEST mode if API key not set, or if init logic changes
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
        # Other metrics could be here
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.33, # Example score
        "evaluation_details": eval_details_with_fallback,
        "error_analysis": []
    }

    evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)
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

    evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)
    evaluator.evaluate(sample_chromosome, "Test task for success logging", {})

    assert sample_chromosome.fitness_score == 0.88

    fallback_log_part = "evaluated using fallback LLM metrics"
    assert fallback_log_part not in caplog.text

    # Check for the standard evaluation log
    final_eval_log = f"FitnessEvaluator: Evaluated chromosome {sample_chromosome.id}, Assigned Fitness: {0.88:.4f}, LLM Analysis Status: success"
    assert final_eval_log in caplog.text


def test_evaluate_logs_missing_status(mock_results_evaluator_agent, sample_chromosome, caplog):
    """Test that evaluate logs a warning if llm_analysis_status is missing."""
    caplog.set_level(logging.WARNING) # We are expecting a warning

    # Scenario 1: content_metrics is present, but llm_analysis_status is missing
    eval_details_missing_status_key = {
        "content_metrics": {
            "llm_assessment_feedback": "Status key missing."
            # llm_analysis_status is deliberately omitted
        }
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.44,
        "evaluation_details": eval_details_missing_status_key,
        "error_analysis": []
    }

    evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent, execution_mode=ExecutionMode.TEST)
    evaluator.evaluate(sample_chromosome, "Test task for missing status key", {})

    expected_warning_log = f"FitnessEvaluator: 'llm_analysis_status' not found in evaluation_details.content_metrics for Chromosome {sample_chromosome.id}."
    assert expected_warning_log in caplog.text

    caplog.clear() # Clear for the next scenario

    # Scenario 2: content_metrics key itself is missing from evaluation_details
    eval_details_missing_content_metrics = {
        "other_metrics": {}
        # content_metrics key is missing
    }
    mock_results_evaluator_agent.process_request.return_value = {
        "fitness_score": 0.22,
        "evaluation_details": eval_details_missing_content_metrics,
        "error_analysis": []
    }
    evaluator.evaluate(sample_chromosome, "Test task for missing content_metrics", {})
    assert expected_warning_log in caplog.text # Same warning should be logged

# --- Test for FitnessEvaluator Picklability ---
import pickle
import json
import os

DUMMY_KNOWLEDGE_FILE_PATH = "dummy_rea_config_for_pickle_test.json"

@pytest.fixture(scope="module")
def real_results_evaluator_agent_for_pickle_test():
    # This fixture runs once per module
    # Create a dummy knowledge file for ResultsEvaluatorAgent
    # AGENT_SETTINGS must be available in config for REA to init its llm_provider/model
    # For this test, we only care that REA can be instantiated and its class/path stored.
    # The actual content of the knowledge file might not be critical if REA's __init__ is robust.
    dummy_config_content = {
        "default_metrics": ["relevance_placeholder"],
        "task_specific": {},
        "llm_assisted_metrics_prompt_template": "Evaluate: {output}",
        "metrics_config": { # Adding this key as ResultsEvaluatorAgent tries to access it.
            "weights": {"constraint_adherence": 0.5, "llm_quality_assessment": 0.5}
        }
    }
    with open(DUMMY_KNOWLEDGE_FILE_PATH, 'w') as f:
        json.dump(dummy_config_content, f)

    agent = ResultsEvaluatorAgent(message_bus=None, knowledge_file_path=DUMMY_KNOWLEDGE_FILE_PATH)
    yield agent # Provide the agent to the test

    # Teardown: remove the dummy file
    if os.path.exists(DUMMY_KNOWLEDGE_FILE_PATH):
        os.remove(DUMMY_KNOWLEDGE_FILE_PATH)

def test_fitness_evaluator_picklable_test_mode(real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in TEST mode."""
    caplog.set_level(logging.INFO) # To see re-initialization logs
    rea = real_results_evaluator_agent_for_pickle_test

    fe_test_mode = FitnessEvaluator(results_evaluator_agent=rea, execution_mode=ExecutionMode.TEST)

    pickled_fe = pickle.dumps(fe_test_mode)
    unpickled_fe_test_mode = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_test_mode, FitnessEvaluator)
    assert unpickled_fe_test_mode.execution_mode == ExecutionMode.TEST
    assert isinstance(unpickled_fe_test_mode.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_test_mode.results_evaluator_agent is not None # Check it's re-created
    assert unpickled_fe_test_mode.results_evaluator_agent.knowledge_file_path == DUMMY_KNOWLEDGE_FILE_PATH
    assert unpickled_fe_test_mode.openai_client is None

    assert "FitnessEvaluator (subprocess): ResultsEvaluatorAgent re-initialized." in caplog.text
    assert "FitnessEvaluator (subprocess): In TEST mode, LLM client not re-initialized." in caplog.text

@patch.object(settings, 'OPENAI_API_KEY', "mock_openai_key_for_pickle_test")
def test_fitness_evaluator_picklable_real_mode_with_key(mock_api_key_setting, real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in REAL mode with an API key."""
    caplog.set_level(logging.INFO)
    rea = real_results_evaluator_agent_for_pickle_test

    fe_real_mode = FitnessEvaluator(results_evaluator_agent=rea, execution_mode=ExecutionMode.REAL)
    # Check client initialized in main process
    assert fe_real_mode.openai_client is not None
    assert fe_real_mode.openai_client.api_key == "mock_openai_key_for_pickle_test"

    pickled_fe = pickle.dumps(fe_real_mode)
    unpickled_fe_real_mode = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_real_mode, FitnessEvaluator)
    assert unpickled_fe_real_mode.execution_mode == ExecutionMode.REAL
    assert isinstance(unpickled_fe_real_mode.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_real_mode.results_evaluator_agent is not None
    assert unpickled_fe_real_mode.results_evaluator_agent.knowledge_file_path == DUMMY_KNOWLEDGE_FILE_PATH

    assert unpickled_fe_real_mode.openai_client is not None
    assert isinstance(unpickled_fe_real_mode.openai_client, openai.OpenAI)
    assert unpickled_fe_real_mode.openai_client.api_key == "mock_openai_key_for_pickle_test"

    assert "FitnessEvaluator (subprocess): ResultsEvaluatorAgent re-initialized." in caplog.text
    assert "FitnessEvaluator (subprocess): OpenAI client re-initialized for REAL mode." in caplog.text

@patch.object(settings, 'OPENAI_API_KEY', None)
def test_fitness_evaluator_picklable_real_mode_no_key(mock_no_api_key_setting, real_results_evaluator_agent_for_pickle_test, caplog):
    """Test FitnessEvaluator is picklable in REAL mode without an API key."""
    caplog.set_level(logging.INFO) # To see init and re-init logs including warnings
    rea = real_results_evaluator_agent_for_pickle_test

    fe_real_mode_no_key = FitnessEvaluator(results_evaluator_agent=rea, execution_mode=ExecutionMode.REAL)
    assert fe_real_mode_no_key.openai_client is None # Check client is None in main process
    assert "OPENAI_API_KEY not found in settings" in caplog.text # From __init__
    caplog.clear() # Clear init logs to focus on subprocess logs

    pickled_fe = pickle.dumps(fe_real_mode_no_key)
    unpickled_fe_real_mode_no_key = pickle.loads(pickled_fe)

    assert isinstance(unpickled_fe_real_mode_no_key, FitnessEvaluator)
    assert unpickled_fe_real_mode_no_key.execution_mode == ExecutionMode.REAL
    assert isinstance(unpickled_fe_real_mode_no_key.results_evaluator_agent, ResultsEvaluatorAgent)
    assert unpickled_fe_real_mode_no_key.results_evaluator_agent is not None
    assert unpickled_fe_real_mode_no_key.openai_client is None

    assert "FitnessEvaluator (subprocess): ResultsEvaluatorAgent re-initialized." in caplog.text
    assert "FitnessEvaluator (subprocess): OpenAI API key not found in settings. Cannot re-initialize client for REAL mode." in caplog.text


# The print statement at the end of the original file is not standard for test files.
# It's removed in this modified version.
# print("New test file created and initial tests added: prompthelix/tests/unit/test_fitness_evaluator.py")
