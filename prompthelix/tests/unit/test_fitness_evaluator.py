import pytest
from unittest.mock import patch, Mock, MagicMock

from prompthelix.genetics.engine import FitnessEvaluator, PromptChromosome
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent # May need to mock this
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
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent)

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
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent)

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
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent)

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

    with patch.object(settings, 'OPENAI_API_KEY', None): # Simulate missing API key
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_evaluator_agent)

    assert "OPENAI_API_KEY not found in settings" in caplog.text
    mock_openai_constructor.assert_not_called() # OpenAI client should not be initialized

    # Now test calling evaluate with no client
    caplog.clear() # Clear previous log messages
    caplog.set_level(logging.WARNING) # For the evaluate warning

    evaluator.evaluate(sample_chromosome, "Test task", {})

    # Check that the results evaluator still gets called, but with an error message
    mock_results_evaluator_agent.process_request.assert_called_once()
    args, _ = mock_results_evaluator_agent.process_request.call_args
    assert args[0]["llm_output"] == "Error: LLM client not initialized."

    # Check log for the specific failure within evaluate (or _call_llm_api)
    assert "LLM client is not initialized. Cannot call LLM API." in caplog.text # This is from _call_llm_api
    assert f"LLM call for prompt ID {sample_chromosome.id}" in caplog.text # This is from evaluate
    assert "failed. Output: Error: LLM client not initialized." in caplog.text # This is from evaluate

print("New test file created and initial tests added: prompthelix/tests/unit/test_fitness_evaluator.py")
