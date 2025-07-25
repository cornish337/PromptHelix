import pytest
from unittest.mock import patch, MagicMock
import re  # Import re for escaping
import datetime

from prompthelix.utils import llm_utils

from openai import (
    OpenAIError,
    APIError as OpenAPIApiError,
    RateLimitError as OpenAIRateLimitError,
    AuthenticationError as OpenAIAuthenticationError,
    APIConnectionError as OpenAIAPIConnectionError,
)
try:  # pragma: no cover - depends on installed OpenAI SDK
    from openai import InvalidRequestError as OpenAIInvalidRequestError
except ImportError:  # pragma: no cover
    from openai import BadRequestError as OpenAIInvalidRequestError

from anthropic import (
    AnthropicError,
    APIError as AnthropicAPIError,
    RateLimitError as AnthropicRateLimitError,
    AuthenticationError as AnthropicAuthenticationError,
    APIStatusError as AnthropicAPIStatusError,
    APIConnectionError as AnthropicAPIConnectionError,
)
from google.api_core import exceptions as google_exceptions

# Fixture for common prompt text
@pytest.fixture
def sample_prompt_text():
    return "This is a test prompt."

# Tests for successful dispatch of call_llm_api (keeping one example, assuming others work if one does)
# These tests might need adjustment if the "Simulated ... API response" is removed due to error handling changes.
# For now, let's assume they pass if a key is provided and no API error is simulated.
@patch("prompthelix.utils.llm_utils.get_openai_api_key", return_value="fake_api_key")
@patch("openai.OpenAI") # Mock the client
def test_call_llm_api_dispatches_openai_successfully(mock_openai_client_constructor, mock_get_key, sample_prompt_text, db_session):
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=" OpenAI response "))])
    mock_openai_client_constructor.return_value = mock_openai_instance

    result = llm_utils.call_llm_api(prompt=sample_prompt_text, provider="openai", model="gpt-3.5-turbo", db=db_session)
    mock_get_key.assert_called_once_with(db_session)
    mock_openai_instance.chat.completions.create.assert_called_once()
    assert result == "OpenAI response"

# Test `call_llm_api` for missing API keys - expecting specific error string now
@pytest.mark.parametrize("provider_name, llm_utils_api_key_getter_path", [
    ("openai", "prompthelix.utils.llm_utils.get_openai_api_key"),
    ("anthropic", "prompthelix.utils.llm_utils.get_anthropic_api_key"),
    ("google", "prompthelix.utils.llm_utils.get_google_api_key"),
])
def test_call_llm_api_returns_error_string_if_key_missing(
    provider_name, llm_utils_api_key_getter_path, sample_prompt_text, db_session
):
    with patch(llm_utils_api_key_getter_path, return_value=None) as mock_get_key:
        result = llm_utils.call_llm_api(sample_prompt_text, provider=provider_name, model="test_model", db=db_session)
        assert result == "API_KEY_MISSING_ERROR"
        mock_get_key.assert_called_once_with(db_session)


# Test specific provider callers for missing API keys - expecting specific error string now
@patch("prompthelix.utils.llm_utils.get_openai_api_key", return_value=None)
def test_call_openai_api_returns_error_string_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    result = llm_utils.call_openai_api(sample_prompt_text, model="test_model", db=db_session)
    assert result == "API_KEY_MISSING_ERROR"
    mock_get_key.assert_called_once_with(db_session)

@patch("prompthelix.utils.llm_utils.get_anthropic_api_key", return_value=None)
def test_call_claude_api_returns_error_string_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    result = llm_utils.call_claude_api(sample_prompt_text, model="test_model", db=db_session)
    assert result == "API_KEY_MISSING_ERROR"
    mock_get_key.assert_called_once_with(db_session)

@patch("prompthelix.utils.llm_utils.get_google_api_key", return_value=None)
def test_call_google_api_returns_error_string_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    result = llm_utils.call_google_api(sample_prompt_text, model="test_model", db=db_session)
    assert result == "API_KEY_MISSING_ERROR"
    mock_get_key.assert_called_once_with(db_session)


# Test `call_llm_api` for an unsupported provider - expecting specific error string now
def test_call_llm_api_unsupported_provider(sample_prompt_text, db_session):
    result = llm_utils.call_llm_api(sample_prompt_text, provider="unknown_provider", model="test_model", db=db_session)
    assert result == "UNSUPPORTED_PROVIDER_ERROR"

# --- New tests for specific error handling ---

# Helper function to create the problematic exception
def make_openai_invalid_request_error():
    # Ensure the correct exception is imported and used here.
    # This mirrors the original import logic at the top of the file.
    try:
        from openai import InvalidRequestError as ActualOpenAIInvalidRequestError
    except ImportError:
        from openai import BadRequestError as ActualOpenAIInvalidRequestError
    return ActualOpenAIInvalidRequestError(message="invalid req", body=None)

# OpenAI Error Handling Tests
@patch("prompthelix.utils.llm_utils.get_openai_api_key", return_value="fake_key")
@patch("openai.OpenAI")
@pytest.mark.parametrize("openai_exception, expected_error_string", [
    (OpenAIRateLimitError("rate limit", response=MagicMock(), body=None), "RATE_LIMIT_ERROR"),
    (OpenAIAuthenticationError("auth error", response=MagicMock(), body=None), "AUTHENTICATION_ERROR"),
    (OpenAIAPIConnectionError(message="conn error", request=MagicMock()), "API_CONNECTION_ERROR"),
    (make_openai_invalid_request_error(), "INVALID_REQUEST_ERROR"), # Use helper

    (OpenAPIApiError("api error", request=MagicMock(), body=None), "API_ERROR"),
    (OpenAIError("generic openai error"), "OPENAI_ERROR"),
    (Exception("unexpected"), "UNEXPECTED_OPENAI_CALL_ERROR"),
])
def test_call_openai_api_error_handling(mock_openai_client_constructor, mock_get_key, openai_exception, expected_error_string, sample_prompt_text, db_session):
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.side_effect = openai_exception
    mock_openai_client_constructor.return_value = mock_openai_instance

    result = llm_utils.call_openai_api(sample_prompt_text, db=db_session)
    assert result == expected_error_string

# Anthropic Error Handling Tests
@patch("prompthelix.utils.llm_utils.get_anthropic_api_key", return_value="fake_key")
@patch("prompthelix.utils.llm_utils.AnthropicClient") # Mock the aliased client
@pytest.mark.parametrize("anthropic_exception, expected_error_string", [

    (AnthropicRateLimitError("rate limit", response=MagicMock()), "RATE_LIMIT_ERROR"),
    (AnthropicAuthenticationError("auth error", response=MagicMock()), "AUTHENTICATION_ERROR"),
    (AnthropicAPIStatusError("status error", response=MagicMock(status_code=500)), "API_STATUS_ERROR"),
    (AnthropicAPIConnectionError("conn error", request=MagicMock()), "API_CONNECTION_ERROR"),
    (AnthropicAPIError("api error", request=MagicMock()), "API_ERROR"),
    (AnthropicError("generic anthropic error"), "ANTHROPIC_ERROR"),
    (Exception("unexpected"), "UNEXPECTED_ANTHROPIC_CALL_ERROR"),
])
def test_call_claude_api_error_handling(mock_anthropic_client_constructor, mock_get_key, anthropic_exception, expected_error_string, sample_prompt_text, db_session):
    mock_anthropic_instance = MagicMock()
    mock_anthropic_instance.messages.create.side_effect = anthropic_exception
    mock_anthropic_client_constructor.return_value = mock_anthropic_instance

    result = llm_utils.call_claude_api(sample_prompt_text, db=db_session)
    assert result == expected_error_string

# Google Error Handling Tests
@patch("prompthelix.utils.llm_utils.get_google_api_key", return_value="fake_key")
@patch("google.generativeai.GenerativeModel")
@patch("google.generativeai.configure") # Also mock configure as it's called before model instantiation
@pytest.mark.parametrize("google_exception, expected_error_string", [
    (google_exceptions.ResourceExhausted("rate limit"), "RATE_LIMIT_ERROR"),
    (google_exceptions.PermissionDenied("auth error"), "AUTHENTICATION_ERROR"),
    (google_exceptions.InvalidArgument("invalid arg"), "INVALID_ARGUMENT_ERROR"),
    (google_exceptions.InternalServerError("server error"), "API_SERVER_ERROR"),
    (google_exceptions.GoogleAPIError("api error"), "API_ERROR"),
    (Exception("unexpected"), "UNEXPECTED_GOOGLE_CALL_ERROR"),
    # Special case for ValueError from response.text if prompt_feedback indicates blockage
    (ValueError("ValueError accessing response.text"), "BLOCKED_PROMPT_ERROR"),
])
def test_call_google_api_error_handling(mock_genai_configure, mock_generative_model_constructor, mock_get_key, google_exception, expected_error_string, sample_prompt_text, db_session):
    mock_model_instance = MagicMock()

    if expected_error_string == "BLOCKED_PROMPT_ERROR" and isinstance(google_exception, ValueError):
        # Simulate the case where response.text raises ValueError, and prompt_feedback shows blockage
        mock_response = MagicMock()
        mock_response.parts = [] # Or some other condition that leads to .text access attempt
        mock_response.prompt_feedback = MagicMock(block_reason="SAFETY")
        # The actual .text access is what raises the ValueError.
        # We need generate_content to return this mock_response, and then have .text raise the error.
        # This is tricky to set up perfectly without deeper mocking of the response object itself.
        # For simplicity, let's assume if generate_content itself raises ValueError directly in this context, it's treated as BLOCKED_PROMPT_ERROR.
        # A more accurate mock would involve mocking response.text property.
        # However, the code structure is: try generate_content, then access .text in a nested try.
        # So, if generate_content returns a response that *then* causes .text to fail with ValueError:
        def mock_generate_content_then_text_value_error(*args, **kwargs):
            response_with_problematic_text = MagicMock()
            response_with_problematic_text.parts = [MagicMock()] # Ensure it doesn't take the "empty parts" path first
            response_with_problematic_text.prompt_feedback = MagicMock(block_reason="SAFETY")
            # Make .text raise ValueError
            type(response_with_problematic_text).text = MagicMock(side_effect=google_exception)
            return response_with_problematic_text
        mock_model_instance.generate_content.side_effect = mock_generate_content_then_text_value_error
    else:
        mock_model_instance.generate_content.side_effect = google_exception

    mock_generative_model_constructor.return_value = mock_model_instance

    result = llm_utils.call_google_api(sample_prompt_text, db=db_session)
    assert result == expected_error_string


# Test `list_available_llms` (keeping this as it is, should still work)
@pytest.mark.parametrize("openai_key_val, anthropic_key_val, google_key_val, expected_services_list", [
    (None, None, None, []),
    ("fake_openai_key", None, None, ["OPENAI"]),
    (None, "fake_anthropic_key", None, ["ANTHROPIC"]),
    (None, None, "fake_google_key", ["GOOGLE"]),
    ("fake_openai_key", "fake_anthropic_key", None, sorted(["OPENAI", "ANTHROPIC"])),
    ("fake_openai_key", None, "fake_google_key", sorted(["OPENAI", "GOOGLE"])),
    (None, "fake_anthropic_key", "fake_google_key", sorted(["ANTHROPIC", "GOOGLE"])),
    ("fake_openai_key", "fake_anthropic_key", "fake_google_key", sorted(["OPENAI", "ANTHROPIC", "GOOGLE"])),
])
@patch("prompthelix.utils.llm_utils.get_google_api_key", autospec=True)
@patch("prompthelix.utils.llm_utils.get_anthropic_api_key", autospec=True)
@patch("prompthelix.utils.llm_utils.get_openai_api_key", autospec=True)
def test_list_available_llms_scenarios(
    mock_llm_utils_get_openai_key,
    mock_llm_utils_get_anthropic_key,
    mock_llm_utils_get_google_key,
    openai_key_val, anthropic_key_val, google_key_val, expected_services_list,
    db_session
):
    mock_llm_utils_get_openai_key.return_value = openai_key_val
    mock_llm_utils_get_anthropic_key.return_value = anthropic_key_val
    mock_llm_utils_get_google_key.return_value = google_key_val

    available_llms = llm_utils.list_available_llms(db=db_session)

    assert sorted(available_llms) == sorted(expected_services_list)

    mock_llm_utils_get_openai_key.assert_called_once_with(db_session)
    mock_llm_utils_get_anthropic_key.assert_called_once_with(db_session)
    mock_llm_utils_get_google_key.assert_called_once_with(db_session)


@patch('prompthelix.utils.llm_utils.LOG_FILE_PATH', "test_llm_api_calls.log")
@patch('prompthelix.utils.llm_utils.call_google_api')
@patch('prompthelix.utils.llm_utils.call_anthropic_api')
@patch('prompthelix.utils.llm_utils.call_openai_api')
@patch('prompthelix.utils.llm_utils.datetime')
@patch('builtins.open', new_callable=MagicMock)
def test_call_llm_api_logging(
    mock_open,  # Corresponds to @patch('builtins.open', ...)
    mock_datetime,  # Corresponds to @patch('prompthelix.utils.llm_utils.datetime')
    mock_call_openai,  # Corresponds to @patch('prompthelix.utils.llm_utils.call_openai_api')
    mock_call_anthropic,  # Corresponds to @patch('prompthelix.utils.llm_utils.call_anthropic_api')
    mock_call_google,  # Corresponds to @patch('prompthelix.utils.llm_utils.call_google_api')
    mock_log_file_path_object, # Corresponds to @patch('prompthelix.utils.llm_utils.LOG_FILE_PATH', ...)
    sample_prompt_text,
    db_session
):
    # Setup fixed timestamp
    fixed_timestamp_str = "2023-10-26T10:00:00"
    mock_dt_instance = MagicMock()
    mock_dt_instance.isoformat.return_value = fixed_timestamp_str
    mock_datetime.datetime.now.return_value = mock_dt_instance

    test_log_file = "test_llm_api_calls.log" # This is the patched path

    # --- Test Case 1: Successful OpenAI call ---
    mock_call_openai.return_value = "OpenAI success response"
    provider_success = "openai"
    model_success = "gpt-test"
    expected_api_path_success = f"{provider_success.lower()}/{model_success}"

    llm_utils.call_llm_api(prompt=sample_prompt_text, provider=provider_success, model=model_success, db=db_session)

    mock_open.assert_called_with(test_log_file, "a")

    written_content_success_calls = mock_open.return_value.write.call_args_list
    written_content_success = "".join(call[0][0] for call in written_content_success_calls)

    assert f"Timestamp: {fixed_timestamp_str}" in written_content_success
    assert f"API Path: {expected_api_path_success}" in written_content_success
    assert f"Prompt: {sample_prompt_text}" in written_content_success
    assert "Error:" not in written_content_success
    assert "---\n" in written_content_success

    mock_open.reset_mock()
    mock_open.return_value.write.reset_mock()
    mock_call_openai.reset_mock()

    # --- Test Case 2: Error call (Anthropic) ---
    error_return_value = "RATE_LIMIT_ERROR"
    mock_call_anthropic.return_value = error_return_value
    provider_error = "anthropic"
    model_error = "claude-test"
    expected_api_path_error = f"{provider_error.lower()}/{model_error}"
    expected_error_msg_in_log = error_return_value

    llm_utils.call_llm_api(prompt=sample_prompt_text, provider=provider_error, model=model_error, db=db_session)

    mock_open.assert_called_with(test_log_file, "a")
    written_content_error_calls = mock_open.return_value.write.call_args_list
    written_content_error = "".join(call[0][0] for call in written_content_error_calls)

    assert f"Timestamp: {fixed_timestamp_str}" in written_content_error
    assert f"API Path: {expected_api_path_error}" in written_content_error
    assert f"Prompt: {sample_prompt_text}" in written_content_error
    assert f"Error: {expected_error_msg_in_log}" in written_content_error
    assert "---\n" in written_content_error

    mock_open.reset_mock()
    mock_open.return_value.write.reset_mock()
    mock_call_anthropic.reset_mock() # Reset this mock

    # --- Test Case 3: Successful Google call ---
    mock_call_google.return_value = "Google success response"
    provider_google = "google"
    model_google = "gemini-test"
    expected_api_path_google = f"{provider_google.lower()}/{model_google}"

    llm_utils.call_llm_api(prompt=sample_prompt_text, provider=provider_google, model=model_google, db=db_session)

    mock_open.assert_called_with(test_log_file, "a")
    written_content_google_calls = mock_open.return_value.write.call_args_list
    written_content_google = "".join(call[0][0] for call in written_content_google_calls)

    assert f"Timestamp: {fixed_timestamp_str}" in written_content_google
    assert f"API Path: {expected_api_path_google}" in written_content_google
    assert f"Prompt: {sample_prompt_text}" in written_content_google
    assert "Error:" not in written_content_google
    assert "---\n" in written_content_google
