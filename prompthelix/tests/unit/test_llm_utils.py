import pytest
from unittest.mock import patch, MagicMock
import re # Import re for escaping

from prompthelix.utils import llm_utils

# Fixture for common prompt text
@pytest.fixture
def sample_prompt_text():
    return "This is a test prompt."

# Tests for successful dispatch of call_llm_api
@pytest.mark.parametrize("provider_name, llm_utils_api_key_getter_path, expected_model, simulated_response_template_part", [
    ("openai", "prompthelix.utils.llm_utils.get_openai_api_key", "gpt-3.5-turbo", "Simulated OpenAI API response"),
    ("anthropic", "prompthelix.utils.llm_utils.get_anthropic_api_key", "claude-2", "Simulated Claude API response"),
    ("google", "prompthelix.utils.llm_utils.get_google_api_key", "gemini-pro", "Simulated Google API response"),
])
def test_call_llm_api_dispatches_successfully_and_runs_simulated_provider_call(
    provider_name, llm_utils_api_key_getter_path, expected_model, simulated_response_template_part, sample_prompt_text, db_session
):
    # Patching get_..._api_key where it is looked up by call_llm_api (i.e., in llm_utils's namespace)
    # No patch on the specific provider call function (e.g. call_openai_api) for this test.
    with patch(llm_utils_api_key_getter_path, return_value="fake_api_key", autospec=True) as mock_get_key_in_llm_utils:

        # When model is None, call_llm_api uses the default model for the provider
        result = llm_utils.call_llm_api(prompt=sample_prompt_text, provider=provider_name, model=None, db=db_session)

        # Assert that the key getter (mocked in llm_utils) was called by llm_utils.call_llm_api
        mock_get_key_in_llm_utils.assert_called_once_with(db_session)

        # Assert that the result matches the expected simulated output from the real specific provider function
        expected_response = f"{simulated_response_template_part} for model {expected_model}: '{sample_prompt_text}'"
        assert result == expected_response

# Test `call_llm_api` for missing API keys
@pytest.mark.parametrize("provider_name, llm_utils_api_key_getter_path", [
    ("openai", "prompthelix.utils.llm_utils.get_openai_api_key"),
    ("anthropic", "prompthelix.utils.llm_utils.get_anthropic_api_key"),
    ("google", "prompthelix.utils.llm_utils.get_google_api_key"),
])
def test_call_llm_api_raises_error_if_key_missing(
    provider_name, llm_utils_api_key_getter_path, sample_prompt_text, db_session
):
    # Patch the key getter where it's used (in llm_utils)
    with patch(llm_utils_api_key_getter_path, return_value=None, autospec=True) as mock_get_key_in_llm_utils:
        raw_error_message = ""
        if provider_name == "openai":
            raw_error_message = "OpenAI API key not configured. Please set it in the settings or environment."
        elif provider_name == "anthropic":
            raw_error_message = "Anthropic (Claude) API key not configured. Please set it in the settings or environment."
        elif provider_name == "google":
            raw_error_message = "Google API key not configured. Please set it in the settings or environment."

        expected_error_message = re.escape(raw_error_message)

        with pytest.raises(ValueError, match=expected_error_message):
            llm_utils.call_llm_api(sample_prompt_text, provider=provider_name, model="test_model", db=db_session)

        mock_get_key_in_llm_utils.assert_called_once_with(db_session)


# Test specific provider callers for missing API keys
@patch("prompthelix.utils.llm_utils.get_openai_api_key", return_value=None, autospec=True)
def test_call_openai_api_raises_error_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    with pytest.raises(ValueError, match=re.escape("OpenAI API key not configured. Please set it in the settings or environment.")):
        llm_utils.call_openai_api(sample_prompt_text, model="test_model", db=db_session)
    mock_get_key.assert_called_once_with(db_session)

@patch("prompthelix.utils.llm_utils.get_anthropic_api_key", return_value=None, autospec=True)
def test_call_claude_api_raises_error_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    with pytest.raises(ValueError, match=re.escape("Anthropic (Claude) API key not configured. Please set it in the settings or environment.")):
        llm_utils.call_claude_api(sample_prompt_text, model="test_model", db=db_session)
    mock_get_key.assert_called_once_with(db_session)

@patch("prompthelix.utils.llm_utils.get_google_api_key", return_value=None, autospec=True)
def test_call_google_api_raises_error_if_key_missing_direct(mock_get_key, sample_prompt_text, db_session):
    with pytest.raises(ValueError, match=re.escape("Google API key not configured. Please set it in the settings or environment.")):
        llm_utils.call_google_api(sample_prompt_text, model="test_model", db=db_session)
    mock_get_key.assert_called_once_with(db_session)


# Test `call_llm_api` for an unsupported provider
def test_call_llm_api_unsupported_provider(sample_prompt_text, db_session):
    with pytest.raises(ValueError, match="Unsupported LLM provider: unknown_provider"):
        llm_utils.call_llm_api(sample_prompt_text, provider="unknown_provider", model="test_model", db=db_session)


# Test `list_available_llms`
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
