import pytest
from unittest.mock import patch, call
from prompthelix.tests.test_llm_connectivity import test_llm_connectivity

# Since the original script uses print for output and pytest.fail for errors,
# we'll need to capture stdout/stderr or adapt the script if we want to assert specific error messages
# For now, we will focus on the call_llm_api mock and its behavior.

class TestLLMConnectivity:

    @pytest.fixture
    def mock_call_llm_api(self, mocker):
        return mocker.patch('prompthelix.tests.test_llm_connectivity.call_llm_api')

    def test_successful_connection(self, mock_call_llm_api, capsys):
        """
        Tests a successful connection scenario.
        """
        mock_call_llm_api.return_value = "Test response"
        provider = "test_provider"
        model = "test_model"

        test_llm_connectivity(provider, model)

        mock_call_llm_api.assert_called_once_with("Hello, this is a test.", provider, model)

        captured = capsys.readouterr()
        assert f"LLM Provider: {provider}" in captured.out
        assert f"Model: {model}" in captured.out
        assert "Response: Test response" in captured.out
        # The original script's assertions will run, if they pass, this test passes.

    def test_api_key_error_simulation(self, mock_call_llm_api, capsys):
        """
        Tests a scenario where call_llm_api raises a ValueError (simulating an API key issue).
        The original script catches Exception and calls pytest.fail.
        """
        provider = "error_provider"
        model = "error_model"
        mock_call_llm_api.side_effect = ValueError("Simulated API Key Error")

        with pytest.raises(pytest.fail.Exception): # Check if pytest.fail was called
             test_llm_connectivity(provider, model)

        mock_call_llm_api.assert_called_once_with("Hello, this is a test.", provider, model)

        captured = capsys.readouterr()
        assert f"Error during LLM API call: Simulated API Key Error" in captured.out
        # The original script's pytest.fail will be triggered.

    def test_empty_response(self, mock_call_llm_api, capsys):
        """
        Tests a scenario where the LLM API returns an empty string.
        The original script's assertion `assert response != ""` should fail.
        """
        mock_call_llm_api.return_value = ""
        provider = "empty_provider"
        model = "empty_model"

        with pytest.raises(AssertionError, match="Response should not be empty"):
            test_llm_connectivity(provider, model)

        mock_call_llm_api.assert_called_once_with("Hello, this is a test.", provider, model)

        captured = capsys.readouterr()
        assert f"LLM Provider: {provider}" in captured.out
        assert f"Model: {model}" in captured.out
        assert "Response: " in captured.out # Response is empty

    def test_none_response(self, mock_call_llm_api, capsys):
        """
        Tests a scenario where the LLM API returns None.
        The original script's assertion `assert response is not None` should fail.
        """
        mock_call_llm_api.return_value = None
        provider = "none_provider"
        model = "none_model"

        with pytest.raises(AssertionError, match="Response should not be None"):
            test_llm_connectivity(provider, model)

        mock_call_llm_api.assert_called_once_with("Hello, this is a test.", provider, model)

        captured = capsys.readouterr()
        assert f"LLM Provider: {provider}" in captured.out
        assert f"Model: {model}" in captured.out
        assert "Response: None" in captured.out

# To run these tests, you would typically use pytest in the terminal:
# pytest prompthelix/tests/unit/test_llm_connectivity_tool.py
# Ensure that `prompthelix.utils.llm_utils` is available in the PYTHONPATH
# and `prompthelix.tests.test_llm_connectivity.call_llm_api` can be patched.

# Note: The original script `test_llm_connectivity.py` has its own assertions
# and calls `pytest.fail`. These tests are designed to work with that structure.
# If `test_llm_connectivity` were to return values or raise specific custom exceptions,
# these tests could be more direct in their assertions.
