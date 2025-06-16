import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import httpx # For spec
from sqlalchemy.orm import Session # For typing if needed for mocks

from prompthelix.main import app # Your FastAPI app
from prompthelix import schemas
from prompthelix.enums import ExecutionMode
from urllib.parse import urlparse, parse_qs # For robust query param checking


@patch('prompthelix.ui_routes.httpx.AsyncClient') # Target for patching
def test_run_experiment_ui_submit_success(MockedAsyncClient, client: TestClient): # Added client type hint
    # Mock the crud.get_prompts call, which might be called if an error occurs before redirect
    with patch('prompthelix.ui_routes.crud.get_prompts') as mock_get_prompts:
        mock_get_prompts.return_value = [] # Return an empty list

        # Configure the mock AsyncClient instance that the constructor will return
        mock_client_instance = MockedAsyncClient.return_value
        # Configure the __aenter__ method for 'async with' to return an object
        # that has an async 'post' method.
        mock_async_context_manager = mock_client_instance.__aenter__.return_value

        # Mock the response from the API call made by httpx.AsyncClient.post
        mock_api_response = MagicMock(spec=httpx.Response)
        mock_api_response.status_code = 200
        # The API is expected to return data that matches schemas.PromptVersion
        api_response_json = {
            "id": 123,
            "prompt_id": 1,
            "content": "Generated prompt from GA",
            "version_number": 2,
            # Ensure created_at is a string in ISO format as FastAPI would serialize it
            "created_at": "2023-10-26T10:00:00+00:00", # Example ISO format
            "parameters_used": {"param": "value"},
            "fitness_score": 0.95
        }
        mock_api_response.json.return_value = api_response_json

        # Configure the 'post' method of the async context manager
        mock_post_method = AsyncMock(return_value=mock_api_response)
        mock_async_context_manager.post = mock_post_method

        form_data = {
            "task_description": "Test experiment for UI fix",
            "keywords": "ui,fix,test", # Will be split into a list by the route
            "execution_mode": ExecutionMode.TEST.value,
            "num_generations": "5", # Form data are strings
            "population_size": "10",# Form data are strings
            "elitism_count": "2",  # Form data are strings
            "parent_prompt_id": "", # Optional int, empty string becomes None
            "prompt_name": "Test GA Prompt",
            "prompt_description": "A prompt generated via UI test."
        }

        # Simulate logged-in user by setting the auth cookie
        test_token = "test_auth_token_for_ui_call"
        cookies = {"prompthelix_access_token": test_token}

        # Make the request to the UI endpoint using the TestClient
        # The TestClient is synchronous but calls the async route handler correctly.
        response = client.post(
            "/ui/experiments/new",
            data=form_data,
            cookies=cookies, # Pass the cookies with the request
            follow_redirects=False
        )

        # --- Assertions ---
        assert response.status_code == 303, f"Expected redirect (303), got {response.status_code}. Response text: {response.text}"

        # Check that the mocked post method was called
        mock_post_method.assert_called_once()

        # Inspect arguments passed to the mocked httpx.AsyncClient.post
        args, kwargs = mock_post_method.call_args

        # **Crucial Check**: Ensure the URL is a string
        assert isinstance(args[0], str), f"URL should be a string, but got {type(args[0])}"

        # Optional: Check the content of the URL
        # TestClient's default base_url is http://testserver
        assert args[0].endswith("/api/experiments/run-ga"), f"URL '{args[0]}' does not end with '/api/experiments/run-ga'"

        # ----> New/Modified Assertions for Auth Header <----
        assert "headers" in kwargs, "Expected 'headers' in httpx call keyword arguments."
        assert "Authorization" in kwargs["headers"], "Expected 'Authorization' header in httpx call."
        assert kwargs["headers"]["Authorization"] == f"Bearer {test_token}", \
            f"Authorization header incorrect. Expected 'Bearer {test_token}', Got '{kwargs['headers']['Authorization']}'"
        # ----> End of New/Modified Assertions <----

        # Verify the JSON payload sent to the API
        expected_ga_params_payload = schemas.GAExperimentParams(
            task_description="Test experiment for UI fix",
            keywords=["ui", "fix", "test"], # Processed from form_data string
            execution_mode=ExecutionMode.TEST,
            num_generations=5, # Converted to int by Pydantic
            population_size=10, # Converted to int by Pydantic
            elitism_count=2,   # Converted to int by Pydantic
            parent_prompt_id=None, # Pydantic converts "" to None for Optional[int]
            prompt_name="Test GA Prompt",
            prompt_description="A prompt generated via UI test."
        ).model_dump(exclude_none=True) # Use .model_dump() for Pydantic v2+

        assert kwargs['json'] == expected_ga_params_payload, "GA parameters JSON payload mismatch"

        # Verify the redirect URL structure
        # The redirect URL is constructed using request.url_for, so it will be absolute
        actual_redirect_location = response.headers["location"]

        # Construct the expected path and query parameters for the redirect URL
        # The prompt_id and version_id come from the mocked API response
        expected_prompt_id = api_response_json['prompt_id']
        expected_version_id = api_response_json['id']

        expected_redirect_path = f"/ui/prompts/{expected_prompt_id}"
        assert expected_redirect_path in actual_redirect_location, \
            f"Redirect location '{actual_redirect_location}' does not contain path '{expected_redirect_path}'"

        # Check for query parameters more robustly

        parsed_redirect_url = urlparse(actual_redirect_location)
        query_params = parse_qs(parsed_redirect_url.query)

        assert query_params.get("new_version_id") == [str(expected_version_id)], \
            f"Redirect query param 'new_version_id' mismatch. Got {query_params.get('new_version_id')}"

        expected_message = f"New version (ID: {expected_version_id}) created successfully from experiment."
        assert query_params.get("message") == [expected_message], \
            f"Redirect query param 'message' mismatch. Got {query_params.get('message')}"


@patch('prompthelix.ui_routes.httpx.AsyncClient')
def test_run_experiment_ui_submit_no_token(MockedAsyncClient, client: TestClient):
    with patch('prompthelix.ui_routes.crud.get_prompts') as mock_get_prompts:
        mock_get_prompts.return_value = [] # For re-rendering the form page

        mock_client_instance = MockedAsyncClient.return_value
        mock_async_context_manager = mock_client_instance.__aenter__.return_value

        # Simulate API returning 401 when no token is provided
        mock_api_response = MagicMock(spec=httpx.Response)
        mock_api_response.status_code = 401
        mock_api_response.text = '{"detail":"Not authenticated"}' # Example 401 response text
        mock_api_response.json.return_value = {"detail":"Not authenticated"} # If .json() is called

        # Configure the client's post method to raise an HTTPStatusError for 401
        # This simulates how httpx would behave with response.raise_for_status()
        # Ensure the mock request object has a URL attribute
        mock_request = MagicMock(spec=httpx.Request)
        mock_request.url = "http://testserver/api/experiments/run-ga" # Example URL

        mock_post_method = AsyncMock(side_effect=httpx.HTTPStatusError(
            message="Client error '401 Unauthorized' for url http://testserver/api/experiments/run-ga",
            request=mock_request,
            response=mock_api_response
        ))
        mock_async_context_manager.post = mock_post_method

        form_data = { # Same form data as the success test
            "task_description": "Test experiment no token", "keywords": "ui,auth,test",
            "execution_mode": ExecutionMode.TEST.value, "num_generations": "5",
            "population_size": "10", "elitism_count": "2", "parent_prompt_id": "",
            "prompt_name": "Test GA No Token", "prompt_description": "A prompt generated via UI no token."
        }

        # Make the request *without* setting the auth cookie
        response = client.post(
            "/ui/experiments/new",
            data=form_data,
            follow_redirects=False # We expect to stay on the same page with an error
        )

        # Assert that we didn't redirect, but re-rendered the page (status 200) with an error
        assert response.status_code == 200, f"Expected status 200 (re-render with error), got {response.status_code}"

        # Check that the error message is present in the response content
        # This depends on how error messages are rendered in experiment.html
        content = response.text
        assert "Authentication failed" in content
        assert "Please ensure you are logged in and try again" in content

        # Verify that the httpx call was made without Authorization header
        mock_post_method.assert_called_once()
        _, kwargs = mock_post_method.call_args
        if "headers" in kwargs:
            assert "Authorization" not in kwargs["headers"], \
                "Authorization header should not be present when no token cookie is set."
        # If no headers dict is passed at all, kwargs might not have 'headers' key, which is also fine.


def test_get_login_page_ui(client: TestClient):
    """Test that the login page UI loads correctly."""
    response = client.get("/ui/login")
    assert response.status_code == 200
    content = response.text
    assert "<h1>Login</h1>" in content
    assert '<label for="username">Username:</label>' in content
    assert '<input type="text" id="username" name="username" required>' in content
    assert '<label for="password">Password:</label>' in content
    assert '<input type="password" id="password" name="password" required>' in content
    assert '<button type="submit">Login</button>' in content
    assert 'id="error-message"' in content


@patch('prompthelix.ui_routes.httpx.AsyncClient') # Patching at the right place
def test_run_experiment_ui_with_auth_cookie(MockedAsyncClient, client: TestClient, test_user_token: str):
    """
    Test submitting an experiment via UI after obtaining an auth token
    by simulating a login and using the token in a cookie.
    """
    access_token = test_user_token # Using the fixture

    # Configure the mock AsyncClient for the API call within the UI route
    mock_client_instance = MockedAsyncClient.return_value
    mock_async_context_manager = mock_client_instance.__aenter__.return_value

    mock_api_response = MagicMock(spec=httpx.Response)
    mock_api_response.status_code = 200
    api_response_json = {
        "id": 125, "prompt_id": 3, "content": "Generated by logged in user",
        "version_number": 1, "created_at": "2023-11-01T10:00:00+00:00",
        "parameters_used": {"simulated_param": "value"}, "fitness_score": 0.99
    }
    mock_api_response.json.return_value = api_response_json
    mock_post_method = AsyncMock(return_value=mock_api_response)
    mock_async_context_manager.post = mock_post_method

    # Mock crud.get_prompts for error case in experiment submission
    with patch('prompthelix.ui_routes.crud.get_prompts') as mock_get_prompts:
        mock_get_prompts.return_value = []

        form_data = {
            "task_description": "Test experiment with UI login token",
            "keywords": "login,cookie,test",
            "execution_mode": ExecutionMode.SIMULATION.value,
            "num_generations": "3",
            "population_size": "8",
            "elitism_count": "1",
            "parent_prompt_id": "",
            "prompt_name": "Test Logged In GA Prompt",
            "prompt_description": "A prompt generated via UI test after simulated login."
        }

        # Set the obtained access token in the cookies for the TestClient request
        cookies = {"prompthelix_access_token": access_token}

        response = client.post(
            "/ui/experiments/new",
            data=form_data,
            cookies=cookies, # Pass the token via cookie
            follow_redirects=False
        )

        assert response.status_code == 303, \
            f"Expected redirect (303), got {response.status_code}. Response: {response.text}"

        # Verify the API call was made with the Authorization header derived from the cookie
        mock_post_method.assert_called_once()
        _, kwargs = mock_post_method.call_args
        assert "headers" in kwargs and "Authorization" in kwargs["headers"], \
            "Authorization header missing in API call"
        assert kwargs["headers"]["Authorization"] == f"Bearer {access_token}", \
            "Authorization header did not match the token from simulated login"

        # Verify redirect location
        actual_redirect_location = response.headers["location"]
        expected_prompt_id = api_response_json['prompt_id']
        expected_version_id = api_response_json['id']
        expected_redirect_path = f"/ui/prompts/{expected_prompt_id}"

        assert expected_redirect_path in actual_redirect_location
        parsed_redirect_url = urlparse(actual_redirect_location)
        query_params = parse_qs(parsed_redirect_url.query)
        assert query_params.get("new_version_id") == [str(expected_version_id)]
        expected_message = f"New version (ID: {expected_version_id}) created successfully from experiment."
        assert query_params.get("message") == [expected_message]
