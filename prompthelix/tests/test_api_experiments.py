from fastapi.testclient import TestClient
import pytest
from fastapi.testclient import TestClient

from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient # Import TestClient
from prompthelix.enums import ExecutionMode
from prompthelix import schemas # Moved import here
from prompthelix.genetics.engine import PromptChromosome # Not strictly needed if using MagicMock without spec for return
# from prompthelix.schemas import PromptVersion # Not strictly needed for these tests if just checking status and basic fields

# client will be provided by the test_client fixture from conftest.py

# Store ID of a prompt created for testing experiment association
shared_prompt_id_for_experiment_test = None

from sqlalchemy.orm import Session as SQLAlchemySession # For db_session type hint
from prompthelix.tests.utils import get_auth_headers

def test_create_prompt_for_experiment_tests(test_client: TestClient, db_session: SQLAlchemySession):
    """Helper to create a prompt to be used as a parent in another test."""
    global shared_prompt_id_for_experiment_test
    auth_headers = get_auth_headers(test_client, db_session)
    response = test_client.post(
        "/api/prompts",
        json={"name": "Experiment Parent Prompt", "description": "Prompt for GA experiment to attach to"},
        headers=auth_headers
    )
    assert response.status_code == 200, f"Failed to create prompt: {response.text}"
    data = response.json()
    shared_prompt_id_for_experiment_test = data["id"]
    assert shared_prompt_id_for_experiment_test is not None

def test_run_ga_experiment_new_prompt(test_client: TestClient, db_session: SQLAlchemySession):
    experiment_params = {
        "task_description": "Generate a short story about a robot learning to paint.",
        "keywords": ["robot", "art", "creativity"],
        "num_generations": 1, # Keep low for faster tests
        "population_size": 2, # Keep low for faster tests
        "elitism_count": 1,   # Keep low for faster tests
        "prompt_name": "GA Test - New Prompt",
        "prompt_description": "Result of a GA experiment creating a new prompt.",
        "execution_mode": ExecutionMode.REAL.value # Added
    }
    auth_headers = get_auth_headers(test_client, db_session)
    response = test_client.post("/api/experiments/run-ga", json=experiment_params, headers=auth_headers)

    assert response.status_code == 200, f"API call failed: {response.text}"
    data = response.json()

    assert "id" in data # PromptVersion ID
    assert data["content"] # Check that some content was generated
    assert "fitness_score" in data # Even if None, key should be there
    assert "parameters_used" in data

    # Check if parameters_used reflect input, excluding association fields
    params_in_response = data["parameters_used"]
    assert params_in_response["task_description"] == experiment_params["task_description"]
    assert params_in_response["keywords"] == experiment_params["keywords"]
    assert params_in_response["num_generations"] == experiment_params["num_generations"]
    # ... other params as needed

    # Verify the new prompt was created (by checking its name in the version's parent prompt)
    new_prompt_id = data["prompt_id"]
    prompt_response = test_client.get(f"/api/prompts/{new_prompt_id}", headers=auth_headers) # Added auth_headers
    assert prompt_response.status_code == 200
    prompt_data = prompt_response.json()
    assert prompt_data["name"] == experiment_params["prompt_name"]
    assert len(prompt_data["versions"]) >= 1 # Ensure at least one version

def test_run_ga_experiment_existing_prompt(test_client: TestClient, db_session: SQLAlchemySession, experiment_prompt: int):
    parent_prompt_id = experiment_prompt # Use ID from fixture
    auth_headers = get_auth_headers(test_client, db_session)

    # Ensure the parent_prompt_id from fixture is valid by trying to fetch it.
    check_prompt_response = test_client.get(f"/api/prompts/{parent_prompt_id}", headers=auth_headers)
    assert check_prompt_response.status_code == 200, "Experiment prompt fixture did not create a valid prompt that could be fetched."

    experiment_payload = { # Define experiment_payload here
        "task_description": "Generate a limerick about Python",
        "keywords": ["python", "limerick", "code"],
        "num_generations": 1, "population_size": 2, "elitism_count": 1,
        "parent_prompt_id": parent_prompt_id,
        "execution_mode": ExecutionMode.REAL.value
    }
    response = test_client.post("/api/experiments/run-ga", json=experiment_payload, headers=auth_headers)
    assert response.status_code == 200, f"API call failed: {response.text}"
    result_version_data = response.json() # This is the new prompt version

    assert result_version_data["prompt_id"] == parent_prompt_id

    # Verify by fetching the parent prompt and checking its versions
    prompt_response_after_exp = test_client.get(f"/api/prompts/{parent_prompt_id}", headers=auth_headers)
    assert prompt_response_after_exp.status_code == 200
    prompt_data_after_exp = prompt_response_after_exp.json()

    found_new_version = any(v["id"] == result_version_data["id"] for v in prompt_data_after_exp["versions"])
    assert found_new_version, "New version from experiment not found in parent prompt's versions list."
    # Additional checks for content can be added if needed, e.g.
    # new_version_data = next(v for v in prompt_data_after_exp["versions"] if v["id"] == result_version_data["id"])
    # assert new_version_data["content"] == result_version_data["content"]
    # assert new_version_data["parameters_used"]["keywords"] == experiment_payload["keywords"]


def test_run_experiment_associates_with_existing_prompt(test_client: TestClient, db_session: SQLAlchemySession, experiment_prompt: int):
    # experiment_prompt fixture now provides the parent_prompt_id
    parent_prompt_id = experiment_prompt
    # Need to import schemas here or at the top of the file if not already present
    # from prompthelix import schemas # Make sure schemas is imported # Moved to top
    auth_headers = get_auth_headers(test_client, db_session)

    experiment_payload_existing = {
        "task_description": "Generate a haiku about nature",
        "keywords": ["nature", "haiku", "serenity"],
        "num_generations": 1,
        "population_size": 2,
        "elitism_count": 1,
        "parent_prompt_id": parent_prompt_id, # Use the ID from the fixture
        "execution_mode": ExecutionMode.REAL.value
    }

    response = test_client.post("/api/experiments/run-ga", json=experiment_payload_existing, headers=auth_headers)
    assert response.status_code == 200, f"API call failed: {response.text}"
    data = response.json()

    assert "id" in data
    assert data["content"]
    assert "fitness_score" in data
    assert data["prompt_id"] == parent_prompt_id # Check against the fixture ID

    # Verify the version was added to the existing prompt
    prompt_response = test_client.get(f"/api/prompts/{parent_prompt_id}", headers=auth_headers)
    assert prompt_response.status_code == 200
    prompt_data = prompt_response.json()

    found_new_version = False
    for v in prompt_data["versions"]:
        if v["id"] == data["id"]:
            found_new_version = True
            break
    assert found_new_version, "New version from experiment not found in parent prompt's versions list."


def test_run_ga_experiment_invalid_parent_prompt(test_client: TestClient, db_session: SQLAlchemySession):
    invalid_prompt_id = 999999
    experiment_params = {
        "task_description": "Test with invalid parent.",
        "keywords": ["test"],
        "num_generations": 1, "population_size": 2, "elitism_count": 1,
        "parent_prompt_id": invalid_prompt_id,
        "execution_mode": ExecutionMode.REAL.value # Added
    }
    auth_headers = get_auth_headers(test_client, db_session)
    response = test_client.post("/api/experiments/run-ga", json=experiment_params, headers=auth_headers)
    assert response.status_code == 404 # Expect Not Found for parent_prompt_id
    # Ensure the error message is somewhat informative
    error_data = response.json()
    assert "detail" in error_data
    assert str(invalid_prompt_id) in error_data["detail"]
    assert "not found" in error_data["detail"].lower()

# It might be good to also test the case where main_ga_loop returns None or not a PromptChromosome
# This would require mocking main_ga_loop within the API route, which is more involved.
# For now, assume main_ga_loop behaves as expected (returns PromptChromosome or raises error).

# Cleanup the prompt created for experiment tests
# This is tricky with Pytest's execution order if not using explicit ordering plugins.
# A fixture with finalizer could also do this, or a separate cleanup script/test.
# For now, this test will run last if named appropriately or if it's the last one defined.
# However, pytest doesn't guarantee order by default.
# A better way is to use pytest-ordering or specific fixtures for setup/teardown of this shared resource.

# @pytest.mark.run(after='test_run_ga_experiment_existing_prompt') # Example if using pytest-ordering
# def test_cleanup_experiment_parent_prompt(test_client: TestClient): # TestClient type hint removed for consistency
# This test is no longer needed as the fixture handles cleanup.
# If a specific test for cleanup operation is desired, it should be designed differently,
# perhaps by creating a prompt and then immediately trying to delete it.
# For now, commenting out as its original purpose is covered by the fixture.
# pass


# New tests with mocked main_ga_loop
@patch('prompthelix.api.routes.main_ga_loop')
def test_run_ga_experiment_api_test_mode(mock_main_ga_loop, test_client: TestClient, db_session: SQLAlchemySession):
    # Configure mock_main_ga_loop to return a mock PromptChromosome
    mock_chromosome = MagicMock(spec=PromptChromosome)
    mock_chromosome.to_prompt_string.return_value = "Test prompt content from TEST mode"
    mock_chromosome.fitness_score = 0.95
    mock_main_ga_loop.return_value = mock_chromosome

    payload = {
        "task_description": "API Test task for TEST mode",
        "keywords": ["api", "test", "mock"],
        "num_generations": 1, # These params will be passed to the mocked loop
        "population_size": 2,
        "elitism_count": 1,
        "prompt_name": "Test API Prompt (TEST mode)", # Create a new prompt
        "execution_mode": ExecutionMode.TEST.value
    }
    auth_headers = get_auth_headers(test_client, db_session) # Ensure this line is present and correct
    response = test_client.post("/api/experiments/run-ga", json=payload, headers=auth_headers) # Ensure headers are used

    assert response.status_code == 200, f"API call failed: {response.text}"
    response_data = response.json()
    assert response_data["content"] == "Test prompt content from TEST mode"
    assert response_data["fitness_score"] == 0.95

    # Check that main_ga_loop was called with the correct execution_mode
    mock_main_ga_loop.assert_called_once()
    args, kwargs = mock_main_ga_loop.call_args
    assert kwargs.get('execution_mode') == ExecutionMode.TEST
    assert kwargs.get('task_desc') == payload["task_description"]
    # db_session_for_tests fixture is not explicitly used here as CRUD operations are within the endpoint.
    # If direct DB verification were needed here, it would be passed.

@patch('prompthelix.api.routes.main_ga_loop')
def test_run_ga_experiment_api_real_mode_mocked_loop(mock_main_ga_loop, test_client: TestClient, db_session: SQLAlchemySession): # Added db_session
    # Configure mock_main_ga_loop for REAL mode specifics if any
    mock_chromosome = MagicMock(spec=PromptChromosome)
    mock_chromosome.to_prompt_string.return_value = "Real prompt content from REAL mode mock"
    mock_chromosome.fitness_score = 0.88
    mock_main_ga_loop.return_value = mock_chromosome

    payload = {
        "task_description": "API Test task for REAL mode (mocked loop)",
        "keywords": ["api", "real", "mock"],
        "num_generations": 1,
        "population_size": 2,
        "elitism_count": 1,
        "prompt_name": "Test API Prompt (REAL mode - mocked)",
        "execution_mode": ExecutionMode.REAL.value
    }
    auth_headers = get_auth_headers(test_client, db_session) # Corrected indentation and ensured presence
    response = test_client.post("/api/experiments/run-ga", json=payload, headers=auth_headers)

    assert response.status_code == 200, f"API call failed: {response.text}"
    response_data = response.json()
    assert response_data["content"] == "Real prompt content from REAL mode mock"
    assert response_data["fitness_score"] == 0.88

    # Check that main_ga_loop was called with the correct execution_mode
    mock_main_ga_loop.assert_called_once()
    args, kwargs = mock_main_ga_loop.call_args
    assert kwargs.get('execution_mode') == ExecutionMode.REAL
    assert kwargs.get('keywords') == payload["keywords"]
