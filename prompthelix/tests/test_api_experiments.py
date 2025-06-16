import pytest

from unittest.mock import patch, MagicMock
from prompthelix.enums import ExecutionMode
# from prompthelix.genetics.engine import PromptChromosome # Not strictly needed if using MagicMock without spec for return
# from prompthelix.schemas import PromptVersion # Not strictly needed for these tests if just checking status and basic fields


def test_run_experiment_creates_new_prompt_and_version(test_client, db_session):
    experiment_payload = {
        "task_description": "Generate a short story about a robot learning to paint",
        "keywords": ["robot", "art", "creativity"],

        "num_generations": 1, # Keep low for faster tests
        "population_size": 2, # Keep low for faster tests
        "elitism_count": 1,   # Keep low for faster tests
        "prompt_name": "GA Test - New Prompt",
        "prompt_description": "Result of a GA experiment creating a new prompt.",
        "execution_mode": ExecutionMode.REAL.value # Added

    }

    response = test_client.post("/api/experiments/run-ga", json=experiment_payload)

    assert response.status_code == 200, response.text

    result_version_data = response.json()
    assert "id" in result_version_data
    assert "prompt_id" in result_version_data
    assert result_version_data["content"] is not None
    assert result_version_data["fitness_score"] is not None
    # Ensure GA parameters are stored in the version's parameters_used field
    assert "task_description" in result_version_data["parameters_used"]
    assert result_version_data["parameters_used"]["task_description"] == experiment_payload["task_description"]
    assert result_version_data["parameters_used"]["num_generations"] == experiment_payload["num_generations"]


    # Verify in DB
    from prompthelix.api import crud

    created_prompt = crud.get_prompt(db_session, prompt_id=result_version_data["prompt_id"])
    assert created_prompt is not None
    assert created_prompt.name == experiment_payload["prompt_name"]
    assert created_prompt.description == experiment_payload["prompt_description"]

    created_version = crud.get_prompt_version_by_id(db_session, version_id=result_version_data["id"])
    assert created_version is not None
    assert created_version.prompt_id == created_prompt.id
    assert created_version.content == result_version_data["content"]
    assert created_version.fitness_score == result_version_data["fitness_score"]
    assert created_version.parameters_used["keywords"] == experiment_payload["keywords"]

def test_run_experiment_associates_with_existing_prompt(test_client, db_engine, db_session): # Added db_engine
    from prompthelix.api import crud
    from sqlalchemy.orm import sessionmaker

    # Create parent prompt in a separate session that is committed
    SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    db_setup_session = SessionLocalTest()
    try:
        parent_prompt_create = schemas.PromptCreate(name="Parent Prompt for Experiment Test", description="A parent prompt.")
        # crud.create_prompt already commits, so this should be fine.
        parent_prompt = crud.create_prompt(db=db_setup_session, prompt=parent_prompt_create)
        # db_setup_session.commit() # crud.create_prompt handles its own commit
    finally:
        db_setup_session.close()

    assert parent_prompt.id is not None, "Parent prompt should have an ID after creation"

    experiment_payload_existing = {
        "task_description": "Generate a haiku about nature",
        "keywords": ["nature", "haiku", "serenity"],
        "num_generations": 1,
        "population_size": 2,
        "elitism_count": 1,

        "parent_prompt_id": shared_prompt_id_for_experiment_test,
        "execution_mode": ExecutionMode.REAL.value # Added
    }

    response = test_client.post("/api/experiments/run-ga", json=experiment_params)
    assert response.status_code == 200, f"API call failed: {response.text}"
    data = response.json()

    assert "id" in data
    assert data["content"]
    assert "fitness_score" in data
    assert data["prompt_id"] == shared_prompt_id_for_experiment_test

    # Verify the version was added to the existing prompt
    prompt_response = test_client.get(f"/api/prompts/{shared_prompt_id_for_experiment_test}")
    assert prompt_response.status_code == 200
    prompt_data = prompt_response.json()
    # The prompt now should have its original versions (if any) + this new one.
    # If test_create_prompt_for_experiment_tests created a clean prompt, it will have 1 version.
    # If it's run after other tests that might have added versions to it, this count needs care.
    # For simplicity, assume it's 1 if this test runs immediately after its setup.
    # A better way is to count versions before and after.

    found_new_version = False
    for v in prompt_data["versions"]:
        if v["id"] == data["id"]: # data["id"] is the ID of the new PromptVersion
            found_new_version = True
            break
    assert found_new_version, "New version from experiment not found in parent prompt's versions list."


def test_run_ga_experiment_invalid_parent_prompt(test_client: TestClient):
    invalid_prompt_id = 999999
    experiment_params = {
        "task_description": "Test with invalid parent.",
        "keywords": ["test"],
        "num_generations": 1, "population_size": 2, "elitism_count": 1,
        "parent_prompt_id": invalid_prompt_id,
        "execution_mode": ExecutionMode.REAL.value # Added
    }
    response = test_client.post("/api/experiments/run-ga", json=experiment_params)
    assert response.status_code == 404 # Expect Not Found for parent_prompt_id
    # Ensure the error message is somewhat informative if possible
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
def test_cleanup_experiment_parent_prompt(test_client: TestClient):
    if shared_prompt_id_for_experiment_test:
        response = test_client.delete(f"/api/prompts/{shared_prompt_id_for_experiment_test}")
        # Assert 200 or 404 (if already deleted by another test, which shouldn't happen with this structure)
        assert response.status_code in [200, 404]
    # No critical assertions needed here, just cleanup.
    pass


# New tests with mocked main_ga_loop
@patch('prompthelix.api.routes.main_ga_loop')
def test_run_ga_experiment_api_test_mode(mock_main_ga_loop, test_client: TestClient):
    # Configure mock_main_ga_loop to return a mock PromptChromosome
    mock_chromosome = MagicMock()
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
    response = test_client.post("/api/experiments/run-ga", json=payload)

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
def test_run_ga_experiment_api_real_mode_mocked_loop(mock_main_ga_loop, test_client: TestClient):
    # Configure mock_main_ga_loop for REAL mode specifics if any
    mock_chromosome = MagicMock()
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
    response = test_client.post("/api/experiments/run-ga", json=payload)

    assert response.status_code == 200, f"API call failed: {response.text}"
    response_data = response.json()
    assert response_data["content"] == "Real prompt content from REAL mode mock"
    assert response_data["fitness_score"] == 0.88

    # Check that main_ga_loop was called with the correct execution_mode
    mock_main_ga_loop.assert_called_once()
    args, kwargs = mock_main_ga_loop.call_args
    assert kwargs.get('execution_mode') == ExecutionMode.REAL
    assert kwargs.get('keywords') == payload["keywords"]

