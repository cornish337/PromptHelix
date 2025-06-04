from fastapi.testclient import TestClient
import pytest

# client will be provided by the test_client fixture from conftest.py

# Store ID of a prompt created for testing experiment association
shared_prompt_id_for_experiment_test = None

def test_create_prompt_for_experiment_tests(test_client: TestClient):
    """Helper to create a prompt to be used as a parent in another test."""
    global shared_prompt_id_for_experiment_test
    response = test_client.post(
        "/api/prompts",
        json={"name": "Experiment Parent Prompt", "description": "Prompt for GA experiment to attach to"},
    )
    assert response.status_code == 200
    data = response.json()
    shared_prompt_id_for_experiment_test = data["id"]
    assert shared_prompt_id_for_experiment_test is not None

def test_run_ga_experiment_new_prompt(test_client: TestClient):
    experiment_params = {
        "task_description": "Generate a short story about a robot learning to paint.",
        "keywords": ["robot", "art", "creativity"],
        "num_generations": 1, # Keep low for faster tests
        "population_size": 2, # Keep low for faster tests
        "elitism_count": 1,   # Keep low for faster tests
        "prompt_name": "GA Test - New Prompt",
        "prompt_description": "Result of a GA experiment creating a new prompt."
    }
    response = test_client.post("/api/experiments/run-ga", json=experiment_params)

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
    prompt_response = test_client.get(f"/api/prompts/{new_prompt_id}")
    assert prompt_response.status_code == 200
    prompt_data = prompt_response.json()
    assert prompt_data["name"] == experiment_params["prompt_name"]
    assert len(prompt_data["versions"]) == 1 # Should have one version, the result

def test_run_ga_experiment_existing_prompt(test_client: TestClient):
    assert shared_prompt_id_for_experiment_test is not None, "Parent prompt ID for experiment not set."

    experiment_params = {
        "task_description": "Write a poem about the sea.",
        "keywords": ["ocean", "waves", "blue"],
        "num_generations": 1,
        "population_size": 2,
        "elitism_count": 1,
        "parent_prompt_id": shared_prompt_id_for_experiment_test
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
        "parent_prompt_id": invalid_prompt_id
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
