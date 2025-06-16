import pytest
from fastapi.testclient import TestClient # For type hinting test_client fixture
from sqlalchemy.orm import Session # For type hinting db_session fixture
import uuid # For unique names

from prompthelix import schemas # For request/response models
from prompthelix.api import crud # For direct DB checks
from prompthelix.enums import ExecutionMode # For GA params

def test_prompt_lifecycle(test_client: TestClient, db_session: Session):
    """
    Tests the complete lifecycle of a prompt and its version via API calls:
    Create -> Retrieve -> Add Version -> Retrieve with Version -> (DB Check) -> Delete
    """
    prompt_name = f"Integration Test Lifecycle Prompt - {uuid.uuid4()}" # Unique name
    prompt_description = "A prompt created during an integration test."

    # 1. Create a Prompt
    create_payload = schemas.PromptCreate(name=prompt_name, description=prompt_description)
    response = test_client.post("/api/prompts", json=create_payload.model_dump())

    assert response.status_code == 200, f"Failed to create prompt: {response.text}"
    created_prompt_data = response.json()
    assert created_prompt_data["name"] == prompt_name
    assert created_prompt_data["description"] == prompt_description
    assert "id" in created_prompt_data
    prompt_id = created_prompt_data["id"]

    # 2. Retrieve the Prompt
    response = test_client.get(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200, f"Failed to retrieve prompt: {response.text}"
    retrieved_prompt_data = response.json()
    assert retrieved_prompt_data["id"] == prompt_id
    assert retrieved_prompt_data["name"] == prompt_name
    assert retrieved_prompt_data["description"] == prompt_description
    assert "versions" in retrieved_prompt_data
    initial_versions_count = len(retrieved_prompt_data["versions"])

    # 3. Add a Version to the Prompt
    version_content = "Version 1 content for integration test."
    version_params = {"temp": 0.7, "source": "integration_test"}
    version_payload = schemas.PromptVersionCreate(
        content=version_content,
        parameters_used=version_params,
        fitness_score=0.85
    )
    response = test_client.post(f"/api/prompts/{prompt_id}/versions", json=version_payload.model_dump())
    assert response.status_code == 200, f"Failed to add version: {response.text}"
    created_version_data = response.json()
    assert created_version_data["content"] == version_content
    assert created_version_data["parameters_used"] == version_params
    assert created_version_data["fitness_score"] == 0.85
    assert "id" in created_version_data
    version_id = created_version_data["id"]
    assert created_version_data["prompt_id"] == prompt_id

    # 4. Retrieve the Prompt with Versions
    response = test_client.get(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200, f"Failed to retrieve prompt with versions: {response.text}"
    prompt_with_versions_data = response.json()
    assert len(prompt_with_versions_data["versions"]) == initial_versions_count + 1

    found_added_version = False
    for version in prompt_with_versions_data["versions"]:
        if version["id"] == version_id:
            assert version["content"] == version_content
            assert version["parameters_used"] == version_params
            assert version["fitness_score"] == 0.85
            found_added_version = True
            break
    assert found_added_version, f"Added version (ID: {version_id}) not found in prompt's versions list."

    # 5. Database Check
    db_prompt = crud.get_prompt(db=db_session, prompt_id=prompt_id)
    assert db_prompt is not None
    assert db_prompt.name == prompt_name
    assert db_prompt.description == prompt_description

    db_session.refresh(db_prompt)

    assert len(db_prompt.versions) == initial_versions_count + 1
    db_version_found = False
    for v in db_prompt.versions:
        if v.id == version_id:
            assert v.content == version_content
            assert v.parameters_used == version_params
            assert v.fitness_score == 0.85
            db_version_found = True
            break
    assert db_version_found, "Added version not found in DB prompt's versions."

    # 6. Cleanup: Delete the Prompt
    response = test_client.delete(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200, f"Failed to delete prompt: {response.text}"
    deleted_prompt_data = response.json()
    assert deleted_prompt_data["id"] == prompt_id

    response = test_client.get(f"/api/prompts/{prompt_id}")
    assert response.status_code == 404, "Prompt was not actually deleted (still found)."

    db_prompt_after_delete = crud.get_prompt(db=db_session, prompt_id=prompt_id)
    assert db_prompt_after_delete is None, "Prompt still found in DB after API delete."


def test_run_ga_experiment_integration_creates_new_prompt(test_client: TestClient, db_session: Session):
    """
    Tests running a GA experiment via API that creates a new prompt.
    Uses TEST execution mode.
    """
    experiment_prompt_name = f"GA Integration Test Prompt - {uuid.uuid4()}"
    experiment_task_desc = "Generate a short poem about unit testing."
    experiment_keywords = ["test", "poem", "pytest"]

    ga_payload = {
        "task_description": experiment_task_desc,
        "keywords": experiment_keywords,
        "num_generations": 1,
        "population_size": 2, # Small for speed
        "elitism_count": 1,   # Small for speed
        "execution_mode": ExecutionMode.TEST.value, # Use TEST mode
        "prompt_name": experiment_prompt_name,
        "prompt_description": "Prompt created by GA integration test."
    }

    # 2. Run Experiment API Call
    response = test_client.post("/api/experiments/run-ga", json=ga_payload)
    assert response.status_code == 200, f"Failed to run GA experiment: {response.text}"

    experiment_result_version_data = response.json()
    assert "id" in experiment_result_version_data # This is the version ID
    assert "prompt_id" in experiment_result_version_data
    assert experiment_result_version_data["content"] is not None
    assert experiment_result_version_data["fitness_score"] is not None
    assert "parameters_used" in experiment_result_version_data

    created_version_id = experiment_result_version_data["id"]
    parent_prompt_id = experiment_result_version_data["prompt_id"]

    # Verify parameters_used in the version
    params_in_version = experiment_result_version_data["parameters_used"]
    assert params_in_version["task_description"] == experiment_task_desc
    assert params_in_version["keywords"] == experiment_keywords
    assert params_in_version["num_generations"] == ga_payload["num_generations"]
    assert params_in_version["execution_mode"] == ExecutionMode.TEST.value # Check if mode is stored

    # 3. Database/API Verification
    # Retrieve the parent prompt
    response = test_client.get(f"/api/prompts/{parent_prompt_id}")
    assert response.status_code == 200, f"Failed to retrieve parent prompt {parent_prompt_id}: {response.text}"
    parent_prompt_data = response.json()
    assert parent_prompt_data["name"] == experiment_prompt_name
    assert parent_prompt_data["description"] == "Prompt created by GA integration test."

    # Verify the new version is present in the prompt's versions list
    assert "versions" in parent_prompt_data
    found_experiment_version = False
    for version in parent_prompt_data["versions"]:
        if version["id"] == created_version_id:
            assert version["content"] == experiment_result_version_data["content"]
            # Check a subset of parameters_used as the GA might add more (like chromosome_id)
            for key in ["task_description", "keywords", "num_generations", "execution_mode"]:
                assert version["parameters_used"][key] == ga_payload[key]
            found_experiment_version = True
            break
    assert found_experiment_version, f"Version {created_version_id} from GA experiment not found in prompt {parent_prompt_id}."

    # (Optional) DB Check
    db_prompt = crud.get_prompt(db=db_session, prompt_id=parent_prompt_id)
    assert db_prompt is not None
    assert db_prompt.name == experiment_prompt_name
    db_session.refresh(db_prompt)
    assert len(db_prompt.versions) > 0 # Should have at least one version

    db_version_found = False
    for v in db_prompt.versions:
        if v.id == created_version_id:
            assert v.content == experiment_result_version_data["content"]
            assert v.parameters_used["task_description"] == experiment_task_desc
            db_version_found = True
            break
    assert db_version_found, "Experiment version not found directly in DB prompt's versions."

    # 4. Cleanup
    response = test_client.delete(f"/api/prompts/{parent_prompt_id}")
    assert response.status_code == 200, f"Failed to delete prompt created by GA experiment: {response.text}"

    # Verify deletion
    response = test_client.get(f"/api/prompts/{parent_prompt_id}")
    assert response.status_code == 404, "Prompt created by GA was not actually deleted."
