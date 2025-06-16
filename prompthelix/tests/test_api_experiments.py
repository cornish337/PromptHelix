import pytest
from fastapi.testclient import TestClient
from prompthelix.main import app # main FastAPI app
from prompthelix import schemas

# Using the test_client fixture from conftest.py which handles DB setup
# client = TestClient(app) # Not needed if using test_client fixture directly in tests

def test_run_experiment_creates_new_prompt_and_version(test_client, db_session):
    experiment_payload = {
        "task_description": "Generate a short story about a robot learning to paint",
        "keywords": ["robot", "art", "creativity"],
        "num_generations": 1, # Small for testing
        "population_size": 2, # Small for testing
        "elitism_count": 1,   # Small for testing
        "prompt_name": "Robot Painting Story Prompt (Test)",
        "prompt_description": "Test prompt for GA experiment creating a new prompt."
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
        "parent_prompt_id": parent_prompt.id
    }

    response = test_client.post("/api/experiments/run-ga", json=experiment_payload_existing)
    assert response.status_code == 200, response.text

    result_version_data_existing = response.json()
    assert result_version_data_existing["prompt_id"] == parent_prompt.id
    assert "task_description" in result_version_data_existing["parameters_used"]
    assert result_version_data_existing["parameters_used"]["task_description"] == experiment_payload_existing["task_description"]


    # Verify in DB
    created_version = crud.get_prompt_version_by_id(db_session, version_id=result_version_data_existing["id"])
    assert created_version is not None
    assert created_version.prompt_id == parent_prompt.id

    # Check that prompt_name and prompt_description were not part of the GA params for this version
    # as they are attributes of the parent prompt.
    assert "prompt_name" not in created_version.parameters_used
    assert "prompt_description" not in created_version.parameters_used
    # parent_prompt_id is also not stored in parameters_used, it's used to link the version.
    assert "parent_prompt_id" not in created_version.parameters_used

    # Verify the parent prompt's name and description were not changed
    retrieved_parent_prompt = crud.get_prompt(db_session, prompt_id=parent_prompt.id)
    assert retrieved_parent_prompt.name == parent_prompt_create.name
    assert retrieved_parent_prompt.description == parent_prompt_create.description
