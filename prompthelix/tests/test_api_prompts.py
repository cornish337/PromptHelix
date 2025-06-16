import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession # For db_session type hint

from prompthelix.schemas import PromptCreate, PromptUpdate, PromptVersionCreate, PromptVersionUpdate
from prompthelix.tests.utils import get_auth_headers # Import the auth helper

# Helper function to create a prompt and return its ID (replaces global created_ids)
def create_test_prompt_via_api(client: TestClient, auth_headers: dict, name: str = "Test Prompt", description: str = "A test description") -> int:
    response = client.post(
        "/api/prompts",
        json={"name": name, "description": description},
        headers=auth_headers
    )
    assert response.status_code == 200 # Or 201 if your route returns that
    return response.json()["id"]

def create_test_prompt_version_via_api(client: TestClient, auth_headers: dict, prompt_id: int, content: str = "Version content", params: dict = None) -> int:
    payload = {"content": content}
    if params:
        payload["parameters_used"] = params
    response = client.post(
        f"/api/prompts/{prompt_id}/versions",
        json=payload,
        headers=auth_headers
    )
    assert response.status_code == 200 # Or 201
    return response.json()["id"]


def test_create_and_get_prompt(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_name = "My Test Prompt"
    prompt_desc = "Description for my test prompt."

    # Create Prompt
    response_create = client.post(
        "/api/prompts",
        json={"name": prompt_name, "description": prompt_desc},
        headers=auth_headers
    )
    assert response_create.status_code == 200
    created_data = response_create.json()
    assert created_data["name"] == prompt_name
    assert created_data["description"] == prompt_desc
    prompt_id = created_data["id"]
    assert prompt_id is not None

    # Get Specific Prompt
    response_get = client.get(f"/api/prompts/{prompt_id}", headers=auth_headers) # Assuming get might also need auth or be public
    assert response_get.status_code == 200
    get_data = response_get.json()
    assert get_data["id"] == prompt_id
    assert get_data["name"] == prompt_name

def test_list_prompts(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id1 = create_test_prompt_via_api(client, auth_headers, name="List Prompt 1")
    prompt_id2 = create_test_prompt_via_api(client, auth_headers, name="List Prompt 2")

    response = client.get("/api/prompts", headers=auth_headers) # Assuming get might also need auth or be public
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2 # There might be other prompts from other tests if DB is not fully isolated per test run

    ids_in_response = [p["id"] for p in data]
    assert prompt_id1 in ids_in_response
    assert prompt_id2 in ids_in_response

def test_update_prompt(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Before Update")

    update_payload = PromptUpdate(name="After Update", description="Updated Description").model_dump(exclude_unset=True)
    response_update = client.put(
        f"/api/prompts/{prompt_id}",
        json=update_payload,
        headers=auth_headers
    )
    assert response_update.status_code == 200
    updated_data = response_update.json()
    assert updated_data["name"] == "After Update"
    assert updated_data["description"] == "Updated Description"

def test_delete_prompt(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="To Be Deleted")

    response_delete = client.delete(f"/api/prompts/{prompt_id}", headers=auth_headers)
    assert response_delete.status_code == 200 # Assuming 200 on successful delete

    response_get_after_delete = client.get(f"/api/prompts/{prompt_id}", headers=auth_headers)
    assert response_get_after_delete.status_code == 404


# --- Prompt Version Tests ---

def test_create_and_get_prompt_version(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Prompt For Versions")

    version_content = "This is version 1 content."
    version_params = {"temp": 0.7}

    # Create Version
    response_create_version = client.post(
        f"/api/prompts/{prompt_id}/versions",
        json={"content": version_content, "parameters_used": version_params},
        headers=auth_headers
    )
    assert response_create_version.status_code == 200
    created_version_data = response_create_version.json()
    assert created_version_data["content"] == version_content
    assert created_version_data["prompt_id"] == prompt_id
    assert created_version_data["parameters_used"] == version_params
    version_id = created_version_data["id"]
    assert version_id is not None
    assert created_version_data["version_number"] == 1 # Assuming first version

    # Get Specific Version
    response_get_version = client.get(f"/api/prompt_versions/{version_id}", headers=auth_headers)
    assert response_get_version.status_code == 200
    get_version_data = response_get_version.json()
    assert get_version_data["id"] == version_id
    assert get_version_data["content"] == version_content

def test_list_versions_for_prompt(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Prompt For Listing Versions")

    v1_id = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="V1")
    v2_id = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="V2")

    response = client.get(f"/api/prompts/{prompt_id}/versions", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    ids_in_response = [v["id"] for v in data]
    assert v1_id in ids_in_response
    assert v2_id in ids_in_response

def test_update_prompt_version(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Prompt For Version Update")
    version_id = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="Old Content")

    update_payload = PromptVersionUpdate(content="New Updated Content", fitness_score=0.99).model_dump(exclude_unset=True)
    response_update = client.put(
        f"/api/prompt_versions/{version_id}",
        json=update_payload,
        headers=auth_headers
    )
    assert response_update.status_code == 200
    updated_data = response_update.json()
    assert updated_data["content"] == "New Updated Content"
    assert updated_data["fitness_score"] == 0.99

def test_delete_prompt_version(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Prompt For Version Delete")
    version_id_to_delete = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="To Be Deleted")
    # version_id_to_keep = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="To Be Kept")


    response_delete = client.delete(f"/api/prompt_versions/{version_id_to_delete}", headers=auth_headers)
    assert response_delete.status_code == 200

    response_get_after_delete = client.get(f"/api/prompt_versions/{version_id_to_delete}", headers=auth_headers)
    assert response_get_after_delete.status_code == 404

    # Check if the other version is still there
    # response_get_kept = client.get(f"/api/prompt_versions/{version_id_to_keep}", headers=auth_headers)
    # assert response_get_kept.status_code == 200


def test_unauthenticated_access_to_protected_prompt_routes(client: TestClient, db_session: SQLAlchemySession):
    # No auth_headers passed
    response_create = client.post("/api/prompts", json={"name": "Unauth Create", "description": "..."})
    assert response_create.status_code == 401

    # Assume a prompt ID 1 exists for other tests; real ID doesn't matter for unauth check
    response_put = client.put("/api/prompts/1", json={"name": "Unauth Update"})
    assert response_put.status_code == 401

    response_delete = client.delete("/api/prompts/1")
    assert response_delete.status_code == 401

    response_create_version = client.post("/api/prompts/1/versions", json={"content": "Unauth Version"})
    assert response_create_version.status_code == 401

    response_put_version = client.put("/api/prompt_versions/1", json={"content": "Unauth Update Version"})
    assert response_put_version.status_code == 401

    response_delete_version = client.delete("/api/prompt_versions/1")
    assert response_delete_version.status_code == 401
