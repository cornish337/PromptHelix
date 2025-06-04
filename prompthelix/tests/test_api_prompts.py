from fastapi.testclient import TestClient
# client will be provided by the test_client fixture from conftest.py

# To store created IDs across test functions in this module
created_ids = {"prompt_id": None}


def test_create_prompt(test_client: TestClient):
    response = test_client.post(
        "/api/prompts",
        json={"name": "Test Prompt", "description": "A test prompt description"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Test Prompt"
    assert data["description"] == "A test prompt description"
    assert "id" in data
    created_ids["prompt_id"] = data["id"] # Save for subsequent tests

def test_get_prompts(test_client: TestClient):
    response = test_client.get("/api/prompts")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    # Check if the previously created prompt is in the list
    if created_ids["prompt_id"]:
        assert any(p["id"] == created_ids["prompt_id"] for p in data)

def test_get_single_prompt(test_client: TestClient):
    prompt_id = created_ids["prompt_id"]
    assert prompt_id is not None, "Prompt ID not set from create test"
    response = test_client.get(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == prompt_id
    assert data["name"] == "Test Prompt"

def test_update_prompt(test_client: TestClient):
    prompt_id = created_ids["prompt_id"]
    assert prompt_id is not None, "Prompt ID not set from create test"
    response = test_client.put(
        f"/api/prompts/{prompt_id}",
        json={"name": "Updated Test Prompt", "description": "Updated description"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Updated Test Prompt"
    assert data["description"] == "Updated description"

    # Verify by getting the prompt again
    response_get = test_client.get(f"/api/prompts/{prompt_id}")
    assert response_get.status_code == 200
    data_get = response_get.json()
    assert data_get["name"] == "Updated Test Prompt"

def test_create_prompt_version(test_client: TestClient):
    prompt_id = created_ids["prompt_id"]
    assert prompt_id is not None, "Prompt ID not set from create test"
    response = test_client.post(
        f"/api/prompts/{prompt_id}/versions",
        json={"content": "This is version 1 content.", "parameters_used": {"temp": 0.7}},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "This is version 1 content."
    assert data["prompt_id"] == prompt_id
    assert data["version_number"] == 1 # Assuming this is the first version
    assert data["parameters_used"]["temp"] == 0.7

    # Create another version to check version_number increment
    response_v2 = test_client.post(
        f"/api/prompts/{prompt_id}/versions",
        json={"content": "This is version 2 content."},
    )
    assert response_v2.status_code == 200
    data_v2 = response_v2.json()
    assert data_v2["version_number"] == 2


def test_get_prompt_with_versions(test_client: TestClient):
    prompt_id = created_ids["prompt_id"]
    assert prompt_id is not None, "Prompt ID not set from create test"
    response = test_client.get(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == prompt_id
    assert "versions" in data
    assert len(data["versions"]) == 2 # From test_create_prompt_version
    assert data["versions"][0]["version_number"] == 1 # Check if sorted or default order
    assert data["versions"][1]["version_number"] == 2


def test_delete_prompt(test_client: TestClient):
    prompt_id = created_ids["prompt_id"]
    assert prompt_id is not None, "Prompt ID not set from create test"
    response = test_client.delete(f"/api/prompts/{prompt_id}")
    assert response.status_code == 200 # Assuming delete returns the deleted object or 200/204

    # Verify by trying to get the prompt again
    response_get = test_client.get(f"/api/prompts/{prompt_id}")
    assert response_get.status_code == 404 # Expect Not Found
    created_ids["prompt_id"] = None # Clear the ID as it's deleted
