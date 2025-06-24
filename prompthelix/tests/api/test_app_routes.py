import pytest
from fastapi.testclient import TestClient

from prompthelix.tests.utils import get_auth_headers
from prompthelix.schemas import User, Token, Prompt, PromptVersion

# Fixtures from prompthelix.tests.conftest are re-exported by tests/conftest.py


def test_create_user_returns_user_schema(test_client: TestClient):
    payload = {
        "username": "user1@example.com",
        "email": "user1@example.com",
        "password": "secret"
    }
    response = test_client.post("/users/", json=payload)
    assert response.status_code == 201
    data = response.json()
    for key in User.model_fields.keys():
        assert key in data


def test_login_returns_token_schema(test_client: TestClient):
    user_payload = {
        "username": "loginuser@example.com",
        "email": "loginuser@example.com",
        "password": "pass123"
    }
    # create user first
    create_response = test_client.post("/users/", json=user_payload)
    assert create_response.status_code == 201

    login_response = test_client.post(
        "/auth/token",
        data={"username": user_payload["username"], "password": user_payload["password"]}
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    for key in Token.model_fields.keys():
        assert key in token_data


def test_prompt_crud_returns_prompt_schema(test_client: TestClient, db_session):
    auth_headers = get_auth_headers(test_client, db_session)

    # create prompt
    prompt_payload = {"name": "Test Prompt", "description": "example"}
    create_resp = test_client.post("/api/prompts", json=prompt_payload, headers=auth_headers)
    assert create_resp.status_code == 200
    prompt_data = create_resp.json()
    for key in Prompt.model_fields.keys():
        assert key in prompt_data
    prompt_id = prompt_data["id"]

    # get prompt
    get_resp = test_client.get(f"/api/prompts/{prompt_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    for key in Prompt.model_fields.keys():
        assert key in get_data

    # update prompt
    update_resp = test_client.put(
        f"/api/prompts/{prompt_id}",
        json={"name": "Updated", "description": "updated desc"},
        headers=auth_headers,
    )
    assert update_resp.status_code == 200
    updated_data = update_resp.json()
    for key in Prompt.model_fields.keys():
        assert key in updated_data

    # delete prompt
    del_resp = test_client.delete(f"/api/prompts/{prompt_id}", headers=auth_headers)
    assert del_resp.status_code == 200


def test_prompt_version_crud_returns_prompt_version_schema(test_client: TestClient, db_session):
    auth_headers = get_auth_headers(test_client, db_session)

    # create prompt first
    prompt_resp = test_client.post("/api/prompts", json={"name": "PV Test", "description": "pv"}, headers=auth_headers)
    prompt_id = prompt_resp.json()["id"]

    # create version
    version_payload = {"content": "v1"}
    version_resp = test_client.post(
        f"/api/prompts/{prompt_id}/versions", json=version_payload, headers=auth_headers
    )
    assert version_resp.status_code == 200
    version_data = version_resp.json()
    for key in PromptVersion.model_fields.keys():
        assert key in version_data
    version_id = version_data["id"]

    # get version
    get_resp = test_client.get(f"/api/prompt_versions/{version_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    get_data = get_resp.json()
    for key in PromptVersion.model_fields.keys():
        assert key in get_data

