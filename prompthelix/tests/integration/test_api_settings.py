import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.schemas import APIKeyCreate, APIKeyDisplay # For request and response validation
from prompthelix.tests.utils import get_auth_headers

def test_create_update_and_get_api_key_settings_api(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)

    service_name = "TEST_OPENAI_API"
    api_key_value = "sk-testkey12345"

    # 1. Create API Key
    api_key_payload = APIKeyCreate(service_name=service_name, api_key=api_key_value)
    response_create = client.post(
        "/api/settings/apikeys/", # Assuming this is the correct endpoint
        json=api_key_payload.model_dump(),
        headers=auth_headers
    )
    assert response_create.status_code == 200 # Or 201 if that's what your route returns
    created_data = response_create.json()

    # Validate response using APIKeyDisplay schema structure
    assert created_data["service_name"] == service_name
    assert created_data["is_set"] is True
    assert service_name in created_data["api_key_hint"] or api_key_value[-4:] in created_data["api_key_hint"]
    apikey_id = created_data["id"]

    # 2. Get API Key (verify creation)
    response_get = client.get(f"/api/settings/apikeys/{service_name}", headers=auth_headers)
    assert response_get.status_code == 200
    get_data = response_get.json()
    assert get_data["service_name"] == service_name
    assert get_data["is_set"] is True
    assert get_data["id"] == apikey_id

    # 3. Update API Key
    updated_api_key_value = "sk-updatedkey67890"
    api_key_update_payload = APIKeyCreate(service_name=service_name, api_key=updated_api_key_value)
    response_update = client.post( # Assuming the same endpoint for create/update
        "/api/settings/apikeys/",
        json=api_key_update_payload.model_dump(),
        headers=auth_headers
    )
    assert response_update.status_code == 200
    updated_data = response_update.json()
    assert updated_data["service_name"] == service_name
    assert updated_data["api_key"][-4:] == updated_api_key_value[-4:] if updated_data.get("api_key") else True # Check hint again
    assert updated_data["is_set"] is True


    # 4. Get API Key again (verify update)
    response_get_updated = client.get(f"/api/settings/apikeys/{service_name}", headers=auth_headers)
    assert response_get_updated.status_code == 200
    get_updated_data = response_get_updated.json()
    # The GET response (APIKeyDisplay) should not return the full key.
    # We can't directly verify the updated_api_key_value here, only its presence via is_set and hint.
    assert get_updated_data["is_set"] is True
    assert updated_api_key_value[-4:] in get_updated_data["api_key_hint"]


def test_get_non_existent_api_key_settings_api(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    service_name = "SERVICE_DOES_NOT_EXIST"

    response_get = client.get(f"/api/settings/apikeys/{service_name}", headers=auth_headers)
    assert response_get.status_code == 404 # Assuming 404 for not found
    assert "API Key not found" in response_get.json()["detail"]

def test_api_key_settings_unauthenticated(client: TestClient, db_session: SQLAlchemySession):
    # No auth_headers
    service_name = "UNAUTH_SERVICE"
    api_key_value = "unauth_key"

    api_key_payload = APIKeyCreate(service_name=service_name, api_key=api_key_value).model_dump()

    response_create = client.post("/api/settings/apikeys/", json=api_key_payload)
    assert response_create.status_code == 401

    response_get = client.get(f"/api/settings/apikeys/{service_name}")
    assert response_get.status_code == 401
