import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session # For type hinting if db fixture is used directly

# from prompthelix.main import app # Client fixture should handle app creation
# from prompthelix.database import get_db, Base, engine # For test DB setup if needed
from prompthelix.api import crud # To verify db changes
from prompthelix.schemas import APIKeyCreate

# Assume 'client' fixture is provided by conftest.py, configured for testing.
# Assume 'db_session' fixture for direct DB assertions if needed.

def test_view_settings_page_loads_correctly(client: TestClient):
    """Test that the settings page loads with a 200 OK status."""
    response = client.get("/ui/settings")
    assert response.status_code == 200
    assert "<h1>Settings</h1>" in response.text # Check for main heading
    assert "<h2>API Key Management</h2>" in response.text # Check for API key section
    assert "<h2>Available Agents</h2>" in response.text # Check for Agents section
    # Check for at least one service input
    assert "OPENAI_api_key" in response.text or "ANTHROPIC_api_key" in response.text or "GOOGLE_api_key" in response.text

# To test agent listing, we might need to ensure some mock agents exist or patch list_available_agents
def test_settings_page_lists_agents(client: TestClient, monkeypatch):
    """Test that the settings page attempts to list agents."""

    # Example: Mock list_available_agents if it's complex or has external dependencies
    # For this test, we'll assume it might return an empty list if no agents are found,
    # or we can patch it to return a predictable list.

    # Patching list_available_agents in prompthelix.ui_routes
    # The actual path to list_available_agents will depend on where it's defined.
    # If it's a global in ui_routes.py, it would be 'prompthelix.ui_routes.list_available_agents'

    def mock_list_available_agents():
        return ["MockAgent1", "TestAgentAlpha"]

    # Assuming list_available_agents is imported and used in prompthelix.ui_routes
    # This path needs to be correct based on the actual module structure.
    # If list_available_agents is a method of a class, the patching target changes.
    # Based on previous steps, it's a standalone function in ui_routes.
    monkeypatch.setattr("prompthelix.ui_routes.list_available_agents", mock_list_available_agents)

    response = client.get("/ui/settings")
    assert response.status_code == 200
    assert "MockAgent1" in response.text
    assert "TestAgentAlpha" in response.text

def test_submit_api_key_form_empty_submission(client: TestClient, db_session: Session):
    """Test submitting the API key form with no changes or empty values."""
    # Initial state: ensure no keys are set for a test service or use one of the existing ones
    service_to_test = "OPENAI" # Or a unique test service
    crud.create_or_update_api_key(
        db_session,
        api_key_create=APIKeyCreate(service_name=service_to_test, api_key=""),
    )  # Ensure it's clear

    response = client.post(
        "/ui/settings/api_keys",
        data={
            f"{service_to_test}_service_name": service_to_test,
            f"{service_to_test}_api_key": "",
            # Add other services as needed to mimic the full form, or ensure backend handles partial submissions gracefully
            "ANTHROPIC_service_name": "ANTHROPIC", "ANTHROPIC_api_key": "",
            "GOOGLE_service_name": "GOOGLE", "GOOGLE_api_key": "",
        },
        allow_redirects=False # We want to inspect the redirect itself
    )
    assert response.status_code == 303 # HTTP_303_SEE_OTHER for redirect
    assert response.headers["location"] == "/ui/settings?message=No changes to API keys were applied." # Or similar message

    # Verify in DB (optional here, more for specific save tests)
    key_in_db = crud.get_api_key(db_session, service_name=service_to_test)
    assert key_in_db is not None # It should exist due to create_or_update
    assert key_in_db.api_key == "" # Should be empty

def test_submit_api_key_form_saves_new_key(client: TestClient, db_session: Session):
    """Test submitting the API key form to save a new key."""
    service_to_test = "OPENAI"
    new_key_value = "test_openai_key_functional_123"

    # Ensure the key is initially not set or different
    crud.create_or_update_api_key(
        db_session,
        api_key_create=APIKeyCreate(service_name=service_to_test, api_key=""),
    )

    response = client.post(
        "/ui/settings/api_keys",
        data={
            f"{service_to_test}_service_name": service_to_test,
            f"{service_to_test}_api_key": new_key_value,
            "ANTHROPIC_service_name": "ANTHROPIC", "ANTHROPIC_api_key": "", # Other services
            "GOOGLE_service_name": "GOOGLE", "GOOGLE_api_key": "",       # Other services
        },
        allow_redirects=False
    )
    assert response.status_code == 303
    expected_message = f"API key settings saved for: {service_to_test}." # Adjust if display name is used
    assert response.headers["location"] == f"/ui/settings?message={expected_message}"


    key_in_db = crud.get_api_key(db_session, service_name=service_to_test)
    assert key_in_db is not None
    assert key_in_db.api_key == new_key_value

def test_submit_api_key_form_clears_existing_key(client: TestClient, db_session: Session):
    """Test submitting the API key form to clear an existing key."""
    service_to_test = "ANTHROPIC"
    initial_key_value = "anthropic_key_to_clear_789"

    # Set an initial key
    crud.create_or_update_api_key(
        db_session,
        api_key_create=APIKeyCreate(service_name=service_to_test, api_key=initial_key_value),
    )
    key_in_db = crud.get_api_key(db_session, service_name=service_to_test)
    assert key_in_db.api_key == initial_key_value


    response = client.post(
        "/ui/settings/api_keys",
        data={
            "OPENAI_service_name": "OPENAI", "OPENAI_api_key": "", # Other services
            f"{service_to_test}_service_name": service_to_test,
            f"{service_to_test}_api_key": "", # Clear this key
            "GOOGLE_service_name": "GOOGLE", "GOOGLE_api_key": "", # Other services
        },
        allow_redirects=False
    )

    assert response.status_code == 303
    # The message might vary based on how "cleared" keys are reported.
    # Based on ui_routes.py, it would be "ANTHROPIC (cleared)"
    expected_message_fragment = f"{service_to_test} (cleared)"
    # The full message includes all processed services. If only one changed:
    # expected_full_message = f"API key settings saved for: {expected_message_fragment}."
    # This needs to match the actual logic in ui_routes.py for message construction.
    # For simplicity, let's check if the fragment is in the location URL.
    assert expected_message_fragment in response.headers["location"]
    assert "message=API key settings saved for:" in response.headers["location"]


    key_in_db_after_clear = crud.get_api_key(db_session, service_name=service_to_test)
    assert key_in_db_after_clear is not None
    assert key_in_db_after_clear.api_key == ""
