import pytest
from sqlalchemy.orm import Session as SQLAlchemySession # Use the specific type hint

from prompthelix.api import crud
from prompthelix.models.settings_models import APIKey # Model for direct assertion
from prompthelix.schemas import APIKeyCreate # Import the schema

# db_session fixture will be provided by conftest.py

def test_create_and_get_api_key(db_session: SQLAlchemySession):
    service_name = "TEST_SERVICE_CREATE_CRUD"
    api_key_value = "test_key_crud_12345"

    api_key_in = APIKeyCreate(service_name=service_name, api_key=api_key_value)

    # 1. Test creation
    # crud.create_or_update_api_key now expects APIKeyCreate schema
    created_key = crud.create_or_update_api_key(db_session, api_key_create=api_key_in)

    assert created_key is not None
    assert created_key.service_name == service_name
    assert created_key.api_key == api_key_value # Assuming the model still stores it directly

    # 2. Test retrieval
    retrieved_key = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key is not None
    assert retrieved_key.id == created_key.id
    assert retrieved_key.service_name == service_name
    assert retrieved_key.api_key == api_key_value

def test_update_api_key(db_session: SQLAlchemySession):
    service_name = "TEST_SERVICE_UPDATE_CRUD"
    initial_api_key_value = "initial_key_crud_67890"
    updated_api_key_value = "updated_key_crud_abcde"

    # 1. Create initial key
    initial_key_in = APIKeyCreate(service_name=service_name, api_key=initial_api_key_value)
    crud.create_or_update_api_key(db_session, api_key_create=initial_key_in)

    # 2. Test update
    updated_key_in = APIKeyCreate(service_name=service_name, api_key=updated_api_key_value)
    updated_key = crud.create_or_update_api_key(db_session, api_key_create=updated_key_in)

    assert updated_key is not None
    assert updated_key.service_name == service_name
    assert updated_key.api_key == updated_api_key_value

    # 3. Verify update by retrieving again
    retrieved_key_after_update = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key_after_update is not None
    assert retrieved_key_after_update.api_key == updated_api_key_value

def test_get_non_existent_api_key(db_session: SQLAlchemySession):
    service_name = "NON_EXISTENT_SERVICE_CRUD"

    retrieved_key = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key is None
