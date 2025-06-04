import pytest
from sqlalchemy.orm import Session

from prompthelix.api import crud
from prompthelix.models.settings_models import APIKey # Model for direct assertion
# from prompthelix.schemas import APIKeyCreate # Schemas might not be directly used here unless crud ops use them

# Assume a fixture named 'db_session' is available from conftest.py that provides a transactional DB session.
# If not, these tests will need adjustment based on how DB sessions are provided in tests.

def test_create_and_get_api_key(db_session: Session):
    service_name = "TEST_SERVICE_CREATE"
    api_key_value = "test_key_12345"

    # 1. Test creation
    created_key = crud.create_or_update_api_key(db_session, service_name=service_name, api_key_value=api_key_value)

    assert created_key is not None
    assert created_key.service_name == service_name
    assert created_key.api_key == api_key_value

    # 2. Test retrieval
    retrieved_key = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key is not None
    assert retrieved_key.id == created_key.id
    assert retrieved_key.service_name == service_name
    assert retrieved_key.api_key == api_key_value

def test_update_api_key(db_session: Session):
    service_name = "TEST_SERVICE_UPDATE"
    initial_api_key_value = "initial_key_67890"
    updated_api_key_value = "updated_key_abcde"

    # 1. Create initial key
    crud.create_or_update_api_key(db_session, service_name=service_name, api_key_value=initial_api_key_value)

    # 2. Test update
    updated_key = crud.create_or_update_api_key(db_session, service_name=service_name, api_key_value=updated_api_key_value)

    assert updated_key is not None
    assert updated_key.service_name == service_name
    assert updated_key.api_key == updated_api_key_value

    # 3. Verify update by retrieving again
    retrieved_key_after_update = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key_after_update is not None
    assert retrieved_key_after_update.api_key == updated_api_key_value

def test_get_non_existent_api_key(db_session: Session):
    service_name = "NON_EXISTENT_SERVICE"

    retrieved_key = crud.get_api_key(db_session, service_name=service_name)

    assert retrieved_key is None

# Clean up any keys created (optional, depends on test DB setup; if transactional, it's handled)
# For example, explicitly delete if needed:
# def teardown_function(function):
#     if "db_session" in function.__globals__: # Crude check
#         db = function.__globals__["db_session"] # This is not how fixtures work directly
#         # Proper teardown would use the fixture context or transactional behavior
#         services_to_delete = ["TEST_SERVICE_CREATE", "TEST_SERVICE_UPDATE"]
#         for service in services_to_delete:
#             key = db.query(APIKey).filter(APIKey.service_name == service).first()
#             if key:
#                 db.delete(key)
#         db.commit()
