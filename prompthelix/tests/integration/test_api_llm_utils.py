import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession
from unittest.mock import patch # To mock the actual LLM call

from prompthelix.schemas import LLMTestRequest, LLMStatistic
from prompthelix.tests.utils import get_auth_headers
from prompthelix.services import user_service # To get user for direct DB manipulation if needed
from prompthelix.models.statistics_models import LLMUsageStatistic # For direct DB verification

# Test data
TEST_LLM_SERVICE = "TEST_LLM_PROVIDER" # A mock or test-only provider name
TEST_PROMPT_TEXT = "Hello, world!"

@pytest.fixture(autouse=True)
def mock_llm_call():
    """Mocks the actual call to an LLM API to avoid external dependencies and costs."""
    with patch("prompthelix.utils.llm_utils.call_llm_api_directly") as mock_call:
        # Configure the mock to return a specific response
        mock_call.return_value = "Mocked LLM response"
        yield mock_call

@pytest.fixture
def setup_llm_key(db_session: SQLAlchemySession):
    """Ensure a dummy API key is set for TEST_LLM_SERVICE to pass validation in call_llm_api."""
    from prompthelix.api import crud
    from prompthelix.schemas import APIKeyCreate
    crud.create_or_update_api_key(db_session, APIKeyCreate(service_name=TEST_LLM_SERVICE, api_key="dummy_key_for_test_provider"))


def test_test_llm_prompt_and_statistics_increment(client: TestClient, db_session: SQLAlchemySession, mock_llm_call, setup_llm_key):
    auth_headers = get_auth_headers(client, db_session)

    # 1. Get initial statistics (or ensure service is not yet in stats)
    initial_stats_response = client.get("/api/llm/statistics", headers=auth_headers)
    assert initial_stats_response.status_code == 200
    initial_stats = initial_stats_response.json()

    initial_count = 0
    for stat in initial_stats:
        if stat["llm_service"] == TEST_LLM_SERVICE:
            initial_count = stat["request_count"]
            break

    # 2. Call /api/llm/test_prompt
    test_request_payload = LLMTestRequest(llm_service=TEST_LLM_SERVICE, prompt_text=TEST_PROMPT_TEXT)
    response_test_prompt = client.post(
        "/api/llm/test_prompt",
        json=test_request_payload.model_dump(),
        headers=auth_headers
    )
    assert response_test_prompt.status_code == 200
    data = response_test_prompt.json()
    assert data["llm_service"] == TEST_LLM_SERVICE
    assert data["response_text"] == "Mocked LLM response" # From our mock
    mock_llm_call.assert_called_once_with(prompt=TEST_PROMPT_TEXT, provider=TEST_LLM_SERVICE, model_name=None)


    # 3. Verify statistics incremented
    updated_stats_response = client.get("/api/llm/statistics", headers=auth_headers)
    assert updated_stats_response.status_code == 200
    updated_stats = updated_stats_response.json()

    found_updated_stat = False
    for stat_data in updated_stats:
        stat = LLMStatistic.model_validate(stat_data) # Validate with Pydantic schema
        if stat.llm_service == TEST_LLM_SERVICE:
            assert stat.request_count == initial_count + 1
            found_updated_stat = True
            break
    assert found_updated_stat, f"Statistics for {TEST_LLM_SERVICE} not found or not incremented."

    # 4. Call test_prompt again for the same service
    client.post("/api/llm/test_prompt", json=test_request_payload.model_dump(), headers=auth_headers).raise_for_status()

    final_stats_response = client.get("/api/llm/statistics", headers=auth_headers)
    final_stats = final_stats_response.json()
    found_final_stat = False
    for stat_data in final_stats:
        stat = LLMStatistic.model_validate(stat_data)
        if stat.llm_service == TEST_LLM_SERVICE:
            assert stat.request_count == initial_count + 2
            found_final_stat = True
            break
    assert found_final_stat


def test_get_llm_statistics_multiple_services(client: TestClient, db_session: SQLAlchemySession, mock_llm_call, setup_llm_key):
    auth_headers = get_auth_headers(client, db_session)

    # Ensure dummy API keys for other services if call_llm_api checks them
    from prompthelix.api import crud
    from prompthelix.schemas import APIKeyCreate
    OTHER_SERVICE_1 = "OTHER_LLM_1"
    OTHER_SERVICE_2 = "OTHER_LLM_2"
    crud.create_or_update_api_key(db_session, APIKeyCreate(service_name=OTHER_SERVICE_1, api_key="dummy1"))
    crud.create_or_update_api_key(db_session, APIKeyCreate(service_name=OTHER_SERVICE_2, api_key="dummy2"))


    # Call test_prompt for TEST_LLM_SERVICE
    client.post("/api/llm/test_prompt", json={"llm_service": TEST_LLM_SERVICE, "prompt_text": "Hello"}, headers=auth_headers).raise_for_status()

    # Call test_prompt for OTHER_SERVICE_1
    client.post("/api/llm/test_prompt", json={"llm_service": OTHER_SERVICE_1, "prompt_text": "Hi"}, headers=auth_headers).raise_for_status()
    client.post("/api/llm/test_prompt", json={"llm_service": OTHER_SERVICE_1, "prompt_text": "Hi again"}, headers=auth_headers).raise_for_status()

    response = client.get("/api/llm/statistics", headers=auth_headers)
    assert response.status_code == 200
    stats_list = response.json()

    stats_map = {stat["llm_service"]: stat["request_count"] for stat in stats_list}

    assert TEST_LLM_SERVICE in stats_map
    assert stats_map[TEST_LLM_SERVICE] >= 1 # Exact count depends on other tests if run in parallel and DB not perfectly isolated per test func

    assert OTHER_SERVICE_1 in stats_map
    assert stats_map[OTHER_SERVICE_1] >= 2

    assert OTHER_SERVICE_2 not in stats_map or stats_map[OTHER_SERVICE_2] == 0 # Since it wasn't called via test_prompt

def test_llm_test_prompt_unauthenticated(client: TestClient):
    test_request_payload = LLMTestRequest(llm_service=TEST_LLM_SERVICE, prompt_text=TEST_PROMPT_TEXT)
    response = client.post("/api/llm/test_prompt", json=test_request_payload.model_dump())
    assert response.status_code == 401

def test_get_llm_statistics_unauthenticated(client: TestClient):
    response = client.get("/api/llm/statistics")
    # This endpoint might be public. If so, this test needs adjustment.
    # The current routes.py makes it use get_current_user, so it's protected.
    # If it's protected:
    assert response.status_code == 401 # Or 200 if it's public
                                      # Based on routes.py, it calls get_llm_statistics_route which is not explicitly protected
                                      # Let's re-check routes.py: get_llm_statistics_route does NOT have Depends(get_current_user)
                                      # So it SHOULD be public.
    # assert response.status_code == 200 # If public
    # If public and test above ran, it might find some stats.
    # For now, I'll assume it's public as per routes.py code for that specific endpoint.
    # This implies the test for unauth above needs to be specific to a protected endpoint.
    # The /api/llm/test_prompt is protected.
    # The /api/llm/statistics is NOT protected in the provided routes.py.
    # So, the test should expect 200.

    # Re-evaluating: The problem description implies general protection for new features.
    # The `get_llm_statistics_route` in provided `routes.py` for previous step did NOT have `Depends(get_current_user)`
    # I will assume it SHOULD be protected for this test suite's consistency.
    # If it's truly public, this test would change to assert 200.
    # Given the context of adding auth broadly, I'll assume it's an oversight and test for 401.
    # If the test fails, it means the route isn't protected as expected by this test.
    assert response.status_code == 401 # Assuming it should be protected.


@patch("prompthelix.utils.llm_utils.list_available_llms_from_config_and_db") # Mock the underlying util
def test_get_available_llms_route(mock_list_llms, client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session) # Assuming this route is also protected
    expected_llms = ["TEST_LLM_PROVIDER", "OTHER_LLM_1"]
    mock_list_llms.return_value = expected_llms

    response = client.get("/api/llm/available", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == expected_llms
    mock_list_llms.assert_called_once_with(db=db_session)

@patch("prompthelix.utils.llm_utils.list_available_llms_from_config_and_db")
def test_get_available_llms_unauthenticated(mock_list_llms, client: TestClient):
    response = client.get("/api/llm/available")
    # Similar to /statistics, this was not explicitly protected in routes.py.
    # Assuming it should be for consistency in this test suite.
    assert response.status_code == 401
