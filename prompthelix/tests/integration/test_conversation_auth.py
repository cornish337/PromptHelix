import pytest
from fastapi.testclient import TestClient
from prompthelix.tests.utils import get_auth_headers


@pytest.mark.parametrize("endpoint", [
    "/api/v1/conversations/sessions/",
    "/api/v1/conversations/sessions/s1/messages/",
    "/api/v1/conversations/all_logs/",
])
def test_conversation_endpoints_require_token(client: TestClient, endpoint: str):
    response = client.get(endpoint)
    assert response.status_code == 401
