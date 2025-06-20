from fastapi.testclient import TestClient

from prompthelix.tests.utils import get_auth_headers


def test_prompt_persistence_between_requests(test_client: TestClient, db_session):
    headers = get_auth_headers(test_client, db_session)
    resp = test_client.post("/api/prompts", json={"name": "Persist", "description": "check"}, headers=headers)
    assert resp.status_code == 200
    prompt_id = resp.json()["id"]

    get_resp = test_client.get(f"/api/prompts/{prompt_id}", headers=headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == prompt_id
