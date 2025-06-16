import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.tests.utils import get_auth_headers


def test_only_owner_can_modify_or_delete_prompt(client: TestClient, db_session: SQLAlchemySession):
    # user1 creates a prompt
    user1_headers = get_auth_headers(client, db_session, username="owner@example.com", password="ownerpass", email="owner@example.com")
    prompt_name = f"Auth Test Prompt {uuid.uuid4()}"
    create_resp = client.post("/api/prompts", json={"name": prompt_name, "description": "owner prompt"}, headers=user1_headers)
    assert create_resp.status_code == 200
    prompt_id = create_resp.json()["id"]

    # user2 attempts update/delete
    user2_headers = get_auth_headers(client, db_session, username="other@example.com", password="otherpass", email="other@example.com")

    update_resp = client.put(f"/api/prompts/{prompt_id}", json={"name": "hacked"}, headers=user2_headers)
    assert update_resp.status_code == 403

    delete_resp = client.delete(f"/api/prompts/{prompt_id}", headers=user2_headers)
    assert delete_resp.status_code == 403

    # owner can update and delete
    update_owner = client.put(f"/api/prompts/{prompt_id}", json={"name": "updated"}, headers=user1_headers)
    assert update_owner.status_code == 200

    delete_owner = client.delete(f"/api/prompts/{prompt_id}", headers=user1_headers)
    assert delete_owner.status_code == 200
