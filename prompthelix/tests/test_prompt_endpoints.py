import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.tests.utils import get_auth_headers
from prompthelix.schemas import Prompt


def test_create_and_list_prompts(client: TestClient, db_session: SQLAlchemySession):
    payload = {"name": "Test Prompt from pytest", "description": "hello"}
    auth_headers = get_auth_headers(client, db_session)

    create_resp = client.post("/api/prompts", json=payload, headers=auth_headers)
    assert create_resp.status_code == 200, create_resp.text
    data = create_resp.json()
    created_prompt = Prompt.model_validate(data)
    assert created_prompt.name == payload["name"]
    assert created_prompt.description == payload["description"]

    list_resp = client.get("/api/prompts", headers=auth_headers)
    assert list_resp.status_code == 200, list_resp.text
    listing = list_resp.json()

    assert any(
        Prompt.model_validate(p).id == created_prompt.id and p["name"] == payload["name"]
        for p in listing
    )

