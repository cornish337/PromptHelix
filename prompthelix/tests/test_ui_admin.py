import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from prompthelix.tests.utils import get_auth_headers
from prompthelix.services import user_service


def _get_token(client: TestClient, db_session: Session) -> str:
    headers = get_auth_headers(client, db_session)
    return headers["Authorization"].split(" ")[1]


def test_create_user_form_requires_login(client: TestClient):
    response = client.get("/ui/admin/users/new")
    assert response.status_code == 401


def test_create_user_form_loads(client: TestClient, db_session: Session):
    token = _get_token(client, db_session)
    cookies = {"prompthelix_access_token": token}
    response = client.get("/ui/admin/users/new", cookies=cookies)
    assert response.status_code == 200
    assert "<h1>Create New User</h1>" in response.text
    assert "name=\"username\"" in response.text
    assert "name=\"email\"" in response.text
    assert "name=\"password\"" in response.text


def test_admin_create_user_success(client: TestClient, db_session: Session):
    token = _get_token(client, db_session)
    cookies = {"prompthelix_access_token": token}
    new_username = "newadmin@example.com"
    form_data = {"username": new_username, "email": new_username, "password": "secret"}
    response = client.post("/ui/admin/users/new", data=form_data, cookies=cookies, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"].startswith("/ui/admin/users/new")
    created_user = user_service.get_user_by_username(db_session, new_username)
    assert created_user is not None
    assert created_user.email == new_username

