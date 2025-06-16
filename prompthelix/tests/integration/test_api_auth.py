import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.schemas import UserCreate, User # Assuming User is the response schema for /users/me
from prompthelix.tests.utils import get_auth_headers, DEFAULT_TEST_USERNAME, DEFAULT_TEST_PASSWORD, DEFAULT_TEST_EMAIL
from prompthelix.services import user_service # For direct session verification if needed

def test_user_registration_success(client: TestClient, db_session: SQLAlchemySession):
    username = "newuser@example.com"
    password = "newpassword"

    response = client.post("/users/", json={"username": username, "email": username, "password": password})
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == username
    assert data["email"] == username
    assert "id" in data
    assert "hashed_password" not in data # Ensure password is not returned

    # Verify user in DB
    db_user = user_service.get_user_by_username(db_session, username=username)
    assert db_user is not None
    assert db_user.email == username

def test_user_registration_duplicate_username(client: TestClient, db_session: SQLAlchemySession):
    # Create user first
    client.post("/users/", json={"username": "dupuser@example.com", "email": "dupuser@example.com", "password": "password"})

    # Attempt to register again with same username
    response = client.post("/users/", json={"username": "dupuser@example.com", "email": "another@example.com", "password": "password"})
    assert response.status_code == 400 # Based on User route
    assert "Username already registered" in response.json()["detail"]

def test_user_registration_duplicate_email(client: TestClient, db_session: SQLAlchemySession):
    client.post("/users/", json={"username": "emailuser1@example.com", "email": "duplicate_email@example.com", "password": "password"})

    response = client.post("/users/", json={"username": "emailuser2@example.com", "email": "duplicate_email@example.com", "password": "password"})
    assert response.status_code == 400 # Based on User route
    assert "Email already registered" in response.json()["detail"]


def test_login_success(client: TestClient, db_session: SQLAlchemySession):
    # User created by get_auth_headers if not exists, or use a known user
    # For this test, let's ensure the default user is created via the helper's side effect
    get_auth_headers(client, db_session) # Ensures default user exists

    login_data = {"username": DEFAULT_TEST_USERNAME, "password": DEFAULT_TEST_PASSWORD}
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_login_failure_wrong_username(client: TestClient, db_session: SQLAlchemySession):
    login_data = {"username": "wronguser@example.com", "password": "testpassword"}
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

def test_login_failure_wrong_password(client: TestClient, db_session: SQLAlchemySession):
    # Ensures default user exists via get_auth_headers side effect
    get_auth_headers(client, db_session)

    login_data = {"username": DEFAULT_TEST_USERNAME, "password": "wrongpassword"}
    response = client.post("/auth/token", data=login_data)
    assert response.status_code == 401
    assert "Incorrect username or password" in response.json()["detail"]

def test_get_current_user_success(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    response = client.get("/users/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == DEFAULT_TEST_USERNAME
    assert data["email"] == DEFAULT_TEST_EMAIL

def test_get_current_user_unauthenticated(client: TestClient):
    response = client.get("/users/me")
    assert response.status_code == 401 # Expecting 401 due to OAuth2PasswordBearer
    assert "Not authenticated" in response.json()["detail"] # Or "Invalid authentication credentials"

def test_logout_success(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session) # User logs in, gets a session

    # Logout
    response_logout = client.post("/auth/logout", headers=auth_headers)
    assert response_logout.status_code == 200
    assert response_logout.json()["message"] == "Successfully logged out"

    # Try to access a protected route with the same token
    response_me_after_logout = client.get("/users/me", headers=auth_headers)
    assert response_me_after_logout.status_code == 401 # Session should be invalid
    assert "Invalid authentication credentials" in response_me_after_logout.json()["detail"] # Or "Session expired" if that's the specific message

    # Verify session is deleted from DB (optional, service test should cover this)
    # token = auth_headers["Authorization"].split(" ")[1]
    # session_in_db = user_service.get_session_by_token(db_session, session_token=token)
    # assert session_in_db is None

def test_logout_unauthenticated(client: TestClient):
    response = client.post("/auth/logout") # No auth headers
    assert response.status_code == 401
    assert "Not authenticated" in response.json()["detail"]


def test_token_expiration(client: TestClient, db_session: SQLAlchemySession, monkeypatch):
    monkeypatch.setattr("prompthelix.config.settings.DEFAULT_SESSION_EXPIRE_MINUTES", 0)
    login_data = {"username": DEFAULT_TEST_USERNAME, "password": DEFAULT_TEST_PASSWORD}
    # ensure user exists
    get_auth_headers(client, db_session)
    resp = client.post("/auth/token", data=login_data)
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    me_resp = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 401
