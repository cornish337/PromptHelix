from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.schemas import UserCreate
from prompthelix.services import user_service

# Default test user credentials
DEFAULT_TEST_USERNAME = "testuser@example.com" # Using email as username for simplicity in tests
DEFAULT_TEST_PASSWORD = "testpassword"
DEFAULT_TEST_EMAIL = "testuser@example.com"


def get_auth_headers(client: TestClient, db_session: SQLAlchemySession, username: str = DEFAULT_TEST_USERNAME, password: str = DEFAULT_TEST_PASSWORD, email: str = None) -> dict:
    """
    Logs in a test user (or creates if not exists) and returns auth headers.
    Uses email as username if username is None and email is provided.
    """
    actual_username = username
    actual_email = email if email else actual_username # If email not given, assume username is also email

    # Ensure user exists or create one
    user = user_service.get_user_by_username(db_session, username=actual_username)
    if not user:
        # If the username looks like an email and the email parameter was not set, use it for the email field.
        user_in = UserCreate(username=actual_username, email=actual_email, password=password)
        user_service.create_user(db_session, user_create=user_in)

    login_data = {"username": actual_username, "password": password}
    response = client.post("/auth/token", data=login_data)

    if response.status_code != 200:
        print(f"Login failed for user {actual_username}. Status: {response.status_code}, Detail: {response.text}")
        response.raise_for_status() # Raise an exception for bad status codes

    token_data = response.json()
    return {"Authorization": f"Bearer {token_data['access_token']}"}
