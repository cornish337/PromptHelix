import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession # Renamed for clarity
from fastapi.testclient import TestClient

# Import Base and all models to ensure they are registered with Base.metadata
from prompthelix.models.base import Base
import prompthelix.models # This should trigger imports in models/__init__.py

from prompthelix.main import app # Import your FastAPI app
from prompthelix.database import get_db # The dependency to override

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    # All models should be imported before this line so Base.metadata knows about them
    Base.metadata.create_all(bind=engine) # Create tables once per session
    yield engine
    Base.metadata.drop_all(bind=engine) # Drop tables at the end of the test session

@pytest.fixture(scope="function")
def db_session(db_engine) -> SQLAlchemySession: # Use the specific type hint
    """
    Provides a transactional scope for tests. Each test gets a new session,
    and any changes are rolled back at the end of the test.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    # Use a session that is bound to the connection, within the transaction
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False) # Bind to engine is not needed here
    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()

@pytest.fixture(scope="session") # test_client can be session-scoped if get_db override provides function-scoped sessions
def test_client_fixture(db_engine): # Renamed to avoid conflict if 'test_client' is used as a function name
    """
    Provides a TestClient for the FastAPI application, with the database
    dependency overridden to use the test database.
    """
    # This TestingSessionLocal is for the FastAPI app when it calls get_db
    AppTestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

    def override_get_db():
        db = AppTestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    original_get_db = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app)
    yield client

    # Clean up overrides after tests
    if original_get_db:
        app.dependency_overrides[get_db] = original_get_db
    else:
        del app.dependency_overrides[get_db]


@pytest.fixture
def client(test_client):
    """Alias used by some tests."""
    yield test_client


@pytest.fixture(scope="function")
def experiment_prompt(db_engine, test_client: TestClient): # Use db_engine for own session
    """
    Fixture to create a prompt for experiment tests, and clean it up afterwards.
    Yields the ID of the created prompt.
    Uses its own database session to ensure the commit is visible across test client requests.
    """
    from prompthelix.api import crud
    from prompthelix import schemas
    from sqlalchemy.orm import sessionmaker

    SessionLocalFixture = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    fixture_db_session = SessionLocalFixture()

    created_prompt_model = None  # To store the SQLAlchemy model instance
    try:
        prompt_create_schema = schemas.PromptCreate(
            name="Test Experiment Parent Prompt",
            description="Prompt for testing experiment parent association."
        )
        # Create prompt using the fixture's own session
        created_prompt_model = crud.create_prompt(db=fixture_db_session, prompt=prompt_create_schema)
        fixture_db_session.commit()  # Commit this session

        yield created_prompt_model.id  # Yield the ID

    finally:
        # Teardown: delete the prompt using its ID via an API call
        if created_prompt_model:
            # print(f"Cleaning up prompt ID: {created_prompt_model.id}")
            response = test_client.delete(f"/api/prompts/{created_prompt_model.id}")
            # Assert 200 (deleted) or 404 (already deleted, e.g. by the test itself)
            assert response.status_code in [200, 404], \
                f"Failed to cleanup prompt {created_prompt_model.id}: {response.text}"
            # print(f"Cleanup response: {response.status_code}")

        if fixture_db_session:
            fixture_db_session.close()
