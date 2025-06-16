import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prompthelix.database import Base, DATABASE_URL, init_db as main_init_db
from fastapi.testclient import TestClient
from prompthelix.main import app # Import your FastAPI app

# Use a separate SQLite database for testing
TEST_DATABASE_URL = "sqlite:///./test_prompthelix.db"

@pytest.fixture(scope="session")
def db_engine():
    # Remove old test database file if it exists
    if os.path.exists("./test_prompthelix.db"):
        os.remove("./test_prompthelix.db")

    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine) # Create tables for the test DB
    yield engine
    # Teardown: optionally remove the test database file after tests
    # if os.path.exists("./test_prompthelix.db"):
    #     os.remove("./test_prompthelix.db")

@pytest.fixture(scope="function") # function scope for session to ensure clean state for each test
def db_session(db_engine):
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = SessionLocalTest()

    yield session # Provide the session to the test

    session.close()
    transaction.rollback() # Rollback any changes after each test
    connection.close()


@pytest.fixture(scope="session")
def test_client(db_engine):
    # Override the app's dependency for get_db to use the test session
    from prompthelix.database import get_db as app_get_db

    def override_get_db():
        # This is a simplified override. For a full override ensuring transactions
        # are handled correctly per test, the db_session fixture logic might need
        # to be integrated more directly or use a pattern where TestClient manages
        # the transaction lifecycle with the test DB.
        # For now, we'll use a new session for each override_get_db call,
        # relying on the db_session fixture to manage overall test db state (like table creation).
        # This is a common pattern but might need refinement for complex scenarios.

        # Recreate SessionLocal for test DB if not already configured for it
        # This setup assumes DATABASE_URL in database.py is NOT changed globally by tests.
        # A more robust approach might involve app factory pattern for FastAPI.
        TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[app_get_db] = override_get_db

    # It's important that main_init_db() (which creates tables based on DATABASE_URL)
    # is not called here again if db_engine fixture already created tables for TEST_DATABASE_URL.
    # Base.metadata.create_all(bind=db_engine) in db_engine fixture handles table creation.

    client = TestClient(app)
    yield client

    # Clean up overrides after tests
    app.dependency_overrides.clear()


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
