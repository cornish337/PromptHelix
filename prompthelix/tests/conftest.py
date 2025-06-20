import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as SQLAlchemySession # Renamed for clarity
from sqlalchemy.pool import StaticPool # Import StaticPool
from fastapi.testclient import TestClient

# Import Base and all models to ensure they are registered with Base.metadata
from prompthelix.models.base import Base
# Explicitly import all models to ensure Base.metadata is populated
from prompthelix.models.prompt_models import Prompt, PromptVersion
from prompthelix.models.settings_models import APIKey
from prompthelix.models.statistics_models import LLMUsageStatistic
from prompthelix.models.user_models import User, Session
from prompthelix.models.performance_models import PerformanceMetric
from prompthelix.models.conversation_models import ConversationLog
from prompthelix.models.evolution_models import GAExperimentRun, GAChromosome
# import prompthelix.models # This should trigger imports in models/__init__.py


from prompthelix.main import app # Import your FastAPI app
from prompthelix.database import get_db # The dependency to override

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

# Engine is session-scoped: one engine for the entire test session.
@pytest.fixture(scope="session")
def db_engine():
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool  # Use StaticPool for in-memory SQLite
    )
    # Models should be imported by now, populating Base.metadata
    # No create_all here; it will be handled per function in test_client
    yield engine
    # No drop_all here for session scope if tables are managed per function

# test_client is function-scoped. It ensures tables are created for each test.
@pytest.fixture(scope="function")
def test_client(db_engine): # Depends on the session-scoped engine
    """
    Provides a TestClient for the FastAPI application, with the database
    dependency overridden to use the test database.
    """
    # This TestingSessionLocal is for the FastAPI app when it calls get_db
    AppTestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    print(f"DEBUG: conftest.py: AppTestingSessionLocal created for override_get_db, bound to db_engine: {db_engine}")

    def override_get_db():
        print(f"DEBUG: override_get_db CALLED in conftest.py. Using AppTestingSessionLocal bound to engine: {AppTestingSessionLocal.kw['bind']}")
        db = AppTestingSessionLocal()
        try:
            yield db
        finally:
            print(f"DEBUG: override_get_db: closing session {db} in conftest.py")
            db.close()

    original_get_db = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db

    # Ensure tables are created using the test engine for each test using this client
    print("DEBUG: test_client: Tables known to Base.metadata before create_all:", list(Base.metadata.tables.keys()))
    Base.metadata.create_all(bind=db_engine)
    print("DEBUG: test_client: Tables known to Base.metadata after create_all:", list(Base.metadata.tables.keys()))

    AppTestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    print(f"DEBUG: conftest.py: AppTestingSessionLocal created for override_get_db, bound to db_engine: {db_engine}")

    def override_get_db():
        print(f"DEBUG: override_get_db CALLED in conftest.py. Using AppTestingSessionLocal bound to engine: {AppTestingSessionLocal.kw['bind']}")
        db = AppTestingSessionLocal()
        try:
            yield db
        finally:
            print(f"DEBUG: override_get_db: closing session {db} in conftest.py")
            db.close()

    original_get_db = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = override_get_db

    client = TestClient(app) # Initialize TestClient after override is set

    yield client

    # Clean up: drop tables and remove override
    print("DEBUG: test_client: Dropping all tables post-test.")
    Base.metadata.drop_all(bind=db_engine)
    if original_get_db:
        app.dependency_overrides[get_db] = original_get_db
    else:
        del app.dependency_overrides[get_db]


@pytest.fixture
def client(test_client):
    """Alias used by some tests."""
    yield test_client


# db_session is function-scoped and now depends on test_client to ensure create_all has run.
@pytest.fixture(scope="function")
def db_session(test_client, db_engine) -> SQLAlchemySession: # Added test_client dependency
    """
    Provides a transactional scope for tests. Each test gets a new session,
    and any changes are rolled back at the end of the test.
    Ensures that create_all from test_client has run.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    # Use a session that is bound to the connection, within the transaction
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False)
    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        # Rollback the transaction first, then close session, then connection
        transaction.rollback()
        session.close()
        connection.close()

@pytest.fixture
def client(test_client):
    """Alias used by some tests."""
    yield test_client


@pytest.fixture(scope="function")
def experiment_prompt(db_engine, test_client: TestClient, db_session: SQLAlchemySession): # Added db_session
    """
    Fixture to create a prompt for experiment tests, and clean it up afterwards.
    Yields the ID of the created prompt.
    Uses its own database session to ensure the commit is visible across test client requests.
    """
    from prompthelix.api import crud
    from prompthelix import schemas
    from sqlalchemy.orm import sessionmaker
    from prompthelix.tests.utils import get_auth_headers # For auth in teardown

    SessionLocalFixture = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    fixture_db_session = SessionLocalFixture()

    created_prompt_model = None  # To store the SQLAlchemy model instance
    try:
        prompt_create_schema = schemas.PromptCreate(
            name="Test Experiment Parent Prompt",
            description="Prompt for testing experiment parent association."
        )
        # Create prompt using the fixture's own session
        created_prompt_model = crud.create_prompt(
            db=fixture_db_session,
            prompt=prompt_create_schema,
            owner_id=1,
        )
        fixture_db_session.commit()  # Commit this session

        yield created_prompt_model.id  # Yield the ID

    finally:
        # Teardown: delete the prompt using its ID via an API call
        if created_prompt_model:
                try:
                    auth_headers = get_auth_headers(test_client, db_session)
                    # print(f"Cleaning up prompt ID: {created_prompt_model.id} with headers {auth_headers}")
                    response = test_client.delete(
                        f"/api/prompts/{created_prompt_model.id}", # Path confirmed from routes.py
                        headers=auth_headers
                    )
                    # Assert 200 (deleted) or 404 (already deleted, e.g. by the test itself)
                    assert response.status_code in [200, 404], \
                        f"Failed to cleanup prompt {created_prompt_model.id}: {response.text}"
                    # print(f"Cleanup response: {response.status_code}")
                except Exception as e:
                    print(f"Error during experiment_prompt teardown: {e}") # To see errors during teardown

        if fixture_db_session:
            fixture_db_session.close()
