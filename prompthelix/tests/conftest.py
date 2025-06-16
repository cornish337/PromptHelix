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


@pytest.fixture(scope="function") # Changed to function scope to align with db_session if needed for overrides
def client(test_client_fixture: TestClient): # Use the renamed fixture
    """Provides a TestClient instance for function-scoped tests."""
    # If tests using this client need transactional behavior tied to db_session,
    # this setup might need further refinement, e.g., by having the TestClient
    # use the same session provided by db_session. However, for service unit tests,
    # we'll use db_session directly. TestClient is more for API/integration tests.
    yield test_client_fixture
