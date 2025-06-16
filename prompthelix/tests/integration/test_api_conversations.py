import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session as DbSession
from datetime import datetime, timezone

# Adjust imports
from prompthelix.main import app # Main FastAPI app
from prompthelix.database import Base, get_db
from prompthelix.models import ConversationLog
from prompthelix.schemas import ConversationSession, ConversationLogEntry # For response validation

# Setup for a test database (SQLite in-memory for example)
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:" # Use a unique name for each test run if needed, or ensure clean state
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Override get_db dependency for tests
def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

# Fixture to set up and tear down the database for the test session (module scope)
@pytest.fixture(scope="module", autouse=True)
def setup_teardown_database():
    Base.metadata.create_all(bind=engine) # Create tables
    # Pre-populate data
    db = TestingSessionLocal()
    # Sample data to preload - ensure timestamps are timezone-aware if your app uses them (e.g. UTC)
    # FastAPI/Pydantic often default to UTC if not specified. For consistency:
    utc_now = datetime.now(timezone.utc)

    db.add(ConversationLog(id=1, session_id="s1", sender_id="a1", content='{"msg": "c1"}', timestamp=datetime(2023,1,1,10,0,0, tzinfo=timezone.utc), message_type="mt1", recipient_id="r1"))
    db.add(ConversationLog(id=2, session_id="s1", sender_id="a2", content='{"msg": "c2"}', timestamp=datetime(2023,1,1,10,1,0, tzinfo=timezone.utc), message_type="mt2", recipient_id="r2"))
    db.add(ConversationLog(id=3, session_id="s2", sender_id="a3", content='{"msg": "c3"}', timestamp=datetime(2023,1,1,11,0,0, tzinfo=timezone.utc), message_type="mt3", recipient_id="r3"))
    db.add(ConversationLog(id=4, session_id="s1", sender_id="a1", content='{"msg": "c4_older_in_s1"}', timestamp=datetime(2023,1,1,9,59,0, tzinfo=timezone.utc), message_type="mt4", recipient_id="r2")) # Older message in s1
    db.commit()
    db.close()

    yield # This is where the testing happens

    Base.metadata.drop_all(bind=engine) # Drop tables after tests are done

# Apply the dependency override globally for this test module
app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


def test_get_conversation_sessions_api():
    response = client.get("/api/v1/conversations/sessions/")
    assert response.status_code == 200
    data = response.json()

    assert len(data) == 2 # s2, then s1 (ordered by last_message_at desc)

    session_s2 = next(s for s in data if s["session_id"] == "s2")
    session_s1 = next(s for s in data if s["session_id"] == "s1")

    assert session_s2["message_count"] == 1
    assert datetime.fromisoformat(session_s2["last_message_at"]) == datetime(2023,1,1,11,0,0, tzinfo=timezone.utc)

    assert session_s1["message_count"] == 3 # s1 has 3 messages now
    assert datetime.fromisoformat(session_s1["last_message_at"]) == datetime(2023,1,1,10,1,0, tzinfo=timezone.utc)

    # Check order (s2's last message is later than s1's last message)
    assert data[0]["session_id"] == "s2"
    assert data[1]["session_id"] == "s1"

def test_get_conversation_sessions_api_pagination():
    response_limit1 = client.get("/api/v1/conversations/sessions/?limit=1")
    assert response_limit1.status_code == 200
    data_limit1 = response_limit1.json()
    assert len(data_limit1) == 1
    assert data_limit1[0]["session_id"] == "s2" # Most recent

    response_skip1_limit1 = client.get("/api/v1/conversations/sessions/?skip=1&limit=1")
    assert response_skip1_limit1.status_code == 200
    data_skip1_limit1 = response_skip1_limit1.json()
    assert len(data_skip1_limit1) == 1
    assert data_skip1_limit1[0]["session_id"] == "s1"


def test_get_messages_by_session_id_api_found():
    response = client.get("/api/v1/conversations/sessions/s1/messages/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3 # s1 has 3 messages
    # Messages are ordered by timestamp ASC
    assert data[0]["content"] == '{"msg": "c4_older_in_s1"}' # id=4, oldest in s1
    assert data[1]["content"] == '{"msg": "c1"}' # id=1
    assert data[2]["content"] == '{"msg": "c2"}' # id=2

def test_get_messages_by_session_id_api_pagination():
    # Test limit
    response = client.get("/api/v1/conversations/sessions/s1/messages/?limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == '{"msg": "c4_older_in_s1"}'

    # Test skip and limit
    response = client.get("/api/v1/conversations/sessions/s1/messages/?skip=1&limit=1")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["content"] == '{"msg": "c1"}'

def test_get_messages_by_session_id_api_not_found():
    response = client.get("/api/v1/conversations/sessions/non_existent_session/messages/")
    assert response.status_code == 404
    assert response.json()["detail"] == "Session ID 'non_existent_session' not found or session has no messages."


def test_get_all_logs_api():
    response = client.get("/api/v1/conversations/all_logs/")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 4 # Total logs
    # Default order is timestamp desc
    assert data[0]["content"] == '{"msg": "c3"}' # id=3, session s2, latest overall
    assert data[1]["content"] == '{"msg": "c2"}' # id=2, session s1
    assert data[2]["content"] == '{"msg": "c1"}' # id=1, session s1
    assert data[3]["content"] == '{"msg": "c4_older_in_s1"}' # id=4, session s1, oldest overall

def test_get_all_logs_api_pagination():
    # Test limit
    response = client.get("/api/v1/conversations/all_logs/?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["content"] == '{"msg": "c3"}'
    assert data[1]["content"] == '{"msg": "c2"}'

    # Test skip and limit
    response = client.get("/api/v1/conversations/all_logs/?skip=1&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["content"] == '{"msg": "c2"}'
    assert data[1]["content"] == '{"msg": "c1"}'
