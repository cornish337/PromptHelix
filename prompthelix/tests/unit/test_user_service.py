import pytest
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.exc import IntegrityError # For testing duplicate entries if service doesn't catch it
from datetime import datetime, timedelta

from prompthelix.services import user_service
from prompthelix.schemas import UserCreate, UserUpdate, SessionCreate # SessionCreate might not be used directly
from prompthelix.models.user_models import User, Session as SessionModel # Alias SessionModel to avoid conflict
from passlib.context import CryptContext

# Helper to verify password, can also use user_service.verify_password
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def test_create_user_success(db_session: SQLAlchemySession):
    user_data = UserCreate(username="testuser", email="test@example.com", password="password123")
    db_user = user_service.create_user(db_session, user_create=user_data)

    assert db_user is not None
    assert db_user.username == "testuser"
    assert db_user.email == "test@example.com"
    assert pwd_context.verify("password123", db_user.hashed_password)
    assert db_user.id is not None
    assert db_user.created_at is not None

def test_create_user_duplicate_username(db_session: SQLAlchemySession):
    user_data1 = UserCreate(username="dupuser", email="test1@example.com", password="password123")
    user_service.create_user(db_session, user_create=user_data1)

    user_data2 = UserCreate(username="dupuser", email="test2@example.com", password="password456")
    # Assuming service doesn't catch IntegrityError and DB throws it
    # If service handles it and returns None or raises custom error, adjust test
    with pytest.raises(IntegrityError): # Or custom exception if service handles it
        # The service currently does not catch this, relies on DB unique constraint
        # For a pure unit test of the service, you might mock the db.add/commit to throw.
        # But here we test service with a real (in-memory) DB session.
        user_service.create_user(db_session, user_create=user_data2)
        db_session.commit() # commit is needed to trigger constraint if service doesn't

def test_get_user(db_session: SQLAlchemySession):
    user_data = UserCreate(username="getuser", email="get@example.com", password="password")
    created_user = user_service.create_user(db_session, user_create=user_data)

    retrieved_user = user_service.get_user(db_session, user_id=created_user.id)
    assert retrieved_user is not None
    assert retrieved_user.id == created_user.id
    assert retrieved_user.username == "getuser"

    non_existent_user = user_service.get_user(db_session, user_id=99999)
    assert non_existent_user is None

def test_get_user_by_username(db_session: SQLAlchemySession):
    user_data = UserCreate(username="getbyusername", email="getuname@example.com", password="password")
    user_service.create_user(db_session, user_create=user_data)

    retrieved_user = user_service.get_user_by_username(db_session, username="getbyusername")
    assert retrieved_user is not None
    assert retrieved_user.username == "getbyusername"

    non_existent_user = user_service.get_user_by_username(db_session, username="nosuchuser")
    assert non_existent_user is None

def test_get_user_by_email(db_session: SQLAlchemySession):
    user_data = UserCreate(username="getbyemailuser", email="getbyemail@example.com", password="password")
    user_service.create_user(db_session, user_create=user_data)

    retrieved_user = user_service.get_user_by_email(db_session, email="getbyemail@example.com")
    assert retrieved_user is not None
    assert retrieved_user.email == "getbyemail@example.com"

    non_existent_user = user_service.get_user_by_email(db_session, email="nosuch@example.com")
    assert non_existent_user is None

def test_verify_password():
    hashed_password = pwd_context.hash("correctpassword")
    assert user_service.verify_password("correctpassword", hashed_password) is True
    assert user_service.verify_password("wrongpassword", hashed_password) is False

def test_update_user(db_session: SQLAlchemySession):
    user_data = UserCreate(username="updateuser", email="update@example.com", password="oldpassword")
    db_user = user_service.create_user(db_session, user_create=user_data)

    # Update email
    update_data_email = UserUpdate(email="updated@example.com")
    updated_user_email = user_service.update_user(db_session, user_id=db_user.id, user_update=update_data_email)
    assert updated_user_email.email == "updated@example.com"
    assert pwd_context.verify("oldpassword", updated_user_email.hashed_password) # Password should be unchanged

    # Update password
    update_data_password = UserUpdate(password="newpassword")
    updated_user_password = user_service.update_user(db_session, user_id=db_user.id, user_update=update_data_password)
    assert pwd_context.verify("newpassword", updated_user_password.hashed_password)
    assert not pwd_context.verify("oldpassword", updated_user_password.hashed_password) # Old password should not work
    assert updated_user_password.email == "updated@example.com" # Email should remain as previously updated

    # Update both
    update_data_both = UserUpdate(email="final@example.com", password="finalpassword")
    updated_user_both = user_service.update_user(db_session, user_id=db_user.id, user_update=update_data_both)
    assert updated_user_both.email == "final@example.com"
    assert pwd_context.verify("finalpassword", updated_user_both.hashed_password)

    # Test update non-existent user
    update_non_existent = UserUpdate(email="nosuch@example.com")
    result = user_service.update_user(db_session, user_id=9999, user_update=update_non_existent)
    assert result is None

def test_create_session(db_session: SQLAlchemySession):
    user_data = UserCreate(username="sessionuser", email="session@example.com", password="password")
    db_user = user_service.create_user(db_session, user_create=user_data)

    session = user_service.create_session(db_session, user_id=db_user.id, expires_delta_minutes=30)
    assert session is not None
    assert session.user_id == db_user.id
    assert session.session_token is not None
    assert len(session.session_token) > 20 # Check for a reasonable token
    assert session.created_at is not None
    expected_expires_at = datetime.utcnow() + timedelta(minutes=30)
    # Allow a small delta for execution time
    assert abs((session.expires_at - expected_expires_at).total_seconds()) < 5

def test_get_session_by_token(db_session: SQLAlchemySession):
    user_data = UserCreate(username="getsessionuser", email="getsession@example.com", password="password")
    db_user = user_service.create_user(db_session, user_create=user_data)
    created_session = user_service.create_session(db_session, user_id=db_user.id, expires_delta_minutes=10)

    # Valid, non-expired
    retrieved_session = user_service.get_session_by_token(db_session, session_token=created_session.session_token)
    assert retrieved_session is not None
    assert retrieved_session.id == created_session.id

    # Invalid token
    invalid_session = user_service.get_session_by_token(db_session, session_token="invalidtoken")
    assert invalid_session is None

    # Expired token
    expired_session_token = user_service.create_session(db_session, user_id=db_user.id, expires_delta_minutes=-1).session_token # Negative delta for immediate expiry
    # db_session.query(SessionModel).filter(SessionModel.session_token == expired_session_token).update({"expires_at": datetime.utcnow() - timedelta(minutes=1)})
    # db_session.commit()
    retrieved_expired = user_service.get_session_by_token(db_session, session_token=expired_session_token)
    assert retrieved_expired is None
    # Verify it was deleted
    assert db_session.query(SessionModel).filter(SessionModel.session_token == expired_session_token).first() is None


def test_delete_session(db_session: SQLAlchemySession):
    user_data = UserCreate(username="delsessionuser", email="delsession@example.com", password="password")
    db_user = user_service.create_user(db_session, user_create=user_data)
    session = user_service.create_session(db_session, user_id=db_user.id)

    result = user_service.delete_session(db_session, session_token=session.session_token)
    assert result is True
    assert user_service.get_session_by_token(db_session, session_token=session.session_token) is None

    result_non_existent = user_service.delete_session(db_session, session_token="nosuchtoken")
    assert result_non_existent is False

def test_delete_all_user_sessions(db_session: SQLAlchemySession):
    user1_data = UserCreate(username="user1sessions", email="user1s@example.com", password="password")
    user1 = user_service.create_user(db_session, user_create=user1_data)
    user2_data = UserCreate(username="user2sessions", email="user2s@example.com", password="password")
    user2 = user_service.create_user(db_session, user_create=user2_data)

    user_service.create_session(db_session, user_id=user1.id)
    user_service.create_session(db_session, user_id=user1.id)
    session_user2 = user_service.create_session(db_session, user_id=user2.id)

    num_deleted = user_service.delete_all_user_sessions(db_session, user_id=user1.id)
    assert num_deleted == 2
    assert db_session.query(SessionModel).filter(SessionModel.user_id == user1.id).count() == 0
    assert db_session.query(SessionModel).filter(SessionModel.user_id == user2.id).count() == 1
    assert db_session.query(SessionModel).filter(SessionModel.session_token == session_user2.session_token).first() is not None

    num_deleted_non_existent = user_service.delete_all_user_sessions(db_session, user_id=99999)
    assert num_deleted_non_existent == 0
