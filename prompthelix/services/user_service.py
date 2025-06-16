from datetime import datetime, timedelta
import secrets
from typing import Optional

from sqlalchemy.orm import Session as DbSession # Renamed to avoid conflict with model name
from passlib.context import CryptContext

from prompthelix.models.user_models import User, Session # Model import
# Schemas will be defined elsewhere, using them in type hints for now
from prompthelix.schemas import UserCreateSchema, UserUpdateSchema # Assuming schemas.py will exist

# Initialize password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# User Service Functions

def create_user(db: DbSession, user_create: UserCreateSchema) -> User:
    """
    Creates a new user.
    """
    hashed_password = pwd_context.hash(user_create.password)
    db_user = User(
        username=user_create.username,
        email=user_create.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def get_user(db: DbSession, user_id: int) -> Optional[User]:
    """
    Retrieves a user by ID.
    """
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_username(db: DbSession, username: str) -> Optional[User]:
    """
    Retrieves a user by username.
    """
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: DbSession, email: str) -> Optional[User]:
    """
    Retrieves a user by email.
    """
    return db.query(User).filter(User.email == email).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain password against a hashed password.
    """
    return pwd_context.verify(plain_password, hashed_password)

def update_user(db: DbSession, user_id: int, user_update: UserUpdateSchema) -> Optional[User]:
    """
    Updates user information.
    """
    db_user = get_user(db, user_id)
    if not db_user:
        return None

    if user_update.email:
        db_user.email = user_update.email
    if user_update.password:
        db_user.hashed_password = pwd_context.hash(user_update.password)

    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Session Service Functions

def create_session(db: DbSession, user_id: int, expires_delta_minutes: int = 60) -> Session:
    """
    Creates a new session for a user.
    """
    session_token = secrets.token_urlsafe(32)
    expires_at = datetime.utcnow() + timedelta(minutes=expires_delta_minutes)
    db_session = Session(
        user_id=user_id,
        session_token=session_token,
        expires_at=expires_at
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

def get_session_by_token(db: DbSession, session_token: str) -> Optional[Session]:
    """
    Retrieves a session by its token. Returns None if expired or not found.
    """
    db_session = db.query(Session).filter(Session.session_token == session_token).first()
    if db_session and db_session.expires_at < datetime.utcnow():
        # Session is expired
        db.delete(db_session) # Optionally delete expired session
        db.commit()
        return None
    return db_session

def delete_session(db: DbSession, session_token: str) -> bool:
    """
    Deletes a session by its token.
    """
    db_session = db.query(Session).filter(Session.session_token == session_token).first()
    if db_session:
        db.delete(db_session)
        db.commit()
        return True
    return False

def delete_all_user_sessions(db: DbSession, user_id: int) -> int:
    """
    Deletes all sessions for a given user.
    """
    num_deleted = db.query(Session).filter(Session.user_id == user_id).delete()
    db.commit()
    return num_deleted
