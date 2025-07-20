import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import HTTPException

from prompthelix.models.base import Base
from prompthelix.schemas import UserCreate
from prompthelix.services import user_service
from prompthelix.api.dependencies import get_current_user


@pytest.fixture
def db_session():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.asyncio
async def test_get_current_user_valid(db_session):
    user = user_service.create_user(db_session, UserCreate(username='bob', email='bob@example.com', password='pw'))
    session_model = user_service.create_session(db_session, user.id, expires_delta_minutes=10)
    result = await get_current_user(token=session_model.session_token, db=db_session)
    assert result.id == user.id


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(db_session):
    with pytest.raises(HTTPException):
        await get_current_user(token='badtoken', db=db_session)
