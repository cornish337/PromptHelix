import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from prompthelix.models.base import Base
from prompthelix.services import user_service
from prompthelix.schemas import UserCreate, UserUpdate

@pytest.fixture()
def db():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_user_crud(db):
    user_in = UserCreate(username='alice', email='a@example.com', password='pw')
    user = user_service.create_user(db, user_in)
    assert user.id is not None

    fetched = user_service.get_user(db, user.id)
    assert fetched.email == 'a@example.com'

    updated = user_service.update_user(db, user.id, UserUpdate(email='b@example.com'))
    assert updated.email == 'b@example.com'

    session_model = user_service.create_session(db, user.id, expires_delta_minutes=1)
    token = session_model.session_token
    assert user_service.get_session_by_token(db, token)

    assert user_service.delete_session(db, token) is True
    assert user_service.get_session_by_token(db, token) is None
