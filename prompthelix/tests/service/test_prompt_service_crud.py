import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from prompthelix.models.base import Base
from prompthelix.services.prompt_service import PromptService
from prompthelix.services import user_service
from prompthelix.schemas import (
    PromptCreate, PromptUpdate,
    PromptVersionCreate, PromptVersionUpdate,
    UserCreate
)

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


def test_prompt_crud(db):
    user = user_service.create_user(db, UserCreate(username='u1', email='u1@e.com', password='pw'))
    service = PromptService()

    prompt = service.create_prompt(db, PromptCreate(name='p1', description='d'), user.id)
    assert prompt.id is not None

    fetched = service.get_prompt(db, prompt.id)
    assert fetched.name == 'p1'

    updated = service.update_prompt(db, prompt.id, PromptUpdate(name='p2'))
    assert updated.name == 'p2'

    version = service.create_prompt_version(db, prompt.id, PromptVersionCreate(content='c'))
    assert version.id is not None

    v_fetched = service.get_prompt_version(db, version.id)
    assert v_fetched.content == 'c'

    v_updated = service.update_prompt_version(db, version.id, PromptVersionUpdate(content='c2'))
    assert v_updated.content == 'c2'

    assert service.delete_prompt_version(db, version.id)
    assert service.get_prompt_version(db, version.id) is None

    assert service.delete_prompt(db, prompt.id)
    assert service.get_prompt(db, prompt.id) is None
