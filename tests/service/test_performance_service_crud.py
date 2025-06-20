import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from prompthelix.models.base import Base
from prompthelix.services import performance_service, user_service
from prompthelix.services.prompt_service import PromptService
from prompthelix.schemas import (
    UserCreate,
    PromptCreate,
    PromptVersionCreate,
    PerformanceMetricCreate,
    PerformanceMetricUpdate,
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


def test_performance_crud(db):
    user = user_service.create_user(db, UserCreate(username='u', email='u@e.com', password='pw'))
    pservice = PromptService()
    prompt = pservice.create_prompt(db, PromptCreate(name='p', description='d'), user.id)
    version = pservice.create_prompt_version(db, prompt.id, PromptVersionCreate(content='c'))

    metric = performance_service.record_performance_metric(db, PerformanceMetricCreate(prompt_version_id=version.id, metric_name='acc', metric_value=0.5))
    assert metric.id is not None

    fetched = performance_service.get_performance_metric(db, metric.id)
    assert fetched.metric_value == 0.5

    updated = performance_service.update_performance_metric(db, metric.id, PerformanceMetricUpdate(metric_value=0.6))
    assert updated.metric_value == 0.6

    assert performance_service.delete_performance_metric(db, metric.id) is True
    assert performance_service.get_performance_metric(db, metric.id) is None
