import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime

from prompthelix.models.base import Base
from prompthelix.models.conversation_models import ConversationLog
from prompthelix.services.conversation_service import ConversationService

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


def test_conversation_crud(db):
    service = ConversationService()
    log1 = ConversationLog(session_id='s1', sender_id='a', recipient_id='b', message_type='t', content='m1', timestamp=datetime.utcnow())
    log2 = ConversationLog(session_id='s1', sender_id='b', recipient_id='a', message_type='t', content='m2', timestamp=datetime.utcnow())
    db.add_all([log1, log2])
    db.commit()

    sessions = service.get_conversation_sessions(db)
    assert sessions[0].session_id == 's1'
    assert sessions[0].message_count == 2

    messages = service.get_messages_by_session_id(db, 's1')
    assert len(messages) == 2

    log1.content = 'm1-upd'
    db.commit()
    messages = service.get_messages_by_session_id(db, 's1')
    assert messages[0].content == 'm1-upd'

    db.delete(log1)
    db.commit()
    messages = service.get_messages_by_session_id(db, 's1')
    assert len(messages) == 1
