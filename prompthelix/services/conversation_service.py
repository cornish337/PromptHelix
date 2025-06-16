from typing import List, Optional
from sqlalchemy.orm import Session as DbSession
from sqlalchemy import func, desc
from prompthelix.models import ConversationLog
from prompthelix.schemas import ConversationLogEntry, ConversationSession # Ensure schemas are importable

class ConversationService:
    def get_conversation_sessions(self, db: DbSession, skip: int = 0, limit: int = 100) -> List[ConversationSession]:
        """Retrieves a list of unique conversation sessions with counts and timestamps."""
        sessions_query = (
            db.query(
                ConversationLog.session_id,
                func.count(ConversationLog.id).label("message_count"),
                func.min(ConversationLog.timestamp).label("first_message_at"),
                func.max(ConversationLog.timestamp).label("last_message_at"),
            )
            .group_by(ConversationLog.session_id)
            .order_by(desc(func.max(ConversationLog.timestamp))) # Show most recent sessions first
            .offset(skip)
            .limit(limit)
        )

        results = sessions_query.all()

        # Convert results to ConversationSession schema
        return [
            ConversationSession(
                session_id=row.session_id,
                message_count=row.message_count,
                first_message_at=row.first_message_at,
                last_message_at=row.last_message_at,
            )
            for row in results
        ]

    def get_messages_by_session_id(
        self, db: DbSession, session_id: str, skip: int = 0, limit: int = 1000 # Default to more messages for a session
    ) -> List[ConversationLogEntry]:
        """Retrieves all messages for a given session_id, ordered by timestamp."""
        return (
            db.query(ConversationLog)
            .filter(ConversationLog.session_id == session_id)
            .order_by(ConversationLog.timestamp)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_all_logs(self, db: DbSession, skip: int = 0, limit: int = 100) -> List[ConversationLogEntry]:
        """Retrieves all conversation logs, ordered by timestamp descending."""
        return (
            db.query(ConversationLog)
            .order_by(desc(ConversationLog.timestamp))
            .offset(skip)
            .limit(limit)
            .all()
        )

conversation_service = ConversationService()
