from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from prompthelix.models.base import Base

class ConversationLog(Base):
    __tablename__ = "conversation_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    session_id = Column(String, index=True, nullable=False) # To group messages
    sender_id = Column(String, nullable=False) # Agent ID, 'SYSTEM', 'LLM_PROVIDER_NAME'
    recipient_id = Column(String, nullable=True) # Agent ID, 'SYSTEM', 'LLM_PROVIDER_NAME', or Null for broadcast/log
    message_type = Column(String, nullable=True) # e.g., 'agent_to_agent', 'agent_to_llm', 'llm_response', 'system_event'
    content = Column(Text, nullable=False) # The actual message content (e.g., JSON string, text)
    # Optional: Add if threading/strict request-response is needed
    # parent_message_id = Column(Integer, ForeignKey("conversation_logs.id"), nullable=True)
    # children_messages = relationship("ConversationLog", backref=backref('parent_message', remote_side=[id]))

    def __repr__(self):
        return f"<ConversationLog(id={self.id}, session_id='{self.session_id}', sender='{self.sender_id}', recipient='{self.recipient_id}', type='{self.message_type}')>"
