from sqlalchemy import Column, Integer, String, DateTime, UniqueConstraint
from datetime import datetime


from prompthelix.models.base import Base

class LLMUsageStatistic(Base):
    """Tracks how many times each LLM provider is used."""

    __tablename__ = "llm_usage_statistics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    llm_service = Column(String, nullable=False, unique=True, index=True)
    request_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("llm_service", name="uq_llm_usage_service"),
    )

    def __repr__(self) -> str:
        return f"<LLMUsageStatistic(service={self.llm_service}, count={self.request_count})>"