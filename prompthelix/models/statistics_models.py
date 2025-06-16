from sqlalchemy import Column, Integer, String

from prompthelix.models.base import Base

class LLMUsageStatistic(Base):
    """Tracks how often a particular LLM service is used."""

    __tablename__ = "llm_usage_statistics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    llm_service = Column(String, unique=True, index=True, nullable=False)
    request_count = Column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:  # pragma: no cover - simple representation
        return f"<LLMUsageStatistic(service='{self.llm_service}', count={self.request_count})>"
