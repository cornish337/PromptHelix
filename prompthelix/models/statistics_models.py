from sqlalchemy import Column, Integer, String
from prompthelix.models.base import Base

class LLMUsageStatistic(Base):
    __tablename__ = "llm_usage_statistics"

    id = Column(Integer, primary_key=True, index=True)
    llm_service = Column(String, unique=True, nullable=False)
    request_count = Column(Integer, nullable=False, default=0)

    def __repr__(self) -> str:
        return f"<LLMUsageStatistic(service={self.llm_service}, count={self.request_count})>"
