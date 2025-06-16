from sqlalchemy import Column, Integer, String
from prompthelix.models.base import Base # Ensure Base is imported

class LLMUsageStatistic(Base):
    __tablename__ = "llm_usage_statistics"

    id = Column(Integer, primary_key=True, index=True)
    llm_service = Column(String, unique=True, index=True, nullable=False)
    request_count = Column(Integer, default=0, nullable=False)

    def __repr__(self):
        return f"<LLMUsageStatistic(llm_service='{self.llm_service}', request_count={self.request_count})>"
