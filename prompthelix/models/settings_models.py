from sqlalchemy import Column, Integer, String, UniqueConstraint
from prompthelix.models.base import Base

class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    service_name = Column(String, unique=True, index=True, nullable=False)
    api_key = Column(String, nullable=False)

    __table_args__ = (UniqueConstraint('service_name', name='uq_service_name'),)

    def __repr__(self):
        return f"<APIKey(service_name='{self.service_name}')>"
