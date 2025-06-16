from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from prompthelix.models.base import Base

class Prompt(Base):
    __tablename__ = "prompts"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    versions = relationship("PromptVersion", back_populates="prompt", cascade="all, delete-orphan")

class PromptVersion(Base):
    __tablename__ = "prompt_versions"

    id = Column(Integer, primary_key=True, index=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False)
    version_number = Column(Integer, nullable=False, default=1)
    content = Column(Text, nullable=False)
    parameters_used = Column(JSON, nullable=True)
    fitness_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    prompt = relationship("Prompt", back_populates="versions")
    performance_metrics = relationship("PerformanceMetric", back_populates="prompt_version", cascade="all, delete-orphan")
