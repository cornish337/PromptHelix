from sqlalchemy import Column, Integer, DateTime, Float, ForeignKey, String, JSON
from sqlalchemy.orm import relationship
from datetime import datetime

from prompthelix.models.base import Base

class GAExperimentRun(Base):
    __tablename__ = "ga_experiment_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    parameters = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    prompt_version_id = Column(Integer, ForeignKey("prompt_versions.id"), nullable=True)

    prompt_version = relationship("PromptVersion")
    chromosomes = relationship("GAChromosome", back_populates="run", cascade="all, delete-orphan")

class GAChromosome(Base):
    __tablename__ = "ga_chromosomes"

    id = Column(String, primary_key=True)
    run_id = Column(Integer, ForeignKey("ga_experiment_runs.id"), nullable=False)
    generation_number = Column(Integer, nullable=False)
    genes = Column(JSON, nullable=False)
    fitness_score = Column(Float, nullable=False)
    evaluation_details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("GAExperimentRun", back_populates="chromosomes")


class GAGenerationMetric(Base):
    __tablename__ = "ga_generation_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("ga_experiment_runs.id"), nullable=False)
    generation_number = Column(Integer, nullable=False)
    best_fitness = Column(Float, nullable=False)
    avg_fitness = Column(Float, nullable=False)
    population_diversity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    run = relationship("GAExperimentRun")

