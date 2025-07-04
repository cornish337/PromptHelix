from sqlalchemy import (
    Column,
    Integer,
    DateTime,
    Float,
    ForeignKey,
    String,
    JSON,
)
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
    parent_ids = Column(JSON, nullable=True)
    mutation_operator = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("GAExperimentRun", back_populates="chromosomes")



class GAGenerationMetrics(Base):
    """Stores summary metrics for each GA generation."""

    __tablename__ = "ga_generation_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("ga_experiment_runs.id"), nullable=False)
    generation_number = Column(Integer, nullable=False)
    best_fitness = Column(Float, nullable=False)
    avg_fitness = Column(Float, nullable=False)

    population_diversity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    population_size = Column(Integer, nullable=False)
    diversity = Column(JSON, nullable=True)


    run = relationship("GAExperimentRun")


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ga_run_id = Column(Integer, ForeignKey("ga_experiment_runs.id"), nullable=True)
    # Assuming ga_chromosomes.id is a UUID stored as a string
    chromosome_id_str = Column(String, nullable=True)
    prompt_content_snapshot = Column(String, nullable=False) # Storing the actual prompt text
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True) # Assuming a users table
    rating = Column(Integer, nullable=False) # e.g., 1-5
    feedback_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships (optional, but good practice)
    run = relationship("GAExperimentRun") # If ga_run_id is not null
    # user = relationship("User") # If you have a User model and want to link back