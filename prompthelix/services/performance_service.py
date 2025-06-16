from typing import List, Optional

from sqlalchemy.orm import Session as DbSession # Renamed to avoid conflict

from prompthelix.models.performance_models import PerformanceMetric
# Schemas will be defined elsewhere, using them in type hints for now
from prompthelix.schemas import PerformanceMetricCreate, PerformanceMetricUpdate

# Performance Metric Service Functions

def record_performance_metric(db: DbSession, metric_create: PerformanceMetricCreate) -> PerformanceMetric:
    """
    Creates a new performance metric and saves it to the database.
    """
    db_metric = PerformanceMetric(
        prompt_version_id=metric_create.prompt_version_id,
        metric_name=metric_create.metric_name,
        metric_value=metric_create.metric_value
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric

def get_metrics_for_prompt_version(db: DbSession, prompt_version_id: int) -> List[PerformanceMetric]:
    """
    Retrieves all performance metrics associated with a specific prompt_version_id.
    """
    return db.query(PerformanceMetric).filter(PerformanceMetric.prompt_version_id == prompt_version_id).all()

def get_performance_metric(db: DbSession, metric_id: int) -> Optional[PerformanceMetric]:
    """
    Retrieves a specific performance metric by its ID.
    """
    return db.query(PerformanceMetric).filter(PerformanceMetric.id == metric_id).first()

def delete_performance_metric(db: DbSession, metric_id: int) -> bool:
    """
    Deletes a specific performance metric by its ID. Returns True if successful, False otherwise.
    """
    db_metric = get_performance_metric(db, metric_id)
    if db_metric:
        db.delete(db_metric)
        db.commit()
        return True
    return False

def update_performance_metric(db: DbSession, metric_id: int, metric_update: PerformanceMetricUpdate) -> Optional[PerformanceMetric]:
    """
    Updates an existing performance metric.
    """
    db_metric = get_performance_metric(db, metric_id)
    if not db_metric:
        return None

    if metric_update.metric_name is not None:
        db_metric.metric_name = metric_update.metric_name
    if metric_update.metric_value is not None:
        db_metric.metric_value = metric_update.metric_value

    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    return db_metric
