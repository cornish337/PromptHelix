import pytest
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.services import performance_service
from prompthelix.schemas import PerformanceMetricCreate, PerformanceMetricUpdate, PromptCreate, PromptVersionCreate
# Need models for creating prerequisites and verifying results
from prompthelix.models.prompt_models import Prompt, PromptVersion
from prompthelix.models.performance_models import PerformanceMetric

# Helper function to create a prompt and a version for tests
def create_test_prompt_version(db_session: SQLAlchemySession, name: str = "Test Prompt", content: str = "Test content v1") -> PromptVersion:
    prompt_model = Prompt(name=name)
    db_session.add(prompt_model)
    db_session.commit() # Commit to get prompt_model.id

    version_model = PromptVersion(
        prompt_id=prompt_model.id,
        content=content,
        version_number=1
    )
    db_session.add(version_model)
    db_session.commit() # Commit to get version_model.id
    db_session.refresh(version_model)
    return version_model

def test_record_performance_metric(db_session: SQLAlchemySession):
    pv = create_test_prompt_version(db_session)
    metric_data = PerformanceMetricCreate(
        prompt_version_id=pv.id,
        metric_name="accuracy",
        metric_value=0.95
    )
    db_metric = performance_service.record_performance_metric(db_session, metric_create=metric_data)

    assert db_metric is not None
    assert db_metric.prompt_version_id == pv.id
    assert db_metric.metric_name == "accuracy"
    assert db_metric.metric_value == 0.95
    assert db_metric.id is not None
    assert db_metric.created_at is not None

def test_get_metrics_for_prompt_version(db_session: SQLAlchemySession):
    pv1 = create_test_prompt_version(db_session, name="PV1")
    pv2 = create_test_prompt_version(db_session, name="PV2") # Another version for isolation

    performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv1.id, metric_name="latency", metric_value=100.0))
    performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv1.id, metric_name="quality", metric_value=0.8))

    # Metric for another prompt version, should not be retrieved
    performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv2.id, metric_name="cost", metric_value=0.01))

    metrics_pv1 = performance_service.get_metrics_for_prompt_version(db_session, prompt_version_id=pv1.id)
    assert len(metrics_pv1) == 2
    metric_names_pv1 = sorted([m.metric_name for m in metrics_pv1])
    assert metric_names_pv1 == ["latency", "quality"]

    metrics_non_existent_pv = performance_service.get_metrics_for_prompt_version(db_session, prompt_version_id=9999)
    assert len(metrics_non_existent_pv) == 0

    metrics_pv2 = performance_service.get_metrics_for_prompt_version(db_session, prompt_version_id=pv2.id)
    assert len(metrics_pv2) == 1
    assert metrics_pv2[0].metric_name == "cost"


def test_get_performance_metric(db_session: SQLAlchemySession):
    pv = create_test_prompt_version(db_session)
    created_metric = performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv.id, metric_name="recall", metric_value=0.88))

    retrieved_metric = performance_service.get_performance_metric(db_session, metric_id=created_metric.id)
    assert retrieved_metric is not None
    assert retrieved_metric.id == created_metric.id
    assert retrieved_metric.metric_name == "recall"

    non_existent_metric = performance_service.get_performance_metric(db_session, metric_id=9999)
    assert non_existent_metric is None

def test_delete_performance_metric(db_session: SQLAlchemySession):
    pv = create_test_prompt_version(db_session)
    metric_to_delete = performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv.id, metric_name="precision", metric_value=0.75))

    result = performance_service.delete_performance_metric(db_session, metric_id=metric_to_delete.id)
    assert result is True
    assert performance_service.get_performance_metric(db_session, metric_id=metric_to_delete.id) is None

    result_non_existent = performance_service.delete_performance_metric(db_session, metric_id=9999)
    assert result_non_existent is False

def test_update_performance_metric(db_session: SQLAlchemySession):
    pv = create_test_prompt_version(db_session)
    db_metric = performance_service.record_performance_metric(db_session, PerformanceMetricCreate(prompt_version_id=pv.id, metric_name="f1_score", metric_value=0.8))

    update_data = PerformanceMetricUpdate(metric_name="f1-score-macro", metric_value=0.82)
    updated_metric = performance_service.update_performance_metric(db_session, metric_id=db_metric.id, metric_update=update_data)

    assert updated_metric is not None
    assert updated_metric.id == db_metric.id
    assert updated_metric.metric_name == "f1-score-macro"
    assert updated_metric.metric_value == 0.82

    # Test updating only one field
    update_only_value = PerformanceMetricUpdate(metric_value=0.85)
    updated_metric_value_only = performance_service.update_performance_metric(db_session, metric_id=db_metric.id, metric_update=update_only_value)
    assert updated_metric_value_only.metric_name == "f1-score-macro" # Name should persist
    assert updated_metric_value_only.metric_value == 0.85

    update_only_name = PerformanceMetricUpdate(metric_name="F1 Score (Macro Averaged)")
    updated_metric_name_only = performance_service.update_performance_metric(db_session, metric_id=db_metric.id, metric_update=update_only_name)
    assert updated_metric_name_only.metric_name == "F1 Score (Macro Averaged)"
    assert updated_metric_name_only.metric_value == 0.85 # Value should persist

    # Test update non-existent metric
    update_non_existent = PerformanceMetricUpdate(metric_name="ghost_metric")
    result_non_existent = performance_service.update_performance_metric(db_session, metric_id=9999, metric_update=update_non_existent)
    assert result_non_existent is None
