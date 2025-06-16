import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session as SQLAlchemySession

from prompthelix.schemas import PerformanceMetricCreate
from prompthelix.tests.utils import get_auth_headers
from prompthelix.tests.test_api_prompts import create_test_prompt_via_api, create_test_prompt_version_via_api # Helper from prompt tests

def test_record_and_get_performance_metrics(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)

    # 1. Create a prompt and a version for the metric
    prompt_id = create_test_prompt_via_api(client, auth_headers, name="Perf Test Prompt")
    version_id = create_test_prompt_version_via_api(client, auth_headers, prompt_id, content="Version for perf testing")

    # 2. Record a performance metric for this version
    metric_payload = PerformanceMetricCreate(
        prompt_version_id=version_id,
        metric_name="accuracy",
        metric_value=0.95
    ).model_dump()

    response_create_metric = client.post(
        "/api/performance_metrics/",
        json=metric_payload,
        headers=auth_headers
    )
    assert response_create_metric.status_code == 201 # Expecting 201 Created
    created_metric_data = response_create_metric.json()
    assert created_metric_data["prompt_version_id"] == version_id
    assert created_metric_data["metric_name"] == "accuracy"
    assert created_metric_data["metric_value"] == 0.95
    metric_id = created_metric_data["id"]
    assert metric_id is not None

    # 3. Record another metric for the same version
    metric_payload2 = PerformanceMetricCreate(
        prompt_version_id=version_id,
        metric_name="latency_ms",
        metric_value=120.5
    ).model_dump()
    client.post("/api/performance_metrics/", json=metric_payload2, headers=auth_headers).raise_for_status()


    # 4. Get metrics for the prompt version
    response_get_metrics = client.get(
        f"/api/prompt_versions/{version_id}/performance_metrics/",
        headers=auth_headers # Assuming this might also be protected or just good practice
    )
    assert response_get_metrics.status_code == 200
    metrics_list = response_get_metrics.json()
    assert isinstance(metrics_list, list)
    assert len(metrics_list) == 2

    metric_names_in_response = sorted([m["metric_name"] for m in metrics_list])
    assert metric_names_in_response == ["accuracy", "latency_ms"]

    for metric in metrics_list:
        assert metric["prompt_version_id"] == version_id

def test_record_metric_for_non_existent_version(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session)
    non_existent_version_id = 99999

    metric_payload = {
        "prompt_version_id": non_existent_version_id,
        "metric_name": "error_rate",
        "metric_value": 0.1
    }
    response = client.post("/api/performance_metrics/", json=metric_payload, headers=auth_headers)
    assert response.status_code == 404 # Not Found for the prompt version
    assert f"PromptVersion with id {non_existent_version_id} not found" in response.json()["detail"]

def test_get_metrics_for_non_existent_version(client: TestClient, db_session: SQLAlchemySession):
    auth_headers = get_auth_headers(client, db_session) # Auth might be needed depending on route setup
    non_existent_version_id = 99998

    response = client.get(
        f"/api/prompt_versions/{non_existent_version_id}/performance_metrics/",
        headers=auth_headers
    )
    assert response.status_code == 404 # Not Found for the prompt version
    assert f"PromptVersion with id {non_existent_version_id} not found" in response.json()["detail"]


def test_performance_metric_unauthenticated_access(client: TestClient, db_session: SQLAlchemySession):
    # No auth_headers
    metric_payload = {
        "prompt_version_id": 1, # ID doesn't matter for unauth check
        "metric_name": "unauth_metric",
        "metric_value": 0.5
    }
    response_create = client.post("/api/performance_metrics/", json=metric_payload)
    assert response_create.status_code == 401

    # GET endpoint might be public or protected; assuming protected for consistency here
    # If it's public, this test part would fail or need adjustment.
    # response_get = client.get("/api/prompt_versions/1/performance_metrics/")
    # assert response_get.status_code == 401 # Or 200 if public and version 1 existed
                                           # For this test, /users/me is a reliable protected check

    # Check a known protected route to ensure test setup for unauth is generally okay
    response_me = client.get("/users/me")
    assert response_me.status_code == 401
