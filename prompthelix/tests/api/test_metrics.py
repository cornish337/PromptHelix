from fastapi.testclient import TestClient


def test_metrics_endpoint_works(test_client: TestClient):
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "prompthelix_ga_generation" in response.text
