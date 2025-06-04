import unittest
from fastapi.testclient import TestClient
# Assuming your FastAPI app instance is named 'app' in 'prompthelix.main'
from prompthelix.main import app

class TestApi(unittest.TestCase):

    def setUp(self):
        """Set up the test client before each test."""
        # The test_client fixture from conftest.py should be used for pytest-style tests.
        # However, this file uses unittest.TestCase. For consistency and to use the
        # conftest.py setup, this whole class should ideally be refactored to pytest style.
        # For a minimal change, we'll keep it but acknowledge it doesn't use the shared test_client fixture.
        self.client = TestClient(app)

    # def test_run_ga_endpoint(self):
    #     """Test the /api/run-ga endpoint.
    #     This endpoint is deprecated or its functionality changed.
    #     The new GA endpoint /api/experiments/run-ga is tested in test_api_experiments.py.
    #     """
    #     # response = self.client.get("/api/run-ga")
    #     # self.assertEqual(response.status_code, 200, f"API request failed with status {response.status_code}: {response.text}")
    #     # try:
    #     #     data = response.json()
    #     # except ValueError:
    #     #     self.fail("API response is not valid JSON.")
    #     # self.assertIn("best_prompt", data)
    #     # self.assertIn("fitness", data)
    #     pass # Test removed / passing trivially

    def test_root_endpoint(self):
        """Test the root / endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Welcome to PromptHelix API"})


if __name__ == '__main__':
    # This is for running unittest directly, pytest execution will not use this.
    unittest.main()
