import unittest
from fastapi.testclient import TestClient
# Assuming your FastAPI app instance is named 'app' in 'prompthelix.main'
from prompthelix.main import app

class TestApi(unittest.TestCase):

    def setUp(self):
        """Set up the test client before each test."""
        self.client = TestClient(app)

    def test_run_ga_endpoint(self):
        """Test the /api/run-ga endpoint."""
        response = self.client.get("/api/run-ga")

        # Check for successful response code
        self.assertEqual(response.status_code, 200, f"API request failed with status {response.status_code}: {response.text}")

        # Parse the JSON response
        try:
            data = response.json()
        except ValueError:
            self.fail("API response is not valid JSON.")

        # Check for expected keys in the response
        self.assertIn("best_prompt", data, "Response JSON missing 'best_prompt' key.")
        self.assertIn("fitness", data, "Response JSON missing 'fitness' key.")

        # Check the types of the values
        self.assertIsInstance(data["best_prompt"], str, "'best_prompt' should be a string.")
        self.assertIsInstance(data["fitness"], (int, float), "'fitness' should be a number (int or float).")

        # Optional: Check if the prompt string is not empty and fitness is non-negative
        self.assertTrue(len(data["best_prompt"]) > 0, "'best_prompt' should not be empty.")
        self.assertTrue(data["fitness"] >= 0, "'fitness' should be non-negative.")

if __name__ == '__main__':
    unittest.main()
