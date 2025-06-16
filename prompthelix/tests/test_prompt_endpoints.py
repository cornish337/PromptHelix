import unittest
from fastapi.testclient import TestClient
from prompthelix.main import app

class TestPromptEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_create_and_list_prompts(self):
        payload = {"name": "Test Prompt from unittest", "description": "hello"}
        create_resp = self.client.post("/api/prompts", json=payload)
        self.assertEqual(create_resp.status_code, 200, create_resp.text) # Added response text for debugging
        data = create_resp.json()
        self.assertIn("id", data)
        self.assertEqual(data["name"], payload["name"])
        self.assertEqual(data["description"], payload["description"])

        list_resp = self.client.get("/api/prompts")
        self.assertEqual(list_resp.status_code, 200, list_resp.text) # Added response text
        listing = list_resp.json()
        # The endpoint returns a list of prompts directly
        self.assertTrue(any(p["id"] == data["id"] and p["name"] == payload["name"] for p in listing))

if __name__ == '__main__':
    unittest.main()
