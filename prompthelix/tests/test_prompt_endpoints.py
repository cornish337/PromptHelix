import unittest
from fastapi.testclient import TestClient
from prompthelix.main import app

class TestPromptEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_create_and_list_prompts(self):
        create_resp = self.client.post("/api/prompts", json={"content": "hello"})
        self.assertEqual(create_resp.status_code, 200)
        data = create_resp.json()
        self.assertIn("id", data)
        self.assertEqual(data["content"], "hello")

        list_resp = self.client.get("/api/prompts")
        self.assertEqual(list_resp.status_code, 200)
        listing = list_resp.json()
        self.assertIn("prompts", listing)
        self.assertTrue(any(p["id"] == data["id"] for p in listing["prompts"]))

if __name__ == '__main__':
    unittest.main()
