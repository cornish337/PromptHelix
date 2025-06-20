import os
import tempfile
import unittest
from unittest.mock import patch

from prompthelix.agents.domain_expert import DomainExpertAgent

class TestDomainExpertAgent(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.kfile = os.path.join(self.tmpdir.name, "domain.json")
        self.agent = DomainExpertAgent(knowledge_file_path=self.kfile)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_initialization_creates_file(self):
        self.assertTrue(os.path.exists(self.kfile))
        self.assertTrue(self.agent.knowledge_base)

    def test_save_and_load(self):
        self.agent.knowledge_base["new"] = {"keywords": ["a"]}
        self.agent.save_knowledge()
        other = DomainExpertAgent(knowledge_file_path=self.kfile)
        self.assertIn("new", other.knowledge_base)

    @patch("prompthelix.agents.domain_expert.call_llm_api")
    def test_process_request_success(self, mock_call):
        mock_call.return_value = '["kw1","kw2"]'
        result = self.agent.process_request({"domain": "medical", "query_type": "keywords"})
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["data"], ["kw1", "kw2"])

    @patch("prompthelix.agents.domain_expert.call_llm_api", side_effect=Exception("fail"))
    def test_process_request_fallback(self, mock_call):
        result = self.agent.process_request({"domain": "medical", "query_type": "keywords"})
        self.assertEqual(result["source"], "knowledge_base")
        self.assertTrue(len(result["data"]) > 0)

if __name__ == "__main__":
    unittest.main()
