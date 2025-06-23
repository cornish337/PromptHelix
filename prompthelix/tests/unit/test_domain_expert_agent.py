import unittest
import asyncio # Added import
from prompthelix.agents.domain_expert import DomainExpertAgent

class TestDomainExpertAgent(unittest.TestCase):
    """Test suite for the DomainExpertAgent."""

    def setUp(self):
        """Instantiate the DomainExpertAgent for each test."""
        self.expert = DomainExpertAgent(knowledge_file_path=None)

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.expert)
        self.assertEqual(self.expert.agent_id, "DomainExpert")
        self.assertTrue(self.expert.knowledge_base, "Knowledge base should be loaded and not empty.")

    def test_load_domain_knowledge(self):
        """Test if the domain knowledge base is loaded correctly."""
        self.assertIsInstance(self.expert.knowledge_base, dict)
        self.assertTrue(len(self.expert.knowledge_base) > 0)
        
        # Check for expected domains
        expected_domains = ["medical", "legal", "coding_python", "general_knowledge"]
        for domain in expected_domains:
            self.assertIn(domain, self.expert.knowledge_base)
            self.assertIsInstance(self.expert.knowledge_base[domain], dict)
        
        # Check structure of a specific domain (e.g., medical)
        medical_domain = self.expert.knowledge_base.get("medical")
        self.assertIsNotNone(medical_domain)
        expected_sub_keys = ["keywords", "constraints", "evaluation_tips", "sample_prompt_starters"]
        for sub_key in expected_sub_keys:
            self.assertIn(sub_key, medical_domain)
            self.assertIsInstance(medical_domain[sub_key], list) # Most of these are lists

    def test_process_request_medical_keywords(self):
        """Test process_request for specific info (medical keywords)."""
        request_data = {"domain": "medical", "query_type": "keywords"}
        result = asyncio.run(self.expert.process_request(request_data))

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("domain"), "medical")
        self.assertEqual(result.get("query_type"), "keywords")
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], list)
        self.assertIn("patient", result["data"], "Specific medical keyword 'patient' not found.")
        self.assertIn("EHR", result["data"], "Specific medical keyword 'EHR' not found.")

    def test_process_request_all_info_coding(self):
        """Test process_request for 'all_info' for a specific domain (coding_python)."""
        request_data = {"domain": "coding_python", "query_type": "all_info"}
        result = asyncio.run(self.expert.process_request(request_data))

        self.assertIsInstance(result, dict)
        self.assertEqual(result.get("domain"), "coding_python")
        self.assertEqual(result.get("query_type"), "all_info")
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], dict)
        
        coding_data = result["data"]
        self.assertIn("keywords", coding_data)
        self.assertIn("constraints", coding_data)
        self.assertIn("evaluation_tips", coding_data)
        self.assertIn("sample_prompt_starters", coding_data)
        self.assertIn("def", coding_data["keywords"]) # Check a specific keyword

    def test_process_request_unknown_domain(self):
        """Test process_request for an unknown domain that doesn't fallback to general."""
        # The agent currently falls back to "general_knowledge" for many unknown domains.
        # To test a true "not found", we'd need to ensure it's not a generic query or make
        # the fallback more specific.
        # Let's test a specific unknown domain that is NOT "general_knowledge" itself.
        request_data = {"domain": "underwater_basket_weaving", "query_type": "keywords"}
        result = asyncio.run(self.expert.process_request(request_data))
        
        # Current implementation falls back to 'general_knowledge' if domain is not one of the main specific ones.
        # So we check if it correctly falls back to general_knowledge
        self.assertEqual(result.get("domain"), "general_knowledge", "Should fallback to general_knowledge for unknown specific domains.")
        self.assertEqual(result.get("query_type"), "keywords")
        self.assertIn("explain", result.get("data", [])) # A keyword from general_knowledge

        # To test a true error, we'd need a scenario where general_knowledge is also not applicable,
        # or if the knowledge_base for general_knowledge was missing the query_type.
        # For now, the fallback covers most "unknown domain" cases by design.

    def test_process_request_unknown_query_type(self):
        """Test process_request for an unknown query_type for a known domain."""
        request_data = {"domain": "medical", "query_type": "non_existent_query_type"}
        result = asyncio.run(self.expert.process_request(request_data))

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertNotIn("data", result)
        self.assertIn("Query type 'non_existent_query_type' not found for domain 'medical'", result["error"])

    def test_process_request_missing_keys(self):
        """Test process_request with missing essential keys in request_data."""
        # Missing 'query_type'
        request_missing_query = {"domain": "medical"}
        result_missing_query = asyncio.run(self.expert.process_request(request_missing_query))
        self.assertIn("error", result_missing_query)
        self.assertIn("Missing 'domain' or 'query_type'", result_missing_query["error"])

        # Missing 'domain'
        request_missing_domain = {"query_type": "keywords"}
        result_missing_domain = asyncio.run(self.expert.process_request(request_missing_domain))
        self.assertIn("error", result_missing_domain)
        self.assertIn("Missing 'domain' or 'query_type'", result_missing_domain["error"])

        # Empty request
        request_empty = {}
        result_empty = asyncio.run(self.expert.process_request(request_empty))
        self.assertIn("error", result_empty)
        self.assertIn("Missing 'domain' or 'query_type'", result_empty["error"])

if __name__ == '__main__':
    unittest.main()
