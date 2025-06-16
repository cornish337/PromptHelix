import unittest
from prompthelix.agents.critic import PromptCriticAgent
from prompthelix.genetics.engine import PromptChromosome

class TestPromptCriticAgent(unittest.TestCase):
    """Test suite for the PromptCriticAgent."""

    def setUp(self):
        """Instantiate the PromptCriticAgent for each test."""
        self.critic = PromptCriticAgent(knowledge_file_path=None)

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.critic)
        self.assertEqual(self.critic.agent_id, "PromptCritic")
        self.assertTrue(self.critic.critique_rules, "Critique rules should be loaded and not empty.")

    def test_load_critique_rules(self):
        """Test if critique rules are loaded correctly."""
        self.assertIsInstance(self.critic.critique_rules, list)
        self.assertTrue(len(self.critic.critique_rules) > 0)
        for rule in self.critic.critique_rules:
            self.assertIsInstance(rule, dict)
            self.assertIn("name", rule)
            self.assertIn("type", rule)
            self.assertIn("message", rule)

    def test_process_request_good_prompt(self):
        """Test process_request with a well-structured prompt."""
        good_prompt = PromptChromosome(genes=[
            "Instruction: Summarize the provided text about climate change.",
            "Context: The text discusses various impacts and mitigation strategies.",
            "Output Format: A concise summary of 2-3 paragraphs."
        ])
        request_data = {"prompt_chromosome": good_prompt}
        result = self.critic.process_request(request_data)

        self.assertGreaterEqual(result["critique_score"], 0.9, "Score should be high for a good prompt.")
        # It might have a suggestion about missing "instruction" keyword if not explicitly in a gene
        # For the current rules, it should be good.
        # Allowing for very minor feedback, but generally expecting few to no negative points.
        negative_feedback_count = sum(1 for fb in result["feedback_points"] if "Issue" in fb or "Violation" in fb or "missing" in fb.lower())
        self.assertEqual(negative_feedback_count, 0, f"Good prompt should have no negative feedback points, got: {result['feedback_points']}")


    def test_process_request_short_prompt(self):
        """Test process_request with a prompt that is too short."""
        short_prompt = PromptChromosome(genes=["Summarize this."])
        request_data = {"prompt_chromosome": short_prompt}
        result = self.critic.process_request(request_data)

        self.assertLess(result["critique_score"], 0.9, "Score should be lower for a short prompt.")
        self.assertTrue(
            any("too short" in fb.lower() for fb in result["feedback_points"]),
            "Feedback for short prompt missing."
        )

    def test_process_request_negative_phrasing(self):
        """Test process_request with a prompt containing negative phrasing."""
        negative_prompt = PromptChromosome(genes=[
            "Instruction: Don't fail to summarize the text.",
            "Context: The user cannot understand complex language.",
            "Output: Simple summary."
        ])
        request_data = {"prompt_chromosome": negative_prompt}
        result = self.critic.process_request(request_data)
        
        self.assertTrue(
            any("negative phrasing" in fb.lower() for fb in result["feedback_points"]),
            f"Feedback for negative phrasing missing. Got: {result['feedback_points']}"
        )

    def test_process_request_lacks_instruction(self):
        """Test process_request with a prompt that lacks a clear instruction segment."""
        # This test depends on how "LacksInstruction" rule is implemented.
        # Current rule checks if "instruction" keyword is missing from all gene strings.
        no_instruction_prompt = PromptChromosome(genes=[
            "Provide details about the solar system.",
            "The audience is young children.",
            "Format: Bullet points."
        ])
        request_data = {"prompt_chromosome": no_instruction_prompt}
        result = self.critic.process_request(request_data)

        self.assertTrue(
            any("lacks instruction" in fb.lower() or "missing 'instruction' segment" in fb.lower() for fb in result["feedback_points"]),
            f"Feedback for missing instruction missing. Got: {result['feedback_points']}"
        )

    def test_process_request_invalid_input(self):
        """Test process_request with invalid or missing prompt_chromosome."""
        # Test with string instead of PromptChromosome
        request_data_str = {"prompt_chromosome": "This is not a chromosome object."}
        result_str = self.critic.process_request(request_data_str)
        self.assertEqual(result_str["critique_score"], 0.0)
        self.assertTrue(
            any("invalid or missing 'prompt_chromosome' object" in fb.lower() for fb in result_str["feedback_points"]),
            "Error message for invalid string input missing."
        )

        # Test with empty dictionary
        request_data_empty = {}
        result_empty = self.critic.process_request(request_data_empty)
        self.assertEqual(result_empty["critique_score"], 0.0)
        self.assertTrue(
            any("invalid or missing 'prompt_chromosome' object" in fb.lower() for fb in result_empty["feedback_points"]),
            "Error message for empty input missing."
        )

        # Test with prompt_chromosome as None
        request_data_none = {"prompt_chromosome": None}
        result_none = self.critic.process_request(request_data_none)
        self.assertEqual(result_none["critique_score"], 0.0)
        self.assertTrue(
            any("invalid or missing 'prompt_chromosome' object" in fb.lower() for fb in result_none["feedback_points"]),
            "Error message for None input missing."
        )

if __name__ == '__main__':
    unittest.main()
