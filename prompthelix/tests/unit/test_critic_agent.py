import unittest
import json
import re # Not strictly needed for tests unless testing regex patterns themselves
from unittest.mock import patch, mock_open, MagicMock

from prompthelix.agents.critic import PromptCriticAgent
# Removed: from prompthelix.genetics.engine import PromptChromosome

# Define a path for a mock knowledge file if needed for some tests
# However, for process_prompt, directly mocking agent.rules is often easier.
MOCK_RULES_FILE_PATH = "mock_knowledge/best_practices_rules.json"

MOCK_RULES_CONTENT = [
    {
        "name": "TestActiveVoice",
        "pattern": "is tested",
        "feedback": "Mock: Use active voice.",
        "penalty": 1
    },
    {
        "name": "TestUseExamples",
        "pattern": "^(?!.*example).*$", # Simplified: finds if 'example' is missing
        "feedback": "Mock: Add examples.",
        "penalty": 2
    },
    {
        "name": "TestAvoidVague",
        "pattern": "stuff|things",
        "feedback": "Mock: Avoid vague terms.",
        "penalty": 1
    }
]

class TestPromptCriticAgentNew(unittest.TestCase):
    """Test suite for the refactored PromptCriticAgent."""

    def setUp(self):
        """Instantiate the PromptCriticAgent for each test."""
        # Initialize with a non-existent default path to ensure tests mock rules correctly
        # or to test behavior when the real file is not found (if intended).
        # For most process_prompt tests, we'll mock agent.rules directly.
        self.critic = PromptCriticAgent(knowledge_file_path="non_existent_rules.json")


    def test_agent_creation_and_initial_state(self):
        """Test basic creation and initial state of the agent."""
        self.assertIsNotNone(self.critic)
        self.assertEqual(self.critic.agent_id, "PromptCritic")
        self.assertEqual(self.critic.knowledge_file_path, "non_existent_rules.json")
        # Since "non_existent_rules.json" won't be found, self.rules should be empty
        self.assertEqual(self.critic.rules, [])

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps(MOCK_RULES_CONTENT))
    def test_load_knowledge_success(self, mock_file_open):
        """Test if rules are loaded correctly when the JSON file exists."""
        critic_with_rules = PromptCriticAgent(knowledge_file_path=MOCK_RULES_FILE_PATH)
        mock_file_open.assert_called_once_with(MOCK_RULES_FILE_PATH, 'r')
        self.assertEqual(len(critic_with_rules.rules), len(MOCK_RULES_CONTENT))
        self.assertEqual(critic_with_rules.rules[0]["name"], "TestActiveVoice")

    def test_load_knowledge_file_not_found(self):
        """Test behavior when the knowledge file is not found."""
        # setUp already initializes with a non-existent file
        # We expect rules to be empty and an error logged (checking logs is harder here)
        self.assertEqual(self.critic.rules, [])

    @patch("builtins.open", new_callable=mock_open, read_data="invalid json")
    def test_load_knowledge_json_decode_error(self, mock_file_open):
        """Test behavior with invalid JSON content."""
        critic_bad_json = PromptCriticAgent(knowledge_file_path="bad_rules.json")
        mock_file_open.assert_called_once_with("bad_rules.json", 'r')
        self.assertEqual(critic_bad_json.rules, []) # Should be empty on error

    def test_process_prompt_no_rules_loaded(self):
        """Test process_prompt when no rules are loaded."""
        self.critic.rules = [] # Ensure no rules
        prompt = "This is a test prompt."
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 10)
        self.assertIn("Warning: No rules loaded for critique.", result["feedback"])

    def test_process_prompt_invalid_prompt_type(self):
        """Test process_prompt with non-string input."""
        result = self.critic.process_prompt(12345)
        self.assertEqual(result["score"], 0)
        self.assertIn("Error: Invalid prompt type. Expected string.", result["feedback"])

    def test_process_prompt_no_violations(self):
        """Test process_prompt with a prompt that violates no rules."""
        self.critic.rules = MOCK_RULES_CONTENT
        prompt = "This active prompt includes an example and specific terms."
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 10)
        self.assertEqual(result["feedback"], [])

    def test_process_prompt_one_violation(self):
        """Test process_prompt with a prompt that violates one rule."""
        self.critic.rules = MOCK_RULES_CONTENT
        prompt = "This prompt is tested with an example."  # Violates TestActiveVoice only
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 9) # 10 - 1
        self.assertIn("Mock: Use active voice.", result["feedback"])

    def test_process_prompt_multiple_violations(self):
        """Test process_prompt with a prompt that violates multiple rules."""
        self.critic.rules = MOCK_RULES_CONTENT
        prompt = "This stuff is tested." # Violates TestActiveVoice, TestUseExamples, TestAvoidVague
        result = self.critic.process_prompt(prompt)
        # Score: 10 - 1 (active) - 2 (example) - 1 (vague) = 6
        self.assertEqual(result["score"], 6)
        self.assertIn("Mock: Use active voice.", result["feedback"])
        self.assertIn("Mock: Add examples.", result["feedback"])
        self.assertIn("Mock: Avoid vague terms.", result["feedback"])

    def test_process_prompt_penalty_calculation(self):
        """Test that penalties are correctly subtracted."""
        self.critic.rules = [
            {"name": "Rule1", "pattern": "rule1", "feedback": "fb1", "penalty": 3},
            {"name": "Rule2", "pattern": "rule2", "feedback": "fb2", "penalty": 2}
        ]
        prompt = "This prompt triggers rule1 and rule2."
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 5) # 10 - 3 - 2

    def test_process_prompt_score_does_not_go_below_zero(self):
        """Test that the score does not go below zero."""
        self.critic.rules = [
            {"name": "HeavyPenalty", "pattern": ".*", "feedback": "fb", "penalty": 15}
        ]
        prompt = "Any prompt."
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 0)

    def test_process_prompt_empty_prompt_string(self):
        """Test process_prompt with an empty string."""
        self.critic.rules = MOCK_RULES_CONTENT # Use mock rules
        prompt = ""
        # Active voice: no "is Xed" - PASS
        # Use Examples: no "example" - FAIL (penalty 2)
        # Avoid Vague: no "stuff" or "things" - PASS
        result = self.critic.process_prompt(prompt)
        self.assertEqual(result["score"], 8) # 10 - 2
        self.assertIn("Mock: Add examples.", result["feedback"])
        self.assertEqual(len(result["feedback"]), 1)


    def test_process_prompt_rule_with_invalid_regex(self):
        """Test graceful handling of a rule with an invalid regex pattern."""
        self.critic.rules = [
            {"name": "GoodRule", "pattern": "good", "feedback": "Good feedback", "penalty": 1},
            {"name": "InvalidRegexRule", "pattern": "[", "feedback": "Invalid regex feedback", "penalty": 1},
            {"name": "AnotherGoodRule", "pattern": "another", "feedback": "Another good feedback", "penalty": 1}
        ]
        # Mock logger to check for error messages
        with patch.object(self.critic.logger, 'error') as mock_log_error:
            prompt = "This prompt has good content and another."
            result = self.critic.process_prompt(prompt)

        # Score: 10 - 1 (GoodRule) - 1 (AnotherGoodRule) = 8. InvalidRegexRule is skipped.
        self.assertEqual(result["score"], 8)
        self.assertIn("Good feedback", result["feedback"])
        self.assertIn("Another good feedback", result["feedback"])
        self.assertNotIn("Invalid regex feedback", result["feedback"])

        # Check that an error was logged for the invalid regex
        self.assertTrue(any("Regex error in rule 'InvalidRegexRule'" in call_args[0][0] for call_args in mock_log_error.call_args_list))

    def test_process_prompt_rule_missing_keys(self):
        """Test graceful handling of a rule missing required keys."""
        self.critic.rules = [
            {"name": "MissingPattern", "feedback": "fb", "penalty": 1}, # Missing pattern
            {"name": "MissingFeedback", "pattern": "abc", "penalty": 1}, # Missing feedback (though process_prompt doesn't use it if no match)
            {"name": "MissingPenalty", "name": "NoPenaltyRule", "pattern": "xyz", "feedback": "fb_no_penalty"}, # Uses default penalty
            {"name": "ValidRule", "pattern": "valid", "feedback": "fb_valid", "penalty": 1}
        ]
        prompt_text = "trigger valid and xyz"
        with patch.object(self.critic.logger, 'error') as mock_log_error:
            result = self.critic.process_prompt(prompt_text)

        # Expected: ValidRule (-1), NoPenaltyRule (-1, default)
        self.assertEqual(result["score"], 8)
        self.assertIn("fb_valid", result["feedback"])
        self.assertIn("fb_no_penalty", result["feedback"])
        
        # Check logs for errors about malformed rules
        self.assertTrue(any("Invalid rule structure for rule 'MissingPattern'" in call_args[0][0] and "Missing key: 'pattern'" in call_args[0][0] for call_args in mock_log_error.call_args_list))
        # Missing feedback might not log an error unless the pattern matches. If it doesn't match, feedback isn't accessed.
        # If "xyz" matches, and feedback was missing, it would error when trying to append. Let's refine the test.

        # Test case for missing feedback when pattern matches
        self.critic.rules = [{"name": "ActuallyMissingFeedback", "pattern": "match_me", "penalty": 1}]
        with patch.object(self.critic.logger, 'error') as mock_log_error_feedback:
             result_missing_fb = self.critic.process_prompt("match_me")
        self.assertEqual(result_missing_fb["score"], 9) # Penalty applied
        # issues.append(rule["feedback"]) would cause KeyError
        self.assertTrue(any("Invalid rule structure for rule 'ActuallyMissingFeedback'" in call_args[0][0] and "Missing key: 'feedback'" in call_args[0][0] for call_args in mock_log_error_feedback.call_args_list))
        # The feedback list would not contain None or error, it just wouldn't add that specific feedback.
        # The current implementation in critic.py would attempt issues.append(rule["feedback"]) and fail.
        # The provided critic.py code has try-except for KeyError.
        self.assertEqual(result_missing_fb["feedback"], []) # Feedback append fails due to KeyError

if __name__ == '__main__':
    unittest.main()
