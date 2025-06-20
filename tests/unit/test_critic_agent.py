import os
import tempfile
import unittest

from prompthelix.agents.critic import PromptCriticAgent

class TestPromptCriticAgentCustomFile(unittest.TestCase):
    def test_initialization_with_custom_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_path = os.path.join(tmpdir, "custom_rules.json")
            agent = PromptCriticAgent(knowledge_file_path=custom_path)
            self.assertEqual(agent.knowledge_file_path, custom_path)
            self.assertIsInstance(agent.rules, list)

if __name__ == "__main__":
    unittest.main()
