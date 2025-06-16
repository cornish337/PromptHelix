import unittest
from unittest.mock import patch # Added import
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.genetics.engine import PromptChromosome

class TestStyleOptimizerAgent(unittest.TestCase):
    """Test suite for the StyleOptimizerAgent."""

    def setUp(self):
        """Instantiate the StyleOptimizerAgent for each test."""
        self.optimizer = StyleOptimizerAgent(knowledge_file_path=None)

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.agent_id, "StyleOptimizer")
        self.assertTrue(self.optimizer.style_rules, "Style rules should be loaded and not empty.")
        # Check default LLM and knowledge path from fallbacks (as no settings/global AGENT_SETTINGS are mocked here)
        from prompthelix.agents.style_optimizer import FALLBACK_LLM_PROVIDER, FALLBACK_LLM_MODEL, FALLBACK_KNOWLEDGE_FILE
        from prompthelix.config import KNOWLEDGE_DIR
        import os
        self.assertEqual(self.optimizer.llm_provider, FALLBACK_LLM_PROVIDER)
        self.assertEqual(self.optimizer.llm_model, FALLBACK_LLM_MODEL)
        expected_kfp = os.path.join(KNOWLEDGE_DIR, FALLBACK_KNOWLEDGE_FILE)
        self.assertEqual(self.optimizer.knowledge_file_path, expected_kfp)


    def test_agent_creation_with_settings_override(self):
        """Test agent creation with settings override."""
        override_settings = {
            "default_llm_provider": "override_style_provider",
            "default_llm_model": "override_style_model",
            "knowledge_file_path": "override_style_rules.json",
            "custom_key": "custom_style_value"
        }

        optimizer_with_override = StyleOptimizerAgent(settings=override_settings)

        self.assertEqual(optimizer_with_override.settings, override_settings)
        self.assertEqual(optimizer_with_override.llm_provider, "override_style_provider")
        self.assertEqual(optimizer_with_override.llm_model, "override_style_model")

        from prompthelix.config import KNOWLEDGE_DIR
        import os
        expected_kfp_override = os.path.join(KNOWLEDGE_DIR, "override_style_rules.json")
        self.assertEqual(optimizer_with_override.knowledge_file_path, expected_kfp_override)

    def test_agent_creation_no_settings_uses_fallbacks(self):
        """Test agent uses fallbacks if no settings dict is passed and global config is empty for it."""
        # To truly test this, we'd need to ensure AGENT_SETTINGS['StyleOptimizerAgent'] is empty or not set.
        # The current setUp already passes knowledge_file_path=None and settings=None.
        # The agent's __init__ should then use its internal FALLBACK constants.
        from prompthelix.agents.style_optimizer import FALLBACK_LLM_PROVIDER, FALLBACK_LLM_MODEL, FALLBACK_KNOWLEDGE_FILE
        from prompthelix.config import KNOWLEDGE_DIR, AGENT_SETTINGS
        import os

        # Temporarily ensure GLOBAL AGENT_SETTINGS for this agent is empty for a stricter test
        with patch.dict(AGENT_SETTINGS, {"StyleOptimizerAgent": {}}, clear=True):
            optimizer_no_settings = StyleOptimizerAgent(settings=None, knowledge_file_path="specific_style_kfp.json")

            self.assertEqual(optimizer_no_settings.llm_provider, FALLBACK_LLM_PROVIDER)
            self.assertEqual(optimizer_no_settings.llm_model, FALLBACK_LLM_MODEL)

            # kfp param should be used if settings doesn't provide it
            expected_kfp = os.path.join(KNOWLEDGE_DIR, "specific_style_kfp.json")
            self.assertEqual(optimizer_no_settings.knowledge_file_path, expected_kfp)

    def test_load_style_rules(self):
        """Test if style rules are loaded correctly."""
        self.assertIsInstance(self.optimizer.style_rules, dict)
        self.assertTrue(len(self.optimizer.style_rules) > 0)
        self.assertIn("formal", self.optimizer.style_rules)
        self.assertIn("casual", self.optimizer.style_rules)
        self.assertIn("instructional", self.optimizer.style_rules) # Based on current implementation
        self.assertIn("replace", self.optimizer.style_rules["formal"])
        self.assertIn("prepend_politeness", self.optimizer.style_rules["formal"])

    def test_process_request_formal_style(self):
        """Test process_request with 'formal' style transformation."""
        original_genes = [
            "Instruction: don't summarize stuff quickly.",
            "Context: wanna see it done well."
        ]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "formal"}
        result_chromosome = self.optimizer.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertNotEqual(result_chromosome.genes, original_genes, "Genes should have been modified.")
        
        expected_gene1_part1 = "Please Instruction: do not summarize items quickly." # Politeness + replacements
        # The exact output depends on how _load_style_rules and process_request are implemented.
        # We check for key transformations.
        self.assertTrue(result_chromosome.genes[0].startswith("Please "), "First gene should start with 'Please ' for formal style instructions.")
        self.assertIn("do not summarize items quickly.", result_chromosome.genes[0], "Formal transformations (don't->do not, stuff->items) not applied correctly to first gene.")
        self.assertIn("want to see it done well.", result_chromosome.genes[1], "Formal transformation (wanna->want to) not applied correctly to second gene.")

    def test_process_request_casual_style(self):
        """Test process_request with 'casual' style transformation."""
        original_genes = [
            "Instruction: Please do not itemize the documents meticulously.",
            "Context: Kindly provide the stuff."
        ]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "casual"}
        result_chromosome = self.optimizer.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertNotEqual(result_chromosome.genes, original_genes, "Genes should have been modified.")
        
        # Check for casual transformations
        # "Please " and "Kindly " should be removed, "do not" -> "don't", "items" -> "stuff" (though "itemize" might not be in casual rules)
        self.assertNotIn("Please", result_chromosome.genes[0], "'Please' not removed for casual style.")
        self.assertIn("Instruction: don't itemize the documents meticulously.", result_chromosome.genes[0], "Casual transformation (do not->don't) not applied correctly.")
        self.assertNotIn("Kindly", result_chromosome.genes[1], "'Kindly' not removed for casual style.")
        self.assertIn("Context: provide the stuff.", result_chromosome.genes[1], "Casual transformation for 'stuff' or removal of 'Kindly' failed.")


    def test_process_request_unrecognized_style(self):
        """Test process_request with an unrecognized target style."""
        original_genes = ["Instruction: Test this.", "Context: Unchanged."]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "non_existent_style"}
        result_chromosome = self.optimizer.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        # The agent's current implementation returns the original chromosome object if style is not found
        self.assertEqual(result_chromosome, original_prompt, "Should return original chromosome for unrecognized style.")
        self.assertEqual(result_chromosome.genes, original_genes, "Genes should be unchanged for unrecognized style.")

    def test_process_request_invalid_input(self):
        """Test process_request with invalid 'prompt_chromosome' input."""
        invalid_prompt_input = "This is not a PromptChromosome object."
        request_data = {"prompt_chromosome": invalid_prompt_input, "target_style": "formal"}
        result = self.optimizer.process_request(request_data)

        # The agent's current implementation returns the original input if it's not a PromptChromosome
        self.assertEqual(result, invalid_prompt_input, "Should return the original invalid input.")

        request_data_none = {"prompt_chromosome": None, "target_style": "formal"}
        result_none = self.optimizer.process_request(request_data_none)
        self.assertIsNone(result_none, "Should return None for None input.")

if __name__ == '__main__':
    unittest.main()
