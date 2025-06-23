import unittest
from unittest.mock import patch, MagicMock, AsyncMock # Added MagicMock, AsyncMock
import asyncio # Added asyncio
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
from prompthelix.genetics.chromosome import PromptChromosome

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

    async def test_process_request_formal_style(self):
        """Test process_request with 'formal' style transformation."""
        original_genes = [
            "Instruction: don't summarize stuff quickly.",
            "Context: wanna see it done well."
        ]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "formal"}
        # Since process_request is now async
        with patch('prompthelix.agents.style_optimizer.call_llm_api', new_callable=AsyncMock) as mock_llm_api:
            # Simulate LLM returning genes that then get rule-based processed, or just let rule-based run
            mock_llm_api.side_effect = Exception("Simulated LLM failure to test rule-based path")
            result_chromosome = await self.optimizer.process_request(request_data)


        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertNotEqual(result_chromosome.genes, original_genes, "Genes should have been modified.")
        
        self.assertTrue(result_chromosome.genes[0].startswith("Please "), "First gene should start with 'Please ' for formal style instructions.")
        self.assertIn("do not summarize items quickly.", result_chromosome.genes[0], "Formal transformations (don't->do not, stuff->items) not applied correctly to first gene.")
        self.assertIn("want to see it done well.", result_chromosome.genes[1], "Formal transformation (wanna->want to) not applied correctly to second gene.")

    async def test_process_request_casual_style(self):
        """Test process_request with 'casual' style transformation."""
        original_genes = [
            "Instruction: Please do not itemize the documents meticulously.",
            "Context: Kindly provide the stuff."
        ]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "casual"}
        with patch('prompthelix.agents.style_optimizer.call_llm_api', new_callable=AsyncMock) as mock_llm_api:
            mock_llm_api.side_effect = Exception("Simulated LLM failure to test rule-based path")
            result_chromosome = await self.optimizer.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertNotEqual(result_chromosome.genes, original_genes, "Genes should have been modified.")
        
        self.assertNotIn("Please", result_chromosome.genes[0], "'Please' not removed for casual style.")
        self.assertIn("Instruction: don't itemize the documents meticulously.", result_chromosome.genes[0], "Casual transformation (do not->don't) not applied correctly.")
        self.assertNotIn("Kindly", result_chromosome.genes[1], "'Kindly' not removed for casual style.")
        self.assertIn("Context: provide the stuff.", result_chromosome.genes[1], "Casual transformation for 'stuff' or removal of 'Kindly' failed.")


    async def test_process_request_unrecognized_style(self):
        """Test process_request with an unrecognized target style."""
        original_genes = ["Instruction: Test this.", "Context: Unchanged."]
        original_prompt = PromptChromosome(genes=original_genes)
        request_data = {"prompt_chromosome": original_prompt, "target_style": "non_existent_style"}
        with patch('prompthelix.agents.style_optimizer.call_llm_api', new_callable=AsyncMock) as mock_llm_api:
            mock_llm_api.side_effect = Exception("Simulated LLM failure") # Ensure LLM path fails
            result_chromosome = await self.optimizer.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertEqual(result_chromosome, original_prompt, "Should return original chromosome for unrecognized style if LLM fails and no rules.")
        self.assertEqual(result_chromosome.genes, original_genes, "Genes should be unchanged for unrecognized style if LLM fails and no rules.")

    async def test_process_request_invalid_input(self):
        """Test process_request with invalid 'prompt_chromosome' input."""
        invalid_prompt_input = "This is not a PromptChromosome object."
        request_data = {"prompt_chromosome": invalid_prompt_input, "target_style": "formal"}
        result = await self.optimizer.process_request(request_data)
        self.assertEqual(result, invalid_prompt_input, "Should return the original invalid input.")

        request_data_none = {"prompt_chromosome": None, "target_style": "formal"}
        result_none = await self.optimizer.process_request(request_data_none)
        self.assertIsNone(result_none, "Should return None for None input.")

    # --- Tests for the new optimize method ---

    @patch('prompthelix.agents.style_optimizer.call_llm_api', new_callable=AsyncMock)
    async def test_optimize_real_mode(self, mock_call_llm_api):
        """Test optimize method in 'REAL' LLM mode."""
        future_val = asyncio.Future()
        future_val.set_result("Optimized: Be concise and clear.")
        mock_call_llm_api.return_value = future_val

        settings = {"llm_mode": "REAL", "default_llm_provider": "test_provider", "default_llm_model": "test_model"}
        agent = StyleOptimizerAgent(settings=settings)

        original_prompt = "Make this prompt better now."
        tone = "professional"
        expected_llm_template = f"Rephrase the following prompt to be more {tone}: {original_prompt}"

        result = await agent.optimize(original_prompt, tone=tone)

        self.assertEqual(result, "Optimized: Be concise and clear.")
        # For async mocks, assert_called_once_with might need adjustment if args are complex
        # or use await mock_call_llm_api.assert_awaited_once_with(...) if it's an AsyncMock
        mock_call_llm_api.assert_called_once_with(
            prompt=expected_llm_template, # Changed from prompt_text to prompt to match call_llm_api
            provider="test_provider",
            model="test_model",
            db=None # call_llm_api expects db
        )

    async def test_optimize_placeholder_mode(self):
        """Test optimize method in non-REAL (e.g., 'PLACEHOLDER') LLM mode."""
        settings = {"llm_mode": "PLACEHOLDER"} # Explicitly set to non-REAL
        agent = StyleOptimizerAgent(settings=settings)

        original_prompt = "Make this prompt better now."
        tone = "friendly"
        expected_output = f"{original_prompt} [Styled: Placeholder]"

        result = await agent.optimize(original_prompt, tone=tone)
        self.assertEqual(result, expected_output)

    async def test_optimize_default_mode_is_placeholder(self):
        """Test optimize method defaults to placeholder behavior if llm_mode is not set."""
        # Agent initialized with no settings or settings missing llm_mode
        agent_no_settings = StyleOptimizerAgent(settings=None)
        original_prompt = "Another prompt"
        tone = "direct"
        expected_output = f"{original_prompt} [Styled: Placeholder]"
        result = await agent_no_settings.optimize(original_prompt, tone=tone)
        self.assertEqual(result, expected_output)

        agent_empty_settings = StyleOptimizerAgent(settings={})
        result_empty_settings = await agent_empty_settings.optimize(original_prompt, tone=tone)
        self.assertEqual(result_empty_settings, expected_output)


    @patch('prompthelix.agents.style_optimizer.call_llm_api', new_callable=MagicMock)
    async def test_optimize_real_mode_llm_api_error(self, mock_call_llm_api):
        """Test optimize method in 'REAL' mode when call_llm_api raises an error."""
        future_val = asyncio.Future()
        future_val.set_exception(Exception("LLM API failed"))
        mock_call_llm_api.return_value = future_val

        settings = {"llm_mode": "REAL", "default_llm_provider": "test_provider", "default_llm_model": "test_model"}
        agent = StyleOptimizerAgent(settings=settings)

        original_prompt = "A prompt that will cause an error."
        tone = "urgent"
        expected_output = f"{original_prompt} [Styled: Placeholder - Error]"

        result = await agent.optimize(original_prompt, tone=tone)

        self.assertEqual(result, expected_output)
        mock_call_llm_api.assert_called_once()


if __name__ == '__main__':
    unittest.main()
