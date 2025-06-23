import unittest
from unittest.mock import patch, MagicMock
import json # Though not directly used, good for complex mock return values
import asyncio # Ensure asyncio is imported
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.genetics.chromosome import PromptChromosome
from prompthelix.config import AGENT_SETTINGS as GLOBAL_AGENT_SETTINGS

# Default config for architect if not overridden by specific test patches
DEFAULT_ARCHITECT_CONFIG = {
    "default_llm_provider": "test_openai",
    "default_llm_model": "test_gpt-3.5-turbo",
}

class TestPromptArchitectAgent(unittest.TestCase):
    """Test suite for the PromptArchitectAgent."""

    def setUp(self):
        """Instantiate the PromptArchitectAgent for each test, mocking config."""
        # Patch AGENT_SETTINGS for all tests in this class or do it per test
        # For simplicity here, we assume a default mock if a test doesn't provide its own
        self.architect_config_patch = patch.dict(GLOBAL_AGENT_SETTINGS, {'PromptArchitectAgent': DEFAULT_ARCHITECT_CONFIG.copy()}, clear=True)
        self.mock_agent_settings = self.architect_config_patch.start()

        # This architect instance will be used by tests that don't specify their own settings
        self.architect = PromptArchitectAgent(knowledge_file_path=None)

    def tearDown(self):
        self.architect_config_patch.stop()

    def test_agent_creation_and_default_config_loading(self):
        """Test basic creation and initialization of the agent, and config loading."""
        # Stop the class-level patch to set up a specific scenario for this test
        self.architect_config_patch.stop()

        # For this test, we want AGENT_SETTINGS to be effectively empty for PromptArchitectAgent
        # so that module-level fallbacks are used.
        with patch.dict(GLOBAL_AGENT_SETTINGS, {'PromptArchitectAgent': {}}, clear=True):
            # Create a new agent instance within this specific patch context
            architect_for_test = PromptArchitectAgent(knowledge_file_path=None)

        self.assertIsNotNone(architect_for_test)
        self.assertEqual(architect_for_test.agent_id, "PromptArchitect")
        self.assertTrue(architect_for_test.templates, "Templates should be loaded and not empty.")
        self.assertIn("summary_v1", architect_for_test.templates, "Default summary template should be loaded.")

        from prompthelix.agents.architect import FALLBACK_LLM_PROVIDER, FALLBACK_LLM_MODEL
        self.assertEqual(architect_for_test.llm_provider, FALLBACK_LLM_PROVIDER)
        self.assertEqual(architect_for_test.llm_model, FALLBACK_LLM_MODEL)

        # Check default knowledge file path
        from prompthelix.config import KNOWLEDGE_DIR
        import os
        expected_kfp = os.path.join(KNOWLEDGE_DIR, "architect_knowledge.json") # FALLBACK_KNOWLEDGE_FILE
        self.assertEqual(self.architect.knowledge_file_path, expected_kfp)

    def test_agent_creation_with_settings_override(self):
        """Test agent creation with settings override."""
        override_settings = {
            "default_llm_provider": "override_provider",
            "default_llm_model": "override_model",
            "knowledge_file_path": "override_path.json",
            "custom_key": "custom_value"
        }
        # Stop the class-level patch to ensure this test uses its own context for settings
        self.architect_config_patch.stop()
        # We are not patching GLOBAL_AGENT_SETTINGS here, so defaults inside the agent will be used if not in override.
        # The 'settings' param to agent constructor is the *final* settings dict for that agent.

        architect_with_override = PromptArchitectAgent(settings=override_settings)

        self.assertEqual(architect_with_override.settings, override_settings)
        self.assertEqual(architect_with_override.llm_provider, "override_provider")
        self.assertEqual(architect_with_override.llm_model, "override_model")

        from prompthelix.config import KNOWLEDGE_DIR
        import os
        expected_kfp_override = os.path.join(KNOWLEDGE_DIR, "override_path.json")
        self.assertEqual(architect_with_override.knowledge_file_path, expected_kfp_override)

        # Restart the class-level patch for other tests
        self.mock_agent_settings = self.architect_config_patch.start()


    def test_agent_creation_with_settings_override_and_kfp_param(self):
        """Test kfp param takes precedence if settings also has it (agent specific logic)."""
        # The current PromptArchitectAgent __init__ logic is:
        # _knowledge_file = self.settings.get("knowledge_file_path", knowledge_file_path_param)
        # So settings['knowledge_file_path'] wins over knowledge_file_path_param.
        # Let's test this specific precedence.

        settings_with_kfp = {"knowledge_file_path": "settings_kfp.json"}
        direct_kfp_param = "direct_kfp.json"

        architect = PromptArchitectAgent(settings=settings_with_kfp, knowledge_file_path=direct_kfp_param)

        from prompthelix.config import KNOWLEDGE_DIR
        import os
        expected_kfp = os.path.join(KNOWLEDGE_DIR, "settings_kfp.json")
        self.assertEqual(architect.knowledge_file_path, expected_kfp, "knowledge_file_path from settings should take precedence.")

        settings_without_kfp = {"some_other_setting": "value"}
        architect2 = PromptArchitectAgent(settings=settings_without_kfp, knowledge_file_path=direct_kfp_param)
        expected_kfp2 = os.path.join(KNOWLEDGE_DIR, direct_kfp_param)
        self.assertEqual(architect2.knowledge_file_path, expected_kfp2, "knowledge_file_path from param should be used if not in settings.")


    def test_agent_creation_no_settings_uses_fallbacks(self):
        """Test agent uses fallbacks if no settings dict is passed and global config is empty for it."""
        self.architect_config_patch.stop() # Stop class-level patch

        # Patch GLOBAL_AGENT_SETTINGS to be empty for this agent type temporarily
        with patch.dict(GLOBAL_AGENT_SETTINGS, {'PromptArchitectAgent': {}}, clear=True):
            architect_no_settings = PromptArchitectAgent(settings=None, knowledge_file_path="specific_kfp.json")

        # It should use hardcoded fallbacks from the top of architect.py
        from prompthelix.agents.architect import FALLBACK_LLM_PROVIDER, FALLBACK_LLM_MODEL
        from prompthelix.config import KNOWLEDGE_DIR
        import os

        self.assertEqual(architect_no_settings.llm_provider, FALLBACK_LLM_PROVIDER)
        self.assertEqual(architect_no_settings.llm_model, FALLBACK_LLM_MODEL)
        # kfp param should be used
        expected_kfp = os.path.join(KNOWLEDGE_DIR, "specific_kfp.json")
        self.assertEqual(architect_no_settings.knowledge_file_path, expected_kfp)

        self.mock_agent_settings = self.architect_config_patch.start() # Restart class-level patch

    # --- Tests for _parse_requirements ---
    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_parse_requirements_llm_success(self, mock_call_llm_api):
        """Test _parse_requirements with successful LLM call."""
        # Use AsyncMock if the mocked function is async and its await status matters
        # For a simple return value that's awaited, MagicMock can work if its return is an awaitable
        future_val = asyncio.Future()
        future_val.set_result(json.dumps({
            "task_description": "Parsed task",
            "keywords": ["kw1", "kw2"],
            "constraints": {"max_len": 100}
        }))
        mock_call_llm_api.return_value = future_val

        parsed = await self.architect._parse_requirements("Original task", ["kw1"], {"max_len_orig": 150})

        mock_call_llm_api.assert_called_once()
        self.assertEqual(parsed["task_description"], "Parsed task")
        self.assertEqual(parsed["keywords"], ["kw1", "kw2"])
        self.assertEqual(parsed["constraints"], {"max_len": 100})

    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_parse_requirements_llm_failure_falls_back(self, mock_call_llm_api):
        """Test _parse_requirements falls back when LLM fails."""
        future_val = asyncio.Future()
        future_val.set_exception(Exception("LLM Error"))
        mock_call_llm_api.return_value = future_val

        parsed = await self.architect._parse_requirements("Task X", ["kwX"], {})
        mock_call_llm_api.assert_called_once()
        self.assertEqual(parsed["task_description"], "Task X") # Fallback uses original
        self.assertEqual(parsed["keywords"], ["kwX"]) # Fallback uses original
        self.assertIn("error", parsed) # Should contain error info from fallback

    # --- Tests for _select_template ---
    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_select_template_llm_success(self, mock_call_llm_api):
        """Test _select_template with successful LLM call."""
        future_val = asyncio.Future()
        future_val.set_result(json.dumps({"template": "summary_v1"}))
        mock_call_llm_api.return_value = future_val

        parsed_reqs = {"task_description": "Summarize this document."}
        template_name = await self.architect._select_template(parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        self.assertEqual(template_name, "summary_v1")

    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    @patch('prompthelix.agents.architect.PromptArchitectAgent._fallback_select_template', new_callable=MagicMock) # Mock the fallback
    async def test_select_template_llm_invalid_response_falls_back(self, mock_fallback_select, mock_call_llm_api):
        """Test _select_template falls back if LLM returns invalid template name."""
        llm_future = asyncio.Future()
        llm_future.set_result(json.dumps({"template": "invalid_template_name"}))
        mock_call_llm_api.return_value = llm_future

        fallback_future = asyncio.Future()
        fallback_future.set_result("generic_v1") # Fallback returns this
        mock_fallback_select.return_value = fallback_future

        parsed_reqs = {"task_description": "A generic task."}
        template_name = await self.architect._select_template(parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        mock_fallback_select.assert_called_once_with(parsed_reqs)
        self.assertEqual(template_name, "generic_v1")


    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    @patch('prompthelix.agents.architect.PromptArchitectAgent._fallback_select_template', new_callable=MagicMock)
    async def test_select_template_llm_failure_falls_back(self, mock_fallback_select, mock_call_llm_api):
        """Test _select_template falls back when LLM fails."""
        llm_future = asyncio.Future()
        llm_future.set_exception(Exception("LLM Error"))
        mock_call_llm_api.return_value = llm_future

        fallback_future = asyncio.Future()
        fallback_future.set_result("summary_v1")
        mock_fallback_select.return_value = fallback_future

        parsed_reqs = {"task_description": "Summarize this for me."}
        template_name = await self.architect._select_template(parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback_select.assert_called_once_with(parsed_reqs)
        self.assertEqual(template_name, "summary_v1")

    # --- Tests for _populate_genes ---
    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_populate_genes_llm_success(self, mock_call_llm_api):
        """Test _populate_genes with successful LLM call."""
        future_val = asyncio.Future()
        future_val.set_result(json.dumps({
            "genes": [
                "Instruction: LLM generated instruction",
                "Context: LLM context",
                "Output Format: LLM format"
            ]
        }))
        mock_call_llm_api.return_value = future_val

        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something.", "keywords": ["detail"]}
        genes = await self.architect._populate_genes(template, parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        self.assertEqual(len(genes), 3)
        self.assertEqual(genes[0], "Instruction: LLM generated instruction")
        self.assertEqual(genes[1], "Context: LLM context")

    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    @patch('prompthelix.agents.architect.PromptArchitectAgent._fallback_populate_genes', new_callable=MagicMock)
    async def test_populate_genes_llm_malformed_response_falls_back(self, mock_fallback_populate, mock_call_llm_api):
        """Test _populate_genes falls back if LLM returns malformed (but not erroring) data."""
        llm_future = asyncio.Future()
        llm_future.set_result("{invalid_json") # Malformed JSON string
        mock_call_llm_api.return_value = llm_future

        fallback_future = asyncio.Future()
        fallback_future.set_result(["Fallback instruction", "Fallback context"])
        mock_fallback_populate.return_value = fallback_future

        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something complex."}
        genes = await self.architect._populate_genes(template, parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback_populate.assert_called_once_with(template, parsed_reqs)
        self.assertEqual(genes[0], "Fallback instruction")


    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    @patch('prompthelix.agents.architect.PromptArchitectAgent._fallback_populate_genes', new_callable=MagicMock)
    async def test_populate_genes_llm_failure_falls_back(self, mock_fallback_populate, mock_call_llm_api):
        """Test _populate_genes falls back when LLM fails."""
        llm_future = asyncio.Future()
        llm_future.set_exception(Exception("LLM Error"))
        mock_call_llm_api.return_value = llm_future

        fallback_future = asyncio.Future()
        fallback_future.set_result(["Fallback gene"])
        mock_fallback_populate.return_value = fallback_future

        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something."}
        genes = await self.architect._populate_genes(template, parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback_populate.assert_called_once_with(template, parsed_reqs)
        self.assertEqual(genes[0], "Fallback gene")

    # --- Edge Cases for process_request ---
    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_process_request_missing_keywords_and_constraints(self, mock_call_llm_api):
        """Test process_request with missing keywords and constraints."""
        async def mock_llm_side_effect(*args, **kwargs):
            # Simulate different responses for each call based on prompt content or call order
            # This is a simplified example; real logic might inspect args/kwargs
            if mock_call_llm_api.call_count == 1: # For _parse_requirements
                return json.dumps({"task_description": "Default task description", "keywords": [], "constraints": {}})
            elif mock_call_llm_api.call_count == 2: # For _select_template
                return json.dumps({"template": "generic_v1"})
            elif mock_call_llm_api.call_count == 3: # For _populate_genes
                return json.dumps({"genes": [
                    "LLM Instruction: Default",
                    "Context: Default context with no keywords",
                    "Output: Default format"
                ]})
            return "" # Default empty response

        mock_call_llm_api.side_effect = mock_llm_side_effect

        request_data = {"task_description": "A simple task"}
        chromosome = await self.architect.process_request(request_data)

        self.assertIsInstance(chromosome, PromptChromosome)
        self.assertTrue(len(chromosome.genes) > 0)
        self.assertEqual(chromosome.genes[0], "LLM Instruction: Default")
        self.assertEqual(mock_call_llm_api.call_count, 3)

    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_process_request_empty_task_description(self, mock_llm):
        """Test process_request with an empty string for task_description."""
        async def mock_llm_side_effect_empty_task(*args, **kwargs):
            if mock_llm.call_count == 1:
                return json.dumps({"task_description": "empty task", "keywords": [], "constraints": {}})
            elif mock_llm.call_count == 2:
                return json.dumps({"template": "generic_v1"})
            elif mock_llm.call_count == 3:
                return json.dumps({"genes": [
                    "Instruction: Handle empty",
                    "Context: Empty context",
                    "Output: As specified"
                ]})
            return ""
        mock_llm.side_effect = mock_llm_side_effect_empty_task

        request_data = {"task_description": "", "keywords": [], "constraints": {}}
        chromosome = await self.architect.process_request(request_data)

        self.assertIsInstance(chromosome, PromptChromosome)
        self.assertTrue(len(chromosome.genes) > 0)
        self.assertEqual(chromosome.genes[0], "Instruction: Handle empty")

        args_select_template, _ = mock_llm.call_args_list[1] # Second call is _select_template
        prompt_for_select_template = args_select_template[0] # First arg to call_llm_api is the prompt
        self.assertIn('Given the task description: "empty task"', prompt_for_select_template)


    @patch('prompthelix.agents.architect.call_llm_api', new_callable=MagicMock)
    async def test_process_request_full_fallback_due_to_llm_system_down(self, mock_call_llm_api_system_down):
        """Test process_request uses full fallback logic if all LLM calls fail."""
        async def raise_llm_error(*args, **kwargs):
            raise Exception("LLM System Down")
        mock_call_llm_api_system_down.side_effect = raise_llm_error

        request_data = {
            "task_description": "Summarize this story about AI and ethics.",
            "keywords": ["AI", "ethics"],
            "constraints": {"max_length": 150}
        }
        result_chromosome = await self.architect.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertTrue(result_chromosome.genes, "Genes list should not be empty.")
        self.assertTrue(
            any("Summarize the following text:" in gene for gene in result_chromosome.genes),
            "Summary instruction (fallback) not found in genes."
        )
        context_gene_found = False
        for gene in result_chromosome.genes:
            if "Context:" in gene:
                context_gene_found = True
                self.assertIn("AI", gene)
                self.assertIn("ethics", gene)
                self.assertIn("Summarize this story about AI and ethics.", gene) # Fallback populates context
                break
        self.assertTrue(context_gene_found, "Context gene (fallback) not found.")
        self.assertEqual(mock_call_llm_api_system_down.call_count, 3) # LLM was attempted for all 3 steps


if __name__ == '__main__':
    unittest.main()
