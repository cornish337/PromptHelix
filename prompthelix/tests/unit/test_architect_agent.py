import unittest
from unittest.mock import patch, MagicMock
import json # Though not directly used, good for complex mock return values
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.genetics.engine import PromptChromosome
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
        self.architect_config_patch = patch.dict(GLOBAL_AGENT_SETTINGS, {'PromptArchitectAgent': DEFAULT_ARCHITECT_CONFIG.copy()})
        self.mock_agent_settings = self.architect_config_patch.start()
        self.architect = PromptArchitectAgent() # Uses mocked settings

    def tearDown(self):
        self.architect_config_patch.stop()

    def test_agent_creation_and_config_loading(self):
        """Test basic creation and initialization of the agent, and config loading."""
        self.assertIsNotNone(self.architect)
        self.assertEqual(self.architect.agent_id, "PromptArchitect")
        self.assertTrue(self.architect.templates, "Templates should be loaded and not empty.")
        self.assertIn("summary_v1", self.architect.templates, "Default summary template should be loaded.")
        # Check if LLM provider and model are loaded from (mocked) config
        self.assertEqual(self.architect.llm_provider, DEFAULT_ARCHITECT_CONFIG["default_llm_provider"])
        self.assertEqual(self.architect.llm_model, DEFAULT_ARCHITECT_CONFIG["default_llm_model"])

    # --- Tests for _parse_requirements ---
    @patch('prompthelix.utils.llm_utils.call_llm_api')
    def test_parse_requirements_llm_success(self, mock_call_llm_api):
        """Test _parse_requirements with successful LLM call."""
        mock_llm_response = '{"task_description": "Parsed task", "parsed_keywords": ["kw1", "llm_added_keyword"], "parsed_constraints": {"max_len": 100}}'
        # Note: The current _parse_requirements simulates parsing, it doesn't use json.loads on the response.
        # It returns a mix of original inputs and a hardcoded "llm_added_keyword".
        # The mock_llm_response here is what the LLM *would* return, but the method's behavior is different.
        # We'll test the method's actual behavior.
        mock_call_llm_api.return_value = "LLM processed: Parsed task, keywords: kw1, constraints: max_len 100"

        parsed = self.architect._parse_requirements("Original task", ["kw1"], {"max_len_orig": 150})

        mock_call_llm_api.assert_called_once()
        self.assertEqual(parsed["task_description"], "Original task") # Current behavior keeps original task
        self.assertIn("llm_added_keyword", parsed["keywords"]) # Current behavior adds this
        self.assertIn("LLM processed", parsed["llm_raw_response_parsing"])

    @patch('prompthelix.utils.llm_utils.call_llm_api', side_effect=Exception("LLM Error"))
    def test_parse_requirements_llm_failure_falls_back(self, mock_call_llm_api):
        """Test _parse_requirements falls back when LLM fails."""
        parsed = self.architect._parse_requirements("Task X", ["kwX"], {})
        mock_call_llm_api.assert_called_once()
        self.assertEqual(parsed["task_description"], "Task X")
        self.assertEqual(parsed["keywords"], ["kwX"])
        self.assertIn("error", parsed) # Should contain error info

    # --- Tests for _select_template ---
    @patch('prompthelix.utils.llm_utils.call_llm_api')
    def test_select_template_llm_success(self, mock_call_llm_api):
        """Test _select_template with successful LLM call."""
        mock_call_llm_api.return_value = "summary_v1" # LLM suggests a valid template
        parsed_reqs = {"task_description": "Summarize this document."}
        template_name = self.architect._select_template(parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        self.assertEqual(template_name, "summary_v1")

    @patch('prompthelix.utils.llm_utils.call_llm_api', return_value="invalid_template_name")
    def test_select_template_llm_invalid_response_falls_back(self, mock_call_llm_api):
        """Test _select_template falls back if LLM returns invalid template name."""
        parsed_reqs = {"task_description": "A generic task."}
        with patch.object(self.architect, '_fallback_select_template', return_value="generic_v1") as mock_fallback:
            template_name = self.architect._select_template(parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        mock_fallback.assert_called_once_with(parsed_reqs)
        self.assertEqual(template_name, "generic_v1")


    @patch('prompthelix.utils.llm_utils.call_llm_api', side_effect=Exception("LLM Error"))
    def test_select_template_llm_failure_falls_back(self, mock_call_llm_api):
        """Test _select_template falls back when LLM fails."""
        parsed_reqs = {"task_description": "Summarize this for me."}
        with patch.object(self.architect, '_fallback_select_template', return_value="summary_v1") as mock_fallback:
            template_name = self.architect._select_template(parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback.assert_called_once_with(parsed_reqs)
        self.assertEqual(template_name, "summary_v1")

    # --- Tests for _populate_genes ---
    @patch('prompthelix.utils.llm_utils.call_llm_api')
    def test_populate_genes_llm_success(self, mock_call_llm_api):
        """Test _populate_genes with successful LLM call."""
        # LLM returns a string that's split by newline in the current implementation
        mock_call_llm_api.return_value = "Instruction: LLM generated instruction\nContext: LLM context\nOutput Format: LLM format"
        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something.", "keywords": ["detail"]}
        
        genes = self.architect._populate_genes(template, parsed_reqs)
        
        mock_call_llm_api.assert_called_once()
        self.assertEqual(len(genes), 3)
        self.assertEqual(genes[0], "Instruction: LLM generated instruction")
        self.assertEqual(genes[1], "Context: LLM context")

    @patch('prompthelix.utils.llm_utils.call_llm_api', return_value="Not a list\nJust one line") # Malformed (not enough lines for typical structure)
    def test_populate_genes_llm_malformed_response_falls_back(self, mock_call_llm_api):
        """Test _populate_genes falls back if LLM returns malformed (but not erroring) data."""
        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something complex."}

        # The current _populate_genes raises ValueError if genes list is empty after split, triggering fallback.
        # Let's ensure the mock_call_llm_api returns something that leads to an empty list or error.
        mock_call_llm_api.return_value = "" # Empty string will result in empty genes list

        with patch.object(self.architect, '_fallback_populate_genes') as mock_fallback:
            mock_fallback.return_value = ["Fallback instruction", "Fallback context"] # Ensure fallback returns something
            genes = self.architect._populate_genes(template, parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback.assert_called_once_with(template, parsed_reqs)
        self.assertEqual(genes[0], "Fallback instruction")


    @patch('prompthelix.utils.llm_utils.call_llm_api', side_effect=Exception("LLM Error"))
    def test_populate_genes_llm_failure_falls_back(self, mock_call_llm_api):
        """Test _populate_genes falls back when LLM fails."""
        template = self.architect.templates["generic_v1"]
        parsed_reqs = {"task_description": "Do something."}
        with patch.object(self.architect, '_fallback_populate_genes', return_value=["Fallback gene"]) as mock_fallback:
            genes = self.architect._populate_genes(template, parsed_reqs)

        mock_call_llm_api.assert_called_once()
        mock_fallback.assert_called_once_with(template, parsed_reqs)
        self.assertEqual(genes[0], "Fallback gene")

    # --- Edge Cases for process_request ---
    # test_process_request_missing_data can be adapted.
    # The key is to ensure it calls the LLM-enabled internal methods,
    # and those methods' fallbacks are tested above.
    @patch('prompthelix.utils.llm_utils.call_llm_api')
    def test_process_request_missing_keywords_and_constraints(self, mock_call_llm_api):
        """Test process_request with missing keywords and constraints."""
        # Set up mocks for each LLM call within process_request
        # 1. _parse_requirements
        #    Current _parse_requirements returns a mix of original and hardcoded "llm_added_keyword"
        mock_call_llm_api.side_effect = [
            "LLM parsed requirements: Default task description", # For _parse_requirements
            "generic_v1",                                  # For _select_template
            "LLM Instruction: Default\nContext: Default context with no keywords\nOutput: Default format" # For _populate_genes
        ]

        request_data = {"task_description": "A simple task"} # No keywords, no constraints
        chromosome = self.architect.process_request(request_data)

        self.assertIsInstance(chromosome, PromptChromosome)
        self.assertTrue(len(chromosome.genes) > 0)
        # Check that the final genes reflect the (mocked) LLM population if successful
        self.assertEqual(chromosome.genes[0], "LLM Instruction: Default")
        self.assertEqual(mock_call_llm_api.call_count, 3) # One for each internal LLM call

    def test_process_request_empty_task_description(self):
        """Test process_request with an empty string for task_description."""
        # This will test how _parse_requirements (and its LLM call) handles empty task_desc,
        # and subsequently how other methods use that.
        with patch('prompthelix.utils.llm_utils.call_llm_api') as mock_llm:
            mock_llm.side_effect = [
                'LLM parsed: empty task, default keywords, no constraints', # _parse_requirements
                'generic_v1', # _select_template
                'Instruction: Handle empty\nContext: Empty context\nOutput: As specified' # _populate_genes
            ]
            request_data = {"task_description": "", "keywords": [], "constraints": {}}
            chromosome = self.architect.process_request(request_data)

        self.assertIsInstance(chromosome, PromptChromosome)
        self.assertTrue(len(chromosome.genes) > 0)
        self.assertEqual(chromosome.genes[0], "Instruction: Handle empty")
        # Check that the "task_description" in parsed_reqs for _select_template was handled (e.g., became "Default task description")
        # The second call to mock_llm is for _select_template. We can inspect its 'prompt' argument.
        args_select_template, _ = mock_llm.call_args_list[1]
        prompt_for_select_template = args_select_template[0]
        # Current _parse_requirements, if task_desc is empty, uses "Default task description" in its fallback.
        # If LLM for _parse_requirements provides something, that's used.
        # Here, the LLM for _parse_requirements returns "LLM parsed: empty task..."
        # The _parse_requirements method itself, in its current simulated parsing, sets "task_description" to the original task_desc.
        # So, for _select_template, task_desc will be empty.
        self.assertIn('Given the task description: ""', prompt_for_select_template)


    # The original tests like test_process_request_summary, test_process_request_generic,
    # test_process_request_question_answering implicitly test the fallback mechanisms
    # if we *don't* mock call_llm_api or if it's mocked to raise an error for all calls.
    # Let's add one explicit test for full fallback sequence.
    @patch('prompthelix.utils.llm_utils.call_llm_api', side_effect=Exception("LLM System Down"))
    def test_process_request_full_fallback_due_to_llm_system_down(self, mock_call_llm_api_system_down):
        """Test process_request uses full fallback logic if all LLM calls fail."""
        request_data = {
            "task_description": "Summarize this story about AI and ethics.",
            "keywords": ["AI", "ethics"],
            "constraints": {"max_length": 150}
        }
        # The original test_process_request_summary can be a model for assertions here
        # as it was testing the non-LLM path.
        result_chromosome = self.architect.process_request(request_data)

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
