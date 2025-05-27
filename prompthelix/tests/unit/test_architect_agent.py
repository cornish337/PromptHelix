import unittest
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.genetics.engine import PromptChromosome

class TestPromptArchitectAgent(unittest.TestCase):
    """Test suite for the PromptArchitectAgent."""

    def setUp(self):
        """Instantiate the PromptArchitectAgent for each test."""
        self.architect = PromptArchitectAgent()

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.architect)
        self.assertEqual(self.architect.agent_id, "PromptArchitect")
        self.assertTrue(self.architect.templates, "Templates should be loaded and not empty.")
        self.assertIn("summary_v1", self.architect.templates, "Default summary template should be loaded.")

    def test_load_templates(self):
        """Test if templates are loaded correctly."""
        self.assertIn("summary_v1", self.architect.templates)
        self.assertIn("question_answering_v1", self.architect.templates)
        self.assertIn("generic_v1", self.architect.templates)
        self.assertEqual(self.architect.templates["summary_v1"]["instruction"], "Summarize the following text:")

    def test_process_request_summary(self):
        """Test process_request with a basic summarization task."""
        request_data = {
            "task_description": "Summarize this story about AI and ethics.", 
            "keywords": ["AI", "ethics"],
            "constraints": {"max_length": 150}
        }
        result_chromosome = self.architect.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertTrue(result_chromosome.genes, "Genes list should not be empty.")
        
        # Check for instruction from summary template
        self.assertTrue(
            any("Summarize the following text:" in gene for gene in result_chromosome.genes),
            "Summary instruction not found in genes."
        )
        
        # Check for keywords in context
        context_gene_found = False
        for gene in result_chromosome.genes:
            if "Context:" in gene:
                context_gene_found = True
                self.assertIn("AI", gene, "Keyword 'AI' not found in context gene.")
                self.assertIn("ethics", gene, "Keyword 'ethics' not found in context gene.")
                # Check if task_description was used to replace placeholder (simple check)
                self.assertIn("Summarize this story about AI and ethics.", gene, "Task description not found in context gene for summary.")
                break
        self.assertTrue(context_gene_found, "Context gene not found.")

    def test_process_request_generic(self):
        """Test process_request with a generic task description."""
        request_data = {
            "task_description": "Explain quantum physics.", 
            "keywords": ["qubit", "entanglement"],
            "constraints": {}
        }
        result_chromosome = self.architect.process_request(request_data)

        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertTrue(result_chromosome.genes, "Genes list should not be empty.")
        
        # Check for instruction from generic template
        self.assertTrue(
            any("Perform the following task:" in gene for gene in result_chromosome.genes),
            "Generic instruction not found in genes."
        )
        
        # Check for keywords in context
        context_gene_found = False
        for gene in result_chromosome.genes:
            if "Context:" in gene:
                context_gene_found = True
                self.assertIn("qubit", gene, "Keyword 'qubit' not found in context gene.")
                self.assertIn("entanglement", gene, "Keyword 'entanglement' not found in context gene.")
                break
        self.assertTrue(context_gene_found, "Context gene not found.")

    def test_process_request_missing_data(self):
        """Test process_request with missing or minimal data."""
        # Test with completely empty request_data
        request_data_empty = {}
        result_empty = self.architect.process_request(request_data_empty)
        self.assertIsInstance(result_empty, PromptChromosome)
        self.assertTrue(result_empty.genes, "Genes list should not be empty for empty request.")
        # Default behavior uses generic template with "Default task description"
        self.assertTrue(
            any("Perform the following task:" in gene for gene in result_empty.genes) or
            any("Default instruction:" in gene for gene in result_empty.genes) or # Adjusted to current fallback
            any("Default task description" in gene for gene in result_empty.genes), # Check based on current implementation
            "Default instruction not found for empty request."
        )


        # Test with task_description as None
        request_data_none_desc = {"task_description": None, "keywords": ["test"]}
        result_none_desc = self.architect.process_request(request_data_none_desc)
        self.assertIsInstance(result_none_desc, PromptChromosome)
        self.assertTrue(result_none_desc.genes, "Genes list should not be empty for None description.")
        # Should still use a default description
        self.assertTrue(
            any("Default task description" in gene for gene in result_none_desc.genes) or
            any("Default instruction:" in gene for gene in result_none_desc.genes), # Adjusted
            "Default description not handled correctly."
        )
        self.assertTrue(
            any("test" in gene for gene in result_none_desc.genes),
            "Keywords not handled with None description."
        )

    def test_process_request_question_answering(self):
        """Test process_request with a question answering task."""
        request_data = {
            "task_description": "What is the capital of France?",
            "keywords": ["capital", "France"],
            "constraints": {}
        }
        result_chromosome = self.architect.process_request(request_data)
        self.assertIsInstance(result_chromosome, PromptChromosome)
        self.assertTrue(result_chromosome.genes)
        self.assertTrue(
            any("Answer the question based on the provided context:" in gene for gene in result_chromosome.genes),
            "Question answering instruction not found."
        )
        context_gene_found = False
        for gene in result_chromosome.genes:
            if "Context:" in gene:
                context_gene_found = True
                self.assertIn("What is the capital of France?", gene, "Question not found in context gene for QA.")
                self.assertIn("capital", gene)
                self.assertIn("France", gene)
                break
        self.assertTrue(context_gene_found, "Context gene not found for QA task.")


if __name__ == '__main__':
    unittest.main()
