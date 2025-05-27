import unittest
from unittest.mock import Mock, patch, call # Using Mock and patch
from prompthelix.genetics.engine import FitnessEvaluator, PromptChromosome
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent

class TestFitnessEvaluator(unittest.TestCase):
    """Test suite for the FitnessEvaluator class."""

    def test_init_successful(self):
        """Test successful instantiation with a mock ResultsEvaluatorAgent."""
        mock_results_agent = Mock(spec=ResultsEvaluatorAgent)
        evaluator = FitnessEvaluator(results_evaluator_agent=mock_results_agent)
        self.assertIsInstance(evaluator, FitnessEvaluator)
        self.assertEqual(evaluator.results_evaluator_agent, mock_results_agent)

    def test_init_type_error(self):
        """Test that __init__ raises TypeError for incorrect agent type."""
        with self.assertRaisesRegex(TypeError, "results_evaluator_agent must be an instance of ResultsEvaluatorAgent."):
            FitnessEvaluator(results_evaluator_agent=None)
        
        with self.assertRaisesRegex(TypeError, "results_evaluator_agent must be an instance of ResultsEvaluatorAgent."):
            FitnessEvaluator(results_evaluator_agent="not_an_agent")

    def setUp(self):
        """Set up common test data and mocks for evaluate tests."""
        # This setUp will be used by tests for the 'evaluate' method
        self.mock_results_agent = Mock(spec=ResultsEvaluatorAgent)
        self.expected_fitness_score = 0.7
        self.mock_results_agent.process_request.return_value = {
            "fitness_score": self.expected_fitness_score,
            "detailed_metrics": {"relevance": 0.8, "coherence": 0.6},
            "error_analysis": []
        }
        
        self.fitness_evaluator = FitnessEvaluator(results_evaluator_agent=self.mock_results_agent)
        self.sample_genes = ["Instruction: Test.", "Context: This is a test context."]
        self.sample_chromosome = PromptChromosome(genes=self.sample_genes, fitness_score=0.0)
        self.task_description = "Test task description for evaluation."

    @patch('prompthelix.genetics.engine.random.randint') # Mock random.randint in engine.py
    def test_evaluate_successful(self, mock_randint):
        """Test the evaluate method with valid inputs."""
        mock_randint.return_value = 42 # Control randomness in mock LLM output
        success_criteria = {"max_length": 100}

        # Mock chromosome's to_prompt_string to check if it's called
        self.sample_chromosome.to_prompt_string = Mock(return_value="Test prompt string.")

        returned_fitness = self.fitness_evaluator.evaluate(
            self.sample_chromosome, 
            self.task_description, 
            success_criteria
        )

        # 1. Verify chromosome.to_prompt_string() was called
        self.sample_chromosome.to_prompt_string.assert_called_once()

        # 2. Verify results_evaluator_agent.process_request was called correctly
        expected_mock_llm_output = (
            f"Mock LLM output for: Test prompt string.[:50]. " # from to_prompt_string mock
            f"Keywords found: {', '.join(str(g) for g in self.sample_chromosome.genes[:2])}. "
            f"Random number: 42"
        )
        
        self.mock_results_agent.process_request.assert_called_once()
        call_args = self.mock_results_agent.process_request.call_args[0][0] # Get the first positional argument (the dict)

        self.assertEqual(call_args["prompt_chromosome"], self.sample_chromosome)
        self.assertEqual(call_args["llm_output"], expected_mock_llm_output)
        self.assertEqual(call_args["task_description"], self.task_description)
        self.assertEqual(call_args["success_criteria"], success_criteria)

        # 3. Verify chromosome.fitness_score is updated
        self.assertEqual(self.sample_chromosome.fitness_score, self.expected_fitness_score)

        # 4. Verify the method returns the correct fitness score
        self.assertEqual(returned_fitness, self.expected_fitness_score)

    @patch('prompthelix.genetics.engine.random.randint')
    def test_evaluate_success_criteria_none(self, mock_randint):
        """Test evaluate with success_criteria=None."""
        mock_randint.return_value = 10 # Another random value for this test
        
        # No success criteria passed
        returned_fitness = self.fitness_evaluator.evaluate(
            self.sample_chromosome, 
            self.task_description,
            success_criteria=None # Explicitly None
        )
        
        self.mock_results_agent.process_request.assert_called_once()
        call_args = self.mock_results_agent.process_request.call_args[0][0]
        
        # Ensure success_criteria in the call to process_request is an empty dict if None was passed
        self.assertEqual(call_args["success_criteria"], {}) 
        self.assertEqual(self.sample_chromosome.fitness_score, self.expected_fitness_score)
        self.assertEqual(returned_fitness, self.expected_fitness_score)

    def test_evaluate_invalid_chromosome_type(self):
        """Test that evaluate raises TypeError for incorrect chromosome type."""
        with self.assertRaisesRegex(TypeError, "chromosome must be an instance of PromptChromosome."):
            self.fitness_evaluator.evaluate("not_a_chromosome", self.task_description)

if __name__ == '__main__':
    unittest.main()
