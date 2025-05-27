import unittest
from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.genetics.engine import PromptChromosome

class TestMetaLearnerAgent(unittest.TestCase):
    """Test suite for the MetaLearnerAgent."""

    def setUp(self):
        """Instantiate the MetaLearnerAgent for each test."""
        self.learner = MetaLearnerAgent()

    def test_agent_creation(self):
        """Test basic creation and initialization of the agent."""
        self.assertIsNotNone(self.learner)
        self.assertEqual(self.learner.agent_id, "MetaLearner")
        self.assertIsInstance(self.learner.knowledge_base, dict, "Knowledge base should be initialized as a dict.")
        self.assertIn("successful_patterns", self.learner.knowledge_base)
        self.assertIn("common_pitfalls", self.learner.knowledge_base)
        self.assertIn("performance_trends", self.learner.knowledge_base)
        self.assertIsInstance(self.learner.data_log, list, "Data log should be initialized as a list.")

    def test_process_request_high_fitness_eval(self):
        """Test process_request with evaluation data indicating high fitness."""
        prompt = PromptChromosome(genes=["Good prompt gene 1", "Instruction: Be good"], fitness_score=0.0) # Fitness on chromo is not used by MetaLearner
        eval_data = {"prompt_chromosome": prompt, "fitness_score": 0.9}
        request_data = {"data_type": "evaluation_result", "data": eval_data}
        
        initial_pattern_count = len(self.learner.knowledge_base["successful_patterns"])
        result = self.learner.process_request(request_data)

        self.assertEqual(result["status"], "Data processed successfully.")
        self.assertEqual(len(self.learner.knowledge_base["successful_patterns"]), initial_pattern_count + 1)
        added_pattern = self.learner.knowledge_base["successful_patterns"][-1]
        self.assertEqual(added_pattern["fitness"], 0.9)
        self.assertEqual(added_pattern["gene_count"], 2)

    def test_process_request_critique_pitfall(self):
        """Test process_request with critique data indicating a common pitfall."""
        critique_feedback = "Structural Issue (PromptTooShort): Prompt might be too short..."
        critique_data = {"feedback_points": [critique_feedback]}
        request_data = {"data_type": "critique_result", "data": critique_data}
        
        theme = critique_feedback.split(":")[0]
        initial_pitfall_count = self.learner.knowledge_base["common_pitfalls"].get(theme, 0)
        
        result = self.learner.process_request(request_data)

        self.assertEqual(result["status"], "Data processed successfully.")
        self.assertEqual(self.learner.knowledge_base["common_pitfalls"].get(theme), initial_pitfall_count + 1)

    def test_identify_system_patterns_indirectly(self):
        """Test _identify_system_patterns by feeding multiple data points."""
        # Feed enough data to trigger _identify_system_patterns (currently every 3 data points)
        prompt1 = PromptChromosome(genes=["P1 G1", "P1 G2"])
        eval_data1 = {"prompt_chromosome": prompt1, "fitness_score": 0.8}
        self.learner.process_request({"data_type": "evaluation_result", "data": eval_data1})

        critique_data1 = {"feedback_points": ["Structural Issue (TestPitfall): Test."]}
        self.learner.process_request({"data_type": "critique_result", "data": critique_data1})
        
        prompt2 = PromptChromosome(genes=["P2 G1", "P2 G2", "P2 G3"]) # Different gene count
        eval_data2 = {"prompt_chromosome": prompt2, "fitness_score": 0.85} # To ensure avg gene count can be calculated
        # This third call should trigger _identify_system_patterns
        self.learner.process_request({"data_type": "evaluation_result", "data": eval_data2}) 

        self.assertTrue(self.learner.knowledge_base["performance_trends"], 
                        "Performance trends should be populated after several data processing cycles.")
        self.assertTrue(any("Average gene count" in trend for trend in self.learner.knowledge_base["performance_trends"]))
        self.assertTrue(any("Most common pitfall" in trend for trend in self.learner.knowledge_base["performance_trends"]))


    def test_generate_recommendations_indirectly(self):
        """Test _generate_recommendations by feeding data and checking output."""
        prompt = PromptChromosome(genes=["Good prompt"])
        eval_data = {"prompt_chromosome": prompt, "fitness_score": 0.9}
        request_data = {"data_type": "evaluation_result", "data": eval_data}
        
        # Process a few times to potentially generate trends
        self.learner.process_request(request_data)
        self.learner.process_request(request_data) # Log size 2
        result = self.learner.process_request(request_data) # Log size 3, triggers pattern identification

        self.assertIsInstance(result["recommendations"], list)
        self.assertTrue(result["recommendations"], "Recommendations list should not be empty.")
        # Check for a specific type of recommendation that might appear based on the data
        self.assertTrue(
            any("Consider prompts with around" in rec for rec in result["recommendations"]) or
            any("No specific new recommendations" in rec for rec in result["recommendations"]),
            "Expected specific or default recommendation."
        )

    def test_process_request_unknown_data_type(self):
        """Test process_request with an unknown data type."""
        request_data = {"data_type": "unknown_type", "data": {"info": "test"}}
        result = self.learner.process_request(request_data)

        self.assertIn("Warning: Unknown data_type", result["status"])
        self.assertIsInstance(result["recommendations"], list, "Recommendations should still be provided even for unknown type.")
        # Default recommendation when no patterns are strong enough
        self.assertTrue(any("No specific new recommendations" in rec for rec in result["recommendations"]))

    def test_process_request_missing_keys(self):
        """Test process_request with missing data_type or data keys."""
        request_no_type = {"data": {"info": "test"}}
        result_no_type = self.learner.process_request(request_no_type)
        self.assertEqual(result_no_type["status"], "Error: Missing data_type or data.")
        self.assertEqual(len(result_no_type["recommendations"]), 0)

        request_no_data = {"data_type": "evaluation_result"}
        result_no_data = self.learner.process_request(request_no_data)
        self.assertEqual(result_no_data["status"], "Error: Missing data_type or data.")
        self.assertEqual(len(result_no_data["recommendations"]), 0)

if __name__ == '__main__':
    unittest.main()
