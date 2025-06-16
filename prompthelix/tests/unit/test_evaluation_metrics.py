import unittest
from prompthelix.evaluation import metrics

class TestPromptEvaluationMetrics(unittest.TestCase):
    """Test suite for prompt quality evaluation metrics."""

    def test_calculate_clarity_score(self):
        """Test the calculate_clarity_score function."""
        # Test with a clear prompt (high Flesch Reading Ease, no ambiguous phrases)
        # "The cat slept on the rug. A red apple is on the table. Tell me more about this." should be fairly easy to read.
        # Assuming default max_flesch_reading_ease = 60.0. If this prompt scores > 60, clarity_from_ease = 1.0
        clear_prompt = "The cat slept on the rug. A red apple is on the table. Provide details about the items."
        # textstat.flesch_reading_ease(clear_prompt) is likely > 60
        self.assertGreater(metrics.calculate_clarity_score(clear_prompt), 0.7, "Clear prompt should have high clarity.")

        # Test with an unclear prompt (more ambiguous phrases, potentially lower Flesch Reading Ease)
        unclear_prompt = "Maybe you could just sort of tell me some stuff about those things, perhaps ASAP? It's very important basically."
        # This prompt has "maybe", "just", "sort of", "stuff", "things", "perhaps", "asap", "very", "basically" (9 ambiguous)
        # Max penalty is 0.5. (9/5)*0.5 = 0.9, capped at 0.5. So clarity_from_ease * 0.5
        # Flesch score for this might be ok, but ambiguity penalty will be high.
        self.assertLess(metrics.calculate_clarity_score(unclear_prompt), 0.5, "Unclear prompt should have lower clarity due to ambiguity.")

        # Test with a prompt that has good Flesch score but some ambiguous words
        clear_structure_ambiguous_words = "The system might be able to process this request. It seems rather complex, perhaps."
        # "might be", "seems to", "rather", "perhaps" (4 ambiguous) -> (4/5)*0.5 = 0.4 penalty. clarity_from_ease * 0.6
        clarity_score_ambiguous = metrics.calculate_clarity_score(clear_structure_ambiguous_words)
        # print(f"Clarity for 'clear_structure_ambiguous_words': {clarity_score_ambiguous}") # For debugging
        self.assertTrue(0.2 < clarity_score_ambiguous < 0.7, "Prompt with some ambiguity should have mid-range clarity.")


        # Test with an empty prompt
        self.assertEqual(metrics.calculate_clarity_score(""), 0.0, "Empty prompt should have 0 clarity.")
        self.assertEqual(metrics.calculate_clarity_score("   "), 0.0, "Whitespace prompt should have 0 clarity.")

        # Test with very short, clear prompt (textstat might struggle, but ambiguity is low)
        short_clear = "Explain cats." # Flesch score might be low due to shortness, but no ambiguity.
        # textstat.flesch_reading_ease("Explain cats.") is ~85. 85/60 = 1.0 (capped). No ambiguity. Score should be high.
        self.assertAlmostEqual(metrics.calculate_clarity_score(short_clear), 1.0, delta=0.1, msg="Short, clear prompt clarity.")

        # Test with a prompt designed for low Flesch Reading Ease
        complex_prompt = "The utilization of sophisticated methodologies for the ascertainment of efficaciousness is paramount."
        # textstat.flesch_reading_ease(complex_prompt) is low.
        self.assertLess(metrics.calculate_clarity_score(complex_prompt), 0.5, "Complex prompt should have low clarity.")

    def test_calculate_completeness_score(self):
        """Test the calculate_completeness_score function."""
        default_reqs = ["Instruction:", "[context]", "Output format:"] # Default in metrics module

        # All default elements present
        prompt_all_elements = "Instruction: Do this. [context] Some context. Output format: JSON."
        self.assertEqual(metrics.calculate_completeness_score(prompt_all_elements, default_reqs), 1.0)
        self.assertEqual(metrics.calculate_completeness_score(prompt_all_elements), 1.0) # Test with internal default

        # Some elements present (case variations)
        prompt_some_elements = "instruction: Do that. [CONTEXT] is here." # Missing "Output format:"
        self.assertAlmostEqual(metrics.calculate_completeness_score(prompt_some_elements, default_reqs), 2/3)
        self.assertAlmostEqual(metrics.calculate_completeness_score(prompt_some_elements.upper(), default_reqs), 2/3, "Should be case-insensitive for content check")


        # No elements present
        prompt_no_elements = "Just a sentence."
        self.assertEqual(metrics.calculate_completeness_score(prompt_no_elements, default_reqs), 0.0)

        # Custom required elements
        custom_reqs = ["Task:", "[Input Data]", "Desired Output:"]
        prompt_custom = "Task: Analyze. [Input Data] Provided. Desired Output: Analysis."
        self.assertEqual(metrics.calculate_completeness_score(prompt_custom, custom_reqs), 1.0)
        prompt_custom_missing = "Task: Analyze. [Input Data] Provided." # Missing one
        self.assertAlmostEqual(metrics.calculate_completeness_score(prompt_custom_missing, custom_reqs), 2/3)


        # Empty prompt
        self.assertEqual(metrics.calculate_completeness_score("", default_reqs), 0.0)
        self.assertEqual(metrics.calculate_completeness_score("   ", default_reqs), 0.0)

        # Empty required_elements list (should be 1.0)
        self.assertEqual(metrics.calculate_completeness_score(prompt_all_elements, []), 1.0)
        # Test with None for required_elements (should use its internal default)
        self.assertEqual(metrics.calculate_completeness_score(prompt_all_elements, None), 1.0)


    def test_calculate_specificity_score(self):
        """Test the calculate_specificity_score function."""
        # Highly specific, no common placeholders, decent length
        specific_prompt = "Instruction: Generate a Python function that calculates the factorial of a positive integer using recursion. Output format: Code block."
        # word count is high, no placeholders.
        self.assertGreater(metrics.calculate_specificity_score(specific_prompt), 0.7, "Specific prompt should have high score.")

        # Generic prompt with placeholders
        generic_prompt_placeholders = "Explain [topic] using [examples]. Provide details about [subtopic]."
        # 3 placeholders -> 3 * 0.1 = 0.3 penalty. Score = 1.0 - 0.3 = 0.7. Word count is low. Penalty -0.2. Score = 0.5
        self.assertTrue(0.4 < metrics.calculate_specificity_score(generic_prompt_placeholders) < 0.6, "Generic prompt with placeholders should have lower score.")

        # Very short prompt
        short_prompt = "Summarize." # word count 1. Penalty -0.4. Score = 0.6
        self.assertLess(metrics.calculate_specificity_score(short_prompt), 0.7, "Very short prompt should have reduced specificity.")

        # Short prompt with placeholders (double penalty)
        short_generic = "[details]" # 1 placeholder -> 0.1 penalty. word count 1 -> 0.4 penalty. Placeholder + short -> 0.2 penalty. Score = 1-0.1-0.4-0.2 = 0.3
        self.assertAlmostEqual(metrics.calculate_specificity_score(short_generic), 0.3, delta=0.01, msg="Short generic prompt specificity.")

        # Prompt with long sentences (mild penalty)
        long_sentence_prompt = "Considering all the multifaceted aspects and diverse parameters involved, could you perhaps elaborate extensively on the fundamental nature of the subject matter in question, ensuring a comprehensive and thorough exposition?"
        # Word count is high. Sentence count is 1. Avg sentence length is high. Penalty -0.1 or -0.15
        self.assertTrue(0.8 <= metrics.calculate_specificity_score(long_sentence_prompt) <= 0.9, "Prompt with long sentences should have slightly reduced specificity.")

        # Empty prompt
        self.assertEqual(metrics.calculate_specificity_score(""), 0.0, "Empty prompt specificity.")

        # Many placeholders to hit max penalty
        many_placeholders = "[a] [b] [c] [d] [e] [f]" # 6 placeholders -> 0.5 penalty. word count low -> -0.2. Score = 0.3
        self.assertAlmostEqual(metrics.calculate_specificity_score(many_placeholders), 0.3, delta=0.01, msg="Many placeholders max penalty.")


    def test_calculate_prompt_length_score(self):
        """Test the calculate_prompt_length_score function."""
        # Using default params: min_len=20, optimal_min=50, optimal_max=350, max_len=500

        # Too short
        self.assertEqual(metrics.calculate_prompt_length_score("short"), 0.0) # len 5 < 20
        self.assertEqual(metrics.calculate_prompt_length_score("This is 19 chars."), 0.0) # len 19 < 20

        # Between min_len and optimal_min
        # (20 - 20) / (50-20) = 0/30 = 0
        self.assertAlmostEqual(metrics.calculate_prompt_length_score("This is twenty chars."), 0.0) # len 20. (20-20)/(50-20) = 0
        # (35 - 20) / (50-20) = 15/30 = 0.5
        self.assertAlmostEqual(metrics.calculate_prompt_length_score("This prompt is exactly 35 chars long."), 0.5) # len 35

        # Optimal length
        optimal_prompt = "This is an optimal length prompt, it should get a full score of 1.0 because its length is good." # len 90
        self.assertEqual(metrics.calculate_prompt_length_score(optimal_prompt), 1.0)
        self.assertEqual(metrics.calculate_prompt_length_score("a" * 50), 1.0)
        self.assertEqual(metrics.calculate_prompt_length_score("a" * 350), 1.0)

        # Between optimal_max and max_len
        # (500 - 351) / (500 - 350) = 149 / 150 = ~0.993
        self.assertAlmostEqual(metrics.calculate_prompt_length_score("a" * 351), (500-351)/(500-350))
         # (500 - 425) / (500 - 350) = 75 / 150 = 0.5
        self.assertAlmostEqual(metrics.calculate_prompt_length_score("a" * 425), 0.5)

        # Too long
        self.assertEqual(metrics.calculate_prompt_length_score("a" * 501), 0.0)

        # Empty string
        self.assertEqual(metrics.calculate_prompt_length_score(""), 0.0) # len 0 < 20

if __name__ == '__main__':
    unittest.main()
