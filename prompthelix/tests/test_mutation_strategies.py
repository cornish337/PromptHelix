import unittest
import random
from unittest.mock import patch
import logging

from prompthelix.genetics.engine import PromptChromosome
from prompthelix.genetics.mutation_strategies import (
    AppendCharStrategy,
    ReverseSliceStrategy,
    PlaceholderReplaceStrategy,
    NoOperationMutationStrategy
)
from prompthelix.genetics.strategy_base import BaseMutationStrategy # For type hinting if needed

class TestMutationStrategies(unittest.TestCase):

    def setUp(self):
        # Suppress logging during tests for cleaner output, unless testing log messages
        logging.disable(logging.CRITICAL)
        # For specific log testing:
        # self.logger = logging.getLogger('prompthelix.genetics.mutation_strategies')
        # self.logger.handlers = [] # Clear existing handlers
        # self.log_capture_string = io.StringIO()
        # self.ch = logging.StreamHandler(self.log_capture_string)
        # self.logger.addHandler(self.ch)
        # self.logger.setLevel(logging.DEBUG)


    def tearDown(self):
        logging.disable(logging.NOTSET) # Re-enable logging

    def test_append_char_strategy(self):
        strategy = AppendCharStrategy(chars_to_append="@")
        chromosome = PromptChromosome(genes=["gene1", "gene2"], fitness_score=0.5)

        # Predictable random choice for testing
        with patch('random.randrange', return_value=0): # Selects gene 0
            with patch('random.choice', return_value="@") as mock_choice: # Selects char
                 mutated_chromosome = strategy.mutate(chromosome)

        self.assertNotEqual(chromosome.id, mutated_chromosome.id, "Should be a new chromosome instance (cloned).")
        self.assertEqual(mutated_chromosome.fitness_score, 0.0, "Fitness should be reset.")
        self.assertEqual(mutated_chromosome.genes[0], "gene1@", "Character should be appended to the selected gene.")
        self.assertEqual(mutated_chromosome.genes[1], "gene2", "Other genes should remain unchanged.")
        mock_choice.assert_called_once_with("@")


    def test_append_char_strategy_no_genes(self):
        strategy = AppendCharStrategy()
        chromosome = PromptChromosome(genes=[], fitness_score=0.5)
        mutated_chromosome = strategy.mutate(chromosome)
        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.genes, [])
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)

    def test_append_char_strategy_no_chars_to_append(self):
        strategy = AppendCharStrategy(chars_to_append="") # Empty string of chars
        chromosome = PromptChromosome(genes=["gene1"], fitness_score=0.5)
        # It should still clone and reset fitness, even if no change is made to genes
        mutated_chromosome = strategy.mutate(chromosome)
        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.genes[0], "gene1") # Gene remains unchanged
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)


    def test_reverse_slice_strategy(self):
        strategy = ReverseSliceStrategy()
        original_gene_content = "abcdefgh"
        chromosome = PromptChromosome(genes=[original_gene_content], fitness_score=0.6)

        # Mock random choices to make the slice predictable for testing
        # random.randrange(len(genes)) -> 0 (selects the first gene)
        # random.randint(1, max(2, len(original_gene_str) // 2)) -> slice_len = 3
        # random.randint(0, len(original_gene_str) - slice_len) -> start_index = 2
        # Slice: "cde", Reversed: "edc"
        # Expected: "ab" + "edc" + "fgh" = "abedcfgh"
        with patch('random.randrange', return_value=0):
            with patch('random.randint') as mock_randint:
                mock_randint.side_effect = [3, 2] # slice_len=3, start_index=2
                mutated_chromosome = strategy.mutate(chromosome)

        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)
        self.assertEqual(mutated_chromosome.genes[0], "abedcfgh")

    def test_reverse_slice_strategy_short_gene(self):
        strategy = ReverseSliceStrategy()
        chromosome = PromptChromosome(genes=["ab"], fitness_score=0.3) # Gene too short
        mutated_chromosome = strategy.mutate(chromosome)
        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.genes[0], "ab", "Gene should remain unchanged if too short.")
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)

    def test_reverse_slice_strategy_no_genes(self):
        strategy = ReverseSliceStrategy()
        chromosome = PromptChromosome(genes=[], fitness_score=0.3)
        mutated_chromosome = strategy.mutate(chromosome)
        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.genes, [])
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)


    def test_placeholder_replace_strategy(self):
        strategy = PlaceholderReplaceStrategy(placeholder="[REPLACED]")
        chromosome = PromptChromosome(genes=["gene_to_replace", "other_gene"], fitness_score=0.7)

        with patch('random.randrange', return_value=0): # Selects gene 0 for replacement
            mutated_chromosome = strategy.mutate(chromosome)

        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)
        self.assertEqual(mutated_chromosome.genes[0], "[REPLACED]")
        self.assertEqual(mutated_chromosome.genes[1], "other_gene")

    def test_placeholder_replace_strategy_no_genes(self):
        strategy = PlaceholderReplaceStrategy()
        chromosome = PromptChromosome(genes=[], fitness_score=0.7)
        mutated_chromosome = strategy.mutate(chromosome)
        self.assertNotEqual(chromosome.id, mutated_chromosome.id)
        self.assertEqual(mutated_chromosome.genes, [])
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)


    def test_no_operation_mutation_strategy(self):
        strategy = NoOperationMutationStrategy()
        original_genes = ["gene1", "gene2"]
        chromosome = PromptChromosome(genes=original_genes, fitness_score=0.8)
        mutated_chromosome = strategy.mutate(chromosome)

        self.assertNotEqual(chromosome.id, mutated_chromosome.id, "Should be a new chromosome instance (cloned).")
        self.assertEqual(mutated_chromosome.fitness_score, 0.0, "Fitness should be reset.")
        self.assertEqual(mutated_chromosome.genes, original_genes, "Genes should remain unchanged.")
        self.assertListEqual(list(mutated_chromosome.genes), list(original_genes)) # Ensure content equality

if __name__ == '__main__':
    unittest.main()
