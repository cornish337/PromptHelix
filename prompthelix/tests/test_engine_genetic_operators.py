import unittest
from unittest.mock import MagicMock, patch
import logging

from prompthelix.genetics.engine import PromptChromosome, GeneticOperators
from prompthelix.genetics.strategy_base import BaseMutationStrategy
from prompthelix.genetics.mutation_strategies import (
    AppendCharStrategy,
    NoOperationMutationStrategy,
    PlaceholderReplaceStrategy
)
# Assuming StyleOptimizerAgent might be in agents module or a sub-module
# For this test, we only need to mock it, so direct import might not be needed if we patch its path.
# If StyleOptimizerAgent is defined in prompthelix.agents.style_optimizer:
from prompthelix.agents.style_optimizer import StyleOptimizerAgent


# A concrete mock strategy for testing selection
class MockStrategyA(BaseMutationStrategy):
    def __init__(self, settings=None):
        super().__init__(settings)

    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.genes.append("mutated_by_A")
        mutated_chromosome.fitness_score = 0.0
        return mutated_chromosome

class MockStrategyB(BaseMutationStrategy):
    def __init__(self, settings=None):
        super().__init__(settings)

    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.genes.append("mutated_by_B")
        mutated_chromosome.fitness_score = 0.0
        return mutated_chromosome

class TestGeneticOperators(unittest.TestCase):

    def setUp(self):
        # logging.disable(logging.CRITICAL) # Removed for more targeted log testing
        self.chromosome = PromptChromosome(genes=["gene1"], fitness_score=0.5)
        self.mock_style_optimizer_agent = MagicMock(spec=StyleOptimizerAgent)
        self.logger_name = "prompthelix.genetics.engine" # Logger used in GeneticOperators


    def tearDown(self):
        pass
        # logging.disable(logging.NOTSET) # Removed

    def test_init_default_strategies(self):
        # When no strategies are provided, it should default to NoOperationMutationStrategy.
        operators = GeneticOperators()
        self.assertEqual(len(operators.mutation_strategies), 1)
        self.assertIsInstance(operators.mutation_strategies[0], NoOperationMutationStrategy)

    def test_init_custom_strategies(self):
        custom_strategies = [MockStrategyA(), MockStrategyB()]
        operators = GeneticOperators(mutation_strategies=custom_strategies)
        self.assertEqual(operators.mutation_strategies, custom_strategies)
        self.assertEqual(len(operators.mutation_strategies), 2)

    def test_init_empty_custom_strategies_defaults_to_no_op(self):
        # This tests the safeguard added: if an empty list is passed, it defaults to NoOperation.
        operators = GeneticOperators(mutation_strategies=[])
        self.assertEqual(len(operators.mutation_strategies), 1)
        self.assertIsInstance(operators.mutation_strategies[0], NoOperationMutationStrategy)


    def test_mutate_applies_selected_strategy(self):
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        # Configure the mock strategy's mutate method to return a new PromptChromosome
        # (or the same one if that's how it's supposed to work after cloning)
        mutated_clone = self.chromosome.clone()
        mutated_clone.genes.append("mock_mutated")
        mutated_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        # mutation_rate = 1.0 ensures mutation is attempted
        # gene_mutation_prob is not directly used by GeneticOperators.mutate for strategy selection anymore
        result_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0, gene_mutation_prob=1.0)

        mock_strategy.mutate.assert_called_once()
        # The chromosome passed to strategy.mutate should be a clone of self.chromosome
        call_args = mock_strategy.mutate.call_args[0][0]
        self.assertIsInstance(call_args, PromptChromosome)
        self.assertNotEqual(call_args.id, self.chromosome.id, "Strategy should operate on a clone.")
        self.assertListEqual(call_args.genes, self.chromosome.genes)

        self.assertEqual(result_chromosome, mutated_clone, "Returned chromosome should be the one from the strategy.")
        self.assertIn("mock_mutated", result_chromosome.genes)


    def test_mutate_rate_zero(self):
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        # Pass a clone of the original chromosome to ensure we can compare IDs later
        original_chromosome_clone_for_comparison = self.chromosome.clone()

        result_chromosome = operators.mutate(original_chromosome_clone_for_comparison, mutation_rate=0.0)

        mock_strategy.mutate.assert_not_called()
        self.assertNotEqual(result_chromosome.id, original_chromosome_clone_for_comparison.id, "Should return a clone even if no mutation.")
        self.assertEqual(result_chromosome.genes, original_chromosome_clone_for_comparison.genes)
        self.assertEqual(result_chromosome.fitness_score, 0.0, "Fitness should be reset on the clone.")


    def test_mutate_rate_one(self):
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        mutated_clone = self.chromosome.clone()
        mutated_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        result_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0)

        mock_strategy.mutate.assert_called_once()
        self.assertEqual(result_chromosome, mutated_clone)


    def test_mutate_with_style_optimizer_agent_success(self):
        # Setup: A base mutation strategy and a style optimizer
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0 # Strategy resets fitness

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        style_optimized_chromosome = self.chromosome.clone() # Different ID from base_mutated_chromosome
        style_optimized_chromosome.genes.append("style_optimized")
        # Style optimizer might or might not reset fitness; GeneticOperators.mutate should ensure it.
        style_optimized_chromosome.fitness_score = 0.8 # Let's say optimizer sets some fitness

        self.mock_style_optimizer_agent.process_request.return_value = style_optimized_chromosome

        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        target_style = "concise"
        final_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0, target_style=target_style)

        mock_base_strategy.mutate.assert_called_once()
        # The input to style optimizer should be the one returned by mock_base_strategy
        self.mock_style_optimizer_agent.process_request.assert_called_once_with({
            "prompt_chromosome": base_mutated_chromosome,
            "target_style": target_style
        })
        self.assertEqual(final_chromosome, style_optimized_chromosome)
        self.assertEqual(final_chromosome.fitness_score, 0.0, "Fitness should be reset by GeneticOperators.mutate after style optimization.")
        self.assertIn("style_optimized", final_chromosome.genes)

    def test_mutate_with_style_optimizer_agent_returns_non_chromosome(self):
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        self.mock_style_optimizer_agent.process_request.return_value = None # Agent returns None

        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        with self.assertLogs(self.logger_name, level='WARNING') as log_watcher:
            final_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")

        self.mock_style_optimizer_agent.process_request.assert_called_once()
        # Should return the chromosome from the base mutation, not the agent's None
        self.assertEqual(final_chromosome, base_mutated_chromosome)
        self.assertIn("base_mutated", final_chromosome.genes)
        self.assertEqual(final_chromosome.fitness_score, 0.0) # Still reset from base strategy

        self.assertTrue(any("StyleOptimizerAgent did not return a PromptChromosome" in msg for msg in log_watcher.output))


    def test_mutate_with_style_optimizer_agent_raises_exception(self):
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        self.mock_style_optimizer_agent.process_request.side_effect = Exception("Agent Error")

        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        with self.assertLogs(self.logger_name, level='ERROR') as log_watcher:
            final_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")

        self.mock_style_optimizer_agent.process_request.assert_called_once()
        self.assertEqual(final_chromosome, base_mutated_chromosome) # Fallback to pre-style-optimization chromosome
        self.assertIn("base_mutated", final_chromosome.genes)
        self.assertEqual(final_chromosome.fitness_score, 0.0)

        self.assertTrue(any(f"Style optimization failed for {base_mutated_chromosome.id}" in msg and "Agent Error" in msg for msg in log_watcher.output))

    def test_mutate_no_style_optimizer_agent_but_target_style_provided(self):
        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=None # Agent not provided
        )

        with self.assertLogs(self.logger_name, level='WARNING') as log_watcher:
            final_chromosome = operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")

        self.assertEqual(final_chromosome, base_mutated_chromosome)
        self.assertTrue(any(f"target_style 'concise' for {base_mutated_chromosome.id}, but StyleOptimizerAgent not available. Skipping." in msg for msg in log_watcher.output))


if __name__ == '__main__':
    unittest.main()
