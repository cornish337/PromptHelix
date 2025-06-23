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
        self.chromosome = PromptChromosome(genes=["gene1"], fitness_score=0.5)
        self.mock_style_optimizer_agent = MagicMock(spec=StyleOptimizerAgent)
        self.logger_name = "prompthelix.genetics.engine" # Logger used in GeneticOperators

        # Ensure the specific logger is enabled and at a level that allows capturing
        logger_to_test = logging.getLogger(self.logger_name)
        self.original_level = logger_to_test.level
        self.original_disabled_state = logger_to_test.disabled
        self.original_handlers = list(logger_to_test.handlers) # Store a copy

        logger_to_test.setLevel(logging.DEBUG) # Set to DEBUG to capture INFO, WARNING, ERROR
        logger_to_test.disabled = False
        # If we are adding a handler for all tests in the class, do it here.
        # However, assertLogs itself adds a temporary handler.
        # So, ensuring the logger level and enabled status is usually sufficient.


    def tearDown(self):
        logger_to_test = logging.getLogger(self.logger_name)
        logger_to_test.setLevel(self.original_level)
        logger_to_test.disabled = self.original_disabled_state
        logger_to_test.handlers = self.original_handlers # Restore original handlers

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


    async def test_mutate_applies_selected_strategy(self):
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        mutated_clone = self.chromosome.clone()
        mutated_clone.genes.append("mock_mutated")
        mutated_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        result_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, gene_mutation_prob=1.0)

        mock_strategy.mutate.assert_called_once()
        call_args = mock_strategy.mutate.call_args[0][0]
        self.assertIsInstance(call_args, PromptChromosome)
        self.assertNotEqual(call_args.id, self.chromosome.id, "Strategy should operate on a clone.")
        self.assertListEqual(call_args.genes, self.chromosome.genes)

        self.assertEqual(result_chromosome, mutated_clone, "Returned chromosome should be the one from the strategy.")
        self.assertIn("mock_mutated", result_chromosome.genes)

    # Additional tests that were missed in the previous async conversion
    async def test_mutate_chromosome_selected_but_no_gene_mutates(self): # Made async
        """Test when chromosome is selected for mutation, but no individual gene meets gene_mutation_prob."""
        # This test implies that the strategy's mutate IS called, but it internally doesn't change genes.
        # The core GeneticOperators.mutate doesn't use gene_mutation_prob directly, strategies do.
        # So, we'll assume the strategy is called and returns an (effectively) unchanged chromosome.
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        unchanged_clone = self.chromosome.clone() # Clone to represent "no actual gene changes"
        unchanged_clone.fitness_score = 0.0 # Strategy should reset fitness
        mock_strategy.mutate.return_value = unchanged_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        # Pass a clone of the original to ensure ID comparison is meaningful for cloning action
        original_chromosome_clone_for_comparison = self.chromosome.clone()

        mutated_chromosome = await operators.mutate(original_chromosome_clone_for_comparison, mutation_rate=1.0, gene_mutation_prob=0.0) # await

        mock_strategy.mutate.assert_called_once()
        self.assertNotEqual(mutated_chromosome.id, original_chromosome_clone_for_comparison.id)
        self.assertEqual(mutated_chromosome.genes, original_chromosome_clone_for_comparison.genes) # Genes are the same as original's clone
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)

    async def test_mutate_does_not_occur_rate_too_high(self): # Made async
        """Test mutation when overall mutation_rate is not met."""
        # This is essentially the same as test_mutate_rate_zero logic-wise
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        original_clone = self.chromosome.clone()
        # random.random() will be > 0.0, so mutation_rate=0.0 ensures no mutation attempt.
        # To test random.random() > mutation_rate:
        # We need to patch random.random to return something > mutation_rate.
        with patch('random.random', return_value=0.5): # random.random() returns 0.5
            mutated_chromosome = await operators.mutate(original_clone, mutation_rate=0.4) # 0.5 > 0.4 is False, so mutate

        mock_strategy.mutate.assert_called_once() # Strategy should be called

        with patch('random.random', return_value=0.5): # random.random() returns 0.5
            mock_strategy.reset_mock() # Reset for second call
            mutated_chromosome_no_mutate = await operators.mutate(original_clone, mutation_rate=0.6) # 0.5 > 0.6 is False, so should mutate.
                                                                                                   # Ah, the logic is `if random.random() > mutation_rate: return`, so if true, it skips.
                                                                                                   # If random=0.5, mutation_rate=0.6 -> 0.5 > 0.6 is False. Mutation occurs.
                                                                                                   # If random=0.5, mutation_rate=0.4 -> 0.5 > 0.4 is True. Mutation skips.

        # Test skip:
        mock_strategy.reset_mock()
        with patch('random.random', return_value=0.5):
            skipped_clone = original_clone.clone() # Fresh clone for this assertion
            mutated_chromosome_skipped = await operators.mutate(skipped_clone, mutation_rate=0.4)
        mock_strategy.mutate.assert_not_called()
        self.assertNotEqual(mutated_chromosome_skipped.id, skipped_clone.id) # Still a clone
        self.assertEqual(mutated_chromosome_skipped.genes, skipped_clone.genes)


    async def test_mutate_empty_gene_list(self): # Made async
        """Test mutation with an empty gene list."""
        empty_chromosome = PromptChromosome(genes=[])
        mock_strategy = MagicMock(spec=BaseMutationStrategy)

        # Strategy should still receive the cloned empty chromosome and operate on it
        mutated_empty_clone = empty_chromosome.clone()
        mutated_empty_clone.genes.append("added_to_empty") # Example mutation
        mutated_empty_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_empty_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        mutated_chromosome = await operators.mutate(empty_chromosome, mutation_rate=1.0) # await

        mock_strategy.mutate.assert_called_once()
        call_args = mock_strategy.mutate.call_args[0][0]
        self.assertNotEqual(call_args.id, empty_chromosome.id)
        self.assertEqual(call_args.genes, [])

        self.assertNotEqual(mutated_chromosome.id, empty_chromosome.id)
        self.assertIn("added_to_empty", mutated_chromosome.genes)


    async def test_mutate_occurs(self): # Made async
        """Test mutation when it's triggered and at least one gene mutates."""
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        mutated_clone = self.chromosome.clone()
        mutated_clone.genes[0] = "mutated_gene1" # Simulate gene change
        mutated_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        result_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, gene_mutation_prob=1.0) # await

        mock_strategy.mutate.assert_called_once()
        self.assertNotEqual(result_chromosome.id, self.chromosome.id)
        self.assertEqual(result_chromosome.genes, ["mutated_gene1"])
        self.assertNotEqual(result_chromosome.genes, self.chromosome.genes)


    async def test_mutate_sets_mutation_op_and_logs(self): # Made async
        class DummyStrategy(BaseMutationStrategy):
            def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
                # Ensure it returns a new instance or a modified clone
                new_chromo = chromosome.clone()
                new_chromo.genes.append("dummy_mutated")
                return new_chromo

        dummy_strategy_instance = DummyStrategy()
        operators = GeneticOperators(mutation_strategies=[dummy_strategy_instance])

        original_chromosome = PromptChromosome(genes=["original"])

        with patch.object(logging.getLogger("prompthelix.ga_metrics"), 'info') as mock_ga_logger_info:
            result = await operators.mutate(original_chromosome, mutation_rate=1.0, run_id="test_run", generation=1) # await

        self.assertEqual(result.mutation_strategy, 'DummyStrategy')
        mock_ga_logger_info.assert_called_once()
        log_args = mock_ga_logger_info.call_args[0][0]
        self.assertEqual(log_args["operation"], "mutation")
        self.assertEqual(log_args["mutation_strategy"], "DummyStrategy")
        self.assertEqual(log_args["chromosome_id"], str(result.id))
        self.assertIn("dummy_mutated", log_args["prompt_text"])


    async def test_mutate_rate_zero(self): # Was already async, content seems fine
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        original_chromosome_clone_for_comparison = self.chromosome.clone()

        result_chromosome = await operators.mutate(original_chromosome_clone_for_comparison, mutation_rate=0.0)

        mock_strategy.mutate.assert_not_called()
        self.assertNotEqual(result_chromosome.id, original_chromosome_clone_for_comparison.id, "Should return a clone even if no mutation.")
        self.assertEqual(result_chromosome.genes, original_chromosome_clone_for_comparison.genes)
        self.assertEqual(result_chromosome.fitness_score, 0.0, "Fitness should be reset on the clone.")


    async def test_mutate_rate_one(self): # Was already async
        mock_strategy = MagicMock(spec=BaseMutationStrategy)
        mutated_clone = self.chromosome.clone()
        mutated_clone.fitness_score = 0.0
        mock_strategy.mutate.return_value = mutated_clone

        strategies = [mock_strategy]
        operators = GeneticOperators(mutation_strategies=strategies)

        result_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0)

        mock_strategy.mutate.assert_called_once()
        self.assertEqual(result_chromosome, mutated_clone)


    async def test_mutate_with_style_optimizer_agent_success(self):
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        style_optimized_chromosome = self.chromosome.clone()
        style_optimized_chromosome.genes.append("style_optimized")
        style_optimized_chromosome.fitness_score = 0.8

        # Mock process_request to be an async function returning an awaitable
        async_mock_process_request = AsyncMock(return_value=style_optimized_chromosome)
        self.mock_style_optimizer_agent.process_request = async_mock_process_request


        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        target_style = "concise"
        final_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, target_style=target_style)

        mock_base_strategy.mutate.assert_called_once()
        async_mock_process_request.assert_called_once_with({
            "prompt_chromosome": base_mutated_chromosome,
            "target_style": target_style
        })
        self.assertEqual(final_chromosome, style_optimized_chromosome)
        self.assertEqual(final_chromosome.fitness_score, 0.0, "Fitness should be reset by GeneticOperators.mutate after style optimization.")
        self.assertIn("style_optimized", final_chromosome.genes)

    async def test_mutate_with_style_optimizer_agent_returns_non_chromosome(self):
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        async_mock_process_request = AsyncMock(return_value=None) # Agent returns None
        self.mock_style_optimizer_agent.process_request = async_mock_process_request


        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        with patch.object(logging.getLogger(self.logger_name), 'warning') as mock_log_warning:
            final_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")

        async_mock_process_request.assert_called_once()
        self.assertEqual(final_chromosome, base_mutated_chromosome)
        self.assertIn("base_mutated", final_chromosome.genes)
        self.assertEqual(final_chromosome.fitness_score, 0.0)

        mock_log_warning.assert_called_once()
        args, _ = mock_log_warning.call_args
        log_message = args[0]
        self.assertIn(f"StyleOptimizerAgent did not return a PromptChromosome for {base_mutated_chromosome.id}", log_message)


    async def test_mutate_with_style_optimizer_agent_raises_exception(self):
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0

        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        async_mock_process_request = AsyncMock(side_effect=Exception("Agent Error"))
        self.mock_style_optimizer_agent.process_request = async_mock_process_request


        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=self.mock_style_optimizer_agent
        )

        with patch.object(logging.getLogger(self.logger_name), 'error') as mock_log_error:
            final_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")


        async_mock_process_request.assert_called_once()
        self.assertEqual(final_chromosome, base_mutated_chromosome)
        self.assertIn("base_mutated", final_chromosome.genes)
        self.assertEqual(final_chromosome.fitness_score, 0.0)

        mock_log_error.assert_called_once()
        args, _ = mock_log_error.call_args
        log_message = args[0]
        self.assertIn(f"Style optimization failed for {base_mutated_chromosome.id}", log_message)
        self.assertIn("Agent Error", log_message)


    async def test_mutate_no_style_optimizer_agent_but_target_style_provided(self):
        mock_base_strategy = MagicMock(spec=BaseMutationStrategy)
        base_mutated_chromosome = self.chromosome.clone()
        base_mutated_chromosome.genes.append("base_mutated")
        base_mutated_chromosome.fitness_score = 0.0
        mock_base_strategy.mutate.return_value = base_mutated_chromosome

        operators = GeneticOperators(
            mutation_strategies=[mock_base_strategy],
            style_optimizer_agent=None # Agent not provided
        )


        with patch.object(logging.getLogger(self.logger_name), 'warning') as mock_log_warning:
            final_chromosome = await operators.mutate(self.chromosome, mutation_rate=1.0, target_style="concise")

        self.assertEqual(final_chromosome, base_mutated_chromosome)

        # Check if logger.warning was called and with the expected message content
        mock_log_warning.assert_called_once()
        args, _ = mock_log_warning.call_args
        log_message = args[0]
        self.assertIn(f"target_style 'concise' for {base_mutated_chromosome.id}, but StyleOptimizerAgent not available. Skipping.", log_message)


if __name__ == '__main__':
    unittest.main()
