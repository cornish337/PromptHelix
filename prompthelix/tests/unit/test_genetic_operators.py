import unittest
import random
from unittest.mock import patch, Mock, AsyncMock
from prompthelix.genetics.engine import GeneticOperators, PromptChromosome
from prompthelix.genetics.strategy_base import BaseMutationStrategy # Changed import

class TestGeneticOperators(unittest.IsolatedAsyncioTestCase): # Changed to IsolatedAsyncioTestCase
    """Test suite for the GeneticOperators class."""

    async def asyncSetUp(self): # Changed to asyncSetUp
        """Set up common test data."""
        self.operators = GeneticOperators()
        self.population = [
            PromptChromosome(genes=["P1_G1", "P1_G2"], fitness_score=0.7),
            PromptChromosome(genes=["P2_G1"], fitness_score=0.9), # Fittest
            PromptChromosome(genes=["P3_G1", "P3_G2", "P3_G3"], fitness_score=0.5),
            PromptChromosome(genes=["P4_G1", "P4_G2"], fitness_score=0.8)
        ]
        self.parent1 = PromptChromosome(genes=["Parent1_Gene1", "Parent1_Gene2", "Parent1_Gene3"], fitness_score=0.6)
        self.parent2 = PromptChromosome(genes=["Parent2_Gene1", "Parent2_Gene2"], fitness_score=0.7)
        self.chromosome_to_mutate = PromptChromosome(genes=["GeneA", "GeneB", "GeneC"], fitness_score=0.5)

    # --- Test selection ---
    def test_selection_valid(self):
        """Test selection with a valid population and tournament size."""
        # Mock random.sample to control tournament contenders
        # To ensure we test the logic of picking the fittest from the sample
        
        # Scenario 1: Pick P2 (0.9) and P4 (0.8) -> P2 wins
        with patch('random.sample', return_value=[self.population[1], self.population[3]]): # P2, P4
            selected = self.operators.selection(self.population, tournament_size=2)
            self.assertEqual(selected.fitness_score, 0.9) # P2 is fittest in this sample

        # Scenario 2: Pick P1 (0.7), P3 (0.5) -> P1 wins
        with patch('random.sample', return_value=[self.population[0], self.population[2]]): # P1, P3
            selected = self.operators.selection(self.population, tournament_size=2)
            self.assertEqual(selected.fitness_score, 0.7)

    def test_selection_tournament_larger_than_population(self):
        """Test selection when tournament size is larger than population size."""
        # Should select the best from the entire population
        selected = self.operators.selection(self.population, tournament_size=10)
        self.assertEqual(selected.fitness_score, 0.9) # P2 is the fittest overall

    def test_selection_empty_population(self):
        """Test selection with an empty population."""
        with self.assertRaises(ValueError):
            self.operators.selection([], tournament_size=3)

    def test_selection_invalid_tournament_size(self):
        """Test selection with invalid tournament sizes."""
        with self.assertRaises(ValueError):
            self.operators.selection(self.population, tournament_size=0)
        with self.assertRaises(ValueError):
            self.operators.selection(self.population, tournament_size=-1)

    # --- Test crossover ---
    @patch('random.randint') # To control crossover point
    @patch('random.random')  # To control crossover rate
    def test_crossover_occurs(self, mock_random_rate, mock_randint_point):
        """Test crossover when it's triggered by crossover_rate."""
        mock_random_rate.return_value = 0.5 # Ensure crossover_rate (0.7 default) is met
        mock_randint_point.return_value = 1 # Crossover after the first gene of shorter parent

        child1, child2 = self.operators.crossover(self.parent1, self.parent2, crossover_rate=0.7)

        self.assertNotEqual(child1.genes, self.parent1.genes)
        self.assertNotEqual(child2.genes, self.parent2.genes)
        
        # Parent1 genes: ["Parent1_Gene1", "Parent1_Gene2", "Parent1_Gene3"]
        # Parent2 genes: ["Parent2_Gene1", "Parent2_Gene2"]
        # Crossover point 1 (on shorter parent P2):
        # Child1: P1_G1 + P2_G2  = ["Parent1_Gene1", "Parent2_Gene2"]
        # Child2: P2_G1 + P1_G2 + P1_G3 = ["Parent2_Gene1", "Parent1_Gene2", "Parent1_Gene3"]
        self.assertEqual(child1.genes, ["Parent1_Gene1", "Parent2_Gene2"])
        self.assertEqual(child2.genes, ["Parent2_Gene1", "Parent1_Gene2", "Parent1_Gene3"])

        self.assertEqual(child1.fitness_score, 0.0)
        self.assertEqual(child2.fitness_score, 0.0)
        self.assertNotEqual(child1.id, self.parent1.id)
        self.assertNotEqual(child2.id, self.parent2.id)

    @patch('random.random')
    def test_crossover_does_not_occur(self, mock_random_rate):
        """Test crossover when it's not triggered by crossover_rate."""
        mock_random_rate.return_value = 0.9 # Above default crossover_rate (0.7)

        child1, child2 = self.operators.crossover(self.parent1, self.parent2, crossover_rate=0.7)

        self.assertEqual(child1.genes, self.parent1.genes)
        self.assertEqual(child2.genes, self.parent2.genes)
        self.assertNotEqual(child1.id, self.parent1.id, "Child1 should be a clone with a new ID.")
        self.assertNotEqual(child2.id, self.parent2.id, "Child2 should be a clone with a new ID.")
        self.assertEqual(child1.fitness_score, 0.0, "Fitness of cloned child should be reset.")
        self.assertEqual(child2.fitness_score, 0.0, "Fitness of cloned child should be reset.")


    def test_crossover_empty_gene_lists_both_parents(self):
        """Test crossover when both parents have empty gene lists."""
        empty_parent1 = PromptChromosome(genes=[])
        empty_parent2 = PromptChromosome(genes=[])
        
        with patch('random.random', return_value=0.5): # Ensure crossover happens
            child1, child2 = self.operators.crossover(empty_parent1, empty_parent2)
            self.assertEqual(child1.genes, [])
            self.assertEqual(child2.genes, [])
            self.assertEqual(child1.fitness_score, 0.0)
            self.assertEqual(child2.fitness_score, 0.0)

    def test_crossover_one_parent_empty_gene_list(self):
        """Test crossover when one parent has an empty gene list."""
        empty_parent = PromptChromosome(genes=[])
        
        with patch('random.random', return_value=0.0) as mock_rand_rate: # Ensure crossover rate is met for the condition
            # Scenario 1: Parent1 is empty, Parent2 is not.
            # Crossover condition `parent1.genes and parent2.genes` will be FALSE because parent1.genes is empty.
            # So, children will be clones of parents.
            child1, child2 = self.operators.crossover(empty_parent, self.parent2, crossover_rate=1.0)
            self.assertEqual(child1.genes, empty_parent.genes, "Child1 should be clone of empty_parent if crossover condition not met")
            self.assertEqual(child2.genes, self.parent2.genes, "Child2 should be clone of parent2 if crossover condition not met")
            self.assertEqual(child1.fitness_score, 0.0)
            self.assertEqual(child2.fitness_score, 0.0)

            # Scenario 2: Parent1 is not empty, Parent2 is empty.
            # Crossover condition `parent1.genes and parent2.genes` will be FALSE.
            child1_s2, child2_s2 = self.operators.crossover(self.parent1, empty_parent, crossover_rate=1.0)
            self.assertEqual(child1_s2.genes, self.parent1.genes, "Child1_s2 should be clone of parent1 if crossover condition not met")
            self.assertEqual(child2_s2.genes, empty_parent.genes, "Child2_s2 should be clone of empty_parent if crossover condition not met")
            self.assertEqual(child1_s2.fitness_score, 0.0)
            self.assertEqual(child2_s2.fitness_score, 0.0)


    # --- Test mutate ---
    @patch('random.choice') # To control mutation type
    @patch('random.random') # To control mutation rates
    async def test_mutate_occurs(self, mock_random_rates, mock_random_choice): # Added async
        """Test mutation when it's triggered and at least one gene mutates."""
        from prompthelix.genetics.mutation_strategies import PlaceholderReplaceStrategy # Import strategy
        # First random.random() for overall mutation_rate, subsequent for gene_mutation_prob
        mock_random_rates.side_effect = [0.05, 0.05, 0.9, 0.05] # Mutate chromo, Mutate gene1, Don't mutate gene2, Mutate gene3
        mock_random_choice.return_value = PlaceholderReplaceStrategy() # Return an instance

        original_genes = list(self.chromosome_to_mutate.genes) # Deep copy for comparison
        mutated_chromosome = await self.operators.mutate(self.chromosome_to_mutate, mutation_rate=0.1, gene_mutation_prob=0.2) # Added await

        self.assertNotEqual(mutated_chromosome.id, self.chromosome_to_mutate.id)
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)
        
        # Check if genes were actually changed
        # PlaceholderReplaceStrategy replaces a random gene with "[MUTATED_GENE_SEGMENT]"
        # This test assumes at least one gene will be replaced.
        self.assertTrue(any(gene == "[MUTATED_GENE_SEGMENT]" for gene in mutated_chromosome.genes),
                        "At least one gene should have been replaced by PlaceholderReplaceStrategy.")
        self.assertNotEqual(mutated_chromosome.genes, original_genes, "Genes should have been modified by PlaceholderReplaceStrategy.")


    @patch('random.random')
    async def test_mutate_does_not_occur_rate_too_high(self, mock_random_rate): # Added async
        """Test mutation when overall mutation_rate is not met."""
        mock_random_rate.return_value = 0.5 # Higher than mutation_rate
        
        original_genes = list(self.chromosome_to_mutate.genes)
        mutated_chromosome = await self.operators.mutate(self.chromosome_to_mutate, mutation_rate=0.1, gene_mutation_prob=0.2) # Added await
        
        self.assertNotEqual(mutated_chromosome.id, self.chromosome_to_mutate.id, "Should be a clone even if no mutation happens.")
        self.assertEqual(mutated_chromosome.genes, original_genes, "Genes should be unchanged if mutation_rate not met.")
        self.assertEqual(mutated_chromosome.fitness_score, 0.0, "Fitness of cloned chromosome should be reset.")

    @patch('random.random')
    async def test_mutate_chromosome_selected_but_no_gene_mutates(self, mock_random_rates): # Added async
        """Test when chromosome is selected for mutation, but no individual gene meets gene_mutation_prob.
           This should trigger the fallback to mutate at least one gene.
        """
        # First call for mutation_rate (0.05 < 0.1, so mutate)
        # Next calls for gene_mutation_prob (all > 0.2, so no individual gene mutates initially)
        mock_random_rates.side_effect = [0.05, 0.9, 0.9, 0.9] 
        
        original_genes = list(self.chromosome_to_mutate.genes)
        mutated_chromosome = await self.operators.mutate(self.chromosome_to_mutate, mutation_rate=0.1, gene_mutation_prob=0.2) # Added await
        
        self.assertNotEqual(mutated_chromosome.id, self.chromosome_to_mutate.id)
        self.assertEqual(mutated_chromosome.fitness_score, 0.0)
        # If NoOperationMutationStrategy is used (default for self.operators), genes should be identical.
        self.assertEqual(mutated_chromosome.genes, original_genes,
                        "Genes should remain unchanged if NoOperationMutationStrategy is chosen and no fallback exists.")


    async def test_mutate_empty_gene_list(self): # Added async
        """Test mutation with an empty gene list."""
        empty_chromosome = PromptChromosome(genes=[], fitness_score=0.3)
        
        with patch('random.random', return_value=0.05): # Ensure mutation is attempted
            mutated_chromosome = await self.operators.mutate(empty_chromosome, mutation_rate=0.1, gene_mutation_prob=0.2) # Added await
            
            self.assertNotEqual(mutated_chromosome.id, empty_chromosome.id)
            self.assertEqual(mutated_chromosome.genes, []) # Genes should remain empty
            self.assertEqual(mutated_chromosome.fitness_score, 0.0)

    @patch('random.choice')
    @patch('random.random')
    async def test_mutate_with_style_optimizer(self, mock_random, mock_choice): # Added async
        """Mutation should incorporate StyleOptimizerAgent output when provided."""
        from prompthelix.genetics.mutation_strategies import PlaceholderReplaceStrategy # Import strategy
        mock_random.side_effect = [0.05, 0.05]  # Trigger mutation and mutate gene0
        mock_choice.return_value = PlaceholderReplaceStrategy() # Return an instance

        style_mock = AsyncMock() # Changed to AsyncMock
        # Configure the mock for an async process_request
        style_mock.process_request.return_value = PromptChromosome(genes=["styled"], fitness_score=0.0)


        operators = GeneticOperators(style_optimizer_agent=style_mock)
        chromo = PromptChromosome(genes=["gene"], fitness_score=0.1)

        result = await operators.mutate(chromo, mutation_rate=0.1, gene_mutation_prob=0.2, target_style="formal") # Added await

        style_mock.process_request.assert_called_once()
        self.assertEqual(result.genes, ["styled"])

    @patch('random.random')
    @patch('random.randint')
    def test_crossover_sets_parents(self, mock_randint, mock_random): # This is not a mutate test, no async needed unless crossover becomes async
        mock_random.return_value = 0.0
        mock_randint.return_value = 0
        child1, child2 = self.operators.crossover(self.parent1, self.parent2, crossover_rate=1.0)
        self.assertEqual(child1.parent_ids, [str(self.parent1.id), str(self.parent2.id)]) # Changed to parent_ids
        self.assertEqual(child2.parent_ids, [str(self.parent1.id), str(self.parent2.id)]) # Changed to parent_ids

    @patch('random.choice')
    @patch('random.random')
    async def test_mutate_sets_mutation_op_and_logs(self, mock_random, mock_choice): # Added async
        class DummyStrategy(BaseMutationStrategy): # Changed inheritance
            def __init__(self, settings=None, **kwargs): # Added __init__ for abstract class
                super().__init__(settings=settings, **kwargs)

            def mutate(self, chromosome: PromptChromosome) -> PromptChromosome: # This is sync, but called by async mutate
                c = chromosome.clone()
                c.genes.append('x')
                return c

        strategy = DummyStrategy()
        mock_choice.return_value = strategy
        mock_random.return_value = 0.0
        # self.chromosome_to_mutate.parent_ids = ['p1', 'p2'] # parent_ids are set by crossover, not before mutate
        operators = GeneticOperators(mutation_strategies=[strategy])

        with patch('prompthelix.genetics.engine.logger.info') as mock_log_info:
            result = await operators.mutate(self.chromosome_to_mutate, mutation_rate=1.0) # Added await

        self.assertEqual(result.mutation_strategy, 'DummyStrategy')
        # parent_ids for a mutated chromosome should be the ID of the chromosome it was mutated from.
        self.assertEqual(result.parent_ids, [str(self.chromosome_to_mutate.id)])

        found_log = False
        for call_args in mock_log_info.call_args_list:
            log_message = call_args[0][0] # First positional argument
            if f"Applying mutation strategy 'DummyStrategy' to chromosome {self.chromosome_to_mutate.id}" in log_message:
                found_log = True
                break
        self.assertTrue(found_log, "Expected log message for applying mutation strategy not found.")

if __name__ == '__main__':
    unittest.main()
