import unittest
from unittest.mock import patch
import random
import string

from prompthelix.genetics.engine import PromptChromosome, GeneticOperators, PopulationManager, FitnessEvaluator

class TestPromptChromosome(unittest.TestCase):
    """Test suite for the PromptChromosome class."""

    def test_chromosome_creation(self):
        """Test basic creation of a PromptChromosome."""
        gene_string = "testgene"
        chromosome = PromptChromosome(genes=gene_string)
        self.assertIsNotNone(chromosome)
        self.assertEqual(chromosome.genes, gene_string)
        self.assertEqual(chromosome.fitness_score, 0) # Fitness is 0 before calculate_fitness

    def test_str_representation(self):
        """Test the string representation of a chromosome."""
        gene_string = "testgene"
        chromosome = PromptChromosome(genes=gene_string)
        self.assertEqual(str(chromosome), gene_string)

    def test_calculate_fitness(self):
        """Test the calculate_fitness method."""
        gene_string = "abc"
        chromosome = PromptChromosome(genes=gene_string)
        
        # Fitness should be 0 initially
        self.assertEqual(chromosome.fitness_score, 0)
        
        returned_fitness = chromosome.calculate_fitness()
        
        # Assert fitness_score attribute is updated
        self.assertEqual(chromosome.fitness_score, len(gene_string))
        # Assert the method returns the calculated fitness
        self.assertEqual(returned_fitness, len(gene_string))

        gene_string_empty = ""
        chromosome_empty = PromptChromosome(genes=gene_string_empty)
        returned_fitness_empty = chromosome_empty.calculate_fitness()
        self.assertEqual(chromosome_empty.fitness_score, 0)
        self.assertEqual(returned_fitness_empty, 0)


class TestGeneticOperators(unittest.TestCase):
    """Test suite for the GeneticOperators class."""

    def setUp(self):
        self.operators = GeneticOperators()
        self.sample_population = [
            PromptChromosome("short"),      # fitness 5
            PromptChromosome("mediumsize"), # fitness 10
            PromptChromosome("verylonggene") # fitness 12
        ]
        # Calculate fitness for the sample population
        for chrom in self.sample_population:
            chrom.calculate_fitness()

    def test_selection(self):
        """Test the selection operator."""
        num_parents = 2
        selected_parents = self.operators.selection(self.sample_population, num_parents)
        
        self.assertEqual(len(selected_parents), num_parents)
        # Parents should be sorted by fitness, highest first
        self.assertEqual(selected_parents[0].genes, "verylonggene")
        self.assertEqual(selected_parents[1].genes, "mediumsize")

    @patch('random.randint')
    def test_crossover(self, mock_randint):
        """Test the crossover operator with a mocked crossover point."""
        parent1 = PromptChromosome("parentone") # len 9
        parent2 = PromptChromosome("parenttwo") # len 9
        
        # Mock random.randint to return a fixed crossover point
        # Let's say crossover point is 4 (index for 'n' in parentone)
        mock_randint.return_value = 4 
        
        offspring = self.operators.crossover(parent1, parent2)
        
        self.assertIsInstance(offspring, PromptChromosome)
        self.assertEqual(offspring.genes, "pare" + "nttwo") # "parentone"[:4] + "parenttwo"[4:]
        self.assertEqual(offspring.fitness_score, 0) # Fitness not calculated yet

        # Test with different lengths (though current implementation might not handle it well, test it)
        parent3 = PromptChromosome("shortp") # len 6
        parent4 = PromptChromosome("verylongparent") # len 14
        mock_randint.return_value = 3 # crossover at "sho"
        offspring2 = self.operators.crossover(parent3, parent4)
        self.assertEqual(offspring2.genes, "sho" + "ylongparent")


    @patch('random.choice')
    @patch('random.random')
    def test_mutate(self, mock_random_random, mock_random_choice):
        """Test the mutation operator."""
        chromosome = PromptChromosome("testmutate")
        original_genes = chromosome.genes
        gene_list = list(original_genes)

        # Test with mutation_rate = 1.0 (guaranteed mutation)
        mock_random_random.return_value = 0.0 # Ensures random.random() < mutation_rate is true
        mock_random_choice.return_value = 'X' # Mutate to 'X'
        
        mutated_chromosome_guaranteed = self.operators.mutate(PromptChromosome(original_genes), mutation_rate=1.0)
        # Check if all characters are 'X' as per mock_random_choice
        self.assertEqual(mutated_chromosome_guaranteed.genes, 'X' * len(original_genes))

        # Test with mutation_rate = 0.0 (no mutation)
        # No need to mock random.random or random.choice here as the condition random.random() < 0.0 will always be false
        mutated_chromosome_none = self.operators.mutate(PromptChromosome(original_genes), mutation_rate=0.0)
        self.assertEqual(mutated_chromosome_none.genes, original_genes)

        # Test partial mutation
        chromosome_partial = PromptChromosome("abc")
        # Let's say only the first char mutates
        # random.random() calls: 0.0 (mutate), 0.5 (no), 0.5 (no)
        # random.choice() call: 'Z'
        mock_random_random.side_effect = [0.01, 0.5, 0.5] # First mutates, others don't for rate 0.1
        mock_random_choice.return_value = 'Z'
        
        mutated_partial = self.operators.mutate(chromosome_partial, mutation_rate=0.1)
        self.assertEqual(mutated_partial.genes[0], 'Z')
        self.assertEqual(mutated_partial.genes[1:], "bc")


class TestPopulationManager(unittest.TestCase):
    """Test suite for the PopulationManager class."""

    def setUp(self):
        self.params = {
            "population_size": 10,
            "gene_pool_characters": "abc",
            "gene_length": 5,
            "mutation_rate": 0.1,
            "num_parents_for_selection": 4
        }
        self.manager = PopulationManager(**self.params)

    def test_initialize_population(self):
        """Test population initialization."""
        self.manager.initialize_population()
        
        self.assertEqual(len(self.manager.population), self.params["population_size"])
        for chromosome in self.manager.population:
            self.assertIsInstance(chromosome, PromptChromosome)
            self.assertEqual(len(chromosome.genes), self.params["gene_length"])
            # Fitness should be calculated and equal to gene_length for this simple fitness function
            self.assertEqual(chromosome.fitness_score, self.params["gene_length"])
            for gene_char in chromosome.genes:
                self.assertIn(gene_char, self.params["gene_pool_characters"])

    @patch.object(GeneticOperators, 'selection')
    @patch.object(GeneticOperators, 'crossover')
    @patch.object(GeneticOperators, 'mutate')
    def test_evolve_population(self, mock_mutate, mock_crossover, mock_selection):
        """Test population evolution process."""
        self.manager.initialize_population()
        initial_population_size = len(self.manager.population)

        # Mock the genetic operators to return predictable results
        # Create dummy chromosomes for mocking
        dummy_parent = PromptChromosome("parent")
        dummy_parent.calculate_fitness()
        dummy_offspring_before_mutation = PromptChromosome("crssovr")
        dummy_offspring_after_mutation = PromptChromosome("mutated")
        dummy_offspring_after_mutation.calculate_fitness() # Ensure fitness is calculated

        mock_selection.return_value = [dummy_parent] * self.params["num_parents_for_selection"]
        mock_crossover.return_value = dummy_offspring_before_mutation
        mock_mutate.return_value = dummy_offspring_after_mutation
        
        self.manager.evolve_population()
        
        self.assertEqual(len(self.manager.population), initial_population_size)
        self.assertEqual(self.manager.generation_number, 1)
        
        # Check if operators were called
        mock_selection.assert_called_once()
        # Crossover and mutate are called population_size times in the loop
        self.assertEqual(mock_crossover.call_count, self.params["population_size"])
        self.assertEqual(mock_mutate.call_count, self.params["population_size"])

        # Check if fitness is calculated for new offspring
        for chrom in self.manager.population:
            self.assertEqual(chrom.genes, "mutated") # All offspring are the same due to mock
            self.assertEqual(chrom.fitness_score, len("mutated"))


    def test_get_fittest_individual(self):
        """Test retrieving the fittest individual from the population."""
        self.manager.initialize_population() # Populates with genes of length 5, fitness 5
        
        # Manually add a fitter individual
        fitter_gene = "abcdef" # fitness 6
        fittest_chromosome = PromptChromosome(fitter_gene)
        fittest_chromosome.calculate_fitness()
        self.manager.population.append(fittest_chromosome)
        
        # Manually add a less fit individual
        less_fit_gene = "abc" # fitness 3
        less_fit_chromosome = PromptChromosome(less_fit_gene)
        less_fit_chromosome.calculate_fitness()
        self.manager.population.append(less_fit_chromosome)

        retrieved_fittest = self.manager.get_fittest_individual()
        self.assertEqual(retrieved_fittest.genes, fitter_gene)
        self.assertEqual(retrieved_fittest.fitness_score, len(fitter_gene))

        # Test with an empty population (should ideally not happen, but good to check)
        self.manager.population = []
        with self.assertRaises(IndexError): # Or handle gracefully in get_fittest_individual
            self.manager.get_fittest_individual()
        
        # Test with population having one individual
        self.manager.population = [fittest_chromosome]
        retrieved_single = self.manager.get_fittest_individual()
        self.assertEqual(retrieved_single.genes, fitter_gene)


class TestFitnessEvaluator(unittest.TestCase):
    """Test suite for the FitnessEvaluator class."""

    def setUp(self):
        self.evaluator = FitnessEvaluator()

    def test_evaluate_basic_score(self):
        """Test basic score calculation (just length)."""
        genes = "test"
        chromosome = PromptChromosome(genes)
        expected_score = float(len(genes))
        
        score = self.evaluator.evaluate(chromosome)
        
        self.assertEqual(score, expected_score)
        self.assertEqual(chromosome.fitness_score, expected_score)

    def test_evaluate_with_char_bonuses(self):
        """Test score calculation with character bonuses."""
        genes = "lazy fox" # x, z
        chromosome = PromptChromosome(genes)
        expected_score = float(len(genes)) + 5.0 (for 'x') + 5.0 (for 'z')
        
        score = self.evaluator.evaluate(chromosome)
        
        self.assertEqual(score, expected_score)
        self.assertEqual(chromosome.fitness_score, expected_score)

    def test_evaluate_with_keyword_bonuses(self):
        """Test score calculation with keyword bonuses."""
        genes = "my prompthelix project" # prompt, helix
        chromosome = PromptChromosome(genes)
        expected_score = float(len(genes)) + 10.0 (for 'helix') + 8.0 (for 'prompt')
        
        score = self.evaluator.evaluate(chromosome)
        
        self.assertEqual(score, expected_score)
        self.assertEqual(chromosome.fitness_score, expected_score)

    def test_evaluate_all_bonuses(self):
        """Test score calculation with all types of bonuses."""
        genes = "quiz helix prompt" # q, z, helix, prompt
        chromosome = PromptChromosome(genes)
        expected_score = float(len(genes)) + 5.0 (for 'q') + 5.0 (for 'z') + 10.0 (for 'helix') + 8.0 (for 'prompt')
        
        score = self.evaluator.evaluate(chromosome)
        
        self.assertEqual(score, expected_score)
        self.assertEqual(chromosome.fitness_score, expected_score)

    def test_evaluate_case_insensitivity(self):
        """Test that bonuses are case-insensitive."""
        genes = "QuIz HeLiX pRoMpT"
        chromosome = PromptChromosome(genes)
        expected_score = float(len(genes)) + 5.0 (for 'q') + 5.0 (for 'z') + 10.0 (for 'helix') + 8.0 (for 'prompt')
        score = self.evaluator.evaluate(chromosome)
        self.assertEqual(score, expected_score)


class TestPopulationManager(unittest.TestCase):
    """Test suite for the PopulationManager class."""

    def setUp(self):
        self.base_params = {
            "population_size": 10,
            "gene_pool_characters": "abcxyzprompthelix", # Ensure bonus chars are possible
            "gene_length": 15, # Long enough for keywords
            "mutation_rate": 0.1,
            "num_parents_for_selection": 4
        }
        # Manager without external evaluator
        self.manager_default_fitness = PopulationManager(**self.base_params)
        
        # Manager with external evaluator
        self.fitness_evaluator = FitnessEvaluator()
        self.manager_with_evaluator = PopulationManager(**self.base_params, fitness_evaluator=self.fitness_evaluator)


    def test_initialize_population_default_fitness(self):
        """Test population initialization using default chromosome.calculate_fitness()."""
        self.manager_default_fitness.initialize_population()
        
        self.assertEqual(len(self.manager_default_fitness.population), self.base_params["population_size"])
        for chromosome in self.manager_default_fitness.population:
            self.assertIsInstance(chromosome, PromptChromosome)
            self.assertEqual(len(chromosome.genes), self.base_params["gene_length"])
            # Fitness should be calculated by PromptChromosome.calculate_fitness()
            self.assertEqual(chromosome.fitness_score, float(len(chromosome.genes)))
            for gene_char in chromosome.genes:
                self.assertIn(gene_char, self.base_params["gene_pool_characters"])

    def test_initialize_population_with_evaluator(self):
        """Test population initialization using the external FitnessEvaluator."""
        self.manager_with_evaluator.initialize_population()
        
        self.assertEqual(len(self.manager_with_evaluator.population), self.base_params["population_size"])
        # Check at least one chromosome to see if FitnessEvaluator was likely used
        # This is probabilistic but simpler than mocking gene generation for specific bonuses
        found_bonus_score = False
        for chromosome in self.manager_with_evaluator.population:
            self.assertIsInstance(chromosome, PromptChromosome)
            self.assertEqual(len(chromosome.genes), self.base_params["gene_length"])
            # Fitness should be calculated by FitnessEvaluator
            # It's hard to predict exact score without knowing the genes,
            # but if it has bonus characters/keywords, it should be > len(genes)
            expected_eval_score = self.fitness_evaluator.evaluate(PromptChromosome(chromosome.genes)) # Recalculate to compare
            self.assertEqual(chromosome.fitness_score, expected_eval_score)
            if chromosome.fitness_score > len(chromosome.genes):
                found_bonus_score = True
        
        # Given the gene pool, it's highly probable some chromosomes get bonuses
        # If gene_length is small or pool is limited, this might need adjustment
        self.assertTrue(found_bonus_score, "Expected at least one chromosome to have a bonus-affected score by FitnessEvaluator.")


    @patch.object(GeneticOperators, 'selection')
    @patch.object(GeneticOperators, 'crossover')
    @patch.object(GeneticOperators, 'mutate')
    def test_evolve_population_default_fitness(self, mock_mutate, mock_crossover, mock_selection):
        """Test population evolution with default fitness calculation."""
        self.manager_default_fitness.initialize_population()
        initial_population_size = len(self.manager_default_fitness.population)

        dummy_parent = PromptChromosome("parentgenes")
        dummy_parent.calculate_fitness()
        dummy_offspring_before_mutation = PromptChromosome("crossovergenes")
        dummy_offspring_after_mutation = PromptChromosome("mutatedgenes")
        # Fitness calculated by PromptChromosome.calculate_fitness()
        dummy_offspring_after_mutation.calculate_fitness() 

        mock_selection.return_value = [dummy_parent] * self.base_params["num_parents_for_selection"]
        mock_crossover.return_value = dummy_offspring_before_mutation
        mock_mutate.return_value = dummy_offspring_after_mutation
        
        self.manager_default_fitness.evolve_population()
        
        self.assertEqual(len(self.manager_default_fitness.population), initial_population_size)
        self.assertEqual(self.manager_default_fitness.generation_number, 1)
        
        for chrom in self.manager_default_fitness.population:
            self.assertEqual(chrom.genes, "mutatedgenes")
            self.assertEqual(chrom.fitness_score, len("mutatedgenes")) # Default fitness

    @patch.object(GeneticOperators, 'selection')
    @patch.object(GeneticOperators, 'crossover')
    @patch.object(GeneticOperators, 'mutate')
    def test_evolve_population_with_evaluator(self, mock_mutate, mock_crossover, mock_selection):
        """Test population evolution with external FitnessEvaluator."""
        self.manager_with_evaluator.initialize_population() # Uses evaluator
        initial_population_size = len(self.manager_with_evaluator.population)

        dummy_parent = PromptChromosome("parentgenesXQZ") # Contains bonus chars
        self.fitness_evaluator.evaluate(dummy_parent) # Evaluated by FitnessEvaluator

        dummy_offspring_before_mutation = PromptChromosome("crossoverPROMPT") # Contains bonus word
        dummy_offspring_after_mutation = PromptChromosome("mutatedHELIX")    # Contains bonus word
        # Fitness will be calculated by FitnessEvaluator inside evolve_population

        mock_selection.return_value = [dummy_parent] * self.base_params["num_parents_for_selection"]
        mock_crossover.return_value = dummy_offspring_before_mutation
        mock_mutate.return_value = dummy_offspring_after_mutation
        
        self.manager_with_evaluator.evolve_population()
        
        self.assertEqual(len(self.manager_with_evaluator.population), initial_population_size)
        self.assertEqual(self.manager_with_evaluator.generation_number, 1)
        
        expected_fitness_for_mutated_helix = self.fitness_evaluator.evaluate(PromptChromosome("mutatedHELIX"))
        for chrom in self.manager_with_evaluator.population:
            self.assertEqual(chrom.genes, "mutatedHELIX")
            # Fitness should reflect FitnessEvaluator's calculation
            self.assertEqual(chrom.fitness_score, expected_fitness_for_mutated_helix)


    def test_get_fittest_individual(self):
        """Test retrieving the fittest individual from the population (applies to both managers)."""
        # This test uses the default fitness for simplicity, but logic is same for manager_with_evaluator
        # as get_fittest_individual only sorts based on pre-calculated fitness_score
        self.manager_default_fitness.initialize_population() 
        
        fitter_gene = "abcdef" 
        fittest_chromosome = PromptChromosome(fitter_gene)
        fittest_chromosome.calculate_fitness() # Default fitness
        self.manager_default_fitness.population.append(fittest_chromosome)
        
        less_fit_gene = "abc" 
        less_fit_chromosome = PromptChromosome(less_fit_gene)
        less_fit_chromosome.calculate_fitness() # Default fitness
        self.manager_default_fitness.population.append(less_fit_chromosome)

        retrieved_fittest = self.manager_default_fitness.get_fittest_individual()
        self.assertEqual(retrieved_fittest.genes, fitter_gene)

        # Test with an empty population
        self.manager_default_fitness.population = []
        with self.assertRaises(IndexError):
            self.manager_default_fitness.get_fittest_individual()
        
        # Test with population having one individual
        self.manager_default_fitness.population = [fittest_chromosome]
        retrieved_single = self.manager_default_fitness.get_fittest_individual()
        self.assertEqual(retrieved_single.genes, fitter_gene)


if __name__ == '__main__':
    unittest.main()
