import unittest

import uuid
import copy
from prompthelix.genetics.engine import PromptChromosome

class TestPromptChromosome(unittest.TestCase):
    """Comprehensive test suite for the PromptChromosome class."""

    def setUp(self):
        """Set up common test data if needed, though most tests create their own."""
        self.genes1 = ["Instruction: Summarize.", "Context: Long text...", "Output: Short summary."]
        self.fitness1 = 0.75

    def test_init_with_genes_and_fitness(self):
        """Test __init__ with specified genes and fitness score."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        self.assertIsInstance(chromosome.id, uuid.UUID, "ID should be a UUID instance.")
        self.assertEqual(chromosome.genes, self.genes1, "Genes not initialized correctly.")
        self.assertEqual(chromosome.fitness_score, self.fitness1, "Fitness score not initialized correctly.")

    def test_init_default_fitness(self):
        """Test __init__ with default fitness score."""
        chromosome = PromptChromosome(genes=self.genes1)
        self.assertEqual(chromosome.fitness_score, 0.0, "Fitness score should default to 0.0.")

    def test_init_genes_none(self):
        """Test __init__ with genes=None, expecting an empty list."""
        chromosome = PromptChromosome(genes=None)
        self.assertEqual(chromosome.genes, [], "Genes should default to an empty list if None is provided.")
        self.assertIsInstance(chromosome.id, uuid.UUID)

    def test_init_default_genes(self):
        """Test __init__ with no genes argument, expecting an empty list."""
        chromosome = PromptChromosome()
        self.assertEqual(chromosome.genes, [], "Genes should default to an empty list if no argument is provided.")
        self.assertIsInstance(chromosome.id, uuid.UUID)


    def test_calculate_fitness(self):
        """Test calculate_fitness method."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        self.assertEqual(chromosome.calculate_fitness(), self.fitness1, "calculate_fitness should return self.fitness_score.")
        
        chromosome.fitness_score = 0.9
        self.assertEqual(chromosome.calculate_fitness(), 0.9, "calculate_fitness should reflect updated self.fitness_score.")

    def test_to_prompt_string_default_separator(self):
        """Test to_prompt_string with the default newline separator."""
        chromosome = PromptChromosome(genes=self.genes1)
        expected_string = "Instruction: Summarize.\nContext: Long text...\nOutput: Short summary."
        self.assertEqual(chromosome.to_prompt_string(), expected_string)

    def test_to_prompt_string_custom_separator(self):
        """Test to_prompt_string with a custom separator."""
        chromosome = PromptChromosome(genes=self.genes1)
        expected_string = "Instruction: Summarize. Context: Long text... Output: Short summary."
        self.assertEqual(chromosome.to_prompt_string(separator=" "), expected_string)

    def test_to_prompt_string_empty_genes(self):
        """Test to_prompt_string with an empty gene list."""
        chromosome = PromptChromosome(genes=[])
        self.assertEqual(chromosome.to_prompt_string(), "", "Should return an empty string for empty genes.")

    def test_to_prompt_string_non_string_genes(self):
        """Test to_prompt_string with non-string elements in genes list."""
        chromosome = PromptChromosome(genes=["Gene 1", 123, {"type": "Instruction"}])
        expected_string = "Gene 1\n123\n{'type': 'Instruction'}"
        self.assertEqual(chromosome.to_prompt_string(), expected_string)

    def test_clone(self):
        """Test the clone method for deep copying and new ID generation."""
        original_genes = [["MutableGene1"], "Gene2"]
        original_chromosome = PromptChromosome(genes=original_genes, fitness_score=self.fitness1)
        
        cloned_chromosome = original_chromosome.clone()

        # Test for new instance and different ID
        self.assertIsNot(cloned_chromosome, original_chromosome, "Clone should be a new instance.")
        self.assertNotEqual(cloned_chromosome.id, original_chromosome.id, "Clone should have a new ID.")

        # Test fitness score is copied
        self.assertEqual(cloned_chromosome.fitness_score, original_chromosome.fitness_score, "Fitness score should be copied to clone.")

        # Test genes are deep-copied
        self.assertEqual(cloned_chromosome.genes, original_chromosome.genes, "Genes content should be equal after cloning.")
        self.assertIsNot(cloned_chromosome.genes[0], original_chromosome.genes[0], "Mutable gene elements should be different instances (deep copy).")
        
        # Modify original's mutable gene element and check clone is unaffected
        original_chromosome.genes[0].append("ModifiedInOriginal")
        self.assertNotEqual(cloned_chromosome.genes[0], original_chromosome.genes[0], "Modifying original's mutable gene should not affect clone's.")
        self.assertEqual(cloned_chromosome.genes[0], ["MutableGene1"], "Clone's mutable gene element changed unexpectedly.")

        # Modify clone's genes and check original is unaffected
        cloned_chromosome.genes.append("AddedToClone")
        self.assertNotEqual(cloned_chromosome.genes, original_chromosome.genes, "Modifying clone's genes should not affect original's.")
        
        cloned_chromosome.genes[0].append("ModifiedInClone")
        self.assertNotEqual(cloned_chromosome.genes[0], original_chromosome.genes[0], "Modifying clone's mutable gene should not affect original's.")


    def test_str_representation(self):
        """Test the __str__ method for human-readable output."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        chromosome_str = str(chromosome)

        self.assertIn(str(chromosome.id), chromosome_str, "__str__ should include ID.")
        self.assertIn(f"Fitness: {self.fitness1:.4f}", chromosome_str, "__str__ should include formatted fitness.")
        self.assertIn("Genes:", chromosome_str, "__str__ should include 'Genes:' label.")
        for gene in self.genes1:
            self.assertIn(f"  - {gene}", chromosome_str, f"Gene '{gene}' not found in __str__ output.")

    def test_str_representation_empty_genes(self):
        """Test __str__ with an empty gene list."""
        chromosome = PromptChromosome(genes=[])
        chromosome_str = str(chromosome)
        self.assertIn("  - (No genes)", chromosome_str, "__str__ should indicate no genes for an empty list.")

    def test_repr_representation(self):
        """Test the __repr__ method for unambiguous output."""
        chromosome = PromptChromosome(genes=self.genes1, fitness_score=self.fitness1)
        chromosome_repr = repr(chromosome)
        
        # Expected format: PromptChromosome(id='...', genes=[...], fitness_score=...)
        self.assertTrue(chromosome_repr.startswith("PromptChromosome(id="), "__repr__ should start with class name and id.")
        self.assertIn(f"id='{chromosome.id}'", chromosome_repr)
        self.assertIn(f"genes={self.genes1!r}", chromosome_repr, "__repr__ should include repr of genes.")
        self.assertIn(f"fitness_score={self.fitness1:.4f}", chromosome_repr, "__repr__ should include formatted fitness.")
        self.assertTrue(chromosome_repr.endswith(")"), "__repr__ should end with a parenthesis.")


if __name__ == '__main__':
    unittest.main()
