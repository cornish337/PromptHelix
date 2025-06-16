import unittest
from unittest.mock import Mock, patch, call, MagicMock
from prompthelix.genetics.engine import (
    PopulationManager, 
    PromptChromosome, 
    GeneticOperators, 
    FitnessEvaluator
)
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent # Needed for actual FitnessEvaluator

class TestPopulationManager(unittest.TestCase):
    """Test suite for the PopulationManager class."""

    def setUp(self):
        """Set up common mock objects for PopulationManager tests."""
        self.mock_genetic_ops = Mock(spec=GeneticOperators)
        self.mock_fitness_eval = Mock(spec=FitnessEvaluator)
        self.mock_architect_agent = Mock(spec=PromptArchitectAgent)

        # Configure default return values for process_request to avoid errors if not overridden
        self.mock_architect_agent.process_request.return_value = PromptChromosome(genes=["Default gene"])


    # --- Test __init__ ---
    def test_init_successful(self):
        """Test successful instantiation of PopulationManager."""
        manager = PopulationManager(
            genetic_operators=self.mock_genetic_ops,
            fitness_evaluator=self.mock_fitness_eval,
            prompt_architect_agent=self.mock_architect_agent,
            population_size=10,
            elitism_count=1
        )
        self.assertIsInstance(manager, PopulationManager)
        self.assertEqual(manager.population_size, 10)
        self.assertEqual(manager.elitism_count, 1)
        self.assertEqual(manager.generation_number, 0)
        self.assertEqual(len(manager.population), 0)

    def test_init_invalid_types(self):
        """Test __init__ with invalid types for agent/operator arguments."""
        with self.assertRaisesRegex(TypeError, "genetic_operators must be an instance of GeneticOperators."):
            PopulationManager("not_genetic_ops", self.mock_fitness_eval, self.mock_architect_agent)
        with self.assertRaisesRegex(TypeError, "fitness_evaluator must be an instance of FitnessEvaluator."):
            PopulationManager(self.mock_genetic_ops, "not_fitness_eval", self.mock_architect_agent)
        with self.assertRaisesRegex(TypeError, "prompt_architect_agent must be an instance of PromptArchitectAgent."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, "not_architect")

    def test_init_invalid_population_size(self):
        """Test __init__ with invalid population_size."""
        with self.assertRaisesRegex(ValueError, "Population size must be positive."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=0)
        with self.assertRaisesRegex(ValueError, "Population size must be positive."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=-5)

    def test_init_invalid_elitism_count(self):
        """Test __init__ with invalid elitism_count."""
        with self.assertRaisesRegex(ValueError, "Elitism count must be non-negative and not exceed population size."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=10, elitism_count=-1)
        with self.assertRaisesRegex(ValueError, "Elitism count must be non-negative and not exceed population size."):
            PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent, population_size=10, elitism_count=11)

    # --- Test initialize_population ---
    def test_initialize_population(self):
        """Test population initialization."""
        pop_size = 5
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1
        )
        
        # Configure architect to return distinct chromosomes for checking
        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"GeneSet{i}"]) for i in range(pop_size)
        ]
        
        task_desc = "Initial task"
        keywords = ["kw1"]
        constraints = {"max_len": 10}
        
        manager.initialize_population(task_desc, keywords, constraints)

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size)
        
        # Check if process_request was called with correct arguments (check one call)
        expected_request_data = {"task_description": task_desc, "keywords": keywords, "constraints": constraints}
        self.mock_architect_agent.process_request.assert_any_call(expected_request_data)
        
        self.assertEqual(manager.generation_number, 0)
        for i, chromo in enumerate(manager.population):
            self.assertEqual(chromo.genes, [f"GeneSet{i}"])

    def test_initialize_population_with_initial_prompt_str(self):
        """Test population initialization when initial_prompt_str is provided."""
        pop_size = 3
        initial_prompt = "This is a seeded prompt."
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1,
            initial_prompt_str=initial_prompt
        )

        # Architect will be called for pop_size - 1 chromosomes
        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"ArchitectGeneSet{i}"]) for i in range(pop_size - 1)
        ]

        task_desc = "Initial task with seed"
        manager.initialize_population(task_desc)

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size - 1)

        seeded_chromosome_found = False
        architect_chromosomes_found = 0
        for chromo in manager.population:
            if chromo.genes == [initial_prompt]:
                seeded_chromosome_found = True
            elif chromo.genes[0].startswith("ArchitectGeneSet"):
                architect_chromosomes_found +=1

        self.assertTrue(seeded_chromosome_found, "Seeded chromosome not found in population.")
        self.assertEqual(architect_chromosomes_found, pop_size - 1, "Incorrect number of architect-generated chromosomes.")
        self.assertEqual(manager.generation_number, 0)

    def test_initialize_population_with_initial_prompt_str_pop_size_1(self):
        """Test population initialization with initial_prompt_str and population size of 1."""
        pop_size = 1
        initial_prompt = "Only seeded prompt."
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=0, # elitism can be 0 if pop_size is 1
            initial_prompt_str=initial_prompt
        )

        task_desc = "Initial task with seed, pop 1"
        manager.initialize_population(task_desc)

        self.assertEqual(len(manager.population), pop_size)
        self.mock_architect_agent.process_request.assert_not_called() # Architect should not be called

        self.assertTrue(len(manager.population) == 1 and manager.population[0].genes == [initial_prompt])
        self.assertEqual(manager.generation_number, 0)

    def test_initialize_population_without_initial_prompt_str(self):
        """Test population initialization when initial_prompt_str is NOT provided."""
        pop_size = 3
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=1,
            initial_prompt_str=None # Explicitly None
        )

        self.mock_architect_agent.process_request.side_effect = [
            PromptChromosome(genes=[f"ArchitectGeneSet{i}"]) for i in range(pop_size)
        ]

        task_desc = "Initial task no seed"
        manager.initialize_population(task_desc)

        self.assertEqual(len(manager.population), pop_size)
        self.assertEqual(self.mock_architect_agent.process_request.call_count, pop_size)

        for i, chromo in enumerate(manager.population):
            self.assertEqual(chromo.genes, [f"ArchitectGeneSet{i}"])
        self.assertEqual(manager.generation_number, 0)


    # --- Test get_fittest_individual ---
    def test_get_fittest_individual_empty_population(self):
        """Test get_fittest_individual with an empty population."""
        manager = PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent)
        self.assertIsNone(manager.get_fittest_individual())

    def test_get_fittest_individual_populated(self):
        """Test get_fittest_individual with a populated list."""
        manager = PopulationManager(self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent)
        c1 = PromptChromosome(fitness_score=0.5)
        c2 = PromptChromosome(fitness_score=0.9) # Fittest
        c3 = PromptChromosome(fitness_score=0.2)
        manager.population = [c1, c2, c3]
        
        # Manually sort as evolve_population would
        manager.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        
        fittest = manager.get_fittest_individual()
        self.assertEqual(fittest, c2)


    # --- Test evolve_population ---
    def test_evolve_population_flow(self):
        """Test the overall flow of evolve_population."""
        pop_size = 4
        elitism_count = 1
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=pop_size, elitism_count=elitism_count
        )

        # Setup initial population
        initial_chromosomes = [
            PromptChromosome(genes=["P1"], fitness_score=0.1), # Will be updated by mock_fitness_eval
            PromptChromosome(genes=["P2"], fitness_score=0.2),
            PromptChromosome(genes=["P3"], fitness_score=0.3),
            PromptChromosome(genes=["P4"], fitness_score=0.4)
        ]
        manager.population = initial_chromosomes
        manager.generation_number = 0

        # Mock fitness_evaluator.evaluate to assign predictable fitness scores
        # Let's say it assigns fitness based on index for simplicity in checking sort
        def mock_evaluate_side_effect(chromo, task_desc, success_criteria):
            if chromo.genes == ["P1"]: chromo.fitness_score = 0.7
            elif chromo.genes == ["P2"]: chromo.fitness_score = 0.5
            elif chromo.genes == ["P3"]: chromo.fitness_score = 0.9 # Fittest after evaluation
            elif chromo.genes == ["P4"]: chromo.fitness_score = 0.3
            return chromo.fitness_score
        self.mock_fitness_eval.evaluate.side_effect = mock_evaluate_side_effect
        
        # Mock genetic operators
        # Selection: just return one of the parents for simplicity, or specific ones
        self.mock_genetic_ops.selection.side_effect = lambda pop: pop[0] # Always selects the current best after sort
        
        # Crossover: return new distinct chromosomes
        mock_child1 = PromptChromosome(genes=["Child1"])
        mock_child2 = PromptChromosome(genes=["Child2"])
        self.mock_genetic_ops.crossover.return_value = (mock_child1, mock_child2)
        
        # Mutate: return the chromosome passed, potentially modified (or a new mock)
        self.mock_genetic_ops.mutate.side_effect = lambda chromo, *args, **kwargs: PromptChromosome(genes=chromo.genes + ["_mutated"])


        task_desc = "Evolution task"
        manager.evolve_population(task_desc)

        # 1. Test fitness_evaluator.evaluate calls
        self.assertEqual(self.mock_fitness_eval.evaluate.call_count, pop_size)
        for chromo in initial_chromosomes: # Check it was called for each original chromosome
             self.mock_fitness_eval.evaluate.assert_any_call(chromo, task_desc, None)

        # 2. Test elitism (P3 should be carried over as it became fittest)
        self.assertIn(initial_chromosomes[2], manager.population, "Fittest individual (P3) not carried over by elitism.")
        self.assertEqual(manager.population[0].genes, ["P3"]) # P3 should be the first due to sorting and elitism

        # 3. Test offspring generation calls
        # Need pop_size - elitism_count = 4 - 1 = 3 new offspring.
        # Crossover produces 2, so it's called ceil(3/2) = 2 times.
        # Selection is called 2 * (number of crossover calls) = 2 * 2 = 4 times.
        # Mutate is called for each child = 2 * 2 = 4 times.
        num_crossover_calls = (pop_size - elitism_count + 1) // 2 # if 3 needed, 2 calls. if 2 needed, 1 call.
        self.assertEqual(self.mock_genetic_ops.selection.call_count, num_crossover_calls * 2)
        self.assertEqual(self.mock_genetic_ops.crossover.call_count, num_crossover_calls)
        self.assertEqual(self.mock_genetic_ops.mutate.call_count, num_crossover_calls * 2)
        
        # 4. Test generation_number increment
        self.assertEqual(manager.generation_number, 1)

        # 5. Test new population size
        self.assertEqual(len(manager.population), pop_size)

        # 6. Check if new population contains mutated offspring (example)
        # The exact content depends on the mocks, but we expect some mutated children
        found_mutated_child = any("_mutated" in gene for chromo in manager.population for gene in chromo.genes)
        self.assertTrue(found_mutated_child, "Mutated offspring not found in the new population.")

    def test_evolve_population_empty(self):
        """Test evolve_population with an initially empty population."""
        manager = PopulationManager(
            self.mock_genetic_ops, self.mock_fitness_eval, self.mock_architect_agent,
            population_size=5, elitism_count=1
        )
        # manager.population is []
        manager.evolve_population("Test task")
        self.assertEqual(manager.generation_number, 0, "Generation number should not change for empty population.")
        self.assertEqual(len(manager.population), 0, "Population should remain empty.")
        self.mock_fitness_eval.evaluate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
