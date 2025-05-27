import random
import string

class PromptChromosome:
    def __init__(self, genes: str):
        self.genes = genes
        self.fitness_score = 0

    def __str__(self) -> str:
        return self.genes

    def calculate_fitness(self) -> int:
        self.fitness_score = len(self.genes)
        return self.fitness_score

class GeneticOperators:
    def selection(self, population: list[PromptChromosome], num_parents: int) -> list[PromptChromosome]:
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        return population[:num_parents]

    def crossover(self, parent1: PromptChromosome, parent2: PromptChromosome) -> PromptChromosome:
        crossover_point = random.randint(0, len(parent1.genes))
        child_genes = parent1.genes[:crossover_point] + parent2.genes[crossover_point:]
        return PromptChromosome(child_genes)

    def mutate(self, chromosome: PromptChromosome, mutation_rate: float) -> PromptChromosome:
        mutated_genes = list(chromosome.genes)
        for i in range(len(mutated_genes)):
            if random.random() < mutation_rate:
                mutated_genes[i] = random.choice(string.ascii_lowercase)
        chromosome.genes = "".join(mutated_genes)
        return chromosome

class FitnessEvaluator:
    def evaluate(self, chromosome: PromptChromosome) -> float:
        score = float(len(chromosome.genes)) # Start with length
        
        # Bonus for rare characters
        if 'x' in chromosome.genes.lower():
            score += 5
        if 'q' in chromosome.genes.lower():
            score += 5
        if 'z' in chromosome.genes.lower():
            score += 5
            
        # Bonus for keywords
        if "helix" in chromosome.genes.lower():
            score += 10
        if "prompt" in chromosome.genes.lower():
            score += 8
            
        chromosome.fitness_score = score # Update chromosome's score
        return score

class PopulationManager:
    def __init__(self, population_size: int, gene_pool_characters: str, gene_length: int, mutation_rate: float, num_parents_for_selection: int, fitness_evaluator: FitnessEvaluator = None):
        self.population_size = population_size
        self.gene_pool_characters = gene_pool_characters
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.num_parents_for_selection = num_parents_for_selection
        self.fitness_evaluator = fitness_evaluator
        self.population: list[PromptChromosome] = []
        self.generation_number = 0
        self.operators = GeneticOperators()

    def initialize_population(self):
        for _ in range(self.population_size):
            genes = "".join(random.choice(self.gene_pool_characters) for _ in range(self.gene_length))
            chromosome = PromptChromosome(genes)
            if self.fitness_evaluator:
                self.fitness_evaluator.evaluate(chromosome)
            else:
                chromosome.calculate_fitness()
            self.population.append(chromosome)

    def evolve_population(self):
        parents = self.operators.selection(self.population, self.num_parents_for_selection)
        next_population: list[PromptChromosome] = []
        while len(next_population) < self.population_size:
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            offspring = self.operators.crossover(parent1, parent2)
            offspring = self.operators.mutate(offspring, self.mutation_rate)
            if self.fitness_evaluator:
                self.fitness_evaluator.evaluate(offspring)
            else:
                offspring.calculate_fitness() # Calculate fitness for the new offspring
            next_population.append(offspring)
        self.population = next_population
        self.generation_number += 1

    def get_fittest_individual(self) -> PromptChromosome:
        self.population.sort(key=lambda x: x.fitness_score, reverse=True)
        return self.population[0]
