from prompthelix.genetics.engine import PopulationManager

if __name__ == "__main__":
    # Define parameters for the PopulationManager
    population_size = 50
    gene_pool_characters = "abcdefghijklmnopqrstuvwxyz "
    gene_length = 20
    mutation_rate = 0.01
    num_parents_for_selection = 10
    num_generations = 10

    # Instantiate PopulationManager
    manager = PopulationManager(
        population_size=population_size,
        gene_pool_characters=gene_pool_characters,
        gene_length=gene_length,
        mutation_rate=mutation_rate,
        num_parents_for_selection=num_parents_for_selection
    )

    # Initialize the population
    manager.initialize_population()

    # Loop for num_generations
    for i in range(num_generations):
        manager.evolve_population()
        fittest_individual = manager.get_fittest_individual()
        print(f"Generation {i+1}: Best Prompt: '{fittest_individual}', Fitness: {fittest_individual.fitness_score}")

    # Get the overall fittest individual from the final population
    overall_fittest_individual = manager.get_fittest_individual()

    # Print the final results
    print("Finished Genetic Algorithm.")
    print(f"Best prompt found: '{overall_fittest_individual}', Fitness: {overall_fittest_individual.fitness_score}")
