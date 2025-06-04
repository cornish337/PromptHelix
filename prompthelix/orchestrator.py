from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.engine import (
    GeneticOperators, 
    FitnessEvaluator, 
    PopulationManager, 
    PromptChromosome
)

def main_ga_loop():
    """
    Main orchestration loop for running the PromptHelix Genetic Algorithm.
    
    This function initializes agents, GA components, and runs a basic
    evolutionary process for a set number of generations, printing
    information about the fittest individuals.
    """
    # 1. Instantiate Agents
    # print("Initializing agents...")
    prompt_architect = PromptArchitectAgent()
    results_evaluator = ResultsEvaluatorAgent()
    # print("Agents initialized.")

    # 2. Instantiate GA Components
    # print("Initializing GA components...")
    genetic_ops = GeneticOperators()
    fitness_eval = FitnessEvaluator(results_evaluator_agent=results_evaluator)
    # Using a small population for quick testing
    pop_manager = PopulationManager(
        genetic_operators=genetic_ops, 
        fitness_evaluator=fitness_eval, 
        prompt_architect_agent=prompt_architect, 
        population_size=10, 
        elitism_count=1
    )
    # print("GA components initialized.")

    # 3. Define GA Parameters
    # print("Setting GA parameters...")
    num_generations = 5  # Small number for testing
    initial_task_desc = "Explain the concept of photosynthesis to a 5th grader."
    initial_keywords = ["plants", "sunlight", "energy", "leaves"]
    
    # For this basic run, evaluation criteria are the same for all generations
    evaluation_task_desc = initial_task_desc 
    evaluation_success_criteria = {
        "must_include_keywords": ["sunlight", "energy"], 
        "max_length": 250 # Increased max_length slightly for more meaningful mock outputs
    }
    # print(f"Parameters: Generations={num_generations}, Task='{initial_task_desc}'")

    # 4. Initialize Population
    # print("\n--- Initializing Population ---")
    pop_manager.initialize_population(
        initial_task_description=initial_task_desc, 
        initial_keywords=initial_keywords
    )
    # print(f"Population initialized with {len(pop_manager.population)} individuals.")

    # Evaluate the initial population to see a starting best
    if pop_manager.population:
        # print("Evaluating initial population...")
        for chromo in pop_manager.population:
            fitness_eval.evaluate(chromo, evaluation_task_desc, evaluation_success_criteria)
        pop_manager.population.sort(key=lambda c: c.fitness_score, reverse=True)
        # fittest_initial = pop_manager.get_fittest_individual()
        # if fittest_initial:
            # print("\n--- Best in Initial Population ---")
            # print(f"Fitness: {fittest_initial.fitness_score:.4f}")
            # print(f"Prompt ID: {fittest_initial.id}")
            # print(f"Prompt Genes: {fittest_initial.genes}")
            # print(f"Prompt String Snippet: {fittest_initial.to_prompt_string()[:200]}...")
        # else:
            # print("No individuals in the initial population after evaluation (this shouldn't happen if initialized).")
    else:
        # print("Population is empty after initialization. Cannot proceed.")
        return {"best_prompt": "Population initialization failed", "fitness": 0.0}

    # 5. Evolution Loop
    # print("\n--- Starting Evolution Loop ---")
    for i in range(num_generations):
        # print(f"\n--- Generation {pop_manager.generation_number + 1} of {num_generations} ---")
        pop_manager.evolve_population(
            task_description=evaluation_task_desc, 
            success_criteria=evaluation_success_criteria
        )
        
        # fittest_in_gen = pop_manager.get_fittest_individual()
        # if fittest_in_gen:
            # print(f"Fittest in Generation {pop_manager.generation_number}: Fitness={fittest_in_gen.fitness_score:.4f}")
            # print(f"Prompt (ID: {fittest_in_gen.id}): {fittest_in_gen.to_prompt_string()[:200]}...")
        # else:
            # print("No fittest individual found in this generation.")

    # 6. Return Final Best
    final_fittest = pop_manager.get_fittest_individual()
    if final_fittest:
        # print("\n--- Final Best Prompt after all generations ---")
        # print(str(final_fittest)) # Uses the __str__ method of PromptChromosome
        return {"best_prompt": final_fittest.to_prompt_string(), "fitness": final_fittest.fitness_score}
    else:
        # print("\nNo solution found after all generations.")
        return {"best_prompt": "No solution found", "fitness": 0.0}

if __name__ == "__main__":
    # print("Running PromptHelix Genetic Algorithm Orchestrator...")
    result = main_ga_loop()
    print(f"Best Prompt: {result['best_prompt']}")
    print(f"Fitness: {result['fitness']}")
    # print("\nOrchestration complete.")
