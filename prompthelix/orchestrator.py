from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.genetics.engine import (
    GeneticOperators,
    FitnessEvaluator,
    PopulationManager,
    PromptChromosome,
)
from typing import List

def main_ga_loop(
    task_desc: str,
    keywords: List[str],
    num_generations: int,
    population_size: int,
    elitism_count: int,
    return_best: bool = True
):
    """
    Main orchestration loop for running the PromptHelix Genetic Algorithm.
    
    This function initializes agents, GA components, and runs the
    evolutionary process based on the provided parameters.
    """
    # 1. Instantiate Agents
    print("Initializing agents...")
    prompt_architect = PromptArchitectAgent()
    results_evaluator = ResultsEvaluatorAgent()
    print("Agents initialized.")

    # 2. Instantiate GA Components
    print("Initializing GA components...")
    genetic_ops = GeneticOperators()
    fitness_eval = FitnessEvaluator(results_evaluator_agent=results_evaluator)
    pop_manager = PopulationManager(
        genetic_operators=genetic_ops, 
        fitness_evaluator=fitness_eval, 
        prompt_architect_agent=prompt_architect, 
        population_size=population_size,
        elitism_count=elitism_count
    )
    print("GA components initialized.")

    # 3. Use GA Parameters from function arguments
    print("Using GA parameters from input...")
    # For this basic run, evaluation criteria are derived from the primary task description and keywords
    # A more sophisticated approach might have separate evaluation criteria or allow them to be passed in.
    evaluation_task_desc = task_desc
    # Simple success criteria: ensure some of the provided keywords are present.
    # This is a placeholder; real evaluation would be more complex.
    evaluation_success_criteria = {
        "must_include_keywords": keywords[:2] if keywords else [], # e.g., use first two keywords
        "max_length": 500 # Arbitrary max length, could also be a parameter
    }
    print(f"Parameters: Generations={num_generations}, Population Size={population_size}, Elitism Count={elitism_count}, Task='{task_desc}'")

    # 4. Initialize Population
    print("\n--- Initializing Population ---")
    pop_manager.initialize_population(
        initial_task_description=task_desc,
        initial_keywords=keywords
    )
    print(f"Population initialized with {len(pop_manager.population)} individuals.")

    # Evaluate the initial population to see a starting best
    # (evolve_population also does this, but this shows the state *before* any evolution)
    if pop_manager.population:
        print("Evaluating initial population...")
        for chromo in pop_manager.population:
            # Ensure fitness is evaluated based on the primary task and criteria
            fitness_eval.evaluate(chromo, evaluation_task_desc, evaluation_success_criteria)
        pop_manager.population.sort(key=lambda c: c.fitness_score, reverse=True)
        fittest_initial = pop_manager.get_fittest_individual()
        if fittest_initial:
            print("\n--- Best in Initial Population ---")
            print(f"Fitness: {fittest_initial.fitness_score:.4f}")
            print(f"Prompt ID: {fittest_initial.id}")
            # print(f"Prompt Genes: {fittest_initial.genes}") # Genes can be verbose
            print(f"Prompt String Snippet: {fittest_initial.to_prompt_string()[:200]}...")
        else:
            print("No individuals in the initial population after evaluation (this shouldn't happen if initialized).")
    else:
        print("Population is empty after initialization. Cannot proceed.")
        if return_best:
            return None # Return None if population is empty
        else:
            return

    # 5. Evolution Loop
    print("\n--- Starting Evolution Loop ---")
    fittest_individual = None
    for i in range(num_generations):
        current_generation_num = pop_manager.generation_number + 1
        print(f"\n--- Generation {current_generation_num} of {num_generations} ---")
        pop_manager.evolve_population(
            task_description=evaluation_task_desc, # Use consistent task for evaluation
            success_criteria=evaluation_success_criteria
        )
        
        fittest_in_gen = pop_manager.get_fittest_individual()
        if fittest_in_gen:
            print(f"Fittest in Generation {pop_manager.generation_number}: Fitness={fittest_in_gen.fitness_score:.4f}") # pop_manager.generation_number is updated by evolve_population
            print(f"Prompt (ID: {fittest_in_gen.id}): {fittest_in_gen.to_prompt_string()[:200]}...")
            fittest_individual = fittest_in_gen # Keep track of the fittest from the last successful generation
        else:
            print("No fittest individual found in this generation.")
            # If population collapses or no valid individuals, might stop or handle differently

    # 6. Final Best
    # The fittest_individual variable now holds the best from the last generation,
    # or the best from a previous one if the population degraded.
    # PopulationManager.get_fittest_individual() gets the current best.
    final_fittest_overall = pop_manager.get_fittest_individual()

    if final_fittest_overall:
        print("\n--- Overall Best Prompt Found ---")
        print(str(final_fittest_overall))
    else:
        print("\nNo solution found after all generations.")

    if return_best:
        return final_fittest_overall


if __name__ == "__main__":
    print("Running PromptHelix Genetic Algorithm Orchestrator (example run)...")
    # Example parameters for a direct run
    example_task = "Describe quantum entanglement in simple terms."
    example_keywords = ["quantum", "physics", "entanglement", "spooky"]
    example_gens = 3 # Keep it short for example
    example_pop = 5  # Small pop for example
    example_elitism = 1

    best_chromosome = main_ga_loop(
        task_desc=example_task,
        keywords=example_keywords,
        num_generations=example_gens,
        population_size=example_pop,
        elitism_count=example_elitism,
        return_best=True
    )
    if best_chromosome:
        print(f"\nExample Run - Best Chromosome Fitness: {best_chromosome.fitness_score}")
        print(f"Example Run - Best Chromosome Prompt: {best_chromosome.to_prompt_string()}")
    else:
        print("\nExample Run - No best chromosome found.")
    print("\nOrchestration complete.")
