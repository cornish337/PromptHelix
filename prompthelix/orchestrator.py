from prompthelix.message_bus import MessageBus
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.agents.domain_expert import DomainExpertAgent # Added for demo
from prompthelix.agents.critic import PromptCriticAgent # Added for demo
# Import BaseAgent if we need to use it for the example, or a mock agent
from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import (
    GeneticOperators,
    FitnessEvaluator,
    PopulationManager,
    PromptChromosome,
)
from typing import List
from prompthelix.enums import ExecutionMode # Added import

def main_ga_loop(
    task_desc: str,
    keywords: List[str],
    num_generations: int,
    population_size: int,
    elitism_count: int,
    execution_mode: ExecutionMode, # New parameter
    return_best: bool = True
):
    """
    Main orchestration loop for running the PromptHelix Genetic Algorithm.
    
    This function initializes agents, GA components, and runs the
    evolutionary process based on the provided parameters.
    """
    # 0. Instantiate Message Bus
    print("Initializing Message Bus...")
    message_bus = MessageBus()
    print("Message Bus initialized.")

    # 1. Instantiate Agents, passing the message bus
    print("Initializing agents with message bus...")
    prompt_architect = PromptArchitectAgent(message_bus=message_bus, knowledge_file_path="architect_ga_knowledge.json")
    results_evaluator = ResultsEvaluatorAgent(message_bus=message_bus, knowledge_file_path="results_evaluator_ga_config.json")
    # Note: PromptCriticAgent and StyleOptimizerAgent are typically used within FitnessEvaluator or other
    # GA components. If they were instantiated directly here for the GA loop, they'd also get specific paths.

    # Register agents with the message bus
    message_bus.register(prompt_architect.agent_id, prompt_architect)
    message_bus.register(results_evaluator.agent_id, results_evaluator)
    print("Agents initialized and registered.")

    # 2. Instantiate GA Components
    print("Initializing GA components...")
    genetic_ops = GeneticOperators()

    # Ensure OPENAI_API_KEY (and other necessary keys for LLMs like ANTHROPIC_API_KEY, GOOGLE_API_KEY if used)
    # are set in the environment for the FitnessEvaluator to function correctly with actual LLM calls.
    # FitnessEvaluator handles its own OpenAI client initialization using settings from config.py.
    fitness_eval = FitnessEvaluator(
        results_evaluator_agent=results_evaluator,
        execution_mode=execution_mode  # New argument
    )

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

    # --- Message Bus Demonstration ---
    print("\n--- Simple Message Bus Demonstration ---")
    # Create a mock agent for demonstration if existing agents are too complex for a simple ping
    class DemoAgent(BaseAgent): # Inherit from BaseAgent to get send_message and receive_message
        def __init__(self, agent_id, message_bus):
            super().__init__(agent_id, message_bus)

        def process_request(self, request_data: dict) -> dict:
            # Simple echo for demo
            print(f"DemoAgent '{self.agent_id}' process_request called with: {request_data}")
            return {"status": "processed_by_demo_agent", "original_payload": request_data}

        def do_something_and_send_ping(self, target_agent_id, data):
            print(f"DemoAgent '{self.agent_id}' is sending a ping to '{target_agent_id}'.")
            self.send_message(target_agent_id, {"ping_data": data}, "direct_request") # Use direct_request for demo

    demo_bus = MessageBus()
    agent_X = DemoAgent(agent_id="AgentX", message_bus=demo_bus)
    agent_Y = DemoAgent(agent_id="AgentY", message_bus=demo_bus)
    demo_bus.register(agent_X.agent_id, agent_X)
    demo_bus.register(agent_Y.agent_id, agent_Y)

    # --- "Ping" Message Demonstration (using updated send_message) ---
    print(f"\n--- 'Ping' Message Demonstration ---")
    ping_payload = {"content": "AgentX checking in"}
    print(f"{agent_X.agent_id} sending 'ping' to {agent_Y.agent_id} with payload: {ping_payload}")
    pong_response = agent_X.send_message(recipient_agent_id="AgentY", message_content=ping_payload, message_type="ping")
    print(f"{agent_X.agent_id} received pong response: {pong_response}")

    # Agent Y sends a message to a non-existent agent (will be logged by message bus)
    non_existent_response = agent_Y.send_message(recipient_agent_id="AgentZ", message_content={"data": "Test to Z"}, message_type="info_update")
    print(f"{agent_Y.agent_id} sending to non-existent AgentZ, response: {non_existent_response}")
    print("--- End of 'Ping' and Message Bus Demonstration ---\n")


    # --- DomainExpertAgent Persistence Demonstration ---
    print("\n--- DomainExpertAgent Persistence Demonstration ---")
    dea_knowledge_file = "domain_expert_orchestrator_demo.json"
    # Initial instantiation (might create the file with defaults)
    domain_expert_1 = DomainExpertAgent(message_bus=demo_bus, knowledge_file_path=dea_knowledge_file)
    # Ensure knowledge base is populated, especially if file didn't exist
    if not domain_expert_1.knowledge_base:
        print("Warning: DomainExpertAgent knowledge base is empty after init. Loading defaults for demo.")
        domain_expert_1.knowledge_base = domain_expert_1._get_default_knowledge()


    print(f"Initial medical keywords from instance 1: {domain_expert_1.knowledge_base.get('medical', {}).get('keywords', [])}")

    # Modify knowledge
    if 'medical' not in domain_expert_1.knowledge_base:
        domain_expert_1.knowledge_base['medical'] = {'keywords': [], 'constraints': [], 'evaluation_tips': [], 'sample_prompt_starters': []}
    if 'keywords' not in domain_expert_1.knowledge_base['medical']:
        domain_expert_1.knowledge_base['medical']['keywords'] = []

    domain_expert_1.knowledge_base['medical']['keywords'].append("orchestrator_added_keyword")
    print(f"Modified medical keywords in instance 1: {domain_expert_1.knowledge_base['medical']['keywords']}")

    domain_expert_1.save_knowledge()
    print(f"Knowledge saved by instance 1 to '{dea_knowledge_file}'.")

    # New instance loading from the same file
    domain_expert_2 = DomainExpertAgent(message_bus=demo_bus, knowledge_file_path=dea_knowledge_file)
    print(f"Medical keywords from instance 2 (should include modification): {domain_expert_2.knowledge_base.get('medical', {}).get('keywords', [])}")
    print("--- End of DomainExpertAgent Persistence Demonstration ---\n")

    # --- PromptCriticAgent Persistence Demonstration ---
    print("\n--- PromptCriticAgent Persistence Demonstration ---")
    pca_knowledge_file = "critic_orchestrator_demo.json"
    # Initial instantiation
    critic_agent_1 = PromptCriticAgent(message_bus=demo_bus, knowledge_file_path=pca_knowledge_file)
    if not critic_agent_1.critique_rules:
        print("Warning: PromptCriticAgent critique rules are empty after init. Loading defaults for demo.")
        critic_agent_1.critique_rules = critic_agent_1._get_default_critique_rules()

    rule_name_to_check = "PromptTooShort"
    initial_rule = next((rule for rule in critic_agent_1.critique_rules if rule.get("name") == rule_name_to_check), None)
    print(f"Initial '{rule_name_to_check}' rule from instance 1: {initial_rule}")

    # Modify a rule
    modified_min_genes = 1 # Change from default 3
    rule_modified = False
    if critic_agent_1.critique_rules:
        for rule in critic_agent_1.critique_rules:
            if rule.get("name") == rule_name_to_check:
                rule["min_genes"] = modified_min_genes
                rule_modified = True
                break
    if rule_modified:
        print(f"Modified '{rule_name_to_check}' rule in instance 1 to have min_genes: {modified_min_genes}")
    else:
        print(f"Could not find or modify rule '{rule_name_to_check}' in instance 1. Rules: {critic_agent_1.critique_rules}")

    critic_agent_1.save_knowledge()
    print(f"Knowledge saved by Critic instance 1 to '{pca_knowledge_file}'.")

    # New instance loading from the same file
    critic_agent_2 = PromptCriticAgent(message_bus=demo_bus, knowledge_file_path=pca_knowledge_file)
    loaded_rule = next((rule for rule in critic_agent_2.critique_rules if rule.get("name") == rule_name_to_check), None)
    print(f"'{rule_name_to_check}' rule from instance 2 (should reflect modification): {loaded_rule}")
    print("--- End of PromptCriticAgent Persistence Demonstration ---\n")


    # --- MetaLearnerAgent Persistence Demonstration ---
    print("\n--- MetaLearnerAgent Persistence Demonstration ---")
    # MetaLearnerAgent needs a message bus to be instantiated, even if not used in this simple demo part
    meta_learner_bus = MessageBus() # Using demo_bus for consistency if preferred, or a new one.
    meta_learner_agent = MetaLearnerAgent(message_bus=meta_learner_bus, knowledge_file_path="meta_learner_knowledge_orchestrator_demo.json")

    print(f"Initial knowledge base keys: {list(meta_learner_agent.knowledge_base.keys())}")
    print(f"Initial data log size: {len(meta_learner_agent.data_log)}")

    # Simulate some data processing
    # This data is simplified and doesn't fully match real agent outputs, for demo purposes.
    # In a real scenario, these would be actual evaluation/critique results.
    dummy_eval_data_1 = {
        "prompt_chromosome": PromptChromosome(genes=["Evaluable prompt 1 gene 1", "Evaluable prompt 1 gene 2"]), # Dummy chromosome
        "fitness_score": 0.8
    }
    meta_learner_agent.process_request({"data_type": "evaluation_result", "data": dummy_eval_data_1})

    dummy_critique_data_1 = {
        "feedback_points": ["Critique: Too verbose.", "Critique: Lacks clarity."]
    }
    meta_learner_agent.process_request({"data_type": "critique_result", "data": dummy_critique_data_1})

    # Simulate another round of processing
    dummy_eval_data_2 = {
        "prompt_chromosome": PromptChromosome(genes=["Evaluable prompt 2 gene 1"]),
        "fitness_score": 0.92
    }
    # This call should trigger an internal save as data_log length will be 3
    meta_learner_agent.process_request({"data_type": "evaluation_result", "data": dummy_eval_data_2})

    print(f"Knowledge base successful features after processing: {meta_learner_agent.knowledge_base.get('successful_prompt_features')}")
    print(f"Knowledge base common critique themes after processing: {meta_learner_agent.knowledge_base.get('common_critique_themes')}")
    print(f"Data log size after processing: {len(meta_learner_agent.data_log)}")

    # Explicitly call save_knowledge (though it might have been called internally already)
    meta_learner_agent.save_knowledge()
    print("MetaLearnerAgent knowledge explicitly saved.")
    print(f"Knowledge file '{meta_learner_agent.knowledge_file_path}' should now contain the latest data.")
    print("--- End of MetaLearnerAgent Persistence Demonstration ---\n")

    # Example parameters for a direct run of the GA loop
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
        execution_mode=ExecutionMode.TEST, # Added for the example call
        return_best=True
    )
    if best_chromosome:
        print(f"\nExample Run - Best Chromosome Fitness: {best_chromosome.fitness_score}")
        print(f"Example Run - Best Chromosome Prompt: {best_chromosome.to_prompt_string()}")
    else:
        print("\nExample Run - No best chromosome found.")
    print("\nOrchestration complete.")
