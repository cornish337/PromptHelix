from prompthelix.message_bus import MessageBus
import logging
import time # Added
from prompthelix.database import SessionLocal # Added
from prompthelix.websocket_manager import ConnectionManager

websocket_manager = ConnectionManager()
from prompthelix.agents.architect import PromptArchitectAgent
from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
from prompthelix.agents.style_optimizer import StyleOptimizerAgent
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
from typing import List, Optional, Dict
from prompthelix.enums import ExecutionMode
from prompthelix.utils.config_utils import update_settings # Assuming a utility for deep merging configs
from prompthelix import config as global_config # To access global default settings

logger = logging.getLogger(__name__)

def main_ga_loop(
    task_desc: str,
    keywords: List[str],
    num_generations: int,
    population_size: int,
    elitism_count: int,
    execution_mode: ExecutionMode,
    initial_prompt_str: Optional[str] = None,
    agent_settings_override: Optional[Dict] = None,
    llm_settings_override: Optional[Dict] = None,
    parallel_workers: Optional[int] = None, # New parameter
    return_best: bool = True,
    population_path: Optional[str] = None
):
    """
    Main orchestration loop for running the PromptHelix Genetic Algorithm.

    Args:
        task_desc: Description of the task prompts should solve.
        keywords: Keywords used by the PromptArchitectAgent.
        num_generations: Number of generations to evolve.
        population_size: Desired population size.
        elitism_count: Number of top individuals preserved each generation.
        execution_mode: Whether to run in REAL or TEST mode.
        initial_prompt_str: Optional initial prompt string to seed population.
        agent_settings_override: Optional dictionary to override agent settings.
        llm_settings_override: Optional dictionary to override LLM utility settings.
        return_best: If True, return the best chromosome at the end.
        population_path: Optional path for loading/saving population JSON.
    """
    logger.info("--- main_ga_loop started ---")
    logger.info(f"Task Description: {task_desc}")
    logger.info(f"Keywords: {keywords}")
    logger.info(f"Num Generations: {num_generations}, Population Size: {population_size}, Elitism Count: {elitism_count}")
    logger.info(f"Execution Mode: {execution_mode.name}")
    if initial_prompt_str:
        logger.info(f"Initial Prompt String provided: '{initial_prompt_str[:100]}...'")
    if agent_settings_override:
        logger.info(f"Agent Settings Override provided: {agent_settings_override}")
    if llm_settings_override:
        logger.info(f"LLM Settings Override provided: {llm_settings_override}")
    if parallel_workers is not None:
        logger.info(f"Parallel Workers specified: {parallel_workers}")
    else:
        logger.info("Parallel Workers: Using default (None).")

    # 0. Instantiate Message Bus
    logger.debug("Initializing Message Bus...")
    message_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
    logger.debug("Message Bus initialized.")

    # Handle LLM settings override
    # Create a local copy of LLM settings to use for this run
    current_llm_settings = update_settings(global_config.LLM_UTILS_SETTINGS.copy(), llm_settings_override)
    if llm_settings_override:
        logger.info("LLM settings have been updated with overrides for this session.")
        # logger.debug(f"Effective LLM Settings: {current_llm_settings}")


    # 1. Instantiate Agents, passing the message bus
    # Agent settings will be a combination of global defaults and overrides
    logger.debug("Initializing agents with message bus and potentially overridden settings...")

    def get_agent_config(agent_name: str) -> Dict:
        base_settings = global_config.AGENT_SETTINGS.get(agent_name, {}).copy()
        if agent_settings_override and agent_name in agent_settings_override:
            logger.info(f"Applying override settings for agent: {agent_name}")
            # Ensure deep update if settings are nested
            return update_settings(base_settings, agent_settings_override[agent_name])
        return base_settings

    # Pass relevant part of current_llm_settings if agents need direct LLM config
    # Or, agents should use llm_utils which will now be configured by current_llm_settings (if llm_utils is adapted)
    # For now, assuming agents get their specific config, and llm_utils will handle global llm_config

    prompt_architect_config = get_agent_config("PromptArchitectAgent")
    prompt_architect = PromptArchitectAgent(
        message_bus=message_bus,
        knowledge_file_path=prompt_architect_config.get("knowledge_file_path", "architect_ga_knowledge.json")
    )

    results_evaluator_config = get_agent_config("ResultsEvaluatorAgent")
    results_evaluator = ResultsEvaluatorAgent(
        message_bus=message_bus,
        knowledge_file_path=results_evaluator_config.get("knowledge_file_path", "results_evaluator_ga_config.json")
    )

    style_optimizer_config = get_agent_config("StyleOptimizerAgent")
    style_optimizer = StyleOptimizerAgent(
        message_bus=message_bus,
        knowledge_file_path=style_optimizer_config.get("knowledge_file_path") # May be None
    )
    # TODO: Agents need to be updated to accept and use the 'settings' dict.
    # For now, they might only use knowledge_file_path from it or ignore it if not updated.

    message_bus.register(prompt_architect.agent_id, prompt_architect)
    message_bus.register(results_evaluator.agent_id, results_evaluator)
    message_bus.register(style_optimizer.agent_id, style_optimizer)
    logger.debug("Agents initialized and registered.")

    # 2. Instantiate GA Components
    logger.debug("Initializing GA components...")
    # Pass style_optimizer_config to GeneticOperators if it needs settings
    genetic_ops = GeneticOperators(style_optimizer_agent=style_optimizer) # Add settings if needed

    fitness_eval = FitnessEvaluator(
        results_evaluator_agent=results_evaluator,
        execution_mode=execution_mode
    )
    # TODO: FitnessEvaluator needs to be updated to accept and use llm_settings.

    pop_manager = PopulationManager(
        genetic_operators=genetic_ops,
        fitness_evaluator=fitness_eval,
        prompt_architect_agent=prompt_architect, # Architect is used for initial prompt generation
        population_size=population_size,
        elitism_count=elitism_count,
        parallel_workers=parallel_workers, # Pass the new parameter
        population_path=population_path,
        message_bus=message_bus # Added
        # TODO: Pass agent_settings_override or specific agent configs if PopulationManager
        # is responsible for creating/configuring more agents during its operations.
        # For now, agents are configured above.
    )
    # TODO: PopulationManager needs to be updated to accept and use initial_prompt_str.
    #       Actually, initial_prompt_str is passed to initialize_population and also handled by PopulationManager's __init__
    logger.debug("GA components initialized.")

    # 3. Use GA Parameters (already logged)
    evaluation_task_desc = task_desc
    evaluation_success_criteria = {
        "must_include_keywords": keywords[:2] if keywords else [],
        "max_length": 500 # This could also come from config or overrides
    }
    logger.info(f"Evaluation criteria (example): {evaluation_success_criteria}")


    # 4. Initialize Population
    if not pop_manager.population:
        logger.info("--- Initializing Population ---")
        # PopulationManager will now use initial_prompt_str if provided (once updated)
        pop_manager.initialize_population(
            initial_task_description=task_desc, # This might be redundant if initial_prompt_str is primary
            initial_keywords=keywords
        )
        logger.info(f"Population initialized with {len(pop_manager.population)} individuals.")
    else:
        print(f"Loaded population with {len(pop_manager.population)} individuals.")

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
    logger.info("Starting evolution loop for %d generations", num_generations)
    fittest_individual = None

    if pop_manager.population: # Ensure there's a population to manage
        pop_manager.status = "RUNNING" # Initial status before first generation
        pop_manager.broadcast_ga_update(event_type="ga_run_started")

    for i in range(num_generations):
        # --- Start of new control logic ---
        while pop_manager.is_paused and not pop_manager.should_stop:
            logger.info("GA run is paused. Waiting for resume or stop command...")
            pop_manager.broadcast_ga_update(event_type="ga_status_heartbeat") # Optional: send periodic updates while paused
            time.sleep(1) # Check every second

        if pop_manager.should_stop:
            logger.info("GA run stopping as per request.")
            pop_manager.status = "STOPPED" # Status already set by stop_evolution, but ensure it's reflected
            pop_manager.broadcast_ga_update(event_type="ga_run_stopped") # Broadcast again to confirm orchestrator loop stop
            break # Exit the generation loop
        # --- End of new control logic ---

        current_generation_num = pop_manager.generation_number + 1
        print(f"\n--- Generation {current_generation_num} of {num_generations} ---")
        logger.info("Generation %d start", current_generation_num)

        # evolve_population now checks for should_stop internally as well.
        pop_manager.evolve_population(
            task_description=evaluation_task_desc,  # Use consistent task for evaluation
            success_criteria=evaluation_success_criteria,
            target_style="formal" # Example, ensure this is handled or removed if not dynamic
        )

        # If evolve_population was skipped due to should_stop, this check handles it
        if pop_manager.should_stop and pop_manager.status == "STOPPED": # Status might be set by evolve_population's end or by stop_evolution
            logger.info(f"GA run stopped during generation {current_generation_num} processing.")
            # An event for ga_run_stopped would have been sent by stop_evolution or the check above.
            # If evolve_population itself sets status to STOPPED and broadcasts, this might be redundant.
            # For clarity, let's ensure a final "run_stopped" is sent if loop breaks here.
            # pop_manager.broadcast_ga_update(event_type="ga_run_stopped_in_gen") # Or rely on evolve_population's broadcast
            break


        fittest_in_gen = pop_manager.get_fittest_individual()
        if fittest_in_gen:
            print(f"Fittest in Generation {pop_manager.generation_number}: Fitness={fittest_in_gen.fitness_score:.4f}")
            print(f"Prompt (ID: {fittest_in_gen.id}): {fittest_in_gen.to_prompt_string()[:200]}...")
            fittest_individual = fittest_in_gen
        else:
            print("No fittest individual found in this generation.")
        logger.info(
            "Generation %d end - population size: %d, best fitness: %s",
            pop_manager.generation_number,
            len(pop_manager.population),
            f"{fittest_in_gen.fitness_score:.4f}" if fittest_in_gen else "N/A",
        )

        # If loop completes normally (all generations run for this iteration)
        if i == num_generations - 1 and not pop_manager.should_stop:
             pop_manager.status = "COMPLETED" # Mark as completed if it finished all generations
             pop_manager.broadcast_ga_update(event_type="ga_run_completed")


    # After the loop (if it broke due to stop or completed naturally)
    final_fittest_overall = pop_manager.get_fittest_individual() # Get the best regardless of how loop ended

    # Consolidate final status update logic
    if pop_manager.should_stop and pop_manager.status != "STOPPED":
        # This case might occur if stop_evolution was called but loop exited before status was updated by main loop logic
        pop_manager.status = "STOPPED"
        logger.info("GA run has been externally stopped. Final status set to STOPPED.")
    elif not pop_manager.should_stop and pop_manager.status != "COMPLETED":
        # If not stopped and not already marked COMPLETED (e.g. if num_generations was 0 or loop exited unexpectedly)
        # This could also be an ERROR state if population became empty, etc.
        # For now, if it wasn't explicitly stopped, and didn't complete all gens, assume it finished its course.
        pop_manager.status = "COMPLETED" # Or "UNKNOWN_EXIT_STATUS"
        logger.info(f"GA run finished. Final status set to {pop_manager.status}.")

    pop_manager.broadcast_ga_update(event_type="ga_run_final_status") # Send final status


    if final_fittest_overall:
        print("\n--- Overall Best Prompt Found ---")
        print(str(final_fittest_overall))
        logger.info(
            "GA finished - final population size: %d, best fitness: %.4f, status: %s",
            len(pop_manager.population),
            final_fittest_overall.fitness_score,
            pop_manager.status
        )
    else:
        print("\nNo solution found after all generations.")
        logger.info(
            "GA finished - final population size: %d, no valid solution, status: %s",
            len(pop_manager.population),
            pop_manager.status
        )

    if population_path:
        pop_manager.save_population(population_path)

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

    demo_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
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
    meta_learner_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager) # Using demo_bus for consistency if preferred, or a new one.
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
        parallel_workers=None, # Example: use default for this specific call
        return_best=True
    )
    if best_chromosome:
        print(f"\nExample Run - Best Chromosome Fitness: {best_chromosome.fitness_score}")
        print(f"Example Run - Best Chromosome Prompt: {best_chromosome.to_prompt_string()}")
    else:
        print("\nExample Run - No best chromosome found.")
    print("\nOrchestration complete.")
