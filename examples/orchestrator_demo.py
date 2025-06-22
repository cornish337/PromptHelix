"""Example demonstrations previously embedded in prompthelix.orchestrator.

This script showcases the MessageBus, persistence for several agents and an
example GA run using ``main_ga_loop``. It is intended for development and
educational purposes and is not required for production use.
"""

from prompthelix.orchestrator import main_ga_loop, SessionLocal
from prompthelix.message_bus import MessageBus
from prompthelix.globals import websocket_manager
from prompthelix.agents.base import BaseAgent
from prompthelix.agents.domain_expert import DomainExpertAgent
from prompthelix.agents.critic import PromptCriticAgent
from prompthelix.agents.meta_learner import MetaLearnerAgent
from prompthelix.enums import ExecutionMode
from prompthelix.genetics.engine import PromptChromosome


class DemoAgent(BaseAgent):
    """Minimal agent used for ping demonstrations."""

    def __init__(self, agent_id: str, message_bus: MessageBus):
        super().__init__(agent_id, message_bus)

    def process_request(self, request_data: dict) -> dict:
        print(f"DemoAgent '{self.agent_id}' process_request called with: {request_data}")
        return {"status": "processed_by_demo_agent", "original_payload": request_data}

    def do_something_and_send_ping(self, target_agent_id: str, data: dict) -> None:
        print(f"DemoAgent '{self.agent_id}' is sending a ping to '{target_agent_id}'.")
        self.send_message(target_agent_id, {"ping_data": data}, "direct_request")


def message_bus_demo() -> None:
    print("\n--- Simple Message Bus Demonstration ---")
    if SessionLocal is not None:
        demo_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
    else:
        print("Warning: SessionLocal not available. Demo MessageBus will not log to the database.")
        demo_bus = MessageBus(connection_manager=websocket_manager)

    agent_x = DemoAgent(agent_id="AgentX", message_bus=demo_bus)
    agent_y = DemoAgent(agent_id="AgentY", message_bus=demo_bus)
    demo_bus.register(agent_x.agent_id, agent_x)
    demo_bus.register(agent_y.agent_id, agent_y)

    ping_payload = {"content": "AgentX checking in"}
    print(f"{agent_x.agent_id} sending 'ping' to {agent_y.agent_id} with payload: {ping_payload}")
    pong_response = agent_x.send_message(recipient_agent_id="AgentY", message_content=ping_payload, message_type="ping")
    print(f"{agent_x.agent_id} received pong response: {pong_response}")

    non_existent_response = agent_y.send_message(recipient_agent_id="AgentZ", message_content={"data": "Test to Z"}, message_type="info_update")
    print(f"{agent_y.agent_id} sending to non-existent AgentZ, response: {non_existent_response}")
    print("--- End of 'Ping' and Message Bus Demonstration ---\n")


def domain_expert_demo(bus: MessageBus) -> None:
    print("\n--- DomainExpertAgent Persistence Demonstration ---")
    dea_file = "domain_expert_orchestrator_demo.json"
    domain_expert_1 = DomainExpertAgent(message_bus=bus, knowledge_file_path=dea_file)
    if not domain_expert_1.knowledge_base:
        print("Warning: DomainExpertAgent knowledge base is empty after init. Loading defaults for demo.")
        domain_expert_1.knowledge_base = domain_expert_1._get_default_knowledge()

    print(f"Initial medical keywords from instance 1: {domain_expert_1.knowledge_base.get('medical', {}).get('keywords', [])}")

    if 'medical' not in domain_expert_1.knowledge_base:
        domain_expert_1.knowledge_base['medical'] = {'keywords': [], 'constraints': [], 'evaluation_tips': [], 'sample_prompt_starters': []}
    if 'keywords' not in domain_expert_1.knowledge_base['medical']:
        domain_expert_1.knowledge_base['medical']['keywords'] = []

    domain_expert_1.knowledge_base['medical']['keywords'].append("orchestrator_added_keyword")
    print(f"Modified medical keywords in instance 1: {domain_expert_1.knowledge_base['medical']['keywords']}")

    domain_expert_1.save_knowledge()
    print(f"Knowledge saved by instance 1 to '{dea_file}'.")

    domain_expert_2 = DomainExpertAgent(message_bus=bus, knowledge_file_path=dea_file)
    print(f"Medical keywords from instance 2 (should include modification): {domain_expert_2.knowledge_base.get('medical', {}).get('keywords', [])}")
    print("--- End of DomainExpertAgent Persistence Demonstration ---\n")


def prompt_critic_demo(bus: MessageBus) -> None:
    print("\n--- PromptCriticAgent Persistence Demonstration ---")
    pca_file = "critic_orchestrator_demo.json"
    critic_agent_1 = PromptCriticAgent(message_bus=bus, knowledge_file_path=pca_file)
    if not critic_agent_1.critique_rules:
        print("Warning: PromptCriticAgent critique rules are empty after init. Loading defaults for demo.")
        critic_agent_1.critique_rules = critic_agent_1._get_default_critique_rules()

    rule_name = "PromptTooShort"
    initial_rule = next((rule for rule in critic_agent_1.critique_rules if rule.get("name") == rule_name), None)
    print(f"Initial '{rule_name}' rule from instance 1: {initial_rule}")

    modified_min_genes = 1
    for rule in critic_agent_1.critique_rules:
        if rule.get("name") == rule_name:
            rule["min_genes"] = modified_min_genes
            break
    else:
        print(f"Could not find or modify rule '{rule_name}' in instance 1. Rules: {critic_agent_1.critique_rules}")

    critic_agent_1.save_knowledge()
    print(f"Knowledge saved by Critic instance 1 to '{pca_file}'.")

    critic_agent_2 = PromptCriticAgent(message_bus=bus, knowledge_file_path=pca_file)
    loaded_rule = next((rule for rule in critic_agent_2.critique_rules if rule.get("name") == rule_name), None)
    print(f"'{rule_name}' rule from instance 2 (should reflect modification): {loaded_rule}")
    print("--- End of PromptCriticAgent Persistence Demonstration ---\n")


def meta_learner_demo() -> None:
    print("\n--- MetaLearnerAgent Persistence Demonstration ---")
    if SessionLocal is not None:
        bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
    else:
        print("Warning: SessionLocal not available. MetaLearner MessageBus will not log to the database.")
        bus = MessageBus(connection_manager=websocket_manager)

    meta_learner = MetaLearnerAgent(message_bus=bus, knowledge_file_path="meta_learner_knowledge_orchestrator_demo.json")
    print(f"Initial knowledge base keys: {list(meta_learner.knowledge_base.keys())}")
    print(f"Initial data log size: {len(meta_learner.data_log)}")

    dummy_eval_data_1 = {"prompt_chromosome": PromptChromosome(genes=["Evaluable prompt 1 gene 1", "Evaluable prompt 1 gene 2"]), "fitness_score": 0.8}
    meta_learner.process_request({"data_type": "evaluation_result", "data": dummy_eval_data_1})

    dummy_critique_data_1 = {"feedback_points": ["Critique: Too verbose.", "Critique: Lacks clarity."]}
    meta_learner.process_request({"data_type": "critique_result", "data": dummy_critique_data_1})

    dummy_eval_data_2 = {"prompt_chromosome": PromptChromosome(genes=["Evaluable prompt 2 gene 1"],), "fitness_score": 0.92}
    meta_learner.process_request({"data_type": "evaluation_result", "data": dummy_eval_data_2})

    print(f"Knowledge base successful features after processing: {meta_learner.knowledge_base.get('successful_prompt_features')}")
    print(f"Knowledge base common critique themes after processing: {meta_learner.knowledge_base.get('common_critique_themes')}")
    print(f"Data log size after processing: {len(meta_learner.data_log)}")

    meta_learner.save_knowledge()
    print("MetaLearnerAgent knowledge explicitly saved.")
    print(f"Knowledge file '{meta_learner.knowledge_file_path}' should now contain the latest data.")
    print("--- End of MetaLearnerAgent Persistence Demonstration ---\n")


def example_ga_run() -> None:
    print("\n--- Example GA Run ---")
    example_task = "Describe quantum entanglement in simple terms."
    example_keywords = ["quantum", "physics", "entanglement", "spooky"]
    example_gens = 3
    example_pop = 5
    example_elitism = 1

    best = main_ga_loop(
        task_desc=example_task,
        keywords=example_keywords,
        num_generations=example_gens,
        population_size=example_pop,
        elitism_count=example_elitism,
        execution_mode=ExecutionMode.TEST,
        parallel_workers=None,
        population_path=None,
        save_frequency_override=None,
        return_best=True,
    )
    if best:
        print(f"Best Chromosome Fitness: {best.fitness_score}")
        print(f"Best Chromosome Prompt: {best.to_prompt_string()}")
    else:
        print("No best chromosome found.")
    print("\nOrchestration complete.")


if __name__ == "__main__":
    message_bus_demo()

    if SessionLocal is not None:
        demo_bus = MessageBus(db_session_factory=SessionLocal, connection_manager=websocket_manager)
    else:
        demo_bus = MessageBus(connection_manager=websocket_manager)

    domain_expert_demo(demo_bus)
    prompt_critic_demo(demo_bus)
    meta_learner_demo()
    example_ga_run()
