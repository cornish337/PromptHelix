from prompthelix.agents.base import BaseAgent

class DomainExpertAgent(BaseAgent):
    """Provides domain-specific knowledge and constraints."""
    def __init__(self):
        super().__init__(agent_id="DomainExpert")
        self.recommendations = [
            "Ensure the prompt uses terminology accurately according to this domain.",
            "Are there any implicit assumptions in the prompt that might be incorrect for this specific domain?",
            "Consider common pitfalls or misconceptions in this domain that the prompt should avoid triggering.",
            "What are the key entities and relationships in this domain that the prompt should be aware of?",
            "Suggest relevant data sources or knowledge bases that could inform the prompt.",
        ]

    def process_request(self, request_data: dict) -> dict:
        """Processes a request for domain expertise."""

        # print(f"{self.agent_id} processing request: {request_data}")
        raise NotImplementedError(f"Processing logic for {self.agent_id} is not implemented yet.")