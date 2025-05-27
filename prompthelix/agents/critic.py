from prompthelix.agents.base import BaseAgent

class PromptCriticAgent(BaseAgent):
    """Evaluates and critiques prompts."""
    def __init__(self):
        super().__init__(agent_id="PromptCritic")

    def process_request(self, request_data: dict) -> dict:
        """Processes a request to critique a prompt."""
        # print(f"{self.agent_id} processing request: {request_data}")
        raise NotImplementedError(f"Processing logic for {self.agent_id} is not implemented yet.")
