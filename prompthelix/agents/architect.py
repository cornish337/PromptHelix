from prompthelix.agents.base import BaseAgent

class PromptArchitectAgent(BaseAgent):
    """Designs initial prompt structures."""
    def __init__(self):
        super().__init__(agent_id="PromptArchitect")

    def process_request(self, request_data: dict) -> dict:
        """Processes a request to design a prompt structure."""
        # print(f"{self.agent_id} processing request: {request_data}")
        raise NotImplementedError(f"Processing logic for {self.agent_id} is not implemented yet.")
