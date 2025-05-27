from prompthelix.agents.base import BaseAgent

class PromptCriticAgent(BaseAgent):
    """Evaluates and critiques prompts."""
    def __init__(self):
        super().__init__(agent_id="PromptCritic")
        self.recommendations = [
            "Is the prompt too leading? Does it overly bias the expected answer?",
            "Check for ambiguity: Are there phrases or terms that could be interpreted in multiple ways?",
            "Evaluate clarity: Is the language concise and easy to understand?",
            "Assess specificity: Is the prompt detailed enough to get the desired output, or is it too vague?",
            "Consider if the prompt requests information it hasn't provided sufficient context for.",
        ]

    def process_request(self, request_data: dict) -> dict:
        """Processes a request to critique a prompt."""
        print(f"{self.agent_id} processing request: {request_data}")
        # Actual implementation will follow
        return {"recommendations": self.recommendations}
