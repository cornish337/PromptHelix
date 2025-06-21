import json
import re
import logging
from prompthelix.agents.base import BaseAgent
# Removed imports: PromptChromosome, call_llm_api, AGENT_SETTINGS, prompt_metrics, asyncio

logger = logging.getLogger(__name__)

# KNOWLEDGE_DIR might be needed if knowledge_file_path is just a filename
# For now, assuming knowledge_file_path will be relative or absolute.
# from prompthelix.config import KNOWLEDGE_DIR # Example if KNOWLEDGE_DIR is used

class PromptCriticAgent(BaseAgent):
    agent_id = "PromptCritic"
    agent_description = "Evaluates and critiques prompts for structure, clarity, and adherence to best practices using a set of predefined rules."

    def __init__(self, message_bus=None, knowledge_file_path="knowledge/best_practices_rules.json"):
        """
        Initializes the PromptCriticAgent.
        Loads critique rules from a JSON file.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            knowledge_file_path (str, optional): Path to the JSON file storing
                critique rules. Defaults to "knowledge/best_practices_rules.json".
        """
        super().__init__(agent_id=self.agent_id, message_bus=message_bus)
        self.logger = logger  # Expose module logger as instance attribute for tests
        self.knowledge_file_path = knowledge_file_path
        self.rules = []  # Renamed from self.critique_rules
        self.critique_rules = self.rules  # Backwards compatibility
        self.load_knowledge()

    def _get_default_critique_rules(self):
        """Returns the default critique rules."""
        # Provide a simple default rule to satisfy tests expecting non-empty defaults.
        return [
            {
                "name": "Default Placeholder Rule",
                "pattern": ".*", # Matches anything, effectively a no-op for critique
                "feedback": "This is a default placeholder critique rule. Consider defining specific rules.",
                "penalty": 0
            }
        ]

    def load_knowledge(self):
        """
        Loads critique rules from the specified JSON file.
        If the file is not found or is invalid, it loads default rules and attempts to save them.
        """
        try:
            effective_path = self.knowledge_file_path
            with open(effective_path, 'r') as f:
                self.rules = json.load(f)
            self.critique_rules = self.rules # Keep backwards compatibility reference
            logger.info(f"Agent '{self.agent_id}': Rules loaded successfully from '{effective_path}'.")
        except FileNotFoundError:
            logger.warning(f"Agent '{self.agent_id}': Knowledge file '{effective_path}' not found. Loading default rules.")
            self.rules = self._get_default_critique_rules()
            self.critique_rules = self.rules
            self.save_knowledge() # Attempt to save defaults
        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{effective_path}': {e}. Loading default rules.",
                exc_info=True,
            )
            self.rules = self._get_default_critique_rules()
            self.critique_rules = self.rules
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}': Failed to load rules from '{effective_path}': {e}. Loading default rules.",
                exc_info=True,
            )
            self.rules = self._get_default_critique_rules()
            self.critique_rules = self.rules

    def save_knowledge(self):
        """
        Saves the current critique rules to the specified JSON file.
        """
        effective_path = self.knowledge_file_path
        try:
            # Create directory if it doesn't exist
            import os
            os.makedirs(os.path.dirname(effective_path), exist_ok=True)

            with open(effective_path, 'w') as f:
                json.dump(self.rules, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Rules saved successfully to '{effective_path}'.")
        except IOError as e:
            logger.error(f"Agent '{self.agent_id}': IOError saving rules to '{effective_path}': {e}", exc_info=True)
        except Exception as e: # Catch any other unexpected errors during save
            logger.error(f"Agent '{self.agent_id}': Unexpected error saving rules to '{effective_path}': {e}", exc_info=True)


    def process_request(self, request_data: dict) -> dict:
        """Handle a direct critique request."""
        prompt = request_data.get("prompt")
        if not isinstance(prompt, str):
            logger.error(
                f"Agent '{self.agent_id}': 'prompt' missing or not a string in request_data."
            )
            return {"score": 0, "feedback": ["Error: Invalid or missing 'prompt'."]}

        return self.process_prompt(prompt)

    def process_prompt(self, prompt: str) -> dict:
        """
        Analyzes a given prompt string based on loaded rules.

        Args:
            prompt (str): The prompt string to analyze.

        Returns:
            dict: A dictionary with 'score' (integer) and 'feedback' (list of strings).
        """
        if not isinstance(prompt, str):
            logger.error(f"Agent '{self.agent_id}': Invalid prompt type. Expected string, got {type(prompt)}.")
            return {"score": 0, "feedback": ["Error: Invalid prompt type. Expected string."]}

        logger.info(f"Agent '{self.agent_id}': Processing prompt: \"{prompt[:100]}...\"")
        
        issues = []
        score = 10  # Starting score

        if not self.rules:
            logger.warning(f"Agent '{self.agent_id}': No rules loaded. Prompt processing will not apply any checks.")
            return {"score": score, "feedback": ["Warning: No rules loaded for critique."]}

        for rule in self.rules:
            pattern = rule.get("pattern")
            if not pattern:
                logger.warning( # Changed to warning as it's a data issue, not critical error
                    f"Agent '{self.agent_id}': Invalid rule structure for rule '{rule.get('name', 'Unnamed')}'. Missing key: 'pattern'. Skipping rule."
                )
                continue
            try:
                regex = re.compile(pattern)
            except re.error as e:
                logger.warning( # Changed to warning
                    f"Agent '{self.agent_id}': Regex error in rule '{rule.get('name', 'Unnamed')}': {e}. Skipping rule."
                )
                continue

            if regex.search(prompt):
                penalty = rule.get("penalty", 1)
                feedback_template = rule.get("feedback") # Renamed for clarity
                if feedback_template is not None:
                    # Simple placeholder replacement, can be expanded
                    feedback_message = feedback_template.replace("{pattern}", pattern)
                    issues.append(feedback_message)
                else:
                    logger.warning( # Changed to warning
                        f"Agent '{self.agent_id}': Invalid rule structure for rule '{rule.get('name', 'Unnamed')}'. Missing key: 'feedback'."
                    )
                score -= penalty


        final_score = max(score, 0)
        
        result = {
            "score": final_score,
            "feedback": issues
        }

        logger.info(f"Agent '{self.agent_id}': Prompt processing complete. Score: {final_score}, Issues: {len(issues)}")
        # Message bus notification can be added here if needed, similar to the old process_request
        # For example:
        # if self.message_bus:
        #     # Decide on message format and topic
        #     self.message_bus.broadcast_message("prompt_critique_result", result, sender_id=self.agent_id)

        return result
