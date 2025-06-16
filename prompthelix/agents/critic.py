from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS
from prompthelix.evaluation import metrics as prompt_metrics  # Import new metrics
import logging
import asyncio

logger = logging.getLogger(__name__)

# Default provider from config if specific agent setting is not found
FALLBACK_LLM_PROVIDER = AGENT_SETTINGS.get("PromptCriticAgent", {}).get("default_llm_provider", "openai")
FALLBACK_LLM_MODEL = AGENT_SETTINGS.get("PromptCriticAgent", {}).get("default_llm_model", "gpt-3.5-turbo")

class PromptCriticAgent(BaseAgent):
    agent_id = "PromptCritic"
    agent_description = "Evaluates and critiques prompts for structure, clarity, and adherence to best practices."
    """
    Evaluates and critiques prompts based on their structure, content,
    and adherence to best practices, without necessarily executing them.
    It acts as a "static analyzer" for prompts.
    """
    def __init__(self, message_bus=None, knowledge_file_path=None):
        """
        Initializes the PromptCriticAgent.
        Loads critique rules or heuristics and agent configuration.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            knowledge_file_path (str, optional): Path to the JSON file storing
                critique rules. Defaults to "critic_rules.json" if not
                provided.
        """
        super().__init__(agent_id=self.agent_id, message_bus=message_bus)

        agent_config = AGENT_SETTINGS.get(self.agent_id, {})
        self.llm_provider = agent_config.get("default_llm_provider", FALLBACK_LLM_PROVIDER)
        self.llm_model = agent_config.get("default_llm_model", FALLBACK_LLM_MODEL)
        logger.info(f"Agent '{self.agent_id}' initialized with LLM provider: {self.llm_provider} and model: {self.llm_model}")

        self.knowledge_file_path = knowledge_file_path or "critic_rules.json"

        self.critique_rules = [] # Initialize before loading
        self.load_knowledge()

    def _get_default_critique_rules(self) -> list:
        """
        Provides default mock critique rules.

        In a real scenario, this would load from a configuration file,
        a database, or be dynamically updated by the MetaLearnerAgent.

        Returns:
            list: A list of critique rule dictionaries.
        """
        logger.info(f"Agent '{self.agent_id}': Using default critique rules.")
        return [
            {"name": "PromptTooShort", "type": "length_check", "min_genes": 3, "message": "Prompt might be too short to be effective (less than 3 gene segments)."},
            {"name": "PromptTooLong", "type": "length_check", "max_genes": 7, "message": "Prompt might be too long and complex (more than 7 gene segments)."},
            {
                "name": "LacksInstruction",
                "type": "gene_keyword_check",
                "gene_keyword_missing": "instruction",
                "message": "Prompt is missing 'instruction' segment. Consider adding one.",
            },
            {"name": "LacksContext", "type": "gene_keyword_check", "gene_keyword_missing": "context", "message": "Prompt may benefit from a 'context' segment. Ensure sufficient background is provided."},
            {
                "name": "UsesNegativePhrasing",
                "type": "keyword_check",
                "keywords": ["don't", "cannot", "avoid", "won't", "not able to"],
                "message": "Avoid negative phrasing in prompts to keep instructions positive and direct.",
            },
        ]

    def load_knowledge(self):
        """
        Loads critique rules from the specified JSON file.
        If the file is not found or is invalid, it loads default rules
        and saves them to a new file.
        """
        try:
            with open(self.knowledge_file_path, 'r') as f:
                self.critique_rules = json.load(f)
            logger.info(f"Agent '{self.agent_id}': Critique rules loaded successfully from '{self.knowledge_file_path}'.")
        except FileNotFoundError:
            logger.warning(f"Agent '{self.agent_id}': Knowledge file '{self.knowledge_file_path}' not found. Using default rules and creating the file.")
            self.critique_rules = self._get_default_critique_rules()
            self.save_knowledge() # Save defaults if file not found
        except json.JSONDecodeError as e:
            logger.error(f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default rules.", exc_info=True)
            self.critique_rules = self._get_default_critique_rules()
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to load critique rules from '{self.knowledge_file_path}': {e}. Using default rules.", exc_info=True)
            self.critique_rules = self._get_default_critique_rules()

    def save_knowledge(self):
        """
        Saves the current critique rules to the specified JSON file.
        """
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(self.critique_rules, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Critique rules saved successfully to '{self.knowledge_file_path}'.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to save critique rules to '{self.knowledge_file_path}': {e}", exc_info=True)

    def _structural_analysis(self, prompt_chromosome: PromptChromosome) -> list:
        """
        Analyzes the overall structure of the prompt chromosome.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings based on structural issues.
        """
        prompt_str = "\n".join(prompt_chromosome.genes)
        llm_prompt = f"""
Analyze the following prompt for structural issues:
---
{prompt_str}
---
Consider the following rules:
- Prompt length: Ideal is 3-7 segments.
- Presence of key segments like 'instruction' or 'context' (though not strictly required, their absence can be a structural point).

Based on this, provide a list of structural feedback points.
If no issues, return "No specific structural issues found."
Each feedback point should start with "Structural Issue:" or "Structural Suggestion:".
Example:
Structural Issue: Prompt might be too short (less than 3 segments).
Structural Suggestion: Consider adding a dedicated 'context' segment.
"""
        try:
            response = call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model)
            feedback = [line.strip() for line in response.split('\n') if line.strip() and (line.startswith("Structural Issue:") or line.startswith("Structural Suggestion:"))]
            if not feedback and "No specific structural issues found." not in response:
                feedback.append(f"LLM Structural Analysis: {response}")
            elif not feedback and "No specific structural issues found." in response:
                 logger.info(f"Agent '{self.agent_id}': LLM found no specific structural issues.")
                 return []
            return feedback
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error in LLM structural analysis: {e}. Falling back to rule-based.", exc_info=True)
            feedback = []
            num_genes = len(prompt_chromosome.genes)
            for rule in self.critique_rules:
                if rule["type"] == "length_check":
                    if "min_genes" in rule and num_genes < rule["min_genes"]:
                        feedback.append(f"Structural Issue ({rule['name']}): {rule['message']}")
                    if "max_genes" in rule and num_genes > rule["max_genes"]:
                        feedback.append(f"Structural Issue ({rule['name']}): {rule['message']}")
            return feedback

    def _clarity_check(self, prompt_chromosome: PromptChromosome) -> list:
        """
        Analyzes prompt clarity using an LLM.
        Falls back to basic checks if LLM fails.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings related to clarity.
        """
        prompt_str = "\n".join(prompt_chromosome.genes)
        llm_prompt = f"""
Analyze the following prompt for clarity and conciseness:
---
{prompt_str}
---
Check for:
- Ambiguous phrasing
- Excessive jargon (unless it's domain-specific and expected)
- Overly complex sentences
- Readability

Provide a list of clarity feedback points.
If no issues, return "Prompt appears clear and concise."
Each feedback point should start with "Clarity Issue:" or "Clarity Suggestion:".
Example:
Clarity Issue: The phrase 'perform the thing' is ambiguous.
Clarity Suggestion: Consider rephrasing for better readability.
"""
        try:
            response = call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model)
            feedback = [line.strip() for line in response.split('\n') if line.strip() and (line.startswith("Clarity Issue:") or line.startswith("Clarity Suggestion:"))]
            if not feedback and "Prompt appears clear and concise." not in response:
                feedback.append(f"LLM Clarity Analysis: {response}")
            elif not feedback and "Prompt appears clear and concise." in response:
                logger.info(f"Agent '{self.agent_id}': LLM found no specific clarity issues.")
                return []
            return feedback
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error in LLM clarity check: {e}. Falling back to basic checks.", exc_info=True)
            feedback = []
            # Example basic check (can be expanded)
            # if any("very unclear phrase" in gene.lower() for gene in prompt_chromosome.genes):
            #    feedback.append("Fallback Clarity Issue: Contains potentially unclear phrasing.")
            return feedback

    def _apply_heuristics(self, prompt_chromosome: PromptChromosome) -> list:
        """
        Applies heuristics to the prompt using an LLM, falling back to rule-based checks.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings based on triggered heuristics.
        """
        prompt_str = "\n".join(prompt_chromosome.genes)
        # Convert self.critique_rules to a string format for the LLM prompt
        rules_str = "\n".join([f"- {rule['name']}: {rule['message']}" for rule in self.critique_rules])

        llm_prompt = f"""
Analyze the following prompt based on a set of heuristics:
---
{prompt_str}
---
Heuristics to consider:
{rules_str}

Additionally, check for common best practices:
- Is there a clear call to action?
- Is the desired output format specified or implied?
- Does the prompt avoid leading questions (unless intended)?
- Does it use positive framing (e.g., "do this" vs "don't do that")?

Provide a list of heuristic feedback points.
If no issues, return "Prompt aligns well with heuristics."
Each feedback point should start with "Heuristic Violation:", "Heuristic Suggestion:", or "Best Practice Note:".
Example:
Heuristic Violation (LacksInstruction): Prompt is missing 'instruction' segment.
Best Practice Note: Consider specifying the desired output format.
"""
        try:
            response = call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model)
            feedback = [line.strip() for line in response.split('\n') if line.strip() and
                        (line.startswith("Heuristic Violation:") or
                         line.startswith("Heuristic Suggestion:") or
                         line.startswith("Best Practice Note:"))]
            if not feedback and "Prompt aligns well with heuristics." not in response:
                feedback.append(f"LLM Heuristic Analysis: {response}")
            elif not feedback and "Prompt aligns well with heuristics." in response:
                logger.info(f"Agent '{self.agent_id}': LLM found no specific heuristic violations.")
                return []
            return feedback
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error in LLM heuristic application: {e}. Falling back to rule-based.", exc_info=True)
            feedback = []
            prompt_content_lower = " ".join(str(gene) for gene in prompt_chromosome.genes).lower()
            for rule in self.critique_rules:
                if rule["type"] == "keyword_check":
                    if any(keyword in prompt_content_lower for keyword in rule["keywords"]):
                        feedback.append(f"Heuristic Violation ({rule['name']}): {rule['message']}")
                elif rule["type"] == "gene_keyword_check":
                    keyword_to_check = rule.get("gene_keyword_missing", "").lower()
                    if keyword_to_check:
                        found_keyword_in_genes = False
                        for gene_segment in prompt_chromosome.genes:
                            if keyword_to_check in str(gene_segment).lower():
                                found_keyword_in_genes = True
                                break
                        if not found_keyword_in_genes:
                             feedback.append(f"Heuristic Suggestion ({rule['name']}): {rule['message']}")
            return feedback

    def process_request(self, request_data: dict) -> dict:
        """
        Analyzes and critiques a given prompt chromosome.

        Args:
            request_data (dict): Expected to contain 'prompt_chromosome', 
                                 an instance of PromptChromosome.
                                 Example:
                                 {
                                     "prompt_chromosome": PromptChromosome(genes=["Instruct: ...", "Context: ..."])
                                 }
        Returns:
            dict: A dictionary with 'critique_score' (float from 0.0 to 1.0, higher is better),
                  'feedback_points' (list of strings), and 'suggestions' (list of strings).
        """
        prompt_chromosome = request_data.get("prompt_chromosome")

        if not isinstance(prompt_chromosome, PromptChromosome):
            logger.error(f"Agent '{self.agent_id}': Invalid or missing 'prompt_chromosome' in request_data.")
            return {
                "critique_score": 0.0, 
                "feedback_points": ["Error: Invalid or missing 'prompt_chromosome' object provided."],
                "suggestions": ["Ensure a valid PromptChromosome object is passed under the 'prompt_chromosome' key."]
            }

        logger.info(f"Agent '{self.agent_id}': Critiquing prompt: {str(prompt_chromosome)}")
        
        feedback_points = []
        programmatic_metric_details = {} # To store scores from new metrics

        # --- LLM/Rule-based Analysis (existing) ---
        # These methods already provide qualitative feedback based on LLM or rules
        llm_structural_feedback = self._structural_analysis(prompt_chromosome)
        feedback_points.extend(llm_structural_feedback)

        llm_clarity_feedback = self._clarity_check(prompt_chromosome)
        feedback_points.extend(llm_clarity_feedback)

        llm_heuristic_feedback = self._apply_heuristics(prompt_chromosome)
        feedback_points.extend(llm_heuristic_feedback)

        issues_found_by_llm_rules = len(feedback_points)

        # --- Programmatic Prompt Quality Metrics ---
        prompt_text_for_metrics = prompt_chromosome.to_prompt_string(separator=" ") # Use space for better linguistic analysis by textstat

        clarity_score = prompt_metrics.calculate_clarity_score(prompt_text_for_metrics)
        programmatic_metric_details["clarity_score"] = clarity_score
        if clarity_score < 0.5: # Threshold for adding feedback
            feedback_points.append(f"Programmatic Metric - Low Clarity: Score {clarity_score:.2f}. Consider simplifying language or structure.")

        # Example required elements for completeness - this could be dynamic later
        # For now, using the default from metrics.py: ["Instruction:", "[context]", "Output format:"]
        completeness_score = prompt_metrics.calculate_completeness_score(prompt_text_for_metrics)
        programmatic_metric_details["completeness_score"] = completeness_score
        if completeness_score < 0.67: # e.g., less than 2 out of 3 default elements found
            feedback_points.append(f"Programmatic Metric - Low Completeness: Score {completeness_score:.2f}. Ensure key components like 'Instruction:', '[context]', and 'Output format:' are present.")

        specificity_score = prompt_metrics.calculate_specificity_score(prompt_text_for_metrics)
        programmatic_metric_details["specificity_score"] = specificity_score
        if specificity_score < 0.5:
            feedback_points.append(f"Programmatic Metric - Low Specificity: Score {specificity_score:.2f}. Prompt may be too generic or use too many placeholders.")

        length_score = prompt_metrics.calculate_prompt_length_score(prompt_text_for_metrics)
        programmatic_metric_details["prompt_length_score"] = length_score
        if length_score < 0.75 and length_score > 0.0: # If not 0 (out of abs range) but not optimal
            feedback_points.append(f"Programmatic Metric - Suboptimal Length: Score {length_score:.2f}. Consider adjusting prompt length for optimal performance.")
        elif length_score == 0.0:
             feedback_points.append(f"Programmatic Metric - Extreme Length: Score {length_score:.2f}. Prompt is too short or too long.")
        
        logger.info(f"Agent '{self.agent_id}': Programmatic metric scores - {programmatic_metric_details}")

        # --- Calculate Final Critique Score ---
        # Weighting: 50% for LLM/rule-based qualitative feedback, 50% for programmatic quantitative metrics

        # Score from LLM/rule-based feedback (penalty based)
        qualitative_score_component = max(0.0, 1.0 - (issues_found_by_llm_rules * 0.15)) # Slightly higher penalty per issue

        # Score from programmatic metrics (average of the individual scores)
        programmatic_scores = [clarity_score, completeness_score, specificity_score, length_score]
        quantitative_score_component = sum(programmatic_scores) / len(programmatic_scores) if programmatic_scores else 0.0

        # Combine scores (e.g., 50/50 weighting)
        # This weighting can be moved to config later if needed
        critique_score = (qualitative_score_component * 0.5) + (quantitative_score_component * 0.5)
        critique_score = round(max(0.0, min(1.0, critique_score)), 3)


        # --- Suggestions ---
        suggestions = []
        if critique_score < 0.7: # Overall low score
            suggestions.append("Overall prompt quality appears to have areas for improvement. Review feedback points.")
        if issues_found_by_llm_rules > 0 :
             suggestions.append("Address qualitative feedback from LLM/rule-based analysis.")
        if quantitative_score_component < 0.6: # Average programmatic score is low
            suggestions.append("Review programmatic metric scores (clarity, completeness, specificity, length) for specific improvement areas.")

        if not suggestions:
            suggestions.append("Prompt appears generally well-structured based on current analysis. Consider test execution for dynamic evaluation.")

        result = {
            "critique_score": critique_score,
            "feedback_points": list(set(feedback_points)), # Remove duplicate feedback if any
            "suggestions": suggestions,
            "metric_details": programmatic_metric_details # Include the new scores for transparency
        }
        logger.info(f"Agent '{self.agent_id}': Critique result - Score={critique_score}, Total Feedback Points#={len(feedback_points)}, Programmatic Scores={programmatic_metric_details}")
        if self.message_bus:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.message_bus.broadcast_message("critique_result", result, sender_id=self.agent_id))
            except RuntimeError:
                asyncio.run(self.message_bus.broadcast_message("critique_result", result, sender_id=self.agent_id))
        return result

