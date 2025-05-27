from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome # Assuming PromptChromosome is in this path

class PromptCriticAgent(BaseAgent):
    """
    Evaluates and critiques prompts based on their structure, content,
    and adherence to best practices, without necessarily executing them.
    It acts as a "static analyzer" for prompts.
    """
    def __init__(self):
        """
        Initializes the PromptCriticAgent.
        Loads critique rules or heuristics.
        """
        super().__init__(agent_id="PromptCritic")

        self.critique_rules = self._load_critique_rules()

    def _load_critique_rules(self) -> list:
        """
        Loads mock critique rules.

        In a real scenario, this would load from a configuration file,
        a database, or be dynamically updated by the MetaLearnerAgent.

        Returns:
            list: A list of critique rule dictionaries.
        """
        return [
            {"name": "PromptTooShort", "type": "length_check", "min_genes": 3, "message": "Prompt might be too short to be effective (less than 3 gene segments)."},
            {"name": "PromptTooLong", "type": "length_check", "max_genes": 7, "message": "Prompt might be too long and complex (more than 7 gene segments)."},
            {"name": "LacksInstruction", "type": "gene_keyword_check", "gene_keyword_missing": "instruction", "message": "Prompt should ideally have a clear 'instruction' segment. Consider adding one."},
            {"name": "LacksContext", "type": "gene_keyword_check", "gene_keyword_missing": "context", "message": "Prompt may benefit from a 'context' segment. Ensure sufficient background is provided."},
            {"name": "UsesNegativePhrasing", "type": "keyword_check", "keywords": ["don't", "cannot", "avoid", "won't", "not able to"], "message": "Consider rephrasing negative statements (e.g., 'don't do X') to be positive and direct (e.g., 'do Y')."}
        ]

    def _structural_analysis(self, prompt_chromosome: PromptChromosome) -> list:
        """
        Analyzes the overall structure of the prompt chromosome.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings based on structural issues.
        """
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
        Placeholder method for analyzing prompt clarity.

        In a real implementation, this could involve NLP techniques to assess
        ambiguity, readability scores, etc.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings (currently empty).
        """
        # Placeholder: Future implementation could check for ambiguous phrases,
        # excessive jargon (if not domain-specific), or readability scores.
        # For now, it returns no specific clarity feedback.
        # feedback = []
        # if any("very unclear phrase" in gene.lower() for gene in prompt_chromosome.genes):
        #    feedback.append("Clarity Issue: Contains potentially unclear phrasing.")
        return []

    def _apply_heuristics(self, prompt_chromosome: PromptChromosome) -> list:
        """
        Applies a set of predefined heuristics or rules to the prompt.

        Args:
            prompt_chromosome (PromptChromosome): The prompt to analyze.

        Returns:
            list: A list of feedback strings based on triggered heuristics.
        """
        feedback = []
        prompt_content_lower = " ".join(str(gene) for gene in prompt_chromosome.genes).lower()

        for rule in self.critique_rules:
            if rule["type"] == "keyword_check":
                if any(keyword in prompt_content_lower for keyword in rule["keywords"]):
                    feedback.append(f"Heuristic Violation ({rule['name']}): {rule['message']}")
            
            elif rule["type"] == "gene_keyword_check":
                # Check if a gene segment *containing* the keyword is missing
                # This is a simplified check; more robust would be to check gene *roles* or *types*
                # if the PromptChromosome structure supported that.
                # For now, we check if the keyword appears in any gene string.
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
            return {
                "critique_score": 0.0, 
                "feedback_points": ["Error: Invalid or missing 'prompt_chromosome' object provided."],
                "suggestions": ["Ensure a valid PromptChromosome object is passed under the 'prompt_chromosome' key."]
            }

        print(f"{self.agent_id} - Critiquing prompt: {str(prompt_chromosome)}")
        
        feedback_points = []

        # Call internal analysis methods
        feedback_points.extend(self._structural_analysis(prompt_chromosome))
        feedback_points.extend(self._clarity_check(prompt_chromosome)) # Currently a placeholder
        feedback_points.extend(self._apply_heuristics(prompt_chromosome))

        issues_found = len(feedback_points)
        
        # Score is inversely related to the number of issues. 1.0 is a perfect score.
        # Each issue reduces the score.
        critique_score = max(0.0, 1.0 - (issues_found * 0.1)) # Example: each issue costs 0.1

        suggestions = []
        if issues_found > 0:
            suggestions.append("Consider addressing the feedback points to improve prompt quality and effectiveness.")
            suggestions.append("Review prompt engineering best practices for clarity, conciseness, and positive framing.")
        else:
            suggestions.append("Prompt appears well-structured based on current heuristics. Further evaluation with an LLM is recommended.")

        result = {
            "critique_score": critique_score,
            "feedback_points": feedback_points,
            "suggestions": suggestions
        }
        print(f"{self.agent_id} - Critique result: Score={critique_score}, Feedback#={len(feedback_points)}")
        return result

