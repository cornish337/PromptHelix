from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome

class PromptArchitectAgent(BaseAgent):
    """
    Designs initial prompt structures based on user requirements, 
    system goals, or existing successful prompt patterns.
    """
    def __init__(self):
        """
        Initializes the PromptArchitectAgent.
        Loads prompt templates.
        """
        super().__init__(agent_id="PromptArchitect")

        self.recommendations = [
            "Consider a ReAct-style prompt structure: Thought, Action, Observation.",
            "Explore a Chain-of-Thought approach: break down the problem into sequential steps.",
            "For classification tasks, try a few-shot prompt with clear examples for each category.",
            "Use XML-like tags to delineate different parts of the prompt, like <context>, <question>, <output_format>.",
        ]

        # Load initial templates
        self.templates = self._load_templates()


    def _load_templates(self) -> dict:
        """
        Loads mock prompt templates.

        In a real scenario, this would load from a configuration file or database.

        Returns:
            dict: A dictionary of prompt templates.
        """
        return {
            "summary_v1": {
                "instruction": "Summarize the following text:",
                "context_placeholder": "[text_to_summarize]",
                "output_format": "Concise summary."
            },
            "question_answering_v1": {
                "instruction": "Answer the question based on the provided context:",
                "context_placeholder": "Context: [context_text]\nQuestion: [question_text]",
                "output_format": "Clear and direct answer."
            },
            "generic_v1": {
                "instruction": "Perform the following task:",
                "context_placeholder": "[details]",
                "output_format": "As requested."
            }
        }

    def _parse_requirements(self, task_desc: str, keywords: list, constraints: dict) -> dict:
        """
        Placeholder method to parse and interpret input requirements.

        Args:
            task_desc (str): The description of the task.
            keywords (list): A list of keywords relevant to the task.
            constraints (dict): A dictionary of constraints for the prompt.

        Returns:
            dict: A dictionary of parsed requirements.
        """
        print(f"{self.agent_id} - Parsing requirements: Task='{task_desc}', Keywords='{keywords}', Constraints='{constraints}'")
        # In a real implementation, this would involve more sophisticated NLP
        # and logic to extract structured information.
        return {
            "task_description": task_desc,
            "keywords": keywords,
            "constraints": constraints
        }

    def _select_template(self, parsed_requirements: dict) -> str:
        """
        Placeholder method to select a prompt template based on parsed requirements.

        Args:
            parsed_requirements (dict): The parsed requirements from _parse_requirements.

        Returns:
            str: The name of the selected template.
        """
        task_desc_lower = parsed_requirements.get("task_description", "").lower()
        if "summary" in task_desc_lower or "summarize" in task_desc_lower:
            template_name = "summary_v1"
        elif "question" in task_desc_lower or "answer" in task_desc_lower:
            template_name = "question_answering_v1"
        else:
            template_name = "generic_v1"
        
        print(f"{self.agent_id} - Selected template: {template_name}")
        return template_name

    def _populate_genes(self, template: dict, parsed_requirements: dict) -> list:
        """
        Placeholder method to populate genes for a PromptChromosome based on a template
        and parsed requirements.

        Args:
            template (dict): The selected prompt template.
            parsed_requirements (dict): The parsed requirements.

        Returns:
            list: A list of gene strings for the PromptChromosome.
        """
        genes = []
        genes.append(template["instruction"])

        context = template.get("context_placeholder", "[details_placeholder]")
        keywords = parsed_requirements.get("keywords", [])
        
        # Simple keyword integration for context
        # This could be much more sophisticated, e.g., filling specific placeholders
        if "[text_to_summarize]" in context and parsed_requirements.get("task_description"):
             # Assuming task_description might contain the text for summary in this simple case
            context = context.replace("[text_to_summarize]", parsed_requirements.get("task_description"))
        elif "[question_text]" in context and parsed_requirements.get("task_description"):
            # Assuming task_description might contain the question for QA in this simple case
            context = context.replace("[question_text]", parsed_requirements.get("task_description"))


        if keywords:
            # Add keywords into the context string, if not already a specific placeholder
            if any(kw_placeholder in context for kw_placeholder in ["[keywords]", "{keywords}"]):
                 context = context.replace("[keywords]", ", ".join(keywords)).replace("{keywords}", ", ".join(keywords))
            else:
                context += " Focus on: " + ", ".join(keywords)
        
        genes.append(f"Context: {context}")
        genes.append(f"Output Format: {template['output_format']}")
        
        print(f"{self.agent_id} - Populated genes: {genes}")
        return genes

    def process_request(self, request_data: dict) -> PromptChromosome:
        """
        Designs an initial prompt structure (PromptChromosome) based on the request.

        This method uses a simplified approach of parsing requirements, selecting a template,
        and populating genes. More complex logic would involve deeper NLP, rule engines,
        or even learned models for each step.

        Args:
            request_data (dict): Contains information like 'task_description', 
                                 'keywords', 'constraints'.
                                 Example: 
                                 {
                                     "task_description": "Summarize the provided article about AI.",
                                     "keywords": ["machine learning", "neural networks"],
                                     "constraints": {"max_length": 150}
                                 }
        Returns:
            PromptChromosome: An initial prompt chromosome.
        """
        print(f"{self.agent_id} processing request: {request_data}")

        
        task_desc = request_data.get("task_description", "Default task description")
        keywords = request_data.get("keywords", [])
        constraints = request_data.get("constraints", {})

        # 1. Parse requirements (placeholder)
        parsed_reqs = self._parse_requirements(task_desc, keywords, constraints)

        # 2. Select a template (placeholder)
        template_name = self._select_template(parsed_reqs)
        selected_template = self.templates.get(template_name)

        if not selected_template:
            # Fallback if template is somehow not found (e.g., _select_template returns invalid name)
            print(f"{self.agent_id} - Warning: Template '{template_name}' not found. Using default fallback.")
            fallback_genes = ["Default instruction: " + task_desc]
            if keywords:
                fallback_genes.append("Keywords: " + ", ".join(keywords))
            return PromptChromosome(genes=fallback_genes, fitness_score=0.0) # fitness_score set by evaluator

        # 3. Populate genes based on template and inputs (placeholder)
        genes = self._populate_genes(selected_template, parsed_reqs)
        
        # Fitness score is typically not set by architect, but by evaluator later
        prompt_chromosome = PromptChromosome(genes=genes, fitness_score=0.0) 
        
        print(f"{self.agent_id} - Created PromptChromosome: {str(prompt_chromosome)}")
        return prompt_chromosome

