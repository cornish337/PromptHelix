from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS # Import AGENT_SETTINGS
import logging

logger = logging.getLogger(__name__)

# Default provider from config if specific agent setting is not found
FALLBACK_LLM_PROVIDER = AGENT_SETTINGS.get("PromptArchitectAgent", {}).get("default_llm_provider", "openai")
FALLBACK_LLM_MODEL = AGENT_SETTINGS.get("PromptArchitectAgent", {}).get("default_llm_model", "gpt-3.5-turbo")


class PromptArchitectAgent(BaseAgent):
    agent_id = "PromptArchitect"
    agent_description = "Designs initial prompt structures based on requirements or patterns."
    """
    Designs initial prompt structures based on user requirements, 
    system goals, or existing successful prompt patterns.
    """
    def __init__(self, message_bus=None):
        """
        Initializes the PromptArchitectAgent.
        Loads prompt templates and configuration.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
        """
        super().__init__(agent_id=self.agent_id, message_bus=message_bus)

        agent_config = AGENT_SETTINGS.get(self.agent_id, {})
        self.llm_provider = agent_config.get("default_llm_provider", FALLBACK_LLM_PROVIDER)
        self.llm_model = agent_config.get("default_llm_model", FALLBACK_LLM_MODEL)
        logger.info(f"Agent '{self.agent_id}' initialized with LLM provider: {self.llm_provider} and model: {self.llm_model}")

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
        print(f"{self.agent_id} - Parsing requirements using LLM: Task='{task_desc}', Keywords='{keywords}', Constraints='{constraints}'")

        prompt = f"""
        Parse the following task requirements into a structured format.
        Task Description: "{task_desc}"
        Keywords: {keywords}
        Constraints: {constraints}

        Output a JSON-like structure with keys: 'task_description', 'parsed_keywords', 'parsed_constraints'.
        For example:
        {{
            "task_description": "Summarize a long article.",
            "parsed_keywords": ["summary", "article"],
            "parsed_constraints": {{ "max_length": 200 }}
        }}
        """
        try:
            response = call_llm_api(prompt, provider=self.llm_provider, model=self.llm_model)
            # Basic parsing of a stringified JSON-like response (placeholder)
            # In a robust implementation, ensure the LLM is prompted for valid JSON
            # and use json.loads() here.
            # For now, we'll simulate a structured response based on the LLM text.
            # This is a simplification.
            logger.info(f"Agent '{self.agent_id}' - LLM response for parsing: {response}")
            # Simulate parsing the LLM's text response into a dict
            # This is highly dependent on the LLM's output format and reliability
            # For this placeholder, we'll just return a structured dict based on input + simulated LLM enhancement
            parsed_reqs = {
                "task_description": task_desc, # Or response.get("task_description") if LLM returns structured JSON
                "keywords": keywords + ["llm_added_keyword"], # Simulate LLM adding keywords
                "constraints": constraints,
                "llm_raw_response_parsing": response
            }
            return parsed_reqs
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error calling LLM for parsing requirements: {e}", exc_info=True)
            # Fallback to simpler parsing if LLM call fails
            return {
                "task_description": task_desc if task_desc else "Default task description",
                "keywords": keywords,
                "constraints": constraints,
                "error": str(e)
            }

    def _select_template(self, parsed_requirements: dict) -> str:
        """
        Selects a prompt template based on parsed requirements using an LLM.

        Args:
            parsed_requirements (dict): The parsed requirements from _parse_requirements.

        Returns:
            str: The name of the selected template.
        """
        task_desc = parsed_requirements.get("task_description", "")
        available_templates = list(self.templates.keys())

        prompt = f"""
        Given the task description: "{task_desc}"
        And available prompt templates: {available_templates}
        Which template is most suitable for this task?
        Respond with only the name of the template (e.g., "summary_v1").
        """
        try:
            llm_response = call_llm_api(prompt, provider=self.llm_provider, model=self.llm_model)
            # Ensure the response is a valid template name
            selected_template_name = llm_response.strip()
            if selected_template_name in self.templates:
                logger.info(f"Agent '{self.agent_id}' - LLM selected template: {selected_template_name}")
                return selected_template_name
            else:
                logger.warning(f"Agent '{self.agent_id}' - LLM returned invalid template '{selected_template_name}'. Falling back.")
                # Fallback logic (same as original)
                return self._fallback_select_template(parsed_requirements)
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error calling LLM for template selection: {e}", exc_info=True)
            # Fallback logic (same as original)
            return self._fallback_select_template(parsed_requirements)

    def _fallback_select_template(self, parsed_requirements: dict) -> str:
        """ Fallback template selection logic if LLM fails. """
        task_desc = parsed_requirements.get("task_description") or ""
        task_desc_lower = task_desc.lower()
        if "summary" in task_desc_lower or "summarize" in task_desc_lower:
            template_name = "summary_v1"
        elif "question" in task_desc_lower or "answer" in task_desc_lower or "?" in task_desc:
            template_name = "question_answering_v1"
        else:
            template_name = "generic_v1"
        logger.info(f"Agent '{self.agent_id}' - Fallback selected template: {template_name}")
        return template_name

    def _populate_genes(self, template: dict, parsed_requirements: dict) -> list:
        """
        Populates genes for a PromptChromosome using an LLM based on a template
        and parsed requirements.

        Args:
            template (dict): The selected prompt template.
            parsed_requirements (dict): The parsed requirements.

        Returns:
            list: A list of gene strings for the PromptChromosome.
        """
        task_desc = parsed_requirements.get("task_description", "N/A")
        keywords = parsed_requirements.get("keywords", [])
        constraints = parsed_requirements.get("constraints", {})

        prompt = f"""
        Given the following prompt template:
        Instruction: "{template['instruction']}"
        Context Placeholder: "{template['context_placeholder']}"
        Output Format: "{template['output_format']}"

        And the task requirements:
        Description: "{task_desc}"
        Keywords: {keywords}
        Constraints: {constraints}

        Populate the template to create a specific prompt.
        Fill in the context placeholder. Integrate keywords naturally.
        Adhere to any specified output format or constraints.

        Return the populated prompt as a list of strings, where each string is a gene (a part of the prompt).
        For example:
        [
            "Instruction: Summarize the provided text.",
            "Context: The text is about AI. Focus on machine learning, neural networks. The summary should be concise.",
            "Output Format: A short paragraph."
        ]
        """
        try:
            llm_response = call_llm_api(prompt, provider=self.llm_provider, model=self.llm_model)
            logger.info(f"Agent '{self.agent_id}' - LLM response for gene population: {llm_response}")

            # This is a major simplification. LLM would ideally return a JSON list of strings.
            # For now, let's assume it returns a multi-line string that we can split.
            # A more robust solution would be to ask the LLM for JSON and parse it.
            genes = [line.strip() for line in llm_response.split('\n') if line.strip()]

            if not genes: # Fallback if LLM response is not parsable into lines
                raise ValueError("LLM returned empty or unparsable gene list.")

            logger.info(f"Agent '{self.agent_id}' - LLM populated genes: {genes}")
            return genes
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error calling LLM for gene population: {e}. Falling back to basic population.", exc_info=True)
            # Fallback to original simpler gene population
            return self._fallback_populate_genes(template, parsed_requirements)

    def _fallback_populate_genes(self, template: dict, parsed_requirements: dict) -> list:
        """ Fallback gene population logic if LLM fails. """
        genes = []
        genes.append(template["instruction"])

        task_description = parsed_requirements.get("task_description", "Default task description")
        if task_description == "Default task description" and "Default task description" not in template["instruction"]:
             genes.append(task_description)


        context = template.get("context_placeholder", "[details_placeholder]")
        keywords = parsed_requirements.get("keywords", [])
        
        if "[text_to_summarize]" in context and task_description:
            context = context.replace("[text_to_summarize]", task_description)
        elif "[question_text]" in context and task_description:
            context = context.replace("[question_text]", task_description)
        elif "[details]" in context and task_description:
             context = context.replace("[details]", task_description)


        if keywords:
            keyword_str = ", ".join(keywords)
            if any(kw_placeholder in context for kw_placeholder in ["[keywords]", "{keywords}"]):
                 context = context.replace("[keywords]", keyword_str).replace("{keywords}", keyword_str)
            elif "[details]" not in context : # Avoid appending if details were just replaced
                context += f" Focus on: {keyword_str}"
            elif task_description not in context : # Avoid appending if task_description was the detail
                context += f" Focus on: {keyword_str}"


        genes.append(f"Context: {context}")
        genes.append(f"Output Format: {template['output_format']}")
        
        logger.info(f"Agent '{self.agent_id}' - Fallback populated genes: {genes}")
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
        logger.info(f"Agent '{self.agent_id}' processing request: {request_data}")

        
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
            logger.warning(f"Agent '{self.agent_id}' - Warning: Template '{template_name}' not found. Using default fallback.")
            fallback_genes = ["Default instruction: " + task_desc]
            if keywords:
                fallback_genes.append("Keywords: " + ", ".join(keywords))
            return PromptChromosome(genes=fallback_genes, fitness_score=0.0) # fitness_score set by evaluator

        # 3. Populate genes based on template and inputs (placeholder)
        genes = self._populate_genes(selected_template, parsed_reqs)
        
        # Fitness score is typically not set by architect, but by evaluator later
        prompt_chromosome = PromptChromosome(genes=genes, fitness_score=0.0) 
        
        logger.info(f"Agent '{self.agent_id}' - Created PromptChromosome: {str(prompt_chromosome)}")
        return prompt_chromosome

