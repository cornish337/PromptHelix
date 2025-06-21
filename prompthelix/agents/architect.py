from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS, KNOWLEDGE_DIR # Keep KNOWLEDGE_DIR for default path construction
import os
import json
import logging
from typing import Optional, Dict # Added for type hinting

logger = logging.getLogger(__name__)

# Default knowledge filename if nothing else is provided
FALLBACK_LLM_PROVIDER = "openai"
FALLBACK_LLM_MODEL = "gpt-3.5-turbo"
FALLBACK_KNOWLEDGE_FILE = "architect_knowledge.json"


class PromptArchitectAgent(BaseAgent):
    # agent_id class variable can serve as a default if not overridden by pipeline config.
    agent_id_default = "PromptArchitect"
    agent_description = "Designs initial prompt structures based on requirements or patterns."
    """
    Designs initial prompt structures based on user requirements, 
    system goals, or existing successful prompt patterns.
    """
    def __init__(self, agent_id: Optional[str] = None, message_bus=None, settings: Optional[Dict] = None, knowledge_file_path: Optional[str] = None):
        """
        Initializes the PromptArchitectAgent.
        Loads prompt templates and configuration.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            settings (Optional[Dict], optional): Configuration settings for the agent.
            knowledge_file_path (Optional[str], optional): Path to the knowledge file.
                This is kept for backward compatibility or direct specification,
                but `settings` dictionary can also provide it via 'knowledge_file_path'.
        """
        # Use provided agent_id from orchestrator, fallback to class default if None
        effective_agent_id = agent_id if agent_id is not None else self.agent_id_default
        super().__init__(agent_id=effective_agent_id, message_bus=message_bus, settings=settings)

        # Load defaults from global AGENT_SETTINGS in case settings dict is missing keys
        # The settings_key for this agent type is typically "PromptArchitectAgent"
        settings_key_for_globals = self.settings.get("settings_key_from_pipeline", "PromptArchitectAgent")
        global_defaults = AGENT_SETTINGS.get(settings_key_for_globals, {})
        llm_provider_default = global_defaults.get("default_llm_provider", "openai")
        llm_model_default = global_defaults.get("default_llm_model", "gpt-3.5-turbo")

        # Configuration values will now be sourced from self.settings, with fallbacks
        self.llm_provider = self.settings.get("default_llm_provider", llm_provider_default)
        self.llm_model = self.settings.get("default_llm_model", llm_model_default)

        # knowledge_file_path can be overridden by settings, then by direct param, then fallback
        _knowledge_file = self.settings.get("knowledge_file_path", knowledge_file_path)
        if _knowledge_file:
            self.knowledge_file_path = (
                _knowledge_file
                if os.path.isabs(_knowledge_file)
                else os.path.join(KNOWLEDGE_DIR, _knowledge_file)
            )
        else:
            self.knowledge_file_path = os.path.join(KNOWLEDGE_DIR, FALLBACK_KNOWLEDGE_FILE)

        logger.info(f"Agent '{self.agent_id}' initialized with LLM provider: {self.llm_provider}, model: {self.llm_model}, knowledge: {self.knowledge_file_path}")

        self.recommendations = [
            "Consider a ReAct-style prompt structure: Thought, Action, Observation.",
            "Explore a Chain-of-Thought approach: break down the problem into sequential steps.",
            "For classification tasks, try a few-shot prompt with clear examples for each category.",
            "Use XML-like tags to delineate different parts of the prompt, like <context>, <question>, <output_format>.",
        ]

        os.makedirs(os.path.dirname(self.knowledge_file_path), exist_ok=True)
        self.templates = {} # Initialize before loading
        self.load_knowledge()


    def _get_default_templates(self) -> dict:
        """
        Provides default mock prompt templates.

        Returns:
            dict: A dictionary of prompt templates.
        """
        logger.info(f"Agent '{self.agent_id}': Using default prompt templates.")
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

    def load_knowledge(self):
        """
        Loads prompt templates from the specified JSON file.
        If the file is not found or is invalid, it loads default templates
        and saves them to a new file.
        """
        try:
            with open(self.knowledge_file_path, 'r') as f:
                self.templates = json.load(f)
            logger.info(f"Agent '{self.agent_id}': Prompt templates loaded successfully from '{self.knowledge_file_path}'.")
        except FileNotFoundError:
            logger.warning(f"Agent '{self.agent_id}': Knowledge file '{self.knowledge_file_path}' not found. Using default templates and creating the file.")
            self.templates = self._get_default_templates()
            self.save_knowledge() # Save defaults if file not found
        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default templates.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default templates.",
                exc_info=True,
            )
            self.templates = self._get_default_templates()
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}': Failed to load templates from '{self.knowledge_file_path}': {e}. Using default templates.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Failed to load templates from '{self.knowledge_file_path}': {e}. Using default templates.",
                exc_info=True,
            )
            self.templates = self._get_default_templates()

    def save_knowledge(self):
        """
        Saves the current prompt templates to the specified JSON file.
        """
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(self.templates, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Prompt templates saved successfully to '{self.knowledge_file_path}'.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to save prompt templates to '{self.knowledge_file_path}': {e}", exc_info=True)

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
        Parse the following task requirements into JSON with keys 'task_description', 'keywords', and 'constraints'.
        Task Description: "{task_desc}"
        Keywords: {keywords}
        Constraints: {constraints}

        Respond ONLY with valid JSON.
        """
        try:
            response = call_llm_api(prompt, provider=self.llm_provider, model=self.llm_model)
            # Basic parsing of a stringified JSON-like response (placeholder)
            # In a robust implementation, ensure the LLM is prompted for valid JSON
            logger.info(f"Agent '{self.agent_id}' - LLM response for parsing: {response}")
            data = json.loads(response)
            if not isinstance(data, dict):
                raise ValueError("LLM response for requirements is not a JSON object")

            return {
                "task_description": data.get("task_description", task_desc if task_desc else "Default task description"),
                "keywords": data.get("keywords", keywords),
                "constraints": data.get("constraints", constraints),
            }
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
            data = json.loads(llm_response)
            if not isinstance(data, dict):
                raise ValueError("LLM response for template selection is not a JSON object")

            selected_template_name = data.get("template") or data.get("template_name")
            if selected_template_name in self.templates:
                logger.info(f"Agent '{self.agent_id}' - LLM selected template: {selected_template_name}")
                return selected_template_name
            else:
                logger.warning(f"Agent '{self.agent_id}' - LLM returned invalid template '{selected_template_name}'. Falling back.")
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

            data = json.loads(llm_response)
            if not isinstance(data, dict):
                raise ValueError("LLM response for gene population is not a JSON object")

            genes = data.get("genes")
            if not isinstance(genes, list) or not genes:
                raise ValueError("LLM returned invalid gene list")

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

