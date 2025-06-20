from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS, KNOWLEDGE_DIR
import json  # For parsing LLM response
import logging
import os
from typing import Optional, Dict # Added for type hinting

logger = logging.getLogger(__name__)

# Default knowledge filename if nothing else is provided
FALLBACK_KNOWLEDGE_FILE = "style_optimizer_rules.json"


class StyleOptimizerAgent(BaseAgent):
    agent_id = "StyleOptimizer"
    agent_description = "Improves prompt style and clarity."
    """
    Refines prompts to enhance their style, tone, clarity, and persuasiveness,
    often based on specific target audience or desired communication effect.
    """
    def __init__(self, message_bus=None, settings: Optional[Dict] = None, knowledge_file_path: Optional[str] = None): # Modified signature
        """
        Initializes the StyleOptimizerAgent.
        Loads style transformation rules or lexicons and agent configuration.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            settings (Optional[Dict], optional): Configuration settings for the agent.
            knowledge_file_path (Optional[str], optional): Path to the knowledge file.
                Overrides 'knowledge_file_path' in settings if provided.
        """
        super().__init__(agent_id="StyleOptimizer", message_bus=message_bus, settings=settings)

        global_defaults = AGENT_SETTINGS.get("StyleOptimizerAgent", {})
        llm_provider_default = global_defaults.get("default_llm_provider", "openai")
        llm_model_default = global_defaults.get("default_llm_model", "gpt-3.5-turbo")

        self.llm_provider = self.settings.get("default_llm_provider", llm_provider_default)
        self.llm_model = self.settings.get("default_llm_model", llm_model_default)

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

        os.makedirs(os.path.dirname(self.knowledge_file_path), exist_ok=True)
        self.style_rules = {} # Initialize before loading
        self.load_knowledge()

    def _get_default_style_rules(self) -> dict:
        """
        Provides default mock style transformation rules.

        In a real scenario, this would load from a configuration file,
        a database, or be dynamically updated by other agents like MetaLearnerAgent.

        Returns:
            dict: A dictionary of style rules.
        """
        logger.info(f"Agent '{self.agent_id}': Using default style rules.")
        return {
            "formal": {
                "replace": {"don't": "do not", "stuff": "items", "gonna": "going to", "wanna": "want to"},
                "prepend_politeness": "Please ", # Changed from append to prepend for instructions
                "ensure_ending_punctuation": True
            },
            "casual": {
                "replace": {"do not": "don't", "items": "stuff", "please ": "", "Please ": "", "kindly ": "", "Kindly ": ""},
                "remove_ending_punctuation": False # Usually casual still has punctuation
            },
            "instructional": { # Example of a more specific style
                "prepend_politeness": "Could you ",
                "append_request_marker": "?", # For instructions that are phrased as questions
                "replace": {"tell me": "explain"}
            }
        }

    def load_knowledge(self):
        """
        Loads style rules from the specified JSON file.
        If the file is not found or is invalid, it loads default rules
        and saves them to a new file.
        """
        try:
            with open(self.knowledge_file_path, 'r') as f:
                self.style_rules = json.load(f)
            logger.info(f"Agent '{self.agent_id}': Style rules loaded successfully from '{self.knowledge_file_path}'.")
        except FileNotFoundError:
            logger.warning(f"Agent '{self.agent_id}': Knowledge file '{self.knowledge_file_path}' not found. Using default rules and creating the file.")
            self.style_rules = self._get_default_style_rules()
            self.save_knowledge() # Save defaults if file not found
        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            self.style_rules = self._get_default_style_rules()
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}': Failed to load style rules from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Failed to load style rules from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            self.style_rules = self._get_default_style_rules()

    def optimize(self, prompt: str, tone: str = "concise") -> str:
        """
        Optimizes the style of a given prompt string using an LLM.

        Args:
            prompt (str): The prompt string to optimize.
            tone (str): The desired tone for the optimization (e.g., "concise", "formal", "casual").

        Returns:
            str: The optimized prompt string or a placeholder if not in "REAL" LLM mode.
        """
        llm_template = f"Rephrase the following prompt to be more {tone}: {prompt}"

        # Ensure self.settings is available. BaseAgent should store it.
        if not hasattr(self, 'settings') or self.settings is None:
            logger.warning(f"Agent '{self.agent_id}': Settings not available. Defaulting to non-REAL mode for optimize.")
            # Attempt to fetch from AGENT_SETTINGS if self.settings is missing
            # This is a fallback, ideally settings should be passed during init.
            agent_specific_settings = AGENT_SETTINGS.get(self.agent_id, {})
            llm_mode = agent_specific_settings.get("llm_mode", "PLACEHOLDER")
        else:
            llm_mode = self.settings.get("llm_mode", "PLACEHOLDER")

        if llm_mode == "REAL":
            try:
                logger.info(f"Agent '{self.agent_id}': Calling LLM for style optimization. Tone: {tone}. Prompt: \"{prompt[:100]}...\"")
                optimized_prompt = call_llm_api(
                    prompt_text=llm_template,
                    provider=self.llm_provider,
                    model=self.llm_model
                )
                logger.info(f"Agent '{self.agent_id}': LLM optimization successful. Returning optimized prompt.")
                return optimized_prompt
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}': Error calling LLM API during style optimization: {e}. Returning original prompt with placeholder.", exc_info=True)
                # Fallback to placeholder behavior in case of LLM error even in REAL mode
                return f"{prompt} [Styled: Placeholder - Error]"
        else:
            logger.info(f"Agent '{self.agent_id}': LLM mode is '{llm_mode}'. Returning placeholder styled prompt.")
            return f"{prompt} [Styled: Placeholder]"

    def save_knowledge(self):
        """
        Saves the current style rules to the specified JSON file.
        """
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(self.style_rules, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Style rules saved successfully to '{self.knowledge_file_path}'.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to save style rules to '{self.knowledge_file_path}': {e}", exc_info=True)

    def _tone_analysis_adjustment(self, genes: list, target_tone: str) -> list:
        """
        Placeholder for analyzing and adjusting the tone of prompt genes.

        Args:
            genes (list): The list of gene strings.
            target_tone (str): The desired tone (e.g., "neutral", "enthusiastic").

        Returns:
            list: The list of genes, potentially modified for tone.
        """
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Analyzing/adjusting tone for: {target_tone}")
        # Future: Implement NLP techniques for tone detection and rule-based or
        # model-based transformations.
        return genes

    def _clarity_enhancement(self, genes: list) -> list:
        """
        Placeholder for enhancing the clarity of prompt genes.

        Args:
            genes (list): The list of gene strings.

        Returns:
            list: The list of genes, potentially modified for clarity.
        """
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Enhancing clarity.")
        # Future: Implement checks for ambiguity, complex sentences, jargon reduction, etc.
        # Example:
        # for i, gene in enumerate(genes):
        #     if "utilize" in gene:
        #         genes[i] = gene.replace("utilize", "use")
        return genes

    def _persuasiveness_improvement(self, genes: list) -> list:
        """
        Placeholder for improving the persuasiveness of prompt genes.

        Args:
            genes (list): The list of gene strings.

        Returns:
            list: The list of genes, potentially modified for persuasiveness.
        """
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Improving persuasiveness.")
        # Future: Implement techniques like adding rhetorical questions, benefit statements, etc.
        return genes

    def _compare_chromosomes(self, old_chromo: PromptChromosome, new_chromo: PromptChromosome) -> list:
        """
        Compares two chromosomes and lists the differences in their genes.

        Args:
            old_chromo (PromptChromosome): The original chromosome.
            new_chromo (PromptChromosome): The new chromosome.

        Returns:
            list: A list of strings describing the differences.
        """
        diffs = []
        old_genes = [str(g) for g in old_chromo.genes]
        new_genes = [str(g) for g in new_chromo.genes]

        if len(old_genes) != len(new_genes):
            diffs.append(f"Gene count changed from {len(old_genes)} to {len(new_genes)}.")
        
        for i in range(min(len(old_genes), len(new_genes))):
            if old_genes[i] != new_genes[i]:
                diffs.append(f"Gene {i+1}: '{old_genes[i]}' -> '{new_genes[i]}'")
        
        if len(new_genes) > len(old_genes):
            for i in range(len(old_genes), len(new_genes)):
                diffs.append(f"Gene {i+1} added: '{new_genes[i]}'")
        elif len(old_genes) > len(new_genes):
             for i in range(len(new_genes), len(old_genes)):
                diffs.append(f"Gene {i+1} removed: '{old_genes[i]}'")
        return diffs

    def process_request(self, request_data: dict) -> PromptChromosome:
        """
        Optimizes the style of a given prompt chromosome based on a target style.

        Args:
            request_data (dict): Expected to contain:
                'prompt_chromosome' (PromptChromosome): The prompt to optimize.
                'target_style' (str): The desired style (e.g., "formal", "casual").
                                 Example:
                                 {
                                     "prompt_chromosome": PromptChromosome(genes=["Instruct: don't summarize stuff", "Context: ..."]),
                                     "target_style": "formal"
                                 }

        Returns:
            PromptChromosome: The style-optimized prompt chromosome.
        """
        original_chromosome = request_data.get("prompt_chromosome")
        target_style = request_data.get("target_style")

        if not isinstance(original_chromosome, PromptChromosome):
            logger.error(f"Agent '{self.agent_id}': Invalid or missing 'prompt_chromosome' object provided.")
            return original_chromosome 
        
        if not target_style : # Removed check against self.style_rules as LLM might handle undefined styles
            logger.warning(f"Agent '{self.agent_id}': Target style not provided. Returning original prompt.")
            return original_chromosome
        
        # If target_style is not in self.style_rules, only LLM can handle it. Fallback won't work.
        # This is acceptable as LLM is the primary, rules are fallback.

        logger.info(f"Agent '{self.agent_id}': Optimizing prompt (ID: {original_chromosome.id if original_chromosome else 'N/A'}) for style: {target_style}")
        
        original_genes_str_list = [str(g) for g in original_chromosome.genes]

        # Attempt LLM-based style optimization first
        llm_optimized_genes = None
        llm_prompt = f"""
Rewrite the following prompt segments to adopt a '{target_style}' style.
Ensure the core meaning and placeholders (like [PLACEHOLDER]) are preserved.
Return the rewritten prompt segments as a JSON list of strings, with the same number of segments as the original.

Original Prompt Segments:
{json.dumps(original_genes_str_list, indent=2)}

Target Style: {target_style}

Rewritten Prompt Segments (JSON list of strings):
"""
        try:
            response_str = call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model)
            logger.info(f"Agent '{self.agent_id}': LLM raw response for style optimization: {response_str}")

            parsed_genes = self._parse_llm_gene_response(response_str)
            if parsed_genes and len(parsed_genes) == len(original_genes_str_list):
                llm_optimized_genes = parsed_genes
                logger.info(f"Agent '{self.agent_id}': Successfully optimized style '{target_style}' using LLM.")
            else:
                logger.warning(f"Agent '{self.agent_id}': LLM response for style '{target_style}' was not a valid list of genes or mismatched length. Will use rule-based fallback if rules exist. Parsed: {parsed_genes}")

        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error during LLM style optimization for '{target_style}': {e}. Using rule-based fallback if rules exist.", exc_info=True)

        if llm_optimized_genes:
            modified_genes = llm_optimized_genes
        else:
            # Fallback to rule-based transformations only if target_style is in defined style_rules
            if target_style in self.style_rules:
                logger.info(f"Agent '{self.agent_id}': Using rule-based fallback for style: {target_style}")
                modified_genes = list(original_genes_str_list) # Start with a fresh copy for rule-based
                style_config = self.style_rules[target_style]

                if "replace" in style_config:
                    for i, gene_str in enumerate(modified_genes):
                        # current_gene_val = gene_str # Not needed if replacing on modified_genes[i]
                        for old, new in style_config["replace"].items():
                            modified_genes[i] = modified_genes[i].replace(old, new)

                if "prepend_politeness" in style_config and modified_genes:
                    first_gene_lower = modified_genes[0].lower()
                    instruction_keywords = ["summarize", "generate", "write", "answer", "explain", "describe", "list", "create", "instruct:"]
                    if any(keyword in first_gene_lower for keyword in instruction_keywords) and \
                       not modified_genes[0].startswith(style_config["prepend_politeness"]):
                        modified_genes[0] = style_config["prepend_politeness"] + modified_genes[0]

                if "append_request_marker" in style_config and modified_genes:
                    if not modified_genes[0].endswith(style_config["append_request_marker"]):
                         modified_genes[0] = modified_genes[0].rstrip('.!') + style_config["append_request_marker"]

                if style_config.get("ensure_ending_punctuation", False) and modified_genes:
                    for i, gene_str in enumerate(modified_genes):
                        if gene_str and gene_str[-1] not in ".!?":
                            modified_genes[i] = gene_str + "."
            else:
                # If LLM failed and no rule-based style exists, return original genes
                logger.warning(f"Agent '{self.agent_id}': LLM failed for style '{target_style}' and no rule-based fallback exists. Returning original genes.")
                modified_genes = list(original_genes_str_list)

            # Apply placeholder adjustments after rule-based changes or if LLM failed and no rules applied
            modified_genes = self._tone_analysis_adjustment(modified_genes, target_style)
            modified_genes = self._clarity_enhancement(modified_genes)
            modified_genes = self._persuasiveness_improvement(modified_genes)

        optimized_chromosome = PromptChromosome(genes=modified_genes, fitness_score=0.0)
        
        diff = self._compare_chromosomes(original_chromosome, optimized_chromosome)
        if diff:
            source = "LLM" if llm_optimized_genes else ("Rule-based Fallback" if target_style in self.style_rules and not llm_optimized_genes else "Original (No applicable rules/LLM failed)")
            logger.info(f"Agent '{self.agent_id}': Stylistic changes applied via {source} for target style '{target_style}':")
            for d in diff:
                logger.info(f"  - {d}")
        else:
            logger.info(f"Agent '{self.agent_id}': No stylistic changes applied for target style '{target_style}'.")

        return optimized_chromosome

    def _parse_llm_gene_response(self, response_str: str) -> list[str] | None:
        """ Parses LLM response expected to be a JSON list of strings (genes). """
        try:
            # Handle cases where LLM might add explanatory text before/after JSON
            json_start = response_str.find('[')
            json_end = response_str.rfind(']') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                actual_json = response_str[json_start:json_end]
                parsed_list = json.loads(actual_json)
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    return parsed_list
            print(f"{self.agent_id} - LLM response was not a valid JSON list of strings: {response_str}")
            return None
        except json.JSONDecodeError as e:
            print(f"{self.agent_id} - Failed to parse LLM gene response as JSON list: {e}. Response: {response_str}")
            # Fallback: if not JSON, try splitting by newline if it looks like a list of genes
            # This is less reliable and should ideally be avoided by ensuring LLM sticks to JSON.
            if "\n" in response_str:
                lines = [line.strip() for line in response_str.split('\n') if line.strip()]
                # Basic check if lines look like genes (e.g. not too long, no weird chars)
                # This is very heuristic.
                if lines and all(len(line) < 500 for line in lines): # Arbitrary length check
                    print(f"{self.agent_id} - Attempting to use newline-separated genes from LLM response.")
                    return lines
            return None
