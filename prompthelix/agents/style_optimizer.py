from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.chromosome import PromptChromosome # Updated import
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS, KNOWLEDGE_DIR
import json
import logging
import os
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

FALLBACK_LLM_PROVIDER = "openai"
FALLBACK_LLM_MODEL = "gpt-3.5-turbo"
FALLBACK_KNOWLEDGE_FILE = "style_optimizer_rules.json"


class StyleOptimizerAgent(BaseAgent):
    agent_id_default = "StyleOptimizer"
    agent_description = "Improves prompt style and clarity."

    def __init__(self, agent_id: Optional[str] = None, message_bus=None, settings: Optional[Dict] = None, knowledge_file_path: Optional[str] = None):
        effective_agent_id = agent_id if agent_id is not None else self.agent_id_default
        super().__init__(agent_id=effective_agent_id, message_bus=message_bus, settings=settings)

        settings_key_for_globals = self.settings.get("settings_key_from_pipeline", "StyleOptimizerAgent")
        global_defaults = AGENT_SETTINGS.get(settings_key_for_globals, {})
        llm_provider_default = global_defaults.get("default_llm_provider", FALLBACK_LLM_PROVIDER)
        llm_model_default = global_defaults.get("default_llm_model", FALLBACK_LLM_MODEL)

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
        self.style_rules = {}
        self.load_knowledge()

    def _get_default_style_rules(self) -> dict:
        logger.info(f"Agent '{self.agent_id}': Using default style rules.")
        return {
            "formal": {
                "replace": {"don't": "do not", "stuff": "items", "gonna": "going to", "wanna": "want to"},
                "prepend_politeness": "Please ",
                "ensure_ending_punctuation": True
            },
            "casual": {
                "replace": {"do not": "don't", "items": "stuff", "please ": "", "Please ": "", "kindly ": "", "Kindly ": ""},
                "remove_ending_punctuation": False
            },
            "instructional": {
                "prepend_politeness": "Could you ",
                "append_request_marker": "?",
                "replace": {"tell me": "explain"}
            }
        }

    def load_knowledge(self):
        try:
            with open(self.knowledge_file_path, 'r') as f:
                self.style_rules = json.load(f)
            logger.info(f"Agent '{self.agent_id}': Style rules loaded successfully from '{self.knowledge_file_path}'.")
        except FileNotFoundError:
            logger.warning(f"Agent '{self.agent_id}': Knowledge file '{self.knowledge_file_path}' not found. Using default rules and creating the file.")
            self.style_rules = self._get_default_style_rules()
            self.save_knowledge()
        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            self.style_rules = self._get_default_style_rules()
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}': Failed to load style rules from '{self.knowledge_file_path}': {e}. Using default rules.",
                exc_info=True,
            )
            self.style_rules = self._get_default_style_rules()

    async def optimize(self, prompt: str, tone: str = "concise") -> str: # Changed to async
        llm_template = f"Rephrase the following prompt to be more {tone}: {prompt}"
        if not hasattr(self, 'settings') or self.settings is None:
            logger.warning(f"Agent '{self.agent_id}': Settings not available. Defaulting to non-REAL mode for optimize.")
            agent_specific_settings = AGENT_SETTINGS.get(self.agent_id, {})
            llm_mode = agent_specific_settings.get("llm_mode", "PLACEHOLDER")
        else:
            llm_mode = self.settings.get("llm_mode", "PLACEHOLDER")

        if llm_mode == "REAL":
            try:
                logger.info(f"Agent '{self.agent_id}': Calling LLM for style optimization. Tone: {tone}. Prompt: \"{prompt[:100]}...\"")
                optimized_prompt = await call_llm_api( # Added await
                    prompt=llm_template, # Corrected param name from prompt_text to prompt
                    provider=self.llm_provider,
                    model=self.llm_model
                )
                logger.info(f"Agent '{self.agent_id}': LLM optimization successful. Returning optimized prompt.")
                return optimized_prompt
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}': Error calling LLM API during style optimization: {e}. Returning original prompt with placeholder.", exc_info=True)
                return f"{prompt} [Styled: Placeholder - Error]"
        else:
            logger.info(f"Agent '{self.agent_id}': LLM mode is '{llm_mode}'. Returning placeholder styled prompt.")
            return f"{prompt} [Styled: Placeholder]"

    def save_knowledge(self):
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(self.style_rules, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Style rules saved successfully to '{self.knowledge_file_path}'.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to save style rules to '{self.knowledge_file_path}': {e}", exc_info=True)

    def _tone_analysis_adjustment(self, genes: list, target_tone: str) -> list:
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Analyzing/adjusting tone for: {target_tone}")
        return genes

    def _clarity_enhancement(self, genes: list) -> list:
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Enhancing clarity.")
        return genes

    def _persuasiveness_improvement(self, genes: list) -> list:
        logger.info(f"Agent '{self.agent_id}': (Placeholder) Improving persuasiveness.")
        return genes

    def _compare_chromosomes(self, old_chromo: PromptChromosome, new_chromo: PromptChromosome) -> list:
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

    async def process_request(self, request_data: dict) -> PromptChromosome: # Changed to async
        original_chromosome = request_data.get("prompt_chromosome")
        target_style = request_data.get("target_style")

        if not isinstance(original_chromosome, PromptChromosome):
            logger.error(f"Agent '{self.agent_id}': Invalid or missing 'prompt_chromosome' object provided.")
            return original_chromosome 
        
        if not target_style :
            logger.warning(f"Agent '{self.agent_id}': Target style not provided. Returning original prompt.")
            return original_chromosome
        
        logger.info(f"Agent '{self.agent_id}': Optimizing prompt (ID: {original_chromosome.id if original_chromosome else 'N/A'}) for style: {target_style}")
        
        original_genes_str_list = [str(g) for g in original_chromosome.genes]

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
            response_str = await call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model) # Added await
            logger.info(f"Agent '{self.agent_id}': LLM raw response for style optimization: {response_str}")

            parsed_genes = self._parse_llm_gene_response(response_str) # This is a sync helper, fine
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
            if target_style in self.style_rules:
                logger.info(f"Agent '{self.agent_id}': Using rule-based fallback for style: {target_style}")
                modified_genes = list(original_genes_str_list)
                style_config = self.style_rules[target_style]

                if "replace" in style_config:
                    for i, gene_str in enumerate(modified_genes):
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
                logger.warning(
                    f"Agent '{self.agent_id}': LLM failed for style '{target_style}' and no rule-based fallback exists. Returning original genes."
                )
                return original_chromosome

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
            json_start = response_str.find('[')
            json_end = response_str.rfind(']') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                actual_json = response_str[json_start:json_end]
                parsed_list = json.loads(actual_json)
                if isinstance(parsed_list, list) and all(isinstance(item, str) for item in parsed_list):
                    return parsed_list
            logger.warning(f"{self.agent_id} - LLM response was not a valid JSON list of strings: {response_str}") # Changed print to logger.warning
            return None
        except json.JSONDecodeError as e:
            logger.error(f"{self.agent_id} - Failed to parse LLM gene response as JSON list: {e}. Response: {response_str}", exc_info=True) # Changed print to logger.error
            if "\n" in response_str:
                lines = [line.strip() for line in response_str.split('\n') if line.strip()]
                if lines and all(len(line) < 500 for line in lines):
                    logger.info(f"{self.agent_id} - Attempting to use newline-separated genes from LLM response.") # Changed print to logger.info
                    return lines
            return None
