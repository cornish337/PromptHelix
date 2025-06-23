from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.chromosome import PromptChromosome # Updated import
from prompthelix.utils.llm_utils import call_llm_api
import json
import os
import logging
from typing import Optional, Dict, List # Added List for type hinting

from prompthelix.config import AGENT_SETTINGS, KNOWLEDGE_DIR # Import new config

logger = logging.getLogger(__name__)

# Default file name and persist flag; other defaults are resolved during initialization
DEFAULT_KNOWLEDGE_FILENAME = AGENT_SETTINGS.get("MetaLearnerAgent", {}).get("knowledge_file_path", "meta_learner_knowledge_fallback.json")
PERSIST_ON_UPDATE_DEFAULT = AGENT_SETTINGS.get("MetaLearnerAgent", {}).get("persist_knowledge_on_update", True)

class MetaLearnerAgent(BaseAgent):
    agent_id = "MetaLearner"
    agent_description = "Learns from performance metrics to guide other agents."
    """
    Analyzes the overall performance of the prompt generation and optimization
    process over time. It learns from successful and unsuccessful prompts and
    agents' actions to provide higher-level guidance and adapt strategies.
    Includes persistence for its learned knowledge.
    """
    def __init__(self, message_bus=None, settings: Optional[Dict] = None, knowledge_file_path: str = None): # Added settings to __init__
        """
        Initializes the MetaLearnerAgent.
        Sets up the knowledge base and data log, and loads existing knowledge.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            settings (Optional[Dict], optional): Configuration settings for the agent.
            knowledge_file_path (str, optional): Path to the JSON file for persisting knowledge.
                                                 If None, uses path from config.
        """
        super().__init__(agent_id="MetaLearner", message_bus=message_bus, settings=settings) # Pass settings to super

        agent_config = self.settings if self.settings else AGENT_SETTINGS.get(self.agent_id, {}) # Use self.settings if available
        llm_provider_default = agent_config.get("default_llm_provider", "openai")
        self.llm_provider = llm_provider_default
        self.llm_model = agent_config.get("default_llm_model")

        _knowledge_filename = knowledge_file_path if knowledge_file_path else agent_config.get("knowledge_file_path", DEFAULT_KNOWLEDGE_FILENAME)

        # Ensure KNOWLEDGE_DIR exists
        # Check if KNOWLEDGE_DIR is an absolute path, if not, make it relative to current working dir or a predefined base
        # For simplicity, this example assumes KNOWLEDGE_DIR is correctly configured (e.g., absolute or resolvable relative to execution)
        # A more robust solution would involve checking and creating KNOWLEDGE_DIR if it's relative and doesn't exist.
        # Here, we just ensure the *specific knowledge file's directory* exists.

        self.knowledge_file_path = _knowledge_filename # Store the potentially relative path first
        if not os.path.isabs(self.knowledge_file_path):
             self.knowledge_file_path = os.path.join(KNOWLEDGE_DIR, self.knowledge_file_path)

        knowledge_dir_for_file = os.path.dirname(self.knowledge_file_path)
        if knowledge_dir_for_file and not os.path.exists(knowledge_dir_for_file): # Check if dir_for_file is not empty
            try:
                os.makedirs(knowledge_dir_for_file, exist_ok=True)
                logger.info(f"Agent '{self.agent_id}': Created directory for knowledge file: {knowledge_dir_for_file}")
            except OSError as e:
                logger.error(f"Agent '{self.agent_id}': Error creating directory {knowledge_dir_for_file}: {e}. Knowledge might not save correctly.", exc_info=True)

        self.persist_on_update = agent_config.get("persist_knowledge_on_update", PERSIST_ON_UPDATE_DEFAULT)

        if self.message_bus:
            self.subscribe_to("evaluation_result")
            self.subscribe_to("critique_result")

        self._default_knowledge_base_structure = {
            "successful_prompt_features": [],
            "common_critique_themes": [],
            "prompt_metric_stats": [],
            "llm_identified_trends": [],
            "statistical_prompt_metric_trends": [],
            "legacy_successful_patterns": [],
            "legacy_common_pitfalls": {},
            "legacy_performance_trends": [],
            "agent_effectiveness_signals": {"CriticAgent_score_vs_fitness_trend": "neutral"},
            "ga_parameter_adjustment_suggestions": {"mutation_rate_factor": 1.0, "population_size_factor": 1.0}
        }
        self.knowledge_base = self._default_knowledge_base_structure.copy()
        self.data_log = []

        self.load_knowledge()

    def analyze_generation(self, generation_data: list):
        if not generation_data:
            logger.warning(f"Agent '{self.agent_id}': analyze_generation called with empty generation_data.")
            return

        for individual in generation_data:
            log_entry = {
                "type": "generation_individual_summary",
                "prompt_id": individual.get("prompt_id"),
                "fitness_score": individual.get("fitness_score"),
                "critic_score": individual.get("critic_score"),
            }
            log_entry = {k: v for k, v in log_entry.items() if v is not None}
            self.data_log.append(log_entry)
        logger.info(f"Agent '{self.agent_id}': Logged {len(generation_data)} individuals from current generation.")

        if "agent_effectiveness_signals" not in self.knowledge_base:
            self.knowledge_base["agent_effectiveness_signals"] = self._default_knowledge_base_structure["agent_effectiveness_signals"].copy()

        relevant_individuals = [
            ind for ind in generation_data
            if ind.get("fitness_score") is not None and ind.get("critic_score") is not None
        ]

        if len(relevant_individuals) >= 4:
            relevant_individuals.sort(key=lambda x: x["fitness_score"], reverse=True)
            population_size = len(relevant_individuals)
            top_25_percent_index = max(1, population_size // 4)

            top_individuals = relevant_individuals[:top_25_percent_index]
            bottom_individuals = relevant_individuals[-top_25_percent_index:]

            avg_critic_top = sum(ind["critic_score"] for ind in top_individuals) / len(top_individuals) if top_individuals else 0
            avg_critic_bottom = sum(ind["critic_score"] for ind in bottom_individuals) / len(bottom_individuals) if bottom_individuals else 0

            critic_fitness_correlation_threshold = 0.2

            if avg_critic_top > avg_critic_bottom + critic_fitness_correlation_threshold:
                self.knowledge_base['agent_effectiveness_signals']['CriticAgent_score_vs_fitness_trend'] = "positive"
                logger.info(f"Agent '{self.agent_id}': CriticAgent trend set to 'positive'. AvgCriticTop: {avg_critic_top:.2f}, AvgCriticBottom: {avg_critic_bottom:.2f}")
            elif avg_critic_bottom > avg_critic_top + critic_fitness_correlation_threshold:
                self.knowledge_base['agent_effectiveness_signals']['CriticAgent_score_vs_fitness_trend'] = "negative"
                logger.info(f"Agent '{self.agent_id}': CriticAgent trend set to 'negative'. AvgCriticTop: {avg_critic_top:.2f}, AvgCriticBottom: {avg_critic_bottom:.2f}")
            else:
                self.knowledge_base['agent_effectiveness_signals']['CriticAgent_score_vs_fitness_trend'] = "neutral"
                logger.info(f"Agent '{self.agent_id}': CriticAgent trend set to 'neutral'. AvgCriticTop: {avg_critic_top:.2f}, AvgCriticBottom: {avg_critic_bottom:.2f}")
        else:
            logger.warning(f"Agent '{self.agent_id}': Not enough relevant individuals ({len(relevant_individuals)}) with fitness and critic scores to apply CriticAgent heuristic.")

        if "ga_parameter_adjustment_suggestions" not in self.knowledge_base:
             self.knowledge_base["ga_parameter_adjustment_suggestions"] = self._default_knowledge_base_structure["ga_parameter_adjustment_suggestions"].copy()

        prompt_representations = [ind.get('prompt_text', str(ind.get('prompt_id', ''))) for ind in generation_data]
        unique_prompts = set(prompt_representations)
        num_unique_prompts = len(unique_prompts)
        population_size_for_diversity = len(generation_data)

        diversity_threshold_ratio = 0.1

        if population_size_for_diversity > 0 and (num_unique_prompts / population_size_for_diversity) < diversity_threshold_ratio:
            current_mutation_factor = self.knowledge_base['ga_parameter_adjustment_suggestions'].get('mutation_rate_factor', 1.0)
            new_mutation_factor = min(1.5, current_mutation_factor * 1.2)
            self.knowledge_base['ga_parameter_adjustment_suggestions']['mutation_rate_factor'] = new_mutation_factor
            logger.info(f"Agent '{self.agent_id}': Low diversity detected ({num_unique_prompts}/{population_size_for_diversity}). Increased mutation_rate_factor to {new_mutation_factor:.2f}.")
        else:
            logger.info(f"Agent '{self.agent_id}': Sufficient diversity ({num_unique_prompts}/{population_size_for_diversity}). No change to mutation_rate_factor based on this heuristic.")

        if self.persist_on_update:
            self.save_knowledge()


    def save_knowledge(self):
        logger.info(f"Agent '{self.agent_id}' attempting to save knowledge to {self.knowledge_file_path}")
        knowledge_dir = os.path.dirname(self.knowledge_file_path)
        if knowledge_dir and not os.path.exists(knowledge_dir):
            try:
                os.makedirs(knowledge_dir, exist_ok=True)
                logger.info(f"Agent '{self.agent_id}': Created directory for knowledge file: {knowledge_dir}")
            except OSError as e:
                logger.error(f"Agent '{self.agent_id}': Error creating directory {knowledge_dir} for knowledge file: {e}. Knowledge might not be saved.", exc_info=True)
                return

        data_to_save = {
            "knowledge_base": self.knowledge_base,
            "data_log": self.data_log
        }
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(data_to_save, f, indent=4)
            logger.info(f"Agent '{self.agent_id}' successfully saved knowledge to {self.knowledge_file_path}")
        except IOError as e:
            logger.error(f"Agent '{self.agent_id}' failed to save knowledge to {self.knowledge_file_path} due to IOError: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"Agent '{self.agent_id}' failed to save knowledge to {self.knowledge_file_path} due to TypeError (data not serializable?): {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}' encountered an unexpected error while saving knowledge to {self.knowledge_file_path}: {e}", exc_info=True)

    def load_knowledge(self):
        logger.info(f"Agent '{self.agent_id}' attempting to load knowledge from {self.knowledge_file_path}")
        if not os.path.exists(self.knowledge_file_path):
            logger.info(
                f"Agent '{self.agent_id}': Knowledge file {self.knowledge_file_path} not found. Starting with an empty knowledge base."
            )
            self.knowledge_base = self._default_knowledge_base_structure.copy()
            self.data_log = []
            self.save_knowledge()
            return

        try:
            with open(self.knowledge_file_path, 'r') as f:
                loaded_data = json.load(f)

            loaded_kb = loaded_data.get("knowledge_base")

            if isinstance(loaded_kb, dict):
                for key, default_value in self._default_knowledge_base_structure.items():
                    if key in loaded_kb and isinstance(loaded_kb[key], type(default_value)):
                        self.knowledge_base[key] = loaded_kb[key]
                    else:
                        logger.warning(f"Agent '{self.agent_id}': Key '{key}' missing or type mismatch in loaded knowledge. Using default for this key.")
                        self.knowledge_base[key] = default_value
                extra_keys = set(loaded_kb.keys()) - set(self._default_knowledge_base_structure.keys())
                if extra_keys:
                    logger.info(f"Agent '{self.agent_id}': Found extra keys in loaded knowledge: {extra_keys}. These will be ignored unless handled.")
            else:
                logger.warning(f"Agent '{self.agent_id}': Loaded knowledge_base is not a dictionary. Using default knowledge base.")
                self.knowledge_base = self._default_knowledge_base_structure.copy()

            self.data_log = loaded_data.get("data_log", [])
            if not isinstance(self.data_log, list):
                logger.warning(f"Agent '{self.agent_id}': Loaded data_log is not a list. Initializing to empty list.")
                self.data_log = []

            logger.info(f"Agent '{self.agent_id}' successfully loaded and validated knowledge from {self.knowledge_file_path}")

        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}' failed to load knowledge due to JSON decoding error: {e}. Starting with an empty knowledge base.",
                exc_info=True,
            )
            self.knowledge_base = self._default_knowledge_base_structure.copy()
            self.data_log = []
        except IOError as e:
            logger.error(
                f"Agent '{self.agent_id}' failed to load knowledge due to IOError: {e}. Starting with an empty knowledge base.",
                exc_info=True,
            )
            self.knowledge_base = self._default_knowledge_base_structure.copy()
            self.data_log = []
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}' encountered an unexpected error while loading knowledge: {e}. Starting with an empty knowledge base.",
                exc_info=True,
            )
            self.knowledge_base = self._default_knowledge_base_structure.copy()
            self.data_log = []


    async def _analyze_evaluation_data(self, eval_data: dict): # Changed to async
        prompt_chromosome = eval_data.get("prompt_chromosome")
        fitness_score = eval_data.get("fitness_score")

        if not isinstance(prompt_chromosome, PromptChromosome) or fitness_score is None:
            logger.error(f"{self.agent_id} - Error: Invalid evaluation data received for LLM analysis.")
            self._fallback_analyze_evaluation_data(eval_data)
            return

        logger.info(
            f"{self.agent_id} - LLM Analyzing evaluation data: Fitness={fitness_score}, Prompt: {prompt_chromosome.genes}"
        )
        
        if fitness_score < 0.5:
            logger.info(f"{self.agent_id} - Skipping LLM analysis for low fitness score: {fitness_score}")
            self._fallback_analyze_evaluation_data(eval_data)
            return

        llm_prompt = f"""
Analyze the following prompt, which achieved a high fitness score of {fitness_score}:
Prompt Genes:
{json.dumps(prompt_chromosome.genes, indent=2)}

Identify potential reasons for its success. What structural or content patterns make it effective?
Focus on generalizable features.
Return your analysis as a JSON list of strings, where each string is a feature or pattern.
Example: ["Clear instruction segment", "Concise context", "Specific output format requested"]
If no clear patterns are identifiable, return an empty list.
"""
        try:
            response_str = await call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model) # Added await
            identified_features = self._parse_llm_list_response(response_str)

            if identified_features:
                for feature in identified_features:
                    pattern_data = {
                        "feature_description": feature,
                        "source_prompt_genes": prompt_chromosome.genes,
                        "fitness": fitness_score,
                        "type": "positive"
                    }
                    self._update_knowledge_base("successful_prompt_features", pattern_data)
            else:
                logger.info(
                    f"{self.agent_id} - LLM did not identify specific success features for this prompt. Using fallback."
                )
                self._fallback_analyze_evaluation_data(eval_data)

        except Exception as e:
            logger.error(f"{self.agent_id} - Error during LLM evaluation analysis: {e}. Using fallback.")
            self._fallback_analyze_evaluation_data(eval_data)

    def _fallback_analyze_evaluation_data(self, eval_data: dict):
        prompt_chromosome = eval_data.get("prompt_chromosome")
        fitness_score = eval_data.get("fitness_score")
        if not isinstance(prompt_chromosome, PromptChromosome) or fitness_score is None: return

        if fitness_score > 0.75:
            pattern = {
                "type": "prompt_features",
                "gene_count": len(prompt_chromosome.genes),
                "keywords_example": [str(g)[:20] + "..." for g in prompt_chromosome.genes[:2]],
                "fitness": fitness_score
            }
            self._update_knowledge_base("legacy_successful_patterns", pattern)


    async def _analyze_critique_data(self, critique_data: dict): # Changed to async
        feedback_points = critique_data.get("feedback_points", [])
        metric_details = critique_data.get("metric_details")
        prompt_chromosome = critique_data.get("prompt_chromosome")

        if metric_details and isinstance(metric_details, dict):
            logger.info(f"Agent '{self.agent_id}': Received programmatic prompt metrics: {metric_details}")
            self._update_knowledge_base("prompt_metric_stats", {
                "prompt_id": str(prompt_chromosome.id) if prompt_chromosome else None,
                "metrics": metric_details,
                "associated_critique_feedback_count": len(feedback_points)
            })

        if feedback_points:
            logger.info(f"Agent '{self.agent_id}': LLM Analyzing qualitative critique data: Feedback Points#={len(feedback_points)}")
            prompt_context_str = ""
            if prompt_chromosome and hasattr(prompt_chromosome, 'genes'):
                prompt_context_str = f"For context, the critiqued prompt (ID: {prompt_chromosome.id}) was:\n{json.dumps(prompt_chromosome.genes, indent=2)}\n"

            llm_prompt = f"""
Analyze the following critique feedback points for a prompt:
Critique Feedback:
{json.dumps(feedback_points, indent=2)}

{prompt_context_str}
Identify common themes or underlying issues suggested by this feedback.
Focus on actionable insights.
Return your analysis as a JSON list of strings, where each string is a theme or issue.
Example: ["Lack of clarity in instructions", "Overly complex context", "Missing output format specification"]
If no clear themes emerge, return an empty list.
"""
            try:
                response_str = await call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model) # Added await
                identified_themes = self._parse_llm_list_response(response_str)

                if identified_themes:
                    for theme in identified_themes:
                        pattern_data = {
                            "critique_theme": theme,
                            "source_feedback_points_count": len(feedback_points),
                            "source_prompt_id": str(prompt_chromosome.id) if prompt_chromosome else None
                        }
                        self._update_knowledge_base("common_critique_themes", pattern_data)
                else:
                    logger.info(f"Agent '{self.agent_id}': LLM did not identify specific themes from qualitative feedback.")
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}': Error during LLM analysis of qualitative critique: {e}. Qualitative themes might not be extracted.", exc_info=True)
        else:
            logger.info(f"Agent '{self.agent_id}': No qualitative feedback points to analyze with LLM.")

        self._fallback_analyze_critique_data(critique_data)


    def _fallback_analyze_critique_data(self, critique_data: dict):
        feedback_points = critique_data.get("feedback_points", [])
        if not feedback_points: return

        logger.info(f"Agent '{self.agent_id}': Running fallback analysis on critique feedback points.")
        for point in feedback_points:
            theme = point.split(":")[0]
            if "Heuristic Violation" in theme or "Structural Issue" in theme or "Programmatic Metric" in theme:
                current_pitfalls = self.knowledge_base["legacy_common_pitfalls"]
                current_pitfalls[theme] = current_pitfalls.get(theme, 0) + 1
                logger.debug(f"Agent '{self.agent_id}': Updated legacy_common_pitfalls for theme '{theme}' to count {current_pitfalls[theme]}.")


    async def _identify_system_patterns(self): # Changed to async
        logger.info(f"{self.agent_id} - LLM Identifying system patterns...")
        logger.info(f"Agent '{self.agent_id}': Identifying system patterns using LLM on qualitative data.")
        sample_successful_features = self.knowledge_base["successful_prompt_features"][-10:]
        sample_critique_themes = self.knowledge_base["common_critique_themes"][-10:]
        sample_legacy_trends = self.knowledge_base["legacy_performance_trends"][-5:]

        if sample_successful_features or sample_critique_themes or sample_legacy_trends:
            llm_prompt = f"""
Analyze the following aggregated findings from a prompt engineering system:
Successful Prompt Features Observed (sample): {json.dumps(sample_successful_features, indent=2)}
Common Critique Themes Observed (sample): {json.dumps(sample_critique_themes, indent=2)}
Previously Identified Legacy Trends (sample): {json.dumps(sample_legacy_trends, indent=2)}

Based on this data, identify 2-3 high-level system-wide patterns, insights, or actionable recommendations for improving the overall prompt generation process.
Return your analysis as a JSON list of strings. Example: ["System tends to generate overly verbose prompts."]
If no clear patterns emerge, return an empty list.
"""
            try:
                response_str = await call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model) # Added await
                identified_llm_trends = self._parse_llm_list_response(response_str)
                if identified_llm_trends:
                    for trend in identified_llm_trends:
                        self._update_knowledge_base("llm_identified_trends", {"trend_description": trend, "source": "llm_qualitative_aggregation"})
                else:
                    logger.info(f"Agent '{self.agent_id}': LLM did not identify new patterns from qualitative data.")
            except Exception as e:
                logger.error(f"Agent '{self.agent_id}': Error during LLM system pattern identification from qualitative data: {e}", exc_info=True)
        else:
            logger.info(f"Agent '{self.agent_id}': Not enough qualitative data for LLM-based system pattern identification.")

        logger.info(f"Agent '{self.agent_id}': Identifying system patterns from programmatic prompt metrics.")
        metric_stats = self.knowledge_base.get("prompt_metric_stats", [])
        if len(metric_stats) >= 5:
            avg_metrics = {}
            for metric_name in ["clarity_score", "completeness_score", "specificity_score", "prompt_length_score"]:
                values = [m_set["metrics"].get(metric_name, 0) for m_set in metric_stats if isinstance(m_set.get("metrics"), dict)]
                if values:
                    avg_metrics[metric_name] = sum(values) / len(values)

            logger.info(f"Agent '{self.agent_id}': Average programmatic metrics: {avg_metrics}")
            for metric_name, avg_value in avg_metrics.items():
                threshold = 0.6
                if avg_value < threshold:
                    trend_desc = f"Average {metric_name.replace('_', ' ')} is relatively low ({avg_value:.2f}). Consider focusing on improving this aspect of prompts."
                    is_duplicate = any(trend.get("trend_description") == trend_desc for trend in self.knowledge_base.get("statistical_prompt_metric_trends", []))
                    if not is_duplicate:
                         self._update_knowledge_base("statistical_prompt_metric_trends", {"trend_description": trend_desc, "average_value": avg_value, "source": "statistical_metrics_average"})
        else:
            logger.info(f"Agent '{self.agent_id}': Not enough data in prompt_metric_stats to derive statistical trends (found {len(metric_stats)} entries).")

        self._fallback_identify_system_patterns()


    def _fallback_identify_system_patterns(self):
        logger.info(f"Agent '{self.agent_id}': Fallback: Identifying system patterns from legacy data...")
        new_trends = []
        successful_gene_counts = [p["gene_count"] for p in self.knowledge_base["legacy_successful_patterns"] if p["type"] == "prompt_features"]
        if successful_gene_counts:
            avg_successful_gene_count = sum(successful_gene_counts) / len(successful_gene_counts)
            trend1 = f"Average gene count for successful legacy prompts: {avg_successful_gene_count:.2f}."
            if trend1 not in self.knowledge_base["legacy_performance_trends"]:
                 new_trends.append(trend1)

        most_common_pitfall = None
        max_count = 0
        for pitfall_theme, count in self.knowledge_base["legacy_common_pitfalls"].items():
            if isinstance(pitfall_theme, str) and count > max_count:
                max_count = count
                most_common_pitfall = pitfall_theme
        if most_common_pitfall and max_count >= 1:
            trend2 = f"Most common legacy pitfall: '{most_common_pitfall}' (occurred {max_count} times)."
            if trend2 not in self.knowledge_base["legacy_performance_trends"]:
                new_trends.append(trend2)
        
        if new_trends:
            for trend in new_trends:
                 self.knowledge_base["legacy_performance_trends"].append(trend)
            logger.info(f"Agent '{self.agent_id}': Fallback: New legacy performance trends identified: {new_trends}")
        else:
            logger.info(f"Agent '{self.agent_id}': Fallback: No new significant legacy system patterns identified.")


    def _parse_llm_list_response(self, response_str: str) -> list:
        try:
            json_start = response_str.find('[')
            json_end = response_str.rfind(']') + 1
            if json_start != -1 and json_end != -1 and json_start < json_end:
                actual_json = response_str[json_start:json_end]
                parsed_list = json.loads(actual_json)
                if isinstance(parsed_list, list):
                    return [str(item) for item in parsed_list]
            return []
        except json.JSONDecodeError:
            logger.warning(f"{self.agent_id} - Failed to parse LLM response as JSON list: {response_str}")
            if "\n" in response_str and len(response_str.split("\n")) > 1:
                return [line.strip("-* ") for line in response_str.split("\n") if line.strip()]
            return []


    def _update_knowledge_base(self, kb_key: str, data_item: any):
        if kb_key in self.knowledge_base:
            if isinstance(self.knowledge_base[kb_key], list):
                self.knowledge_base[kb_key].append(data_item)
                logger.debug(f"{self.agent_id} - Knowledge base '{kb_key}' updated with: {data_item}")
            else:
                logger.warning(
                    f"{self.agent_id} - Warning: Update logic for non-list KB type '{kb_key}' not handled here. Data: {data_item}"
                )
        else:
            logger.warning(f"{self.agent_id} - Warning: Unknown knowledge base category '{kb_key}'.")

    def _generate_recommendations(self) -> list:
        recommendations = []
        logger.info(f"Agent '{self.agent_id}': Generating recommendations from LLM-enhanced and statistical knowledge...")

        for trend_item in self.knowledge_base.get("llm_identified_trends", []):
            desc = trend_item.get("trend_description", trend_item if isinstance(trend_item, str) else None)
            if desc:
                recommendations.append(f"LLM Insight: {desc}")

        for trend_item in self.knowledge_base.get("statistical_prompt_metric_trends", []):
            desc = trend_item.get("trend_description")
            if desc:
                 recommendations.append(f"Metric Trend: {desc}")

        for feature_item in self.knowledge_base.get("successful_prompt_features", [])[-3:]:
            desc = feature_item.get("feature_description")
            if desc:
                recommendations.append(f"Observation: Successful prompts often feature '{desc}'.")

        for theme_item in self.knowledge_base.get("common_critique_themes", [])[-3:]:
            desc = theme_item.get("critique_theme")
            if desc:
                recommendations.append(f"Warning: A common critique theme is '{desc}'. Strive to avoid this.")

        if len(recommendations) < 2 and self.knowledge_base.get("legacy_performance_trends"):
            logger.info(f"Agent '{self.agent_id}': Adding legacy recommendations as current ones are sparse.")
            for trend in self.knowledge_base.get("legacy_performance_trends", []):
                 recommendations.append(f"Legacy System Trend: {trend}")

            legacy_pitfalls = self.knowledge_base.get("legacy_common_pitfalls", {})
            if legacy_pitfalls:
                most_common_legacy_pitfall = max(legacy_pitfalls.items(), key=lambda item: item[1], default=(None,0))
                if most_common_legacy_pitfall[0]:
                    recommendations.append(f"Legacy Pitfall: '{most_common_legacy_pitfall[0]}' was common (count: {most_common_legacy_pitfall[1]}).")

        if not recommendations:
            recommendations.append("No specific new recommendations at this time. Continue monitoring system performance.")
        
        logger.info(f"Agent '{self.agent_id}': Generated recommendations: {recommendations}")
        return list(set(recommendations))


    async def process_request(self, request_data: dict) -> dict: # Changed to async
        data_type = request_data.get("data_type")
        data = request_data.get("data")
        knowledge_updated = False

        if not data_type or not data:
            logger.warning(f"Agent '{self.agent_id}': Missing data_type or data in process_request.")
            return {"status": "Error: Missing data_type or data.", "recommendations": []}

        logger.info(f"Agent '{self.agent_id}' received data. Type: {data_type}")
        self.data_log.append(request_data)
        knowledge_updated = True

        if data_type == "evaluation_result":
            await self._analyze_evaluation_data(data) # Added await
            knowledge_updated = True
        elif data_type == "critique_result":
            await self._analyze_critique_data(data) # Added await
            knowledge_updated = True
        else:
            logger.warning(f"Agent '{self.agent_id}': Unknown data_type '{data_type}'. Data logged but not specifically analyzed.")

        if len(self.data_log) % 3 == 0:
             logger.info(f"Agent '{self.agent_id}': Triggering system pattern identification (log size: {len(self.data_log)}).")
             await self._identify_system_patterns() # Added await
             knowledge_updated = True

        if knowledge_updated and self.persist_on_update:
            self.save_knowledge()

        recommendations = self._generate_recommendations()
        
        return {"status": "Data processed successfully.", "recommendations": recommendations}
