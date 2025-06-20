from prompthelix.agents.base import BaseAgent
from prompthelix.utils.llm_utils import call_llm_api
from prompthelix.config import AGENT_SETTINGS, KNOWLEDGE_DIR
import json # For trying to parse LLM responses
import logging
import os

logger = logging.getLogger(__name__)

# Default provider fallbacks will be resolved during initialization

# Known error strings returned by call_llm_api indicating the call failed
LLM_API_ERROR_STRINGS = {
    "RATE_LIMIT_ERROR", "API_KEY_MISSING_ERROR", "AUTHENTICATION_ERROR",
    "API_CONNECTION_ERROR", "INVALID_REQUEST_ERROR", "API_ERROR", "OPENAI_ERROR",
    "UNEXPECTED_OPENAI_CALL_ERROR", "ANTHROPIC_ERROR", "UNEXPECTED_ANTHROPIC_CALL_ERROR",
    "GOOGLE_SDK_ERROR", "UNEXPECTED_GOOGLE_CALL_ERROR", "BLOCKED_PROMPT_ERROR",
    "API_STATUS_ERROR", "API_SERVER_ERROR", "EMPTY_GOOGLE_RESPONSE", "INVALID_ARGUMENT_ERROR",
    "MALFORMED_CLAUDE_RESPONSE_CONTENT", "EMPTY_CLAUDE_RESPONSE", "UNSUPPORTED_PROVIDER_ERROR",
    "GENERATION_STOPPED_SAFETY", "GENERATION_STOPPED_RECITATION"
}


class DomainExpertAgent(BaseAgent):
    agent_id = "DomainExpert"
    agent_description = "Provides domain-specific knowledge and evaluation criteria."
    """
    Provides domain-specific knowledge, constraints, terminology, and
    evaluation criteria to other agents.
    """
    def __init__(self, message_bus=None, knowledge_file_path: str = "domain_expert_knowledge.json"):
        """
        Initializes the DomainExpertAgent.
        Loads domain-specific knowledge and agent configuration.

        Args:
            message_bus (object, optional): The message bus for inter-agent communication.
            knowledge_file_path (str, optional): Path to the JSON file for knowledge persistence.
                                                 Defaults to "domain_expert_knowledge.json".
        """
        super().__init__(agent_id="DomainExpert", message_bus=message_bus)
        if knowledge_file_path:
            self.knowledge_file_path = (
                knowledge_file_path
                if os.path.isabs(knowledge_file_path)
                else os.path.join(KNOWLEDGE_DIR, knowledge_file_path)
            )
        else:
            self.knowledge_file_path = os.path.join(KNOWLEDGE_DIR, "domain_expert_knowledge.json")
        os.makedirs(os.path.dirname(self.knowledge_file_path), exist_ok=True)

        agent_config = AGENT_SETTINGS.get(self.agent_id, {})
        llm_provider_default = agent_config.get("default_llm_provider", "openai")
        llm_model_default = agent_config.get("default_llm_model", "gpt-3.5-turbo")
        self.llm_provider = llm_provider_default
        self.llm_model = llm_model_default
        logger.info(f"Agent '{self.agent_id}' initialized with LLM provider: {self.llm_provider} and model: {self.llm_model}")

        self.knowledge_base = {} # Initialize before loading
        self.load_knowledge()

    def _get_default_knowledge(self) -> dict:
        """
        Provides a default mock knowledge base with domain-specific information.

        In a real system, this would load from configuration files, databases,
        or dedicated knowledge management systems.

        Returns:
            dict: A dictionary containing domain-specific knowledge.
                  Keys are domain names, values are dicts of info
                  (e.g., keywords, constraints, evaluation_tips).
        """
        logger.info(f"Agent '{self.agent_id}': Using default knowledge base.")
        return {
            "medical": {
                "keywords": ["patient", "diagnosis", "treatment", "symptom", "prognosis", "EHR"],
                "constraints": [
                    "Avoid speculative language.", 
                    "Cite sources if possible.",
                    "Ensure patient confidentiality (simulated for prompts, e.g., use placeholders like [PATIENT_NAME])."
                ],
                "evaluation_tips": [
                    "Check for HIPAA compliance implications (even in simulated contexts).", 
                    "Accuracy and clarity are paramount.",
                    "Verify if advice aligns with current medical guidelines (if applicable/possible)."
                ],
                "sample_prompt_starters": [
                    "Summarize the patient's history regarding [CONDITION]...",
                    "Outline a differential diagnosis for a patient presenting with [SYMPTOMS]...",
                    "Explain the standard treatment protocol for [DISEASE]..."
                ]
            },
            "legal": {
                "keywords": ["contract", "liability", "precedent", "jurisdiction", "plaintiff", "defendant"],
                "constraints": [
                    "Do not provide legal advice (state as disclaimer if needed).",
                    "Reference specific laws or cases where appropriate (if known).",
                    "Maintain formal and precise language."
                ],
                "evaluation_tips": [
                    "Check for factual accuracy of legal statements.",
                    "Ensure arguments are logically sound.",
                    "Verify if interpretations are consistent with provided legal context."
                ],
                 "sample_prompt_starters": [
                    "Analyze the provided contract clause regarding [CLAUSE_TOPIC]...",
                    "Discuss potential liabilities in a scenario where [SCENARIO_DETAILS]...",
                    "Research precedents related to [LEGAL_ISSUE]..."
                ]
            },
            "coding_python": {
                "keywords": ["def", "class", "import", "return", "list comprehension", "decorator", "async"],
                "constraints": [
                    "Follow PEP 8 guidelines.", 
                    "Include docstrings for functions/classes.",
                    "Specify Python version if behavior differs significantly."
                ],
                "evaluation_tips": [
                    "Check for code execution (if possible via a sandbox).", 
                    "Assess efficiency (Big O notation if applicable).", 
                    "Evaluate readability and maintainability.",
                    "Ensure correctness for given problem statement."
                ],
                 "sample_prompt_starters": [
                    "Write a Python function to [TASK_DESCRIPTION]...",
                    "Create a Python class `[CLASS_NAME]` that implements [FEATURES]...",
                    "Refactor the following Python code for better readability: [CODE_SNIPPET]..."
                ]
            },
            "general_knowledge": {
                "keywords": ["explain", "compare", "describe", "list", "pros and cons"],
                "constraints": ["Provide factual information.", "Cite sources if the information is obscure or critical."],
                "evaluation_tips": ["Check for accuracy.", "Assess completeness and clarity of explanation."]
            }
        }

    def load_knowledge(self):
        """
        Loads the knowledge base from the specified JSON file.
        If the file is not found or is invalid, it loads default knowledge.
        """
        try:
            with open(self.knowledge_file_path, 'r') as f:
                self.knowledge_base = json.load(f)
            logger.info(f"Agent '{self.agent_id}': Knowledge base loaded successfully from '{self.knowledge_file_path}'.")
        except FileNotFoundError:
            logger.warning(
                f"Agent '{self.agent_id}': Knowledge file '{self.knowledge_file_path}' not found. Using default knowledge and creating the file."
            )
            self.knowledge_base = self._get_default_knowledge()
            self.save_knowledge()
        except json.JSONDecodeError as e:
            logger.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default knowledge.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Error decoding JSON from '{self.knowledge_file_path}': {e}. Using default knowledge.",
                exc_info=True,
            )
            self.knowledge_base = self._get_default_knowledge()
        except Exception as e:
            logger.error(
                f"Agent '{self.agent_id}': Failed to load knowledge from '{self.knowledge_file_path}': {e}. Using default knowledge.",
                exc_info=True,
            )
            logging.error(
                f"Agent '{self.agent_id}': Failed to load knowledge from '{self.knowledge_file_path}': {e}. Using default knowledge.",
                exc_info=True,
            )
            self.knowledge_base = self._get_default_knowledge()

    def save_knowledge(self):
        """
        Saves the current knowledge base to the specified JSON file.
        """
        try:
            with open(self.knowledge_file_path, 'w') as f:
                json.dump(self.knowledge_base, f, indent=4)
            logger.info(f"Agent '{self.agent_id}': Knowledge base saved successfully to '{self.knowledge_file_path}'.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Failed to save knowledge to '{self.knowledge_file_path}': {e}", exc_info=True)

    def process_request(self, request_data: dict) -> dict:
        """
        Provides domain-specific information based on a query.

        Args:
            request_data (dict): Expected to contain:
                                 'domain' (str): The domain of interest (e.g., "medical", "coding_python").
                                 'query_type' (str): The type of information requested 
                                                     (e.g., "keywords", "constraints", 
                                                     "evaluation_tips", "sample_prompt_starters", "all_info").

        Returns:
            dict: Contains the requested domain information or an error message.
                  Example success: {"domain": "medical", "query_type": "keywords", "data": ["patient", ...]}
                  Example error: {"error": "Domain 'xyz' not found in knowledge base."}
        """
        domain = request_data.get("domain")
        query_type = request_data.get("query_type")

        if not domain or not query_type:
            logger.warning(f"Agent '{self.agent_id}': Missing 'domain' or 'query_type' in request.")
            return {"error": "Missing 'domain' or 'query_type' in request."}

        logger.info(f"Agent '{self.agent_id}': Processing query for domain '{domain}', type '{query_type}' using LLM first (model: {self.llm_model}).")

        # Attempt to get information from LLM
        llm_prompt = self._construct_llm_prompt(domain, query_type)

        try:
            llm_response_str = call_llm_api(llm_prompt, provider=self.llm_provider, model=self.llm_model)
            logger.info(f"Agent '{self.agent_id}': LLM response for domain '{domain}', type '{query_type}': {llm_response_str}")
            if isinstance(llm_response_str, str) and (
                llm_response_str in LLM_API_ERROR_STRINGS or llm_response_str.startswith("GENERATION_STOPPED_")
            ):
                logger.warning(
                    f"Agent '{self.agent_id}': LLM returned error code '{llm_response_str}'. Skipping parse and falling back."
                )
                parsed_data = None
            else:
                parsed_data = self._parse_llm_response(llm_response_str, query_type)
                if query_type == "keywords" and isinstance(parsed_data, dict) and "keywords" not in parsed_data:
                    merged_keywords = []
                    for key in ("key_terms", "jargon", "important_concepts"):
                        terms = parsed_data.get(key)
                        if isinstance(terms, list):
                            merged_keywords.extend(terms)
                    if merged_keywords:
                        parsed_data = merged_keywords

            if parsed_data is not None: # Successfully parsed data from LLM
                return {"domain": domain, "query_type": query_type, "data": parsed_data, "source": "llm"}
            else:
                logger.warning(f"Agent '{self.agent_id}': Failed to parse LLM response meaningfully for '{query_type}'. Falling back to knowledge base.")
        except Exception as e:
            logger.error(f"Agent '{self.agent_id}': Error calling LLM or parsing response for domain '{domain}', type '{query_type}': {e}. Falling back to knowledge base.", exc_info=True)

        # Fallback to static knowledge base
        logger.info(f"Agent '{self.agent_id}': Falling back to knowledge base for domain '{domain}', type '{query_type}'.")
        domain_info = self.knowledge_base.get(domain)

        if not domain_info:
            if "general_knowledge" in self.knowledge_base:
                logger.info(
                    f"Agent '{self.agent_id}': Domain '{domain}' not found, using 'general_knowledge' as fallback."
                )
                domain_info = self.knowledge_base.get("general_knowledge")
                effective_domain_for_fallback = "general_knowledge"
            else:
                logger.warning(
                    f"Agent '{self.agent_id}': Domain '{domain}' not found and no 'general_knowledge' available. Returning empty data."
                )
                domain_info = {}
                effective_domain_for_fallback = domain
        else:
            effective_domain_for_fallback = domain


        if query_type == "all_info":
            return {"domain": effective_domain_for_fallback, "query_type": "all_info", "data": domain_info, "source": "knowledge_base"}
        
        requested_data = domain_info.get(query_type)
        
        if requested_data is None:
            return {"error": f"Query type '{query_type}' not found for domain '{effective_domain_for_fallback}' in knowledge base and LLM fallback failed."}

        return {"domain": effective_domain_for_fallback, "query_type": query_type, "data": requested_data, "source": "knowledge_base"}

    def _construct_llm_prompt(self, domain: str, query_type: str) -> str:
        """Constructs a prompt for the LLM to get domain-specific information."""

        prompt_intro = f"Provide domain-specific information for the domain: '{domain}'."

        if query_type == "keywords":
            return f"{prompt_intro} List key terms, jargon, and important concepts relevant to this domain. Return as a JSON list of strings."
        elif query_type == "constraints":
            return f"{prompt_intro} What are common constraints, rules, or guidelines to follow when generating prompts or content for this domain? Return as a JSON list of strings."
        elif query_type == "evaluation_tips":
            return f"{prompt_intro} What are important tips or criteria for evaluating prompts or content generated for this domain? Return as a JSON list of strings."
        elif query_type == "sample_prompt_starters":
            return f"{prompt_intro} Provide a few diverse examples of prompt starters or initial questions suitable for this domain. Return as a JSON list of strings."
        elif query_type == "all_info":
            return f"""
{prompt_intro}
Provide a comprehensive overview including:
1.  Key terms, jargon, and important concepts (as 'keywords').
2.  Common constraints, rules, or guidelines ('constraints').
3.  Important tips or criteria for evaluation ('evaluation_tips').
4.  Examples of prompt starters ('sample_prompt_starters').

Return this information as a single JSON object with keys: "keywords", "constraints", "evaluation_tips", "sample_prompt_starters".
Each key should map to a list of strings.
"""
        else:
            # Generic query if type is not specifically handled
            return f"{prompt_intro} Specifically, provide information about '{query_type}'. Format the response clearly. If possible, use a JSON list of strings or a JSON object."

    def _parse_llm_response(self, response_str: str, query_type: str):
        """
        Attempts to parse the LLM's string response into a structured format.
        Returns None if parsing fails or response is not suitable.
        """
        response_str = response_str.strip()
        
        # Try to parse as JSON first, as requested in prompts
        try:
            # Attempt to find a JSON block if the LLM includes surrounding text
            json_start = response_str.find('{') # Try object first
            json_end = response_str.rfind('}') + 1
            if json_start != -1 and json_end > json_start: # Basic check for object-like structure
                 data = json.loads(response_str[json_start:json_end])
                 # For "all_info", data should be a dict.
                 # For other types, if LLM returns a dict with the query_type as a key holding a list, that's also good.
                 if query_type == "all_info" and isinstance(data, dict):
                     return data
                 elif isinstance(data, dict) and query_type in data and isinstance(data[query_type], list):
                     return data[query_type] # Return the list for specific query types
                 elif isinstance(data, dict) and query_type not in data and len(data.keys()) == 1:
                     # If only one key, and it's a list, assume it's the data for non "all_info" query
                     single_key = list(data.keys())[0]
                     if isinstance(data[single_key], list) and query_type != "all_info":
                         logger.info(f"Agent '{self.agent_id}': LLM returned dict with single key '{single_key}' for query '{query_type}'. Using its list value.")
                         return data[single_key]
                 # If it's an object but not matching above, it might be problematic for non-"all_info" queries
                 logger.warning(f"Agent '{self.agent_id}': LLM returned a JSON object but not in expected format for query_type '{query_type}'. Response: {data}")
                 return data # Return the dict anyway if it's an object, might be useful for some generic query_type

            json_start_list = response_str.find('[') # Then try list
            json_end_list = response_str.rfind(']') + 1
            if json_start_list != -1 and json_end_list > json_start_list: # Basic check for list-like structure
                data = json.loads(response_str[json_start_list:json_end_list])
                if query_type != "all_info" and isinstance(data, list):
                    return data
                # If query_type is "all_info", a list isn't the expected dict.
                logger.warning(f"Agent '{self.agent_id}': LLM returned a JSON list for query_type '{query_type}', but expected format might differ (e.g. dict for 'all_info'). Response: {data}")
                return data if query_type != "all_info" else None


        except json.JSONDecodeError as e:
            logger.error(f"Agent '{self.agent_id}': JSON parsing failed for LLM response: {e}. Response: {response_str}", exc_info=True)
            if query_type == "all_info": # Stricter for "all_info"
                return None

        # Fallback text parsing for list-like query_types if JSON fails (and not "all_info")
        if query_type != "all_info" and ("\n-" in response_str or "\n*" in response_str or (response_str.count("\n") > 1 and not response_str.strip().startswith("{"))):
            lines = [line.strip().lstrip("-* ").rstrip(",.") for line in response_str.split('\n') if line.strip()]
            if lines:
                logger.info(f"Agent '{self.agent_id}': Parsed LLM response as list of lines for query type '{query_type}'.")
                return lines

        # If still nothing, and not "all_info", return raw response if it seems like a single piece of text
        if query_type != "all_info" and response_str and response_str.lower() not in ["null", "none", "n/a", "not applicable", "[]", "{}"]:
            logger.info(f"Agent '{self.agent_id}': Could not parse LLM response into structured data for '{query_type}'. Returning raw text.")
            return response_str

        logger.warning(f"Agent '{self.agent_id}': Could not meaningfully parse LLM response for query_type '{query_type}'. Original response: {response_str[:200]}...")
        return None

