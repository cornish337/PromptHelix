from prompthelix.agents.base import BaseAgent

class DomainExpertAgent(BaseAgent):
    """
    Provides domain-specific knowledge, constraints, terminology, and
    evaluation criteria to other agents.
    """
    def __init__(self):
        """
        Initializes the DomainExpertAgent.
        Loads domain-specific knowledge.
        """
        super().__init__(agent_id="DomainExpert")

        self.knowledge_base = self._load_domain_knowledge()

    def _load_domain_knowledge(self) -> dict:
        """
        Loads a mock knowledge base with domain-specific information.

        In a real system, this would load from configuration files, databases,
        or dedicated knowledge management systems.

        Returns:
            dict: A dictionary containing domain-specific knowledge.
                  Keys are domain names, values are dicts of info
                  (e.g., keywords, constraints, evaluation_tips).
        """
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
            return {"error": "Missing 'domain' or 'query_type' in request."}

        print(f"{self.agent_id} - Processing query for domain '{domain}', type '{query_type}'")

        domain_info = self.knowledge_base.get(domain)
        if not domain_info:
            # Fallback to general_knowledge if specific domain not found and query is generic
            if domain not in ["medical", "legal", "coding_python"] and self.knowledge_base.get("general_knowledge"):
                print(f"{self.agent_id} - Domain '{domain}' not found, attempting fallback to 'general_knowledge'.")
                domain_info = self.knowledge_base.get("general_knowledge")
                domain = "general_knowledge" # Update domain to reflect what's being returned
            else:
                return {"error": f"Domain '{domain}' not found in knowledge base."}

        if query_type == "all_info":
            return {"domain": domain, "query_type": "all_info", "data": domain_info}
        
        requested_data = domain_info.get(query_type)
        
        # Check for None explicitly because an empty list or dict might be valid data for some query_types
        if requested_data is None: 
            return {"error": f"Query type '{query_type}' not found for domain '{domain}'."}
        
        return {"domain": domain, "query_type": query_type, "data": requested_data}

