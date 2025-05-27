from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome # Needed for type hinting and accessing genes

class MetaLearnerAgent(BaseAgent):
    """
    Analyzes the overall performance of the prompt generation and optimization
    process over time. It learns from successful and unsuccessful prompts and
    agents' actions to provide higher-level guidance and adapt strategies.
    """
    def __init__(self):
        """
        Initializes the MetaLearnerAgent.
        Sets up the knowledge base and data log.
        """
        super().__init__(agent_id="MetaLearner")
        self.knowledge_base = {
            "successful_patterns": [], # Stores features of successful prompts
            "common_pitfalls": {},     # Stores common critique feedback points and their counts
            "performance_trends": []   # Stores identified system-wide patterns
        }
        self.data_log = [] # Stores raw data entries for later analysis

    def _analyze_evaluation_data(self, eval_data: dict):
        """
        Analyzes data from ResultsEvaluatorAgent.

        Args:
            eval_data (dict): Data from ResultsEvaluatorAgent, expected to contain
                              'prompt_chromosome' (PromptChromosome) and 'fitness_score' (float).
        """
        prompt_chromosome = eval_data.get("prompt_chromosome")
        fitness_score = eval_data.get("fitness_score")

        if not isinstance(prompt_chromosome, PromptChromosome) or fitness_score is None:
            print(f"{self.agent_id} - Error: Invalid evaluation data received.")
            return

        print(f"{self.agent_id} - Analyzing evaluation data: Fitness={fitness_score}, Prompt Genes={prompt_chromosome.genes}")
        
        # Placeholder logic:
        if fitness_score > 0.75: # Arbitrary threshold for "successful"
            pattern = {
                "type": "prompt_features",
                "gene_count": len(prompt_chromosome.genes),
                "keywords_example": [str(g)[:20] + "..." for g in prompt_chromosome.genes[:2]], # Example feature
                "fitness": fitness_score
            }
            self._update_knowledge_base("successful_patterns", pattern)

    def _analyze_critique_data(self, critique_data: dict):
        """
        Analyzes data from PromptCriticAgent.

        Args:
            critique_data (dict): Data from PromptCriticAgent, expected to contain
                                  'feedback_points' (list of strings).
        """
        feedback_points = critique_data.get("feedback_points", [])
        
        if not feedback_points:
            return

        print(f"{self.agent_id} - Analyzing critique data: Feedback Points#={len(feedback_points)}")

        for point in feedback_points:
            # Simplify feedback point to identify common themes (very basic)
            theme = point.split(":")[0] # E.g., "Structural Issue (PromptTooShort)"
            if "Heuristic Violation" in theme or "Structural Issue" in theme:
                self.knowledge_base["common_pitfalls"][theme] = self.knowledge_base["common_pitfalls"].get(theme, 0) + 1
                self._update_knowledge_base("common_pitfalls", {"theme": theme, "count": self.knowledge_base["common_pitfalls"][theme]})


    def _identify_system_patterns(self):
        """
        Placeholder method to identify broader system patterns from the knowledge base
        or data log. Updates self.knowledge_base["performance_trends"].
        """
        print(f"{self.agent_id} - Identifying system patterns...")
        new_trends = []

        # Example 1: Trend from successful patterns
        successful_gene_counts = [p["gene_count"] for p in self.knowledge_base["successful_patterns"] if p["type"] == "prompt_features"]
        if successful_gene_counts:
            avg_successful_gene_count = sum(successful_gene_counts) / len(successful_gene_counts)
            trend1 = f"Average gene count for successful prompts: {avg_successful_gene_count:.2f}."
            if trend1 not in self.knowledge_base["performance_trends"]:
                 new_trends.append(trend1)

        # Example 2: Trend from common pitfalls
        most_common_pitfall = None
        max_count = 0
        for pitfall_theme, count in self.knowledge_base["common_pitfalls"].items():
            if count > max_count: # Check if this theme is a string, not a dict from _update_knowledge_base
                 if isinstance(pitfall_theme, str) and count > max_count : # Ensure pitfall_theme is a string
                    max_count = count
                    most_common_pitfall = pitfall_theme
        if most_common_pitfall and max_count > 3: # Arbitrary threshold for "common"
            trend2 = f"Most common pitfall observed: '{most_common_pitfall}' (occurred {max_count} times)."
            if trend2 not in self.knowledge_base["performance_trends"]:
                new_trends.append(trend2)
        
        if new_trends:
            for trend in new_trends:
                 self.knowledge_base["performance_trends"].append(trend)
            print(f"{self.agent_id} - New performance trends identified: {new_trends}")
        else:
            print(f"{self.agent_id} - No new significant system patterns identified in this cycle.")


    def _update_knowledge_base(self, pattern_type: str, pattern_data: dict):
        """
        Helper method to add identified patterns or data to the knowledge base.

        Args:
            pattern_type (str): The key in self.knowledge_base (e.g., "successful_patterns").
            pattern_data (dict or any): The data to add.
        """
        if pattern_type in self.knowledge_base:
            if isinstance(self.knowledge_base[pattern_type], list):
                self.knowledge_base[pattern_type].append(pattern_data)
                print(f"{self.agent_id} - Knowledge base '{pattern_type}' updated with: {pattern_data}")
            elif isinstance(self.knowledge_base[pattern_type], dict):
                 # For dicts like common_pitfalls, the update logic is in _analyze_critique_data
                 # This method is more for list-based KB entries or direct updates
                 if "theme" in pattern_data and "count" in pattern_data and pattern_type == "common_pitfalls":
                     # This path is taken if _analyze_critique_data calls this, which it does.
                     # The actual dict update is self.knowledge_base["common_pitfalls"][theme] = ...
                     # So this print is slightly redundant but confirms the call.
                     print(f"{self.agent_id} - Knowledge base '{pattern_type}' (theme: {pattern_data['theme']}) count updated to {pattern_data['count']}.")
                 else:
                     print(f"{self.agent_id} - Warning: Update logic for dict type '{pattern_type}' not fully handled here, pattern_data: {pattern_data}")

        else:
            print(f"{self.agent_id} - Warning: Unknown knowledge base category '{pattern_type}'.")

    def _generate_recommendations(self) -> list:
        """
        Generates recommendations based on the current knowledge base.

        Returns:
            list: A list of recommendation strings.
        """
        recommendations = []
        print(f"{self.agent_id} - Generating recommendations...")

        # Based on performance_trends
        for trend in self.knowledge_base["performance_trends"]:
            if "Average gene count" in trend and "successful prompts" in trend:
                try:
                    avg_gc = float(trend.split(":")[-1].strip().rstrip('.'))
                    if avg_gc > 0:
                         recommendations.append(f"Consider prompts with around {avg_gc:.0f} gene segments, as this is common in successful prompts.")
                except ValueError:
                    pass # Could not parse the trend string
            if "Most common pitfall" in trend:
                pitfall_info = trend.split("'")[1] if "'" in trend else "a common issue"
                recommendations.append(f"Pay attention to '{pitfall_info}', as it's a frequently observed pitfall.")

        # Based on common_pitfalls directly (if not already covered by trends)
        if not any("pitfall" in rec for rec in recommendations): # Avoid redundant pitfall advice
            for pitfall_theme, count in self.knowledge_base["common_pitfalls"].items():
                if isinstance(pitfall_theme, str) and count > 5: # If a pitfall is very common
                    recommendations.append(f"Address the common pitfall: '{pitfall_theme}'.")
                    break # Just one major pitfall recommendation for now

        if not recommendations:
            recommendations.append("No specific new recommendations at this time. Continue monitoring system performance.")
        
        print(f"{self.agent_id} - Generated recommendations: {recommendations}")
        return list(set(recommendations)) # Return unique recommendations


    def process_request(self, request_data: dict) -> dict:
        """
        Processes incoming data, updates knowledge base, and generates recommendations.

        Args:
            request_data (dict): Expected to contain 'data_type' (str) and 'data' (dict).
                                 'data_type' can be "evaluation_result", "critique_result", etc.
                                 'data' contains the specific payload for that data_type.
                                 Example for "evaluation_result":
                                 {
                                     "prompt_chromosome": PromptChromosome(...),
                                     "fitness_score": 0.85,
                                     ...
                                 }
                                 Example for "critique_result":
                                 {
                                     "prompt_chromosome": PromptChromosome(...), # Optional for critic
                                     "feedback_points": ["Feedback 1", "Feedback 2"],
                                     ...
                                 }
        Returns:
            dict: Contains 'status' (str) and 'recommendations' (list of strings).
        """
        data_type = request_data.get("data_type")
        data = request_data.get("data")

        if not data_type or not data:
            return {"status": "Error: Missing data_type or data.", "recommendations": []}

        print(f"{self.agent_id} - Received data. Type: {data_type}")
        self.data_log.append(request_data) # Log the raw data entry

        if data_type == "evaluation_result":
            self._analyze_evaluation_data(data)
        elif data_type == "critique_result":
            self._analyze_critique_data(data)
        # elif data_type == "prompt_design": # Example for future extension
        #     self._analyze_prompt_design_data(data)
        else:
            print(f"{self.agent_id} - Warning: Unknown data_type '{data_type}'. Data logged but not specifically analyzed.")
            return {"status": f"Warning: Unknown data_type '{data_type}'.", "recommendations": self._generate_recommendations()}


        # Periodically or on certain triggers, update overall patterns/knowledge
        # This is a simplified trigger. A more robust system might use time-based triggers
        # or analyze after a certain volume of new, diverse data.
        if len(self.data_log) % 3 == 0: # Analyze patterns every 3 data points for demonstration
             print(f"{self.agent_id} - Triggering system pattern identification (log size: {len(self.data_log)}).")
             self._identify_system_patterns()

        recommendations = self._generate_recommendations()
        
        return {"status": "Data processed successfully.", "recommendations": recommendations}
