from prompthelix.agents.base import BaseAgent
from prompthelix.genetics.engine import PromptChromosome

class StyleOptimizerAgent(BaseAgent):
    """
    Refines prompts to enhance their style, tone, clarity, and persuasiveness,
    often based on specific target audience or desired communication effect.
    """
    def __init__(self):
        """
        Initializes the StyleOptimizerAgent.
        Loads style transformation rules or lexicons.
        """
        super().__init__(agent_id="StyleOptimizer")

        self.style_rules = self._load_style_rules()

    def _load_style_rules(self) -> dict:
        """
        Loads mock style transformation rules.

        In a real scenario, this would load from a configuration file,
        a database, or be dynamically updated by other agents like MetaLearnerAgent.

        Returns:
            dict: A dictionary of style rules.
        """
        return {
            "formal": {
                "replace": {"don't": "do not", "stuff": "items", "gonna": "going to", "wanna": "want to"},
                "prepend_politeness": "Please ", # Changed from append to prepend for instructions
                "ensure_ending_punctuation": True
            },
            "casual": {
                "replace": {"do not": "don't", "items": "stuff", "please ": "", "kindly ": ""},
                "remove_ending_punctuation": False # Usually casual still has punctuation
            },
            "instructional": { # Example of a more specific style
                "prepend_politeness": "Could you ",
                "append_request_marker": "?", # For instructions that are phrased as questions
                "replace": {"tell me": "explain"}
            }
        }

    def _tone_analysis_adjustment(self, genes: list, target_tone: str) -> list:
        """
        Placeholder for analyzing and adjusting the tone of prompt genes.

        Args:
            genes (list): The list of gene strings.
            target_tone (str): The desired tone (e.g., "neutral", "enthusiastic").

        Returns:
            list: The list of genes, potentially modified for tone.
        """
        print(f"{self.agent_id} - (Placeholder) Analyzing/adjusting tone for: {target_tone}")
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
        print(f"{self.agent_id} - (Placeholder) Enhancing clarity.")
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
        print(f"{self.agent_id} - (Placeholder) Improving persuasiveness.")
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
            print(f"{self.agent_id} - Error: Invalid or missing 'prompt_chromosome' object provided.")
            # Consider raising ValueError or returning original if appropriate
            return original_chromosome 
        
        if not target_style or target_style not in self.style_rules:
            print(f"{self.agent_id} - Warning: Target style '{target_style}' not recognized or not provided. Returning original prompt.")
            return original_chromosome

        print(f"{self.agent_id} - Optimizing prompt (ID: {original_chromosome}) for style: {target_style}")
        
        modified_genes = [str(gene) for gene in original_chromosome.genes] # Work on a copy of gene strings

        style_config = self.style_rules[target_style]
        
        # Apply transformations based on target_style
        if "replace" in style_config:
            for i, gene_str in enumerate(modified_genes):
                for old, new in style_config["replace"].items():
                    modified_genes[i] = gene_str.replace(old, new)
                    if modified_genes[i] != gene_str: # Log change for this specific replacement
                        gene_str = modified_genes[i] # Update gene_str for further replacements in the same gene

        if "prepend_politeness" in style_config and modified_genes:
            # Prepend to the first gene if it looks like an instruction.
            # This is a simplified heuristic.
            first_gene_lower = modified_genes[0].lower()
            instruction_keywords = ["summarize", "generate", "write", "answer", "explain", "describe", "list", "create", "instruct:"]
            if any(keyword in first_gene_lower for keyword in instruction_keywords) and \
               not modified_genes[0].startswith(style_config["prepend_politeness"]):
                modified_genes[0] = style_config["prepend_politeness"] + modified_genes[0]
        
        if "append_request_marker" in style_config and modified_genes:
            # Append to the first gene if it's an instruction that should be a question
            first_gene_lower = modified_genes[0].lower()
            if not modified_genes[0].endswith(style_config["append_request_marker"]):
                 modified_genes[0] = modified_genes[0].rstrip('.!') + style_config["append_request_marker"]


        if style_config.get("ensure_ending_punctuation", False) and modified_genes:
            for i, gene_str in enumerate(modified_genes):
                if gene_str and gene_str[-1] not in ".!?":
                    modified_genes[i] = gene_str + "."


        # Call placeholder internal methods (they currently don't modify genes but could in future)
        modified_genes = self._tone_analysis_adjustment(modified_genes, target_style)
        modified_genes = self._clarity_enhancement(modified_genes)
        modified_genes = self._persuasiveness_improvement(modified_genes)
        
        # Fitness score is typically reset or re-evaluated after modification.
        optimized_chromosome = PromptChromosome(genes=modified_genes, fitness_score=0.0) 
        
        diff = self._compare_chromosomes(original_chromosome, optimized_chromosome)
        if diff:
            print(f"{self.agent_id} - Stylistic changes applied for target style '{target_style}':")
            for d in diff:
                print(f"  - {d}")
        else:
            print(f"{self.agent_id} - No stylistic changes applied for target style '{target_style}'.")

        return optimized_chromosome

