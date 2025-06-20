from __future__ import annotations

import uuid
import copy
import json
import os
import random
import statistics # Added import

from typing import TYPE_CHECKING, Optional, Dict  # Ensure Optional is here
import asyncio  # Added
import openai
from openai import OpenAIError
from prompthelix.config import (
    settings as global_sdk_settings,
)  # Renamed to avoid conflict
from prompthelix import config as global_ph_config  # For LLM_UTILS_SETTINGS
import logging
from prompthelix.enums import ExecutionMode  # Added import

# from typing import Optional, Dict # Already imported above and updated

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from prompthelix.agents.architect import PromptArchitectAgent
    from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent
    from prompthelix.agents.style_optimizer import StyleOptimizerAgent
    from prompthelix.message_bus import MessageBus  # Added for type hinting


# PromptChromosome class remains unchanged
class PromptChromosome:
    """
    Represents an individual prompt in the genetic algorithm.

    Each chromosome consists of a list of 'genes' (which are typically strings
    or more structured objects representing parts of a prompt), a fitness score,
    and a unique identifier.
    """

    def __init__(self, genes: list | None = None, fitness_score: float = 0.0):
        """
        Initializes a PromptChromosome.

        Args:
            genes (list | None, optional): A list representing the components (genes)
                                           of the prompt. Defaults to an empty list if None.
            fitness_score (float, optional): The initial fitness score of the chromosome.
                                             Defaults to 0.0.
        """
        self.id = uuid.uuid4()
        self.genes: list = [] if genes is None else genes
        self.fitness_score: float = fitness_score

    def calculate_fitness(self) -> float:
        """
        Returns the current fitness score of the chromosome.

        Note: This method simply returns the stored fitness_score. The actual
        calculation and setting of this score are typically handled externally by
        a FitnessEvaluator or a similar mechanism within the genetic algorithm,
        which then updates self.fitness_score.

        Returns:
            float: The fitness score of the chromosome.
        """
        return self.fitness_score

    def to_prompt_string(self, separator: str = "\n") -> str:
        """
        Concatenates all gene strings into a single prompt string.

        This string is typically what would be sent to an LLM for execution.

        Args:
            separator (str, optional): The separator to use between genes.
                                       Defaults to a newline character.

        Returns:
            str: A single string representing the full prompt.
        """
        return separator.join(str(gene) for gene in self.genes)

    def clone(self) -> "PromptChromosome":
        """
        Creates a deep copy of this chromosome with a new unique ID.

        The genes are deep-copied to ensure the clone is independent of the
        original. The fitness score is also copied.

        Returns:
            PromptChromosome: A new PromptChromosome instance that is a deep copy
                              of the current one, but with a new ID.
        """
        cloned_genes = copy.deepcopy(self.genes)
        cloned_chromosome = PromptChromosome(
            genes=cloned_genes, fitness_score=self.fitness_score
        )
        return cloned_chromosome

    def __str__(self) -> str:
        """
        Returns a human-readable string representation of the chromosome.

        Returns:
            str: A string detailing the chromosome's ID, fitness, and genes.
        """
        gene_representation = "\n".join([f"  - {str(gene)}" for gene in self.genes])
        if not self.genes:
            gene_representation = "  - (No genes)"
        return (
            f"Chromosome ID: {self.id}\n"
            f"Fitness: {self.fitness_score:.4f}\n"
            f"Genes:\n{gene_representation}"
        )

    def __repr__(self) -> str:
        """
        Returns an unambiguous string representation of the chromosome object.

        Returns:
            str: A string that could ideally be used to recreate the object.
        """
        return f"PromptChromosome(id='{self.id}', genes={self.genes!r}, fitness_score={self.fitness_score:.4f})"


from prompthelix.genetics.mutation_strategies import (
    MutationStrategy,
    AppendCharStrategy,
    ReverseSliceStrategy,
    PlaceholderReplaceStrategy,
    NoOperationMutationStrategy,
)


# GeneticOperators class remains unchanged
class GeneticOperators:
    """
    Encapsulates genetic operators like selection, crossover, and mutation
    for PromptChromosome objects.
    """

    def __init__(
        self,
        style_optimizer_agent: "StyleOptimizerAgent" | None = None,
        mutation_strategies: list[MutationStrategy] | None = None,
    ):
        """
        Initializes the operator with an optional StyleOptimizerAgent and mutation strategies.

        Args:
            style_optimizer_agent (StyleOptimizerAgent | None, optional): Agent for style optimization.
            mutation_strategies (list[MutationStrategy] | None, optional):
                A list of mutation strategies to use. If None, default strategies are initialized.
        """
        self.style_optimizer_agent = style_optimizer_agent

        if mutation_strategies is None:
            # If no list is provided at all, use the default set of strategies
            self.mutation_strategies = [
                AppendCharStrategy(),
                ReverseSliceStrategy(),
                PlaceholderReplaceStrategy(),
            ]
            logger.info(
                "GeneticOperators initialized with default mutation strategies."
            )
        else:
            # If a list is provided (even if empty), use that list
            self.mutation_strategies = mutation_strategies
            if self.mutation_strategies:  # Log if the provided list is not empty
                logger.info(
                    f"GeneticOperators initialized with {len(self.mutation_strategies)} custom mutation strategies."
                )
            # If the provided list IS empty, the next check will handle it.

        # If, after the above, mutation_strategies is an empty list (e.g., user passed []),
        # then default to NoOperationMutationStrategy.
        if not self.mutation_strategies:
            logger.warning(
                "GeneticOperators received an empty list for mutation_strategies. Defaulting to NoOperationMutationStrategy."
            )
            self.mutation_strategies = [NoOperationMutationStrategy()]

    def selection(
        self, population: list[PromptChromosome], tournament_size: int = 3
    ) -> PromptChromosome:
        """
        Selects an individual from the population using tournament selection.

        Args:
            population (list[PromptChromosome]): A list of PromptChromosome objects.
            tournament_size (int, optional): The number of individuals to select
                                             for the tournament. Defaults to 3.

        Returns:
            PromptChromosome: The individual with the highest fitness_score from
                              the tournament.

        Raises:
            ValueError: If population is empty or tournament_size is not positive.
        """
        if not population:
            raise ValueError("Population cannot be empty for selection.")
        if tournament_size <= 0:
            raise ValueError("Tournament size must be positive.")

        actual_tournament_size = min(len(population), tournament_size)
        tournament_contenders = random.sample(population, actual_tournament_size)

        winner = tournament_contenders[0]
        for contender in tournament_contenders[1:]:
            if contender.fitness_score > winner.fitness_score:
                winner = contender

        logger.debug(
            f"Selection (tournament size {actual_tournament_size}): Winner Chromosome ID {winner.id}, Fitness {winner.fitness_score:.4f}"
        )
        return winner

    def crossover(
        self,
        parent1: PromptChromosome,
        parent2: PromptChromosome,
        crossover_rate: float = 0.7,
    ) -> tuple[PromptChromosome, PromptChromosome]:
        """
        Performs single-point crossover between two parent chromosomes.

        If random.random() < crossover_rate, crossover occurs. Otherwise, children
        are clones of the parents.


        Args:
            parent1 (PromptChromosome): The first parent chromosome.
            parent2 (PromptChromosome): The second parent chromosome.
            crossover_rate (float, optional): The probability of crossover occurring.
                                             Defaults to 0.7.

        Returns:
            tuple[PromptChromosome, PromptChromosome]: Two new child chromosomes.
        """
        child1_genes = []
        child2_genes = []
        performed_crossover = False

        if random.random() < crossover_rate:
            performed_crossover = True
            len1 = len(parent1.genes)
            len2 = len(parent2.genes)

            if len1 == 0 and len2 == 0:
                child1_genes, child2_genes = [], []
            elif len1 == 0:
                child1_genes, child2_genes = copy.deepcopy(parent2.genes), []
            elif len2 == 0:
                child1_genes, child2_genes = [], copy.deepcopy(parent1.genes)
            else:
                shorter_parent_len = min(len1, len2)
                # Ensure crossover_point allows for segments from both if shorter_parent_len > 0
                crossover_point = (
                    random.randint(0, shorter_parent_len)
                    if shorter_parent_len > 0
                    else 0
                )

                child1_genes.extend(parent1.genes[:crossover_point])
                child1_genes.extend(parent2.genes[crossover_point:])
                child2_genes.extend(parent2.genes[:crossover_point])
                child2_genes.extend(parent1.genes[crossover_point:])

            child1 = PromptChromosome(genes=child1_genes, fitness_score=0.0)
            child2 = PromptChromosome(genes=child2_genes, fitness_score=0.0)
            logger.debug(
                f"Crossover performed between Parent {parent1.id} and Parent {parent2.id}. Child1 ID {child1.id}, Child2 ID {child2.id}."
            )
        else:
            child1 = parent1.clone()
            child2 = parent2.clone()
            child1.fitness_score = 0.0
            child2.fitness_score = 0.0
            logger.debug(
                f"Crossover skipped (rate {crossover_rate}). Cloned Parent {parent1.id} to Child {child1.id}, Parent {parent2.id} to Child {child2.id}."
            )
        return child1, child2

    def mutate(
        self,
        chromosome: PromptChromosome,
        mutation_rate: float = 0.1,
        gene_mutation_prob: float = 0.2,  # This param is currently unused as strategies manage gene selection
        target_style: str | None = None,
    ) -> PromptChromosome:
        """
        Mutates a chromosome based on mutation_rate and gene_mutation_prob.

        Args:
            chromosome (PromptChromosome): The chromosome to mutate.
            mutation_rate (float, optional): The overall probability that any mutation
                                             will occur on the chromosome. Defaults to 0.1.
            gene_mutation_prob (float, optional): The probability that an individual
                                                  gene will be mutated, if the chromosome
                                                  is selected for mutation. Defaults to 0.2.
        Returns:
            PromptChromosome: A new, potentially mutated, PromptChromosome instance.
        """
        # Clone the chromosome at the beginning. Strategies will work on this clone.
        # Individual strategies also clone, this might be redundant but ensures encapsulation.
        # Let's have strategies operate on the passed chromosome and expect them to return a new, mutated clone.

        mutated_chromosome_overall = chromosome  # Start with original
        mutation_applied_this_cycle = False

        if (
            random.random() < mutation_rate
        ):  # Overall probability of any mutation occurring
            if self.mutation_strategies:
                # Apply one strategy per gene, if gene_mutation_prob is met for that gene
                # This is a change from "one strategy for the whole chromosome" to "potentially multiple strategies if multiple genes mutate"
                # Or, pick one strategy and apply it if any gene is chosen for mutation. Let's stick to the latter for now.

                # Create a working clone that strategies will modify or replace
                working_chromosome_clone = chromosome.clone()
                working_chromosome_clone.fitness_score = 0.0  # Reset fitness

                at_least_one_gene_mutated = False
                for i in range(len(working_chromosome_clone.genes)):
                    if (
                        random.random() < gene_mutation_prob
                    ):  # Probability of this specific gene mutating
                        selected_strategy = random.choice(self.mutation_strategies)

                        # Create a temporary chromosome for the single gene to pass to the strategy
                        # This is a bit clunky; strategies might need to be aware of gene indices or operate on gene lists.
                        # For now, let's assume strategies are designed to mutate one gene of a chromosome.
                        # The current strategies (AppendChar, ReverseSlice, PlaceholderReplace) pick a random gene.
                        # This means applying a strategy might mutate a *different* gene than the one selected by index i.
                        # This needs refinement.

                        # Refined approach: Pass the working_chromosome_clone and let the strategy pick a gene.
                        # If a strategy is applied, we consider the chromosome mutated.
                        # This is simpler and closer to the original intent where one mutation operation was chosen.

                        # Let's revert to: if mutation is to occur, pick ONE strategy and apply it to the chromosome.
                        # The strategy itself will then pick which gene(s) to mutate internally if it's gene-specific.
                        pass  # This loop for gene_mutation_prob is not how it should be with current strategies

                # Corrected logic: If chromosome is selected for mutation, pick one strategy and apply it.
                # The chosen strategy will internally handle how it mutates (e.g. which gene).
                selected_strategy = random.choice(self.mutation_strategies)
                logger.debug(
                    f"Applying mutation strategy '{selected_strategy.__class__.__name__}' to chromosome {working_chromosome_clone.id}"
                )
                mutated_chromosome_overall = selected_strategy.mutate(
                    working_chromosome_clone
                )  # Pass the clone
                mutation_applied_this_cycle = True  # A strategy was applied

                # The old logic of appending "*" if no gene was modified by the above loop:
                # This should be handled by ensuring strategies always make a change,
                # or by having a specific strategy that does this (e.g. "MinorPerturbationStrategy").
                # For now, if a strategy is selected, we assume it mutates.
                # The NoOperationMutationStrategy can be used if no change is desired sometimes.
            else:
                logger.warning(
                    "Mutation selected to occur, but no mutation strategies are defined in GeneticOperators."
                )
                mutated_chromosome_overall = (
                    chromosome.clone()
                )  # Still clone, reset fitness
                mutated_chromosome_overall.fitness_score = 0.0

        # If no mutation was applied by random chance (missed mutation_rate),
        # return a fresh clone with reset fitness.
        if not mutation_applied_this_cycle:
            mutated_chromosome_overall = chromosome.clone()
            mutated_chromosome_overall.fitness_score = 0.0
            # No specific mutation strategy was chosen via mutation_rate

        # Style optimization step (if applicable)
        # This happens *after* a primary mutation strategy has been applied (or not, if mutation_rate was missed)
        if target_style and self.style_optimizer_agent:
            # If mutation_applied_this_cycle is false, mutated_chromosome_overall is just a clone.
            # We might only want to apply style optimization if a structural mutation happened.
            # Or, style optimization could be a mutation strategy itself.
            # For now, let's assume it can be applied even to an unmutated (but cloned) chromosome.
            try:
                # Ensure the chromosome passed to style optimizer is the one potentially mutated
                request = {
                    "prompt_chromosome": mutated_chromosome_overall,
                    "target_style": target_style,
                }
                optimized_chromosome = self.style_optimizer_agent.process_request(
                    request
                )
                if isinstance(optimized_chromosome, PromptChromosome):
                    # The style optimizer should return a new chromosome instance.
                    # Fitness of this new chromosome should ideally be reset by the optimizer,
                    # or it should be reset here if the optimizer doesn't guarantee it.
                    # For now, we assume the optimizer returns a valid chromosome,
                    # and we'll ensure its fitness is 0.0 for the new generation.
                    mutated_chromosome_overall = optimized_chromosome
                    mutated_chromosome_overall.fitness_score = (
                        0.0  # Ensure fitness is reset
                    )
                    logger.info(
                        f"Chromosome {mutated_chromosome_overall.id} successfully style-optimized to target style '{target_style}'."
                    )
                else:
                    logger.warning(
                        f"StyleOptimizerAgent did not return a PromptChromosome for chromosome {mutated_chromosome_overall.id} "
                        f"(received {type(optimized_chromosome)}). Skipping style optimization."
                    )
            except Exception as e:
                logger.error(
                    f"Style optimization failed during mutation for chromosome {mutated_chromosome_overall.id} "
                    f"with target style '{target_style}': {e}",
                    exc_info=True,
                )
        elif target_style and not self.style_optimizer_agent:
            logger.warning(
                f"Target style '{target_style}' provided for mutation, but StyleOptimizerAgent is not available. Skipping style optimization."
            )

        return mutated_chromosome_overall


# FitnessEvaluator class remains unchanged
class FitnessEvaluator:
    """
    Evaluates the fitness of PromptChromosome instances.
    This class simulates interaction with an LLM and uses a ResultsEvaluatorAgent
    to determine the fitness score based on the LLM's output.
    """

    def __init__(
        self,
        results_evaluator_agent: "ResultsEvaluatorAgent",
        execution_mode: ExecutionMode,
        llm_settings: Optional[Dict] = None,
    ):  # New parameter
        """
        Initializes the FitnessEvaluator.
        Args:
            results_evaluator_agent (ResultsEvaluatorAgent): An instance of
                ResultsEvaluatorAgent that will be used to assess the quality
                of LLM outputs.
            execution_mode (ExecutionMode): The mode of execution (TEST or REAL).
            llm_settings (Optional[Dict]): Overridden LLM settings.
        """
        from prompthelix.agents.results_evaluator import ResultsEvaluatorAgent

        if not isinstance(results_evaluator_agent, ResultsEvaluatorAgent):
            raise TypeError(
                "results_evaluator_agent must be an instance of ResultsEvaluatorAgent."
            )

        self.results_evaluator_agent = (
            results_evaluator_agent  # Stored for main process
        )
        self.execution_mode = execution_mode

        # Merge provided llm_settings with global defaults
        base_llm_settings = copy.deepcopy(
            global_ph_config.LLM_UTILS_SETTINGS.get("openai", {})
        )  # Assuming 'openai' is the provider here
        if llm_settings:
            # Using a simple update; for deep merge, import and use update_settings utility
            # from prompthelix.utils.config_utils import update_settings
            # self.llm_settings = update_settings(base_llm_settings, llm_settings)
            # For now, simple update for direct keys like api_key, timeout from the passed llm_settings.
            # A more robust solution would involve the update_settings utility for deep merge.
            temp_settings = base_llm_settings.copy()
            temp_settings.update(
                llm_settings
            )  # Shallow update, override top-level keys
            self.llm_settings = temp_settings
        else:
            self.llm_settings = base_llm_settings

        logger.debug(f"FitnessEvaluator effective LLM settings: {self.llm_settings}")

        # Store info for re-instantiation in subprocesses
        self.results_evaluator_agent_class = results_evaluator_agent.__class__
        self.results_evaluator_agent_knowledge_file = getattr(
            results_evaluator_agent, "settings", {}
        ).get("knowledge_file_path")

        self.openai_client = None
        if self.execution_mode == ExecutionMode.REAL:
            # Prioritize API key from llm_settings, then global_sdk_settings
            api_key_to_use = self.llm_settings.get(
                "api_key", global_sdk_settings.OPENAI_API_KEY
            )

            if api_key_to_use:
                try:
                    # Pass other params from self.llm_settings if OpenAI client accepts them
                    client_params = {
                        "api_key": api_key_to_use,
                        "timeout": self.llm_settings.get(
                            "default_timeout", openai.DefaultHttpxClient.DEFAULT_TIMEOUT
                        ),  # openai library default
                        # Add other relevant params like 'max_retries', 'organization' if in settings
                    }
                    if "max_retries" in self.llm_settings:
                        client_params["max_retries"] = self.llm_settings["max_retries"]

                    self.openai_client = openai.OpenAI(**client_params)
                    logger.info(
                        "FitnessEvaluator: OpenAI client initialized successfully for REAL mode in main process."
                    )
                except Exception as e:
                    logger.error(
                        f"FitnessEvaluator: Error initializing OpenAI client in main process: {e}",
                        exc_info=True,
                    )
                    self.openai_client = None
            else:
                logger.error(
                    "FitnessEvaluator: OpenAI API Key not found in settings or global config. LLM calls in REAL mode will fail."
                )
        else:  # TEST mode
            logger.info(
                "FitnessEvaluator: Initialized in TEST mode. LLM calls will be skipped by _call_llm_api."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        # llm_settings is serializable, so it can stay
        # Remove attributes that cannot or should not be pickled
        if "results_evaluator_agent" in state:
            del state["results_evaluator_agent"]
        if "openai_client" in state:
            del state["openai_client"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        # Re-initialize results_evaluator_agent
        # Logger should be available as it's a module-level global
        if (
            hasattr(self, "results_evaluator_agent_class")
            and self.results_evaluator_agent_class
        ):
            try:
                # Pass agent settings if they were part of the original agent's setup
                agent_settings = getattr(self.results_evaluator_agent, "settings", None)
                self.results_evaluator_agent = self.results_evaluator_agent_class(
                    message_bus=None,
                    settings=agent_settings,  # Pass original settings
                    knowledge_file_path=self.results_evaluator_agent_knowledge_file,  # Still needed if not in settings
                )
                self.results_evaluator_agent.db = (
                    None  # Ensure db is None in subprocess
                )
                logger.info(
                    "FitnessEvaluator (subprocess): ResultsEvaluatorAgent re-initialized."
                )
            except Exception as e:
                logger.error(
                    f"FitnessEvaluator (subprocess): Failed to re-initialize ResultsEvaluatorAgent: {e}",
                    exc_info=True,
                )
                self.results_evaluator_agent = None
        else:
            logger.error(
                "FitnessEvaluator (subprocess): results_evaluator_agent_class not found in state. Cannot re-initialize agent."
            )
            self.results_evaluator_agent = None

        # Re-initialize openai_client using self.llm_settings
        self.openai_client = None
        if self.execution_mode == ExecutionMode.REAL:
            api_key_to_use = self.llm_settings.get(
                "api_key", global_sdk_settings.OPENAI_API_KEY
            )
            if api_key_to_use:
                try:
                    client_params = {
                        "api_key": api_key_to_use,
                        "timeout": self.llm_settings.get(
                            "default_timeout", openai.DefaultHttpxClient.DEFAULT_TIMEOUT
                        ),
                    }
                    if "max_retries" in self.llm_settings:
                        client_params["max_retries"] = self.llm_settings["max_retries"]
                    self.openai_client = openai.OpenAI(**client_params)
                    logger.info(
                        "FitnessEvaluator (subprocess): OpenAI client re-initialized for REAL mode."
                    )
                except Exception as e:
                    logger.error(
                        f"FitnessEvaluator (subprocess): Error re-initializing OpenAI client for REAL mode: {e}",
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "FitnessEvaluator (subprocess): OpenAI API key not found. Cannot re-initialize client for REAL mode."
                )
        else:
            logger.info(
                "FitnessEvaluator (subprocess): In TEST mode, LLM client not re-initialized."
            )

    def _call_llm_api(
        self, prompt_string: str, model_name: Optional[str] = None
    ) -> str:
        """
        Calls the LLM API with the given prompt string using settings from self.llm_settings.
        Args:
            prompt_string (str): The prompt to send to the LLM.
            model_name (str, optional): Specific model to use, overrides default from settings.
        Returns:
            str: The LLM's response content, or an error message string if the call fails.
        """
        if self.execution_mode == ExecutionMode.TEST:
            logger.info(
                f"Executing in TEST mode. Returning dummy LLM output for prompt: {prompt_string[:100]}..."
            )
            return "This is a test output from dummy LLM in TEST mode."

        if not self.openai_client:
            logger.error(
                "OpenAI client is not initialized. Cannot call LLM API in REAL mode."
            )
            return "Error: LLM client not initialized for REAL mode."

        # Determine model: use provided, then from self.llm_settings, then fallback
        current_model_name = (
            model_name or self.llm_settings.get("default_model") or "gpt-3.5-turbo"
        )

        # Get other call parameters from self.llm_settings
        timeout = self.llm_settings.get(
            "default_timeout", 60
        )  # Default timeout if not in settings
        max_tokens = self.llm_settings.get(
            "max_tokens", 150
        )  # Example, adjust as needed
        temperature = self.llm_settings.get("temperature", 0.7)  # Example

        logger.info(
            f"Calling OpenAI API model {current_model_name} for prompt (first 100 chars): {prompt_string[:100]}..."
        )
        logger.debug(
            f"LLM call params: timeout={timeout}, max_tokens={max_tokens}, temperature={temperature}"
        )
        try:
            response = self.openai_client.chat.completions.create(
                model=current_model_name,
                messages=[{"role": "user", "content": prompt_string}],
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                # Add other parameters like top_p, frequency_penalty, presence_penalty if in self.llm_settings
            )
            if (
                response.choices
                and response.choices[0].message
                and response.choices[0].message.content
            ):
                content = response.choices[0].message.content.strip()
                logger.info(
                    f"OpenAI API call successful. Response (first 100 chars): {content[:100]}..."
                )
                return content
            else:
                logger.warning(
                    f"OpenAI API call for prompt '{prompt_string[:50]}...' returned no content or unexpected response structure."
                )
                return "Error: No content from LLM."
        except OpenAIError as e:
            logger.error(
                f"OpenAI API error for prompt '{prompt_string[:50]}...': {e}",
                exc_info=True,
            )
            return f"Error: LLM API call failed. Details: {str(e)}"
        except Exception as e:  # Catch any other unexpected errors
            logger.critical(
                f"Unexpected error during OpenAI API call for prompt '{prompt_string[:50]}...': {e}",
                exc_info=True,
            )
            return f"Error: Unexpected issue during LLM API call. Details: {str(e)}"

    def evaluate(
        self,
        chromosome: PromptChromosome,
        task_description: str,
        success_criteria: dict | None = None,
    ) -> float:
        """
        Evaluates the fitness of a given chromosome.
        This involves converting the chromosome to a prompt string, simulating
        an LLM call with this prompt, and then using the ResultsEvaluatorAgent
        to score the LLM's output. The chromosome's fitness_score attribute
        is updated with the result.
        Args:
            chromosome (PromptChromosome): The chromosome to evaluate.
            task_description (str): A description of the task the prompt is for.
            success_criteria (dict | None, optional): Criteria for evaluating the
                success of the LLM output. Defaults to None.
        Returns:
            float: The calculated fitness score for the chromosome.
        """
        if not isinstance(chromosome, PromptChromosome):
            raise TypeError("chromosome must be an instance of PromptChromosome.")
        prompt_string = chromosome.to_prompt_string()

        # The call to _call_llm_api is now logged within that method.
        llm_output = self._call_llm_api(prompt_string)

        if llm_output.startswith("Error:"):
            logger.warning(
                f"LLM call for prompt ID {chromosome.id} (text: {prompt_string[:50]}...) failed. Output: {llm_output}"
            )
        # else:
        # Successful LLM output is logged in _call_llm_api.
        # If further logging of the output snippet is desired here, it can be added.
        # logger.debug(f"FitnessEvaluator: LLM Output for prompt ID {chromosome.id}: {llm_output[:150]}...")

        request_data = {
            "prompt_chromosome": chromosome,
            "llm_output": llm_output,  # Pass the actual or error string from LLM
            "task_description": task_description,
            "success_criteria": success_criteria if success_criteria else {},
        }
        eval_result = self.results_evaluator_agent.process_request(request_data)

        chromosome.fitness_score = eval_result.get("fitness_score", 0.0)
        chromosome.evaluation_details = eval_result.get(
            "detailed_metrics", {}
        )  # Corrected key

        # Debugging: Check for LLM analysis status from ResultsEvaluatorAgent
        logger.info(
            f"FIT_EVAL: Chromosome ID {chromosome.id} -- evaluation_details before status check: {str(chromosome.evaluation_details)}"
        )
        llm_analysis_status = None
        feedback_message = "N/A"

        if chromosome.evaluation_details and isinstance(
            chromosome.evaluation_details, dict
        ):
            llm_analysis_status = chromosome.evaluation_details.get(
                "llm_analysis_status"
            )
            feedback_message = chromosome.evaluation_details.get(
                "llm_assessment_feedback", "N/A"
            )
            logger.info(
                f"FIT_EVAL: Chromosome ID {chromosome.id} -- Found evaluation_details. llm_analysis_status: {llm_analysis_status}, feedback: {feedback_message}"
            )
        else:
            logger.warning(
                f"FIT_EVAL: Chromosome ID {chromosome.id} -- chromosome.evaluation_details is None, not a dict, or empty. Value: {str(chromosome.evaluation_details)}"
            )

        # Original logging logic based on the possibly updated llm_analysis_status
        if llm_analysis_status and llm_analysis_status != "success":
            logger.info(
                f"FitnessEvaluator: Chromosome {chromosome.id} evaluated using fallback LLM metrics. "  # Standard log message
                f"Status: '{llm_analysis_status}'. Assigned fitness: {chromosome.fitness_score:.4f}. "
                f"Feedback: {feedback_message}"
            )
        elif (
            not llm_analysis_status
        ):  # This case implies evaluation_details might be missing or status key itself is absent
            logger.warning(
                f"FitnessEvaluator: 'llm_analysis_status' key was not found or was None in evaluation_details for Chromosome {chromosome.id}. "  # Standard log message
                f"Cannot determine if fallback LLM metrics were used. evaluation_details content: {str(chromosome.evaluation_details)[:200]}..."
            )

        # Final log line for summary
        logger.info(
            f"FitnessEvaluator: Evaluated chromosome {chromosome.id}, Assigned Fitness: {chromosome.fitness_score:.4f}, LLM Analysis Status Logged: {llm_analysis_status if llm_analysis_status else 'Status Unknown (Final Log)'}"
        )

        return chromosome.fitness_score


# Updated PopulationManager class
class PopulationManager:
    """
    Manages the population of prompts, including initialization and evolution
    through generations using genetic operators and fitness evaluation.
    """

    def __init__(
        self,
        genetic_operators: GeneticOperators,
        fitness_evaluator: FitnessEvaluator,
        prompt_architect_agent: "PromptArchitectAgent",
        population_size: int = 50,
        elitism_count: int = 2,
        population_path: str | None = None,
        initial_prompt_str: str | None = None,  # New parameter
        parallel_workers: Optional[int] = None,
        evaluation_timeout: Optional[int] = 60,
        message_bus: Optional["MessageBus"] = None,  # Added message_bus
        agents_used: list[str] | None = None,
    ):
        """
        Initializes the PopulationManager.

        Args:
            genetic_operators (GeneticOperators): Instance of GeneticOperators.
            fitness_evaluator (FitnessEvaluator): Instance of FitnessEvaluator.
            prompt_architect_agent (PromptArchitectAgent): Instance for creating initial prompts.
            population_size (int, optional): The desired size of the population. Defaults to 50.
            elitism_count (int, optional): The number of top individuals to carry over
                                           to the next generation without modification. Defaults to 2.
            population_path (str | None, optional): Path to load/save population JSON.
                If provided and the file exists, the population will be loaded
                from this file on initialization.
            initial_prompt_str (str | None, optional): An initial prompt string to seed
                one chromosome in the population. Defaults to None.
            agents_used (list[str] | None, optional): A list of agent IDs used in the process.
                                                     Defaults to None, resulting in an empty list.
        """

        if not isinstance(genetic_operators, GeneticOperators):
            raise TypeError(
                "genetic_operators must be an instance of GeneticOperators."
            )
        if not isinstance(fitness_evaluator, FitnessEvaluator):
            raise TypeError(
                "fitness_evaluator must be an instance of FitnessEvaluator."
            )
        from prompthelix.agents.architect import PromptArchitectAgent

        if not isinstance(prompt_architect_agent, PromptArchitectAgent):
            raise TypeError(
                "prompt_architect_agent must be an instance of PromptArchitectAgent."

            )
        if population_size <= 0:
            raise ValueError("Population size must be positive.")
        if elitism_count < 0 or elitism_count > population_size:
            raise ValueError(
                "Elitism count must be non-negative and not exceed population size."
            )
        if parallel_workers is not None and parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive if specified.")
        if evaluation_timeout is not None and evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive if specified.")

        self.genetic_operators = genetic_operators
        self.fitness_evaluator = fitness_evaluator
        self.prompt_architect_agent = prompt_architect_agent
        self.population_size = population_size
        self.elitism_count = elitism_count
        self.population_path = population_path
        self.initial_prompt_str = initial_prompt_str  # Store the new parameter
        self.parallel_workers = parallel_workers
        self.evaluation_timeout = evaluation_timeout
        self.message_bus = message_bus  # Added
        self.agents_used: list[str] = (
            agents_used if agents_used is not None else []
        )  # Added

        self.population: list[PromptChromosome] = []
        self.generation_number: int = 0

        self.is_paused: bool = False  # Added
        self.should_stop: bool = False  # Added
        self.status: str = "IDLE"  # Added

        if self.population_path:
            self.load_population(self.population_path)

        self.broadcast_ga_update(
            event_type="ga_manager_initialized"
        )  # Initial broadcast

    def pause_evolution(self):  # Added
        self.is_paused = True
        self.status = "PAUSED"
        logger.info("GA evolution paused.")
        self.broadcast_ga_update(event_type="ga_paused")

    def resume_evolution(self):  # Added
        self.is_paused = False
        self.status = "RUNNING"  # Or should it wait for evolve_population to set it?
        logger.info("GA evolution resumed.")
        self.broadcast_ga_update(event_type="ga_resumed")

    def stop_evolution(self):  # Added
        self.should_stop = True
        self.is_paused = False  # Ensure it's not stuck in pause
        self.status = "STOPPING"
        logger.info("GA evolution stopping...")
        self.broadcast_ga_update(event_type="ga_stopping")

    def broadcast_ga_update(
        self,
        event_type: str = "ga_status_update",
        additional_data: Optional[dict] = None,
    ):  # Added
        if not self.message_bus or not self.message_bus.connection_manager:
            # logger.debug("Message bus or connection manager not available for GA update broadcast.")
            return

        fittest = self.get_fittest_individual()
        payload = {
            "status": self.status,
            "generation": self.generation_number,
            "population_size": len(self.population) if self.population else 0,
            "best_fitness": fittest.fitness_score if fittest else None,
            "is_paused": self.is_paused,
            "should_stop": self.should_stop,
            "agents_used": self.agents_used,
            "fittest_chromosome_string": (
                fittest.to_prompt_string() if fittest else None
            ),
        }
        if additional_data:
            payload.update(additional_data)

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(
                    self.message_bus.connection_manager.broadcast_json(
                        {"type": event_type, "data": payload}
                    )
                )
            else:
                loop.run_until_complete(
                    self.message_bus.connection_manager.broadcast_json(
                        {"type": event_type, "data": payload}
                    )
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    self.message_bus.connection_manager.broadcast_json(
                        {"type": event_type, "data": payload}
                    )
                )
            finally:
                loop.close()
        except Exception as e:
            logger.error(
                f"Unexpected error during GA update broadcast: {e}", exc_info=True
            )

    def initialize_population(
        self,
        initial_task_description: str,
        initial_keywords: list | None = None,
        initial_constraints: dict | None = None,
    ):
        """
        Initializes the population with new prompts. If `self.initial_prompt_str`
        is provided, one chromosome will be seeded from it. The rest are
        created by the PromptArchitectAgent.

        Args:
            initial_task_description (str): The task description for the initial prompts.
            initial_keywords (list | None, optional): Keywords for initial prompts. Defaults to None.
            initial_constraints (dict | None, optional): Constraints for initial prompts. Defaults to None.
        """
        self.status = "INITIALIZING"  # Added
        self.broadcast_ga_update(event_type="ga_initialization_started")  # Added

        logger.info(
            f"PopulationManager: Initializing population of size {self.population_size} for task: '{initial_task_description}'"
        )
        self.population = []

        num_to_generate_randomly = self.population_size

        if self.initial_prompt_str and self.population_size > 0:
            logger.info(
                f"PopulationManager: Seeding one chromosome from provided initial_prompt_str: '{self.initial_prompt_str[:100]}...'"
            )
            # Treat the entire string as a single gene for simplicity
            seeded_chromosome = PromptChromosome(genes=[self.initial_prompt_str])
            self.population.append(seeded_chromosome)
            num_to_generate_randomly -= 1
            if num_to_generate_randomly < 0:  # Should not happen if population_size > 0
                num_to_generate_randomly = 0

        actual_initial_keywords = (
            initial_keywords if initial_keywords is not None else []
        )
        actual_initial_constraints = (
            initial_constraints if initial_constraints is not None else {}
        )

        for i in range(num_to_generate_randomly):
            request_data = {
                "task_description": initial_task_description,
                "keywords": copy.deepcopy(actual_initial_keywords),
                "constraints": copy.deepcopy(actual_initial_constraints),
            }

            chromosome = self.prompt_architect_agent.process_request(request_data)
            if not isinstance(chromosome, PromptChromosome):
                logger.warning(
                    f"PopulationManager: PromptArchitectAgent did not return a PromptChromosome. Got: {type(chromosome)}. Skipping this one."
                )
                continue  # Consider how to handle this to ensure population size is met

            self.population.append(chromosome)

        # Ensure population size is exactly self.population_size, potentially by adding more if architect failed
        # or truncating if somehow too many were added (though current logic prevents over-addition)
        while len(self.population) < self.population_size and self.population_size > 0:
            logger.warning(
                f"PopulationManager: Population size {len(self.population)} is less than target {self.population_size} after initial generation. Attempting to add more."
            )
            # This might happen if PromptArchitectAgent fails to return chromosomes for some iterations
            # and initial_prompt_str was also used.
            request_data = {
                "task_description": initial_task_description,
                "keywords": copy.deepcopy(actual_initial_keywords),
                "constraints": copy.deepcopy(actual_initial_constraints),
            }
            chromosome = self.prompt_architect_agent.process_request(request_data)
            if isinstance(chromosome, PromptChromosome):
                self.population.append(chromosome)
            else:
                logger.error(
                    "PopulationManager: PromptArchitectAgent failed again during fill. Population may be undersized."
                )
                break  # Avoid infinite loop if architect keeps failing

        if len(self.population) > self.population_size:
            self.population = self.population[: self.population_size]

        if len(self.population) != self.population_size and self.population_size > 0:
            logger.warning(
                f"PopulationManager: Final initialized population size {len(self.population)} does not match target {self.population_size}."
            )

        self.generation_number = 0
        logger.info(
            f"PopulationManager: Population initialized. Generation: {self.generation_number}, Size: {len(self.population)}"
        )

        self.status = "IDLE"  # Or "READY_TO_EVOLVE" # Added
        self.broadcast_ga_update(event_type="ga_initialization_complete")  # Added

    def evolve_population(
        self,
        task_description: str,
        success_criteria: dict | None = None,
        target_style: str | None = None,
    ):
        """
        Orchestrates one generation of evolution: evaluation, selection, crossover, and mutation.

        Args:
            task_description (str): The task description for fitness evaluation.
            success_criteria (dict | None, optional): Success criteria for fitness evaluation. Defaults to None.
            target_style (str | None, optional): Desired style used during mutation when
                StyleOptimizerAgent is available. Defaults to None.
        """
        if self.should_stop:  # Moved this check up, and modified its behavior
            logger.info(
                f"PopulationManager.evolve_population: Stop requested before starting generation {self.generation_number + 1}. Aborting evolution for this generation."
            )
            # Do not change self.status here, let the orchestrator loop handle final status.
            # self.status = "STOPPED" # This was here, but task says orchestrator handles final status
            # self.broadcast_ga_update(event_type="ga_stopped_before_generation") # This was here
            return  # Exit early

        if self.is_paused:  # This check is fine
            logger.info("Evolution is paused, skipping generation.")
            # broadcast_ga_update is called by pause_evolution (which sets status to PAUSED)
            return

        self.status = "RUNNING"  # Added
        self.broadcast_ga_update(
            event_type="ga_generation_started",
            additional_data={"generation": self.generation_number + 1},
        )  # Added

        if not self.population:
            logger.warning(
                "PopulationManager: Cannot evolve an empty population. Please initialize first."
            )
            self.status = "ERROR"  # Or IDLE
            self.broadcast_ga_update(
                event_type="ga_error", additional_data={"error": "Empty population"}
            )
            return

        current_generation_number = self.generation_number + 1
        logger.info(
            f"PopulationManager: Starting evolution for generation {current_generation_number}. Population size: {len(self.population)}"
        )

        # 1. Evaluate Population
        logger.info(
            f"Generation {current_generation_number}: Evaluating fitness of {len(self.population)} chromosomes."
        )

        evaluated_chromosomes_count = 0
        failed_evaluations_count = 0

        # Ensure ProcessPoolExecutor is imported only when needed
        from concurrent.futures import ProcessPoolExecutor, TimeoutError

        if self.parallel_workers == 1:
            logger.info(
                f"Generation {current_generation_number}: Running fitness evaluation in serial mode."
            )
            for chromosome in self.population:
                try:
                    fitness_score = self.fitness_evaluator.evaluate(
                        chromosome, task_description, success_criteria
                    )
                    chromosome.fitness_score = fitness_score
                    evaluated_chromosomes_count += 1
                except Exception as e:
                    logger.error(
                        f"Error evaluating chromosome {chromosome.id} in serial mode: {e}",
                        exc_info=True,
                    )
                    chromosome.fitness_score = (
                        0.0  # Assign a default low fitness on error
                    )
                    failed_evaluations_count += 1
        else:
            logger.info(
                f"Generation {current_generation_number}: Running fitness evaluation in parallel mode (workers={self.parallel_workers})."
            )
            futures = []
            with ProcessPoolExecutor(max_workers=self.parallel_workers) as executor:
                for chromosome in self.population:
                    # Submit evaluation task to the executor
                    future = executor.submit(
                        self.fitness_evaluator.evaluate,
                        chromosome,
                        task_description,
                        success_criteria,
                    )
                    futures.append((future, chromosome))  # Store future and chromosome

                # Retrieve results and update fitness scores
                for future, chromosome in futures:
                    try:
                        # Use self.evaluation_timeout; if None, future.result() has no timeout
                        fitness_score = future.result(timeout=self.evaluation_timeout)
                        chromosome.fitness_score = fitness_score
                        evaluated_chromosomes_count += 1
                    except TimeoutError:
                        logger.error(
                            f"Fitness evaluation for chromosome {chromosome.id} timed out after {self.evaluation_timeout} seconds."
                        )
                        chromosome.fitness_score = (
                            0.0  # Assign a default low fitness on error
                        )
                        failed_evaluations_count += 1
                    except Exception as e:
                        logger.error(
                            f"Error evaluating chromosome {chromosome.id} in parallel: {e}",
                            exc_info=True,
                        )
                        chromosome.fitness_score = (
                            0.0  # Assign a default low fitness on error
                        )
                        failed_evaluations_count += 1

        # This logging seems to have a slight logic error in original: evaluated_chromosomes_count already includes failures if we count attempts.
        # Let's adjust to show successful vs attempted.
        successful_evaluations = (
            evaluated_chromosomes_count  # This was already successes
        )
        # failed_evaluations_count is correct


        # Log fitness statistics before sorting
        if self.population: # Ensure population is not empty
            fitness_scores = [c.fitness_score for c in self.population if hasattr(c, 'fitness_score')] # Added check for attribute
            if fitness_scores: # Ensure we have scores to process
                min_fitness = min(fitness_scores)
                max_fitness = max(fitness_scores)
                mean_fitness = statistics.mean(fitness_scores)
                median_fitness = statistics.median(fitness_scores)
                std_dev_fitness = statistics.stdev(fitness_scores) if len(fitness_scores) > 1 else 0.0

                logger.info(
                    f"Generation {current_generation_number}: Fitness Stats - "
                    f"Count: {len(fitness_scores)}, "
                    f"Min: {min_fitness:.4f}, Max: {max_fitness:.4f}, "
                    f"Mean: {mean_fitness:.4f}, Median: {median_fitness:.4f}, "
                    f"StdDev: {std_dev_fitness:.4f}"
                )
            else:
                logger.info(f"Generation {current_generation_number}: No valid fitness scores found in population to report statistics.")
        else:
            logger.info(f"Generation {current_generation_number}: Population is empty, no fitness statistics to report.")


        logger.info(
            f"PopulationManager: Fitness evaluation complete for generation {current_generation_number}."
        )  # Use current_generation_number
        logger.info(
            f"Successfully evaluated: {successful_evaluations}/{len(self.population)} chromosomes."
        )
        if failed_evaluations_count > 0:
            logger.warning(
                f"Failed evaluations (due to timeout or other errors): {failed_evaluations_count}/{len(self.population)} chromosomes."
            )

        self.broadcast_ga_update(
            event_type="ga_evaluation_complete",
            additional_data={
                "generation": current_generation_number,
                "evaluated_count": successful_evaluations,
                "failed_count": failed_evaluations_count,
            },
        )  # Added

        # 2. Sort Population by fitness (descending)
        # Ensure population is not empty before sorting, though it should generally not be.
        # If population is empty and sorted, it will result in an empty list.
        sorted_population = (
            sorted(self.population, key=lambda c: c.fitness_score, reverse=True)
            if self.population
            else []
        )

        if sorted_population:
            fittest_individual = sorted_population[0]
            logger.info(
                f"Generation {current_generation_number}: Fittest individual Chromosome ID {fittest_individual.id} "
                f"with fitness {fittest_individual.fitness_score:.4f}"
            )
        else:
            logger.warning(
                f"Generation {current_generation_number}: Population is empty after evaluation. No fittest individual."
            )

        new_population: list[PromptChromosome] = []

        # 3. Elitism: Carry over the best individuals
        if self.elitism_count > 0 and sorted_population:
            logger.info(
                f"Generation {current_generation_number}: Applying elitism for top {self.elitism_count} individuals."
            )
            new_population.extend(sorted_population[: self.elitism_count])

        # 4. Generate Offspring
        logger.info(
            f"Generation {current_generation_number}: Generating {self.population_size - len(new_population)} offspring through selection, crossover, and mutation."
        )
        num_offspring_needed = self.population_size - len(new_population)

        generated_offspring_count = 0
        # Ensure there's a viable population to select from for breeding
        if sorted_population:  # Check if sorted_population is not empty
            while generated_offspring_count < num_offspring_needed:
                # Selection - logging is inside genetic_operators.selection
                parent1 = self.genetic_operators.selection(sorted_population)
                parent2 = self.genetic_operators.selection(sorted_population)

                # Crossover - logging is inside genetic_operators.crossover
                child1, child2 = self.genetic_operators.crossover(parent1, parent2)

                # Mutation - logging is inside genetic_operators.mutate and strategies
                mutated_child1 = self.genetic_operators.mutate(
                    child1, target_style=target_style
                )  # gene_mutation_prob is unused here
                mutated_child2 = self.genetic_operators.mutate(
                    child2, target_style=target_style
                )  # gene_mutation_prob is unused here

                new_population.append(mutated_child1)
                generated_offspring_count += 1
                if generated_offspring_count < num_offspring_needed:
                    new_population.append(mutated_child2)
                    generated_offspring_count += 1
        elif num_offspring_needed > 0:  # If population was empty and we need offspring
            logger.warning(
                f"Generation {current_generation_number}: Cannot generate offspring as the population became empty after evaluation. New population will be undersized."
            )

        self.population = new_population[: self.population_size]

        self.generation_number = current_generation_number  # Update generation number

        # Update status based on pause/stop flags before final broadcast for the generation
        if self.is_paused:
            self.status = "PAUSED"
        elif self.should_stop:
            self.status = "STOPPED"  # Or "COMPLETED_STOP_REQUESTED"
        else:
            # If not paused or stopped, it's still running (or completed if this was the last gen planned by orchestrator)
            # Orchestrator should set final "COMPLETED" status.
            self.status = "RUNNING"

        logger.info(
            f"PopulationManager: Evolution complete for generation {self.generation_number}. New population size: {len(self.population)}. Status: {self.status}"
        )
        self.broadcast_ga_update(
            event_type="ga_generation_complete",
            additional_data={"generation": self.generation_number},
        )  # Added

    def get_fittest_individual(self) -> PromptChromosome | None:
        """
        Returns the fittest individual from the current population.

        The population is sorted by fitness score in descending order after evaluation,
        so the fittest individual is the first one if the population is not empty.

        Returns:
            PromptChromosome | None: The fittest chromosome, or None if the
                                      population is empty.
        """
        if not self.population:
            return None
        # Ensure population is sorted by fitness in descending order
        self.population.sort(key=lambda chromo: chromo.fitness_score, reverse=True)
        return self.population[0]

    def save_population(self, file_path: str) -> None:
        """Save current population to a JSON file."""
        data = {
            "generation_number": self.generation_number,
            "population": [
                {"genes": c.genes, "fitness_score": c.fitness_score}
                for c in self.population
            ],
        }
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            logger.info(f"PopulationManager: Population saved to {file_path}.")
        except Exception as e:
            logger.error(f"Error saving population to {file_path}: {e}", exc_info=True)

    def load_population(self, file_path: str) -> None:
        """Load population from a JSON file if it exists."""
        if not os.path.exists(file_path):
            logger.info(
                f"PopulationManager: No population file at {file_path}; starting fresh."
            )
            return
        try:
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            individuals = data.get("population", [])
            self.generation_number = data.get("generation_number", 0)
            self.population = [
                PromptChromosome(
                    genes=item.get("genes", []),
                    fitness_score=item.get("fitness_score", 0.0),
                )
                for item in individuals
            ]
            # If loading an empty population, population_size might become 0.
            # Keep original population_size if loaded population is empty,
            # unless original population_size was also 0.
            if not self.population and self.population_size > 0:
                logger.warning(
                    f"Loaded population from {file_path} is empty. Retaining configured population_size: {self.population_size}"
                )
            else:
                self.population_size = len(self.population) or self.population_size

            logger.info(
                f"PopulationManager: Loaded {len(self.population)} individuals from {file_path}. Generation set to {self.generation_number}. Population size set to {self.population_size}."
            )
        except Exception as e:
            logger.error(
                f"Error loading population from {file_path}: {e}", exc_info=True
            )

    def get_ga_status(self) -> dict:
        """
        Returns the current status of the genetic algorithm process.
        """
        fittest_individual = self.get_fittest_individual()
        return {
            "status": self.status,
            "generation": self.generation_number,
            "population_size": len(self.population) if self.population else 0,
            "best_fitness": (
                fittest_individual.fitness_score if fittest_individual else None
            ),
            "fittest_individual_id": (
                fittest_individual.id if fittest_individual else None
            ),
            "fittest_chromosome_string": (
                fittest_individual.to_prompt_string() if fittest_individual else ""
            ),
            "agents_used": self.agents_used,
        }
