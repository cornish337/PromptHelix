from __future__ import annotations
import abc
import random
import logging # Added logging
from typing import TYPE_CHECKING, Type, Dict, List

logger = logging.getLogger(__name__) # Added logger

from typing import TYPE_CHECKING, Type, Dict, List, Optional # Added Optional
from prompthelix.genetics.strategy_base import BaseMutationStrategy # Import new ABC

logger = logging.getLogger(__name__) # Added logger

if TYPE_CHECKING: # pragma: no cover - only for type hints
    from prompthelix.genetics.engine import PromptChromosome

# Registry for mutation strategy classes
strategy_registry: Dict[str, Type[BaseMutationStrategy]] = {} # Use new ABC


def register_strategy(cls: Type[BaseMutationStrategy]) -> Type[BaseMutationStrategy]: # Use new ABC
    """Register a BaseMutationStrategy class in the global registry."""
    strategy_registry[cls.__name__] = cls
    return cls


def load_strategies(paths: List[str]) -> List[BaseMutationStrategy]: # Use new ABC
    """Dynamically load BaseMutationStrategy implementations from modules.

    Args:
        paths: List of module paths to import. Each module is scanned for
            classes that subclass :class:`BaseMutationStrategy`.

    Returns:
        List of instantiated strategy objects found in the provided modules.
    """
    strategies: List[BaseMutationStrategy] = [] # Use new ABC
    for module_path in paths:
        try:
            module = __import__(module_path, fromlist=["dummy"])
        except Exception as e:  # pragma: no cover - import errors logged
            logger.error(f"Failed to import mutation strategy module '{module_path}': {e}")
            continue

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseMutationStrategy) # Use new ABC
                and attr is not BaseMutationStrategy     # Use new ABC
            ):
                register_strategy(attr)
                try:
                    # Pass settings=None or an empty dict if strategies expect it
                    strategies.append(attr(settings=None))
                except Exception as inst_err:  # pragma: no cover - instantiation errors logged
                    logger.error(
                        f"Could not instantiate strategy '{attr.__name__}' from '{module_path}': {inst_err}"
                    )
    return strategies

# Old MutationStrategy ABC is removed. Strategies will inherit from BaseMutationStrategy.

@register_strategy
class AppendCharStrategy(BaseMutationStrategy): # Inherit from new ABC
    """Mutates a gene by appending a random character."""

    def __init__(
        self,
        settings: Optional[Dict] = None,
        chars_to_append: str | None = None,
        **kwargs,
    ):
        super().__init__(settings=settings, **kwargs)
        # Specific initialization for this strategy
        self.chars_to_append = (
            chars_to_append
            if chars_to_append is not None
            else self.settings.get("chars_to_append", "!*?_") if self.settings else "!*?_"
        )


    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        # It's often better to work on a clone to avoid side effects
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.fitness_score = 0.0 # Reset fitness for the new mutated version

        if not mutated_chromosome.genes:
            logger.debug(f"AppendCharStrategy: Chromosome {mutated_chromosome.id} has no genes to mutate.")
            return mutated_chromosome # No genes to mutate

        gene_index_to_mutate = random.randrange(len(mutated_chromosome.genes))
        original_gene_str = str(mutated_chromosome.genes[gene_index_to_mutate])

        if not self.chars_to_append: # Safety check
            logger.warning(f"AppendCharStrategy: No characters configured for appending. Chromosome {mutated_chromosome.id} remains unchanged.")
            return mutated_chromosome

        char_to_append = random.choice(self.chars_to_append)
        mutated_gene_str = original_gene_str + char_to_append
        mutated_chromosome.genes[gene_index_to_mutate] = mutated_gene_str

        logger.debug(
            f"AppendCharStrategy: Appended char '{char_to_append}' to gene {gene_index_to_mutate} "
            f"of chromosome {mutated_chromosome.id}. Original: '{original_gene_str}', Mutated: '{mutated_gene_str}'"
        )
        return mutated_chromosome

@register_strategy
class ReverseSliceStrategy(BaseMutationStrategy): # Inherit from new ABC
    """
    Mutates a gene by reversing a random slice of it.
    """
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # No specific settings for this strategy in this example, but structure is here

    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.fitness_score = 0.0

        if not mutated_chromosome.genes:
            logger.debug(f"ReverseSliceStrategy: Chromosome {mutated_chromosome.id} has no genes to mutate.")
            return mutated_chromosome

        gene_index_to_mutate = random.randrange(len(mutated_chromosome.genes))
        original_gene_str = str(mutated_chromosome.genes[gene_index_to_mutate])

        if len(original_gene_str) <= 2: # Not much to reverse if too short
            logger.debug(
                f"ReverseSliceStrategy: Gene {gene_index_to_mutate} in chromosome {mutated_chromosome.id} is too short ('{original_gene_str}'). "
                "No reversal applied."
            )
            return mutated_chromosome

        slice_len = random.randint(1, max(2, len(original_gene_str) // 2))
        start_index = random.randint(0, len(original_gene_str) - slice_len)

        segment_to_reverse = original_gene_str[start_index : start_index + slice_len]
        reversed_segment = segment_to_reverse[::-1]

        mutated_gene_str = (
            original_gene_str[:start_index] +
            reversed_segment +
            original_gene_str[start_index + slice_len:]
        )
        mutated_chromosome.genes[gene_index_to_mutate] = mutated_gene_str
        logger.debug(
            f"ReverseSliceStrategy: Reversed slice (start: {start_index}, len: {slice_len}, segment: '{segment_to_reverse}') "
            f"in gene {gene_index_to_mutate} of chromosome {mutated_chromosome.id}. "
            f"Original: '{original_gene_str}', Mutated: '{mutated_gene_str}'"
        )
        return mutated_chromosome

@register_strategy
class PlaceholderReplaceStrategy(BaseMutationStrategy): # Inherit from new ABC
    """Mutates a gene by replacing it or a part of it with a placeholder."""

    def __init__(
        self,
        settings: Optional[Dict] = None,
        placeholder: str | None = None,
        **kwargs,
    ):
        super().__init__(settings=settings, **kwargs)
        self.placeholder = (
            placeholder
            if placeholder is not None
            else self.settings.get("placeholder", "[MUTATED_GENE_SEGMENT]") if self.settings else "[MUTATED_GENE_SEGMENT]"
        )

    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        mutated_chromosome = chromosome.clone()
        mutated_chromosome.fitness_score = 0.0

        if not mutated_chromosome.genes:
            logger.debug(f"PlaceholderReplaceStrategy: Chromosome {mutated_chromosome.id} has no genes to mutate.")
            return mutated_chromosome

        gene_index_to_mutate = random.randrange(len(mutated_chromosome.genes))
        original_gene_str = str(mutated_chromosome.genes[gene_index_to_mutate])

        mutated_chromosome.genes[gene_index_to_mutate] = self.placeholder
        logger.debug(
            f"PlaceholderReplaceStrategy: Replaced gene {gene_index_to_mutate} "
            f"of chromosome {mutated_chromosome.id} with placeholder '{self.placeholder}'. "
            f"Original gene content: '{original_gene_str}'"
        )
        return mutated_chromosome

# Example of a NoOperationMutation strategy, could be useful
@register_strategy
class NoOperationMutationStrategy(BaseMutationStrategy): # Inherit from new ABC
    """
    A strategy that performs no mutation. Can be useful in some scenarios.
    """
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        # No specific settings for this strategy

    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        # Returns a clone but without any changes to genes
        # Or, if mutations are expected to sometimes not happen, could return original
        # For consistency with other strategies returning clones, this also returns a clone.
        no_op_mutated_chromosome = chromosome.clone()
        # Fitness is typically reset even if no gene change, as it's a "new" individual in the next generation
        no_op_mutated_chromosome.fitness_score = 0.0
        logger.debug(f"NoOperationMutationStrategy: Applied to chromosome {chromosome.id}. No gene changes made.")
        return no_op_mutated_chromosome
