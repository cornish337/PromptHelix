import abc
from typing import Optional, Dict, List, Tuple
from prompthelix.genetics.engine import PromptChromosome

class BaseMutationStrategy(abc.ABC):
    """
    Abstract base class for all mutation strategies.
    """

    @abc.abstractmethod
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        self.settings = settings if settings is not None else {}
        pass

    @abc.abstractmethod
    def mutate(self, chromosome: PromptChromosome) -> PromptChromosome:
        """
        Applies mutation to a chromosome and returns a new, mutated chromosome.
        The original chromosome should not be modified. Fitness of the new
        chromosome should be reset (e.g., to 0.0).
        """
        pass

class BaseSelectionStrategy(abc.ABC):
    """
    Abstract base class for all selection strategies.
    """

    @abc.abstractmethod
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        self.settings = settings if settings is not None else {}
        pass

    @abc.abstractmethod
    def select(self, population: List[PromptChromosome], **kwargs) -> PromptChromosome:
        """
        Selects an individual from the population.
        """
        pass

class BaseCrossoverStrategy(abc.ABC):
    """
    Abstract base class for all crossover strategies.
    """

    @abc.abstractmethod
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        self.settings = settings if settings is not None else {}
        pass

    @abc.abstractmethod
    def crossover(
        self,
        parent1: PromptChromosome,
        parent2: PromptChromosome,
        **kwargs
    ) -> Tuple[PromptChromosome, PromptChromosome]:
        """
        Performs crossover between two parents and returns two new child chromosomes.
        Fitness of the children should be reset (e.g., to 0.0).
        """
        pass
