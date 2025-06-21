import abc
from typing import Optional, Dict
from prompthelix.genetics.engine import PromptChromosome

class BaseFitnessEvaluator(abc.ABC):
    """
    Abstract base class for all fitness evaluators.
    """

    @abc.abstractmethod
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        """
        Initializes the BaseFitnessEvaluator.

        Args:
            settings (Optional[Dict], optional): Configuration settings for the evaluator.
                                                 Defaults to None.
            **kwargs: Additional keyword arguments for specific implementations.
        """
        self.settings = settings if settings is not None else {}
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        chromosome: PromptChromosome,
        task_description: str,
        success_criteria: Optional[Dict] = None
    ) -> float:
        """
        Evaluates the fitness of a given chromosome.

        The chromosome's fitness_score attribute should be updated by this method.

        Args:
            chromosome (PromptChromosome): The chromosome to evaluate.
            task_description (str): A description of the task the prompt is for.
            success_criteria (Optional[Dict], optional): Criteria for evaluating the
                success of the LLM output. Defaults to None.

        Returns:
            float: The calculated fitness score for the chromosome.
        """
        pass
