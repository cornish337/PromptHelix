import random
import copy
import logging
from typing import Tuple, Optional, Dict
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.genetics.strategy_base import BaseCrossoverStrategy

logger = logging.getLogger(__name__)

class SinglePointCrossoverStrategy(BaseCrossoverStrategy):
    """
    Performs single-point crossover between two parent chromosomes.
    """
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.default_crossover_rate = self.settings.get("crossover_rate", 0.7) if self.settings else 0.7
        if not (0 <= self.default_crossover_rate <= 1):
            logger.warning(f"Invalid crossover_rate '{self.default_crossover_rate}', defaulting to 0.7.")
            self.default_crossover_rate = 0.7

    def crossover(
        self,
        parent1: PromptChromosome,
        parent2: PromptChromosome,
        **kwargs
    ) -> Tuple[PromptChromosome, PromptChromosome]:
        """
        Performs single-point crossover between two parent chromosomes.
        If random.random() < crossover_rate, crossover occurs. Otherwise, children
        are clones of the parents.

        Args:
            parent1 (PromptChromosome): The first parent chromosome.
            parent2 (PromptChromosome): The second parent chromosome.
            **kwargs: Can include 'crossover_rate' to override the default.

        Returns:
            Tuple[PromptChromosome, PromptChromosome]: Two new child chromosomes.
        """
        crossover_rate = kwargs.get("crossover_rate", self.default_crossover_rate)
        if not (0 <= crossover_rate <= 1):
            logger.warning(f"Invalid dynamic crossover_rate '{crossover_rate}', using instance default {self.default_crossover_rate}.")
            crossover_rate = self.default_crossover_rate

        child1_genes = []
        child2_genes = []

        if random.random() < crossover_rate:
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
                f"SinglePointCrossoverStrategy: Crossover performed between Parent {parent1.id} and Parent {parent2.id}. Child1 ID {child1.id}, Child2 ID {child2.id}."
            )
        else:
            child1 = parent1.clone()
            child2 = parent2.clone()
            child1.fitness_score = 0.0 # Ensure fitness is reset for clones too
            child2.fitness_score = 0.0
            logger.debug(
                f"SinglePointCrossoverStrategy: Crossover skipped (rate {crossover_rate}). Cloned Parent {parent1.id} to Child {child1.id}, Parent {parent2.id} to Child {child2.id}."
            )
        return child1, child2
