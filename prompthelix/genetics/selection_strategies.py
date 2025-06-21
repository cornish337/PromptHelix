import random
import logging
from typing import List, Optional, Dict
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.genetics.strategy_base import BaseSelectionStrategy

logger = logging.getLogger(__name__)

class TournamentSelectionStrategy(BaseSelectionStrategy):
    """
    Selects an individual from the population using tournament selection.
    """
    def __init__(self, settings: Optional[Dict] = None, **kwargs):
        super().__init__(settings=settings, **kwargs)
        self.tournament_size = self.settings.get("tournament_size", 3) if self.settings else 3
        if not isinstance(self.tournament_size, int) or self.tournament_size <= 0:
            logger.warning(f"Invalid tournament_size '{self.tournament_size}', defaulting to 3.")
            self.tournament_size = 3


    def select(self, population: List[PromptChromosome], **kwargs) -> PromptChromosome:
        """
        Selects an individual from the population using tournament selection.

        Args:
            population (List[PromptChromosome]): A list of PromptChromosome objects.
            **kwargs: Can include 'tournament_size' to override the default.

        Returns:
            PromptChromosome: The individual with the highest fitness_score from
                              the tournament.

        Raises:
            ValueError: If population is empty.
        """
        if not population:
            raise ValueError("Population cannot be empty for selection.")

        current_tournament_size = kwargs.get("tournament_size", self.tournament_size)
        if not isinstance(current_tournament_size, int) or current_tournament_size <= 0:
            logger.warning(f"Invalid dynamic tournament_size '{current_tournament_size}', using instance default {self.tournament_size}.")
            current_tournament_size = self.tournament_size

        actual_tournament_size = min(len(population), current_tournament_size)

        if actual_tournament_size == 0 : # Should not happen if population is not empty
             raise ValueError("Effective tournament size is 0, cannot select.")

        tournament_contenders = random.sample(population, actual_tournament_size)

        winner = tournament_contenders[0]
        for contender in tournament_contenders[1:]:
            if contender.fitness_score > winner.fitness_score:
                winner = contender

        logger.debug(
            f"TournamentSelectionStrategy (size {actual_tournament_size}): Winner Chromosome ID {winner.id}, Fitness {winner.fitness_score:.4f}"
        )
        return winner
