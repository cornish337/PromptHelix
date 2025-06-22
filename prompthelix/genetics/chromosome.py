import uuid
import copy
from typing import List, Optional

class PromptChromosome:
    """Simple representation of a prompt chromosome used in tests."""

    def __init__(
        self,
        genes: Optional[List[str]] = None,
        fitness_score: float = 0.0,
        parent_ids: Optional[List[str]] = None,
        mutation_strategy: Optional[str] = None
    ):
        self.id = uuid.uuid4()
        self.genes = genes if genes is not None else []
        self.fitness_score = fitness_score
        self.parent_ids = parent_ids if parent_ids is not None else []
        self.mutation_strategy = mutation_strategy

    def clone(self) -> "PromptChromosome":
        """Creates a deep copy of this chromosome with a new ID."""
        cloned = PromptChromosome(
            genes=copy.deepcopy(self.genes),
            fitness_score=self.fitness_score,
            parent_ids=list(self.parent_ids),
            mutation_strategy=self.mutation_strategy
        )
        return cloned

    def to_prompt_string(self, separator: str = "\n") -> str:
        """Converts the chromosome's genes into a single string."""
        return separator.join(map(str, self.genes))

    def __str__(self) -> str:
        """Returns a human-readable string representation of the chromosome."""
        if self.genes:
            genes_str = "\n".join([f"  - {str(g)}" for g in self.genes])
        else:
            genes_str = "  - (No genes)"
        return (
            f"PromptChromosome(\n"
            f"  ID: {self.id},\n"
            f"  Fitness: {self.fitness_score:.4f},\n"
            f"  Genes:\n{genes_str},\n"
            f"  Parents: {self.parent_ids},\n"
            f"  MutationOp: {self.mutation_strategy}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Returns an unambiguous string representation of the chromosome."""
        return (
            f"PromptChromosome(id='{self.id}', genes={self.genes!r}, "
            f"fitness_score={self.fitness_score:.4f}, parent_ids={self.parent_ids!r}, " # Changed format for fitness_score
            f"mutation_strategy={self.mutation_strategy!r})"
        )
