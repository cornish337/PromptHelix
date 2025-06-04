from fastapi import APIRouter
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.engine import PromptChromosome

router = APIRouter()

@router.get("/api/run-ga")
async def run_ga_endpoint():
    """Run the placeholder genetic algorithm and return the best prompt."""
    # Capture printed output? We'll run and capture best result by running orchestrator and capturing return value? main_ga_loop prints but doesn't return best. We'll modify orchestrator maybe.
    best = main_ga_loop(return_best=True)
    if isinstance(best, PromptChromosome):
        return {"best_prompt": best.to_prompt_string(), "fitness": best.fitness_score}
    return {"best_prompt": "", "fitness": 0.0}
