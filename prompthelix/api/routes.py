from fastapi import APIRouter
from prompthelix.orchestrator import main_ga_loop # Assuming main_ga_loop is refactored

router = APIRouter()

@router.get("/run-ga")
async def run_ga_endpoint():
    # This might be time-consuming, consider running in a threadpool if it blocks
    # For now, direct call for simplicity
    result = main_ga_loop()
    return result
