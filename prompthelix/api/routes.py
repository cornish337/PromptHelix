from pathlib import Path
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.engine import PromptChromosome
from prompthelix.services.prompt_manager import PromptManager

router = APIRouter()

# Template configuration for simple HTML interface
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))

# In-memory prompt manager instance
prompt_manager = PromptManager()

@router.get("/api/run-ga")
async def run_ga_endpoint():
    """Run the placeholder genetic algorithm and return the best prompt."""
    # Capture printed output? We'll run and capture best result by running orchestrator and capturing return value? main_ga_loop prints but doesn't return best. We'll modify orchestrator maybe.
    best = main_ga_loop(return_best=True)
    if isinstance(best, PromptChromosome):
        return {"best_prompt": best.to_prompt_string(), "fitness": best.fitness_score}
    return {"best_prompt": "", "fitness": 0.0}


@router.get("/api/prompts")
async def list_prompts():
    """Return all stored prompts."""
    return {"prompts": prompt_manager.list_prompts()}


@router.post("/api/prompts")
async def create_prompt(prompt: dict):
    """Create a new prompt from posted JSON."""
    content = prompt.get("content", "")
    if not content:
        return {"error": "content required"}
    return prompt_manager.add_prompt(content)


@router.get("/ui/prompts", response_class=HTMLResponse)
async def prompts_page(request: Request):
    """Render the HTML interface for managing prompts."""
    return templates.TemplateResponse(
        "prompts.html",
        {"request": request, "prompts": prompt_manager.list_prompts()},
    )


@router.post("/ui/prompts", response_class=RedirectResponse)
async def add_prompt_page(content: str = Form(...)):
    """Handle form submission from the HTML interface."""
    prompt_manager.add_prompt(content)
    return RedirectResponse(url="/ui/prompts", status_code=303)
