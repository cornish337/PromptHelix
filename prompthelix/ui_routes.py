from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import Optional, List
import httpx # For making API calls from UI routes

from prompthelix.templating import templates # Import from templating.py
from prompthelix.database import get_db
from prompthelix.api import crud
from prompthelix import schemas # Import all schemas

router = APIRouter()

@router.get("/ui/prompts", name="list_prompts_ui")
async def list_prompts_ui(request: Request, db: Session = Depends(get_db), new_version_id: Optional[int] = Query(None)):
    db_prompts = crud.get_prompts(db)
    return templates.TemplateResponse(
        "prompts.html",
        {"request": request, "prompts": db_prompts, "new_version_id": new_version_id}
    )

@router.get("/ui/prompts/new", name="create_prompt_ui_form")
async def create_prompt_ui_form(request: Request):
    return templates.TemplateResponse("create_prompt.html", {"request": request})

@router.post("/ui/prompts/new", name="create_prompt_ui_submit")
async def create_prompt_ui_submit(
    request: Request,
    db: Session = Depends(get_db),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    initial_content: str = Form(...)
):
    prompt_data = schemas.PromptCreate(name=name, description=description)
    db_prompt = crud.create_prompt(db, prompt=prompt_data)

    version_data = schemas.PromptVersionCreate(content=initial_content)
    crud.create_prompt_version(db, version=version_data, prompt_id=db_prompt.id)

    # Redirect to the new prompt's detail page
    # Use request.url_for to get the URL for the named route
    redirect_url = request.url_for('view_prompt_ui', prompt_id=db_prompt.id)
    return RedirectResponse(url=redirect_url, status_code=303)


@router.get("/ui/experiments/new", name="run_experiment_ui_form")
async def run_experiment_ui_form(request: Request, db: Session = Depends(get_db)):
    available_prompts = crud.get_prompts(db, limit=1000) # Get a list of prompts for dropdown
    return templates.TemplateResponse(
        "experiment.html",
        {"request": request, "available_prompts": available_prompts, "form_data": {}} # Pass empty form_data initially
    )

@router.post("/ui/experiments/new", name="run_experiment_ui_submit")
async def run_experiment_ui_submit(
    request: Request,
    db: Session = Depends(get_db),
    task_description: str = Form(...),
    keywords: Optional[str] = Form(""), # Comma-separated string
    num_generations: int = Form(10),
    population_size: int = Form(20),
    elitism_count: int = Form(2),
    parent_prompt_id: Optional[int] = Form(None),
    prompt_name: Optional[str] = Form(None),
    prompt_description: Optional[str] = Form(None)
):
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()] if keywords else []

    ga_params = schemas.GAExperimentParams(
        task_description=task_description,
        keywords=keyword_list,
        num_generations=num_generations,
        population_size=population_size,
        elitism_count=elitism_count,
        parent_prompt_id=parent_prompt_id if parent_prompt_id else None, # Ensure None if empty string or 0
        prompt_name=prompt_name,
        prompt_description=prompt_description
    )

    api_experiment_url = request.url_for('api_run_ga_experiment') # Needs name in API route

    async with httpx.AsyncClient(app=request.app, base_url=request.base_url) as client:
        try:
            response = await client.post(api_experiment_url, json=ga_params.model_dump(exclude_none=True))
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses

            returned_prompt_version_data = response.json()
            # Ensure data can be parsed into PromptVersion schema; API returns this.
            created_version = schemas.PromptVersion(**returned_prompt_version_data)

            # Redirect to the prompt detail page, highlighting the new version
            redirect_url = request.url_for('view_prompt_ui', prompt_id=created_version.prompt_id)
            redirect_url += f"?new_version_id={created_version.id}" # Pass as query param
            return RedirectResponse(url=redirect_url, status_code=303)

        except httpx.HTTPStatusError as e:
            error_message = f"API Error: {e.response.status_code} - {e.response.text}"
        except Exception as e: # Catch other errors like connection errors
            error_message = f"An unexpected error occurred: {str(e)}"

    # If error, re-render form with error message and previous data
    available_prompts = crud.get_prompts(db, limit=1000)
    form_data_retained = {
        "task_description": task_description,
        "keywords": keywords,
        "num_generations": num_generations,
        "population_size": population_size,
        "elitism_count": elitism_count,
        "parent_prompt_id": parent_prompt_id,
        "prompt_name": prompt_name,
        "prompt_description": prompt_description,
    }
    return templates.TemplateResponse(
        "experiment.html",
        {
            "request": request,
            "available_prompts": available_prompts,
            "error_message": error_message,
            "form_data": form_data_retained
        }
    )


@router.get("/ui/prompts/{prompt_id}", name="view_prompt_ui") # Keep existing view_prompt_ui
async def view_prompt_ui(request: Request, prompt_id: int, db: Session = Depends(get_db), new_version_id: Optional[int] = Query(None)):
    db_prompt = crud.get_prompt(db, prompt_id=prompt_id)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail=f"Prompt with id {prompt_id} not found.")

    # Sort versions by version_number descending for display
    # This assumes versions are loaded, which they are by lazy loading.
    # If a prompt has many versions, this could be optimized in the query.
    sorted_versions = sorted(db_prompt.versions, key=lambda v: v.version_number, reverse=True)

    return templates.TemplateResponse(
        "prompt_detail.html",
        {"request": request, "prompt": db_prompt, "sorted_versions": sorted_versions, "new_version_id": new_version_id}
    )
