import os
import importlib
import inspect
from typing import List, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from starlette.status import HTTP_303_SEE_OTHER # For POST redirect
import httpx # For making API calls from UI routes

from prompthelix.templating import templates # Import from templating.py
from prompthelix.database import get_db     # Ensure this is imported
from prompthelix.api import crud            # Ensure this is imported
from prompthelix import schemas # Import all schemas
from prompthelix.enums import ExecutionMode # Added import
from prompthelix.agents.base import BaseAgent
from prompthelix.utils import llm_utils # Added import


router = APIRouter()

SUPPORTED_LLM_SERVICES = [
    {"name": "OPENAI", "display_name": "OpenAI", "description": "API key for OpenAI models (e.g., GPT-4, GPT-3.5)."},
    {"name": "ANTHROPIC", "display_name": "Anthropic", "description": "API key for Anthropic models (e.g., Claude)."},
    {"name": "GOOGLE", "display_name": "Google", "description": "API key for Google AI models (e.g., Gemini)."}
]

def list_available_agents() -> List[dict[str, str]]: # Updated type hint
    agents_info = [] # Changed variable name
    # Assuming ui_routes.py is in the 'prompthelix' directory.
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    agents_dir = os.path.join(current_file_dir, "agents")

    if not os.path.isdir(agents_dir):
        print(f"Agents directory not found or is not a directory: {agents_dir}") # Consider proper logging
        return []

    for filename in os.listdir(agents_dir):
        if filename.endswith(".py") and filename not in ["__init__.py", "base.py"]:
            module_name = f"prompthelix.agents.{filename[:-3]}"
            try:
                module = importlib.import_module(module_name)
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if inspect.isclass(attribute) and \
                       issubclass(attribute, BaseAgent) and \
                       attribute is not BaseAgent:
                        agent_id_val = getattr(attribute, 'agent_id', None)
                        description_val = getattr(attribute, 'agent_description', "No description available.")

                        if agent_id_val and isinstance(agent_id_val, str):
                            # Ensure agent_id is not duplicated if multiple classes have same id (should not happen)
                            is_present = False
                            for agent_entry in agents_info:
                                if agent_entry['id'] == agent_id_val:
                                    is_present = True
                                    break
                            if not is_present:
                                agents_info.append({'id': agent_id_val, 'description': description_val})
                        else:
                             print(f"Agent class {attribute.__name__} in {module_name} is missing a valid string agent_id class attribute.")

            except ImportError as e:
                print(f"Error importing agent module {module_name}: {e}") # Proper logging recommended
            except Exception as e:
                print(f"Error processing module {module_name}: {e}") # Proper logging recommended
    return sorted(agents_info, key=lambda x: x['id']) # Return unique (by id), sorted list of dicts

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
    execution_mode: str = Form(ExecutionMode.REAL.value), # Added execution_mode
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
        execution_mode=ExecutionMode(execution_mode), # Added execution_mode
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
        "execution_mode": execution_mode, # Added execution_mode
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

@router.get("/ui/settings", name="view_settings_ui")
async def view_settings_ui(
    request: Request,
    db: Session = Depends(get_db),
    message: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    services_config_display = []
    for service_spec in SUPPORTED_LLM_SERVICES:
        service_info = service_spec.copy()
        db_key = crud.get_api_key(db, service_name=service_spec["name"])
        if db_key and db_key.api_key: # Check if api_key string is non-empty
            service_info["is_set"] = True
            service_info["api_key_hint"] = f"********{db_key.api_key[-4:]}" if len(db_key.api_key) >= 4 else "Set (short key)"
            service_info["current_value"] = db_key.api_key
        else:
            service_info["is_set"] = False
            service_info["api_key_hint"] = "Not set"
            service_info["current_value"] = ""
        services_config_display.append(service_info)

    available_agents = list_available_agents()

    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "services_config": services_config_display,
            "agents": available_agents,
            "message": message,
            "error": error
        }
    )

@router.post("/ui/settings/api_keys", name="save_api_keys_settings")
async def save_api_keys_settings(
    request: Request,
    db: Session = Depends(get_db)
):
    form_data = await request.form()
    processed_services_names = []
    error_messages = []

    for service_spec in SUPPORTED_LLM_SERVICES:
        service_name_in_code = service_spec["name"]

        api_key_form_field_name = f"{service_name_in_code}_api_key"
        service_name_form_field_name = f"{service_name_in_code}_service_name"

        submitted_service_name = form_data.get(service_name_form_field_name)
        submitted_api_key_value = form_data.get(api_key_form_field_name)

        if submitted_service_name == service_name_in_code:
            if submitted_api_key_value is not None: # Field was present in form
                try:
                    crud.create_or_update_api_key(
                        db,
                        service_name=submitted_service_name,
                        api_key_value=submitted_api_key_value
                    )
                    # Add to processed_services_names only if key has some value, or if you want to indicate "processed" even if cleared
                    if submitted_api_key_value: # Key has a value
                         processed_services_names.append(service_spec["display_name"])
                    else: # Key was cleared
                         processed_services_names.append(f"{service_spec['display_name']} (cleared)")
                except Exception as e:
                    print(f"Error saving API key for {submitted_service_name}: {e}") # Log properly
                    error_messages.append(f"Failed to save key for {service_spec['display_name']}.")
            # else: api_key field was not submitted, which is unlikely for this form structure

    # Construct message and error status for redirect
    final_message = ""
    error_status = None

    if error_messages:
        final_message = " ".join(error_messages)
        if processed_services_names:
            final_message += f" Other keys ({', '.join(processed_services_names)}) processed."
        error_status = "true" # Indicate an error occurred
    elif processed_services_names:
        final_message = f"API key settings saved for: {', '.join(processed_services_names)}."
    else:
        # No errors, but no services processed (e.g., form submitted with no changes to actual values, or all keys blank)
        final_message = "No changes to API keys were applied."

    redirect_url = request.url_for('view_settings_ui')
    redirect_url += f"?message={final_message}"
    if error_status:
        redirect_url += f"&error={error_status}"

    return RedirectResponse(url=redirect_url, status_code=HTTP_303_SEE_OTHER)


@router.get("/ui/llm_tester", name="llm_tester_ui")
async def llm_tester_ui(request: Request, db: Session = Depends(get_db)):
    """
    Renders the LLM Tester UI page.
    Fetches available LLMs and current statistics to display.
    """
    available_llms = []
    statistics = []
    error_message = None
    try:
        # Option 1: Call llm_utils directly (if it's simple and doesn't require async client calls)
        available_llms = llm_utils.list_available_llms(db=db)

        # Option 2: Call your own API endpoint (if you prefer to keep UI routes calling API routes)
        # This requires httpx and careful handling of base_url if app is not fully running in this context
        # For simplicity, direct util call is used here.
        # async with httpx.AsyncClient(app=request.app, base_url=request.base_url) as client:
        #     response_available = await client.get(request.url_for('get_available_llms'))
        #     response_available.raise_for_status()
        #     available_llms = response_available.json()

        #     response_stats = await client.get(request.url_for('get_llm_statistics'))
        #     response_stats.raise_for_status()
        #     statistics_raw = response_stats.json()
        #     # Convert to schema if necessary, though API should return them as per schema
        #     statistics = [schemas.LLMStatistic(**stat) for stat in statistics_raw]

        db_statistics = crud.get_all_llm_statistics(db=db)
        # Convert to LLMStatistic schema if needed, but crud returns models.
        # The template might directly use model attributes or you can convert here.
        # For consistency with plan schema names:
        statistics = [schemas.LLMStatistic(llm_service=stat.llm_service, request_count=stat.request_count) for stat in db_statistics]


    except HTTPException as http_exc: # Catch HTTP exceptions from potential API calls
        error_message = f"API Error: {http_exc.detail}"
    except Exception as e:
        error_message = f"An unexpected error occurred while fetching data: {str(e)}"
        # Log this error as well
        print(f"Error in llm_tester_ui: {e}")

    return templates.TemplateResponse(
        "llm_tester.html",
        {
            "request": request,
            "available_llms": available_llms,
            "statistics": statistics,
            "error_message": error_message,
            "llm_response": None, # For initial state
            "submitted_prompt": None # For initial state
        }
    )
