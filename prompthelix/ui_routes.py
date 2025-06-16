import os
import importlib
import inspect
from typing import List, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from starlette.status import HTTP_303_SEE_OTHER # For POST redirect
import httpx # For making API calls from UI routes

from prompthelix.templating import templates # Import from templating.py
from prompthelix.database import get_db     # Ensure this is imported
from prompthelix.api import crud            # Ensure this is imported
from prompthelix import schemas # Import all schemas
from prompthelix.enums import ExecutionMode # Added import
from prompthelix.agents.base import BaseAgent


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
    # UI routes do not handle authentication yet; default owner_id=1
    db_prompt = crud.create_prompt(db, prompt=prompt_data, owner_id=1)

    version_data = schemas.PromptVersionCreate(content=initial_content)
    crud.create_prompt_version(db, version=version_data, prompt_id=db_prompt.id)

    # Redirect to the new prompt's detail page
    # Use request.url_for to get the URL for the named route
    message = f"Prompt '{db_prompt.name}' created successfully."
    redirect_url = str(request.url_for('view_prompt_ui', prompt_id=db_prompt.id)) + f"?message={message}"
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
    parent_prompt_id: Optional[str] = Form(None), # Changed to Optional[str]
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
        parent_prompt_id=int(parent_prompt_id) if parent_prompt_id and parent_prompt_id.isdigit() else None, # Convert to int if not empty/None
        prompt_name=prompt_name,
        prompt_description=prompt_description
    )

    api_experiment_url = request.url_for('api_run_ga_experiment') # Needs name in API route

    async with httpx.AsyncClient(app=request.app, base_url=str(request.base_url)) as client:
        try:
            access_token = request.cookies.get("prompthelix_access_token")
            headers = {}
            if access_token:
                headers["Authorization"] = f"Bearer {access_token}"
            else:
                # Optional: log that token is missing for server-side diagnosis
                print("DEBUG: prompthelix_access_token cookie not found in run_experiment_ui_submit.")

            response = await client.post(
                str(api_experiment_url),
                json=ga_params.model_dump(exclude_none=True),
                headers=headers # Pass the headers
            )
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses

            returned_prompt_version_data = response.json()
            # Ensure data can be parsed into PromptVersion schema; API returns this.
            created_version = schemas.PromptVersion(**returned_prompt_version_data)

            # Redirect to the prompt detail page, highlighting the new version
            message = f"New version (ID: {created_version.id}) created successfully from experiment."
            redirect_url = str(request.url_for('view_prompt_ui', prompt_id=created_version.prompt_id)) # Convert URL to string
            redirect_url += f"?new_version_id={created_version.id}&message={message}" # Pass as query param
            return RedirectResponse(url=redirect_url, status_code=303)

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            error_message = f"API Error: {e.response.status_code} - {error_detail}"
            if e.response.status_code == 401:
                error_message = f"Authentication failed. Please ensure you are logged in and try again. Detail: {error_detail}"
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
        "parent_prompt_id": parent_prompt_id, # Keep as string for re-rendering form
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


@router.get("/ui/playground", name="llm_playground_ui")
async def get_llm_playground_ui(request: Request, db: Session = Depends(get_db)):
    llm_providers = SUPPORTED_LLM_SERVICES # Static list for now
    db_prompts = crud.get_prompts(db, limit=1000) # Fetch available prompts

    prompts_content_map = {}
    if db_prompts:
        for p in db_prompts:
            if p.versions:  # Check if there are any versions
                # Sort versions by version_number descending to get the latest
                latest_version = sorted(p.versions, key=lambda v: v.version_number, reverse=True)[0]
                prompts_content_map[str(p.id)] = latest_version.content
            else:
                prompts_content_map[str(p.id)] = "" # Default content if no versions

    return templates.TemplateResponse(
        "llm_playground.html",
        {
            "request": request,
            "llm_providers": llm_providers,
            "available_prompts": db_prompts, # For dropdown list
            "prompts_content_map": prompts_content_map # For JS to get content
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
                        api_key_create=schemas.APIKeyCreate(
                            service_name=submitted_service_name,
                            api_key=submitted_api_key_value,
                        ),
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

    redirect_url = str(request.url_for('view_settings_ui'))
    redirect_url += f"?message={final_message}"
    if error_status:
        redirect_url += f"&error={error_status}"

    return RedirectResponse(url=redirect_url, status_code=HTTP_303_SEE_OTHER)


@router.get("/ui/conversations", response_class=HTMLResponse, name="ui_list_conversations")
async def ui_list_conversations(request: Request):
    """Serves the UI page for viewing conversation logs."""
    return templates.TemplateResponse("conversations.html", {"request": request})

@router.get("/ui/login", response_class=HTMLResponse, name="ui_login")
async def ui_login_form(request: Request):
    """Serves the UI page for user login."""
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/ui/dashboard", response_class=HTMLResponse, name="ui_dashboard")
async def get_dashboard_ui(request: Request):
    """Serves the UI page for the real-time monitoring dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})
