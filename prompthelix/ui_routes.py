import os
import importlib
import inspect
from typing import List, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, Form, Query, status
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.orm import Session
from starlette.status import HTTP_303_SEE_OTHER  # For POST redirect
import httpx  # For making API calls from UI routes
from datetime import datetime
import unittest
import io
import asyncio

from prompthelix.templating import templates # Import from templating.py
from prompthelix.database import get_db     # Ensure this is imported
from prompthelix.api import crud            # Ensure this is imported
from prompthelix import schemas # Import all schemas
from prompthelix.enums import ExecutionMode  # Added import
from prompthelix.agents.base import BaseAgent
from prompthelix.models.user_models import User as UserModel
from prompthelix.services import user_service


router = APIRouter()

SUPPORTED_LLM_SERVICES = [
    {"name": "OPENAI", "display_name": "OpenAI", "description": "API key for OpenAI models (e.g., GPT-4, GPT-3.5)."},
    {"name": "ANTHROPIC", "display_name": "Anthropic", "description": "API key for Anthropic models (e.g., Claude)."},
    {"name": "GOOGLE", "display_name": "Google", "description": "API key for Google AI models (e.g., Gemini)."}
]


async def get_current_user_ui(
    request: Request, db: Session = Depends(get_db)
) -> UserModel:
    token = request.cookies.get("prompthelix_access_token")
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    db_session = user_service.get_session_by_token(db, session_token=token)
    if not db_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authentication credentials")
    if db_session.expires_at < datetime.utcnow():
        user_service.delete_session(db, session_token=token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")
    user = user_service.get_user(db, user_id=db_session.user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found for session")
    return user


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


def list_interactive_tests() -> List[str]:
    """Discover interactive tests from multiple specified directories."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    # Define paths to the interactive test directories
    # Path 1: PROJECT_ROOT/tests/interactive
    # Path 2: PROJECT_ROOT/prompthelix/tests/interactive
    interactive_test_dirs = [
        os.path.join(project_root, "tests", "interactive"),
        os.path.join(project_root, "prompthelix", "tests", "interactive")
    ]

    all_test_names: List[str] = []
    loader = unittest.TestLoader()

    def _collect_tests_from_suite(suite: unittest.TestSuite, L: List[str]):
        for test in suite:
            if isinstance(test, unittest.TestSuite):
                _collect_tests_from_suite(test, L)
            else:
                L.append(test.id())

    for tests_dir in interactive_test_dirs:
        if os.path.isdir(tests_dir):
            # Discover tests in the current directory
            # Use a pattern that matches typical test files, e.g., test_*.py
            suite = loader.discover(start_dir=tests_dir, pattern="test_*.py")
            current_dir_test_names: List[str] = []
            _collect_tests_from_suite(suite, current_dir_test_names)
            all_test_names.extend(current_dir_test_names)
        else:
            # Optionally log that a configured test directory was not found
            print(f"Warning: Interactive test directory not found: {tests_dir}")

    # Return sorted unique test names
    # Return sorted unique test names
    return sorted(list(set(all_test_names)))


# Configuration for different test suites
TEST_SUITES_CONFIG = [
    {
        "id": "interactive_all",
        "name": "All Interactive Tests",
        "paths": ["tests/interactive", "prompthelix/tests/interactive"], # Relative to project root
        "pattern": "test_*.py",
        "default": True,
    },
    {
        "id": "interactive_general",
        "name": "General Interactive Tests",
        "paths": ["tests/interactive"],
        "pattern": "test_*.py",
    },
    {
        "id": "interactive_prompthelix",
        "name": "PromptHelix Interactive Tests",
        "paths": ["prompthelix/tests/interactive"],
        "pattern": "test_*.py",
    },
    {
        "id": "unit_prompthelix",
        "name": "PromptHelix Unit Tests",
        "paths": ["prompthelix/tests/unit"],
        "pattern": "test_*.py",
    },
    {
        "id": "integration_prompthelix",
        "name": "PromptHelix Integration Tests",
        "paths": ["prompthelix/tests/integration"],
        "pattern": "test_*.py",
    },
]

def get_test_suite_config_by_id(suite_id: str) -> Optional[dict]:
    """Retrieve a test suite's configuration by its ID."""
    for suite_config in TEST_SUITES_CONFIG:
        if suite_config["id"] == suite_id:
            return suite_config
    return None

def list_tests_for_suite(suite_id: str) -> List[str]:
    """Discover tests for a given suite ID based on its configuration."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    suite_config = get_test_suite_config_by_id(suite_id)

    if not suite_config:
        print(f"Warning: Test suite configuration not found for ID: {suite_id}")
        return []

    all_test_names: List[str] = []
    loader = unittest.TestLoader()

    # Helper to recursively collect test names from a suite
    def _collect_from_suite(current_suite: unittest.TestSuite, L: List[str]):
        for test_item in current_suite:
            if isinstance(test_item, unittest.TestSuite):
                _collect_from_suite(test_item, L)
            else:
                L.append(test_item.id())

    for relative_path in suite_config.get("paths", []):
        tests_dir = os.path.join(project_root, relative_path)
        if os.path.isdir(tests_dir):
            discovered_suite = loader.discover(
                start_dir=tests_dir,
                pattern=suite_config.get("pattern", "test_*.py") # Use configured pattern
            )
            _collect_from_suite(discovered_suite, all_test_names)
        else:
            print(f"Warning: Test directory not found for suite '{suite_id}': {tests_dir}")

    return sorted(list(set(all_test_names)))


def run_unittest(test_name: str) -> tuple[str, bool]:
    """Run a single unittest and return output and success status."""
    stream = io.StringIO()
    loader = unittest.TestLoader()
    # loadTestsFromName can load individual tests or entire modules/classes
    suite = loader.loadTestsFromName(test_name)
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    return stream.getvalue(), result.wasSuccessful()


@router.get("/", response_class=HTMLResponse, name="ui_index")
@router.get("/index", response_class=HTMLResponse, name="ui_index_alt") # Optional: alternative path
async def ui_index_page(request: Request):
    """Serves the UI index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/index.html", include_in_schema=False)
async def ui_index_page_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("ui_index")))

@router.get("/prompts", name="list_prompts_ui")
async def list_prompts_ui(request: Request, db: Session = Depends(get_db), new_version_id: Optional[int] = Query(None), message: Optional[str] = Query(None)):
    db_prompts = crud.get_prompts(db)
    return templates.TemplateResponse(
        "prompts.html",
        {"request": request, "prompts": db_prompts, "new_version_id": new_version_id, "message": message}
    )

@router.get("/prompts.html", include_in_schema=False)
async def list_prompts_ui_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("list_prompts_ui")))

@router.get("/prompts/new", name="create_prompt_ui_form")
async def create_prompt_ui_form(request: Request):
    return templates.TemplateResponse("create_prompt.html", {"request": request})

@router.get("/prompts/new.html", include_in_schema=False)
async def create_prompt_ui_form_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("create_prompt_ui_form")))

@router.post("/prompts/new", name="create_prompt_ui_submit")
async def create_prompt_ui_submit(
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user_ui),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    initial_content: str = Form(...)
):
    prompt_data = schemas.PromptCreate(name=name, description=description)
    db_prompt = crud.create_prompt(db, prompt=prompt_data, owner_id=current_user.id)

    version_data = schemas.PromptVersionCreate(content=initial_content)
    crud.create_prompt_version(db, version=version_data, prompt_id=db_prompt.id)

    # Redirect to the new prompt's detail page
    # Use request.url_for to get the URL for the named route
    message = f"Prompt '{db_prompt.name}' created successfully."
    redirect_url = str(request.url_for('view_prompt_ui', prompt_id=db_prompt.id)) + f"?message={message}"
    return RedirectResponse(url=redirect_url, status_code=303)


@router.get("/experiments/new", name="run_experiment_ui_form")
async def run_experiment_ui_form(
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user_ui),
):
    available_prompts = crud.get_prompts(db, limit=1000) # Get a list of prompts for dropdown
    return templates.TemplateResponse(
        "experiment.html",
        {"request": request, "available_prompts": available_prompts, "form_data": {}} # Pass empty form_data initially
    )

@router.get("/experiments/new.html", include_in_schema=False)
async def run_experiment_ui_form_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("run_experiment_ui_form")))

@router.post("/experiments/new", name="run_experiment_ui_submit")
async def run_experiment_ui_submit(
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user_ui),
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

    async with httpx.AsyncClient(app=request.app, base_url=str(request.base_url))as client:
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
                headers=headers  # Pass the headers
            )
            response.raise_for_status()  # Raises an exception for 4XX/5XX responses

            response_data = response.json()
            message = None
            redirect_url = None

            # Preferred format: GARunResponse with task ID
            try:
                ga_run_info = schemas.GARunResponse(**response_data)
                message = ga_run_info.message
                if ga_run_info.task_id:
                    message += f" Task ID: {ga_run_info.task_id}"
                redirect_url = str(request.url_for("ui_dashboard"))
            except Exception:
                # Fallback to old behaviour expecting a PromptVersion structure
                if "id" in response_data and "prompt_id" in response_data:
                    pv = schemas.PromptVersion(**response_data)
                    message = (
                        f"New version (ID: {pv.id}) created successfully from experiment."
                    )
                    redirect_url = str(
                        request.url_for("view_prompt_ui", prompt_id=pv.prompt_id)
                    )
                    redirect_url += f"?new_version_id={pv.id}"
                else:
                    # Unknown response structure
                    message = "GA experiment started"
                    redirect_url = str(request.url_for("ui_dashboard"))

            redirect_url += f"?message={message}"
            return RedirectResponse(url=redirect_url, status_code=303)

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                # Attempt to parse error detail if it's JSON
                error_json = e.response.json()
                error_message = f"API Error: {e.response.status_code} - {error_json.get('detail', error_detail)}"
            except ValueError: # if error_detail is not JSON
                error_message = f"API Error: {e.response.status_code} - {error_detail}"

            if e.response.status_code == 401:
                error_message = f"Authentication failed. Please ensure you are logged in and try again. Detail: {error_detail}"
        except Exception as e: # Catch other errors like connection errors
            error_message = f"An unexpected error occurred: {str(e)}"

    # If error, re-render form with error message and previous data
    available_prompts = crud.get_prompts(db, limit=1000) # Get a list of prompts for dropdown
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


@router.get("/playground", name="llm_playground_ui")
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

@router.get("/playground.html", include_in_schema=False)
async def llm_playground_ui_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("llm_playground_ui")))


@router.get("/prompts/{prompt_id}", name="view_prompt_ui")
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

@router.get("/prompts/{prompt_id}.html", include_in_schema=False)
async def view_prompt_ui_alias(request: Request, prompt_id: int):
    return RedirectResponse(url=str(request.url_for("view_prompt_ui", prompt_id=prompt_id)))

@router.get("/settings", name="view_settings_ui")
@router.get("/settings/", name="view_settings_ui_alt") # This path with trailing slash will also be prefixed
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

@router.get("/settings.html", include_in_schema=False)
async def view_settings_ui_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("view_settings_ui")))


@router.get("/admin/users/new", response_class=HTMLResponse, name="admin_create_user_form")
async def admin_create_user_form(
    request: Request,
    current_user: UserModel = Depends(get_current_user_ui),
    message: Optional[str] = Query(None),
    error: Optional[str] = Query(None)
):
    return templates.TemplateResponse(
        "create_user.html",
        {"request": request, "message": message, "error": error}
    )


@router.post("/admin/users/new", name="admin_create_user_submit")
async def admin_create_user_submit(
    request: Request,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user_ui),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    if user_service.get_user_by_username(db, username=username):
        redirect_url = str(request.url_for("admin_create_user_form")) + "?error=Username already exists."
        return RedirectResponse(url=redirect_url, status_code=HTTP_303_SEE_OTHER)
    if user_service.get_user_by_email(db, email=email):
        redirect_url = str(request.url_for("admin_create_user_form")) + "?error=Email already exists."
        return RedirectResponse(url=redirect_url, status_code=HTTP_303_SEE_OTHER)

    user_data = schemas.UserCreate(username=username, email=email, password=password)
    user_service.create_user(db, user_create=user_data)

    redirect_url = str(request.url_for("admin_create_user_form")) + f"?message=User {username} created successfully."
    return RedirectResponse(url=redirect_url, status_code=HTTP_303_SEE_OTHER)

@router.post("/settings/api_keys", name="save_api_keys_settings")
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


@router.get("/conversations", response_class=HTMLResponse, name="ui_list_conversations")
async def ui_list_conversations(request: Request):
    """Serves the UI page for viewing conversation logs."""
    return templates.TemplateResponse("conversations.html", {"request": request})

@router.get("/conversations.html", include_in_schema=False)
async def ui_list_conversations_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("ui_list_conversations")))

@router.get("/login", response_class=HTMLResponse, name="ui_login")
async def ui_login_form(request: Request):
    """Serves the UI page for user login."""
    return templates.TemplateResponse("login.html", {"request": request})

@router.get("/login.html", include_in_schema=False)
async def ui_login_form_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("ui_login")))


@router.post("/login", name="ui_login_submit")
async def ui_login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
):
    """Process login credentials and set auth cookie."""
    form_data = {"username": username, "password": password}
    async with httpx.AsyncClient(base_url=str(request.base_url)) as client:
        response = await client.post("/auth/token", data=form_data)

    if response.status_code == 200:
        token = response.json().get("access_token")
        redirect = RedirectResponse(
            url=str(request.url_for("list_prompts_ui")),
            status_code=HTTP_303_SEE_OTHER,
        )
        if token:
            redirect.set_cookie(
                key="prompthelix_access_token",
                value=token,
                httponly=True,
                path="/",
            )
        return redirect

    error = response.json().get("detail", "Login failed. Please check your credentials.")
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error": error},
        status_code=response.status_code,
    )


@router.get("/logout", name="ui_logout")
async def ui_logout(request: Request):
    """Logs out the current user and clears the auth cookie."""
    token = request.cookies.get("prompthelix_access_token")
    if token:
        # The /auth/logout path is absolute
        async with httpx.AsyncClient(base_url=str(request.base_url)) as client:
            await client.post(
                "/auth/logout",
                headers={"Authorization": f"Bearer {token}"},
            )
    redirect = RedirectResponse(url=str(request.url_for("ui_login")), status_code=HTTP_303_SEE_OTHER)
    redirect.delete_cookie("prompthelix_access_token", path="/") # Cookie path root
    return redirect

@router.get("/logout.html", include_in_schema=False)
async def ui_logout_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("ui_logout")))


@router.get("/dashboard", response_class=HTMLResponse, name="ui_dashboard")
async def get_dashboard_ui(request: Request):
    """Serves the UI page for the real-time monitoring dashboard."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@router.get("/dashboard.html", include_in_schema=False)
async def get_dashboard_ui_alias(request: Request):
    return RedirectResponse(url=str(request.url_for("ui_dashboard")))


@router.get("/tests", response_class=HTMLResponse, name="ui_list_tests")
async def ui_list_tests(
    request: Request,
    suite_id: Optional[str] = Query(None),
    selected_tests: List[str] = Query(None) # To retain selection if page reloads with query params
):
    if suite_id is None:
        # Find the default suite or fallback to the first one
        default_suite_config = next((s for s in TEST_SUITES_CONFIG if s.get("default")), None)
        if not default_suite_config and TEST_SUITES_CONFIG:
            default_suite_config = TEST_SUITES_CONFIG[0]
        suite_id = default_suite_config["id"] if default_suite_config else None

    current_tests_list = list_tests_for_suite(suite_id) if suite_id else []

    # Ensure selected_tests is a list, even if None or empty from query
    active_selection = selected_tests if selected_tests is not None else []

    return templates.TemplateResponse(
        "interactive_tests.html",
        {
            "request": request,
            "test_suites": TEST_SUITES_CONFIG,
            "current_suite_id": suite_id,
            "tests": current_tests_list,
            "selected_tests": active_selection,
            "test_results": [],
            "overall_message": None,
            "error_occurred": False,
        },
    )

@router.get("/tests.html", include_in_schema=False)
async def ui_list_tests_alias(request: Request):
    # This simple alias might not correctly carry over query parameters like suite_id
    # For a better UX, ensure any navigation to /tests.html preserves parameters or redirects to the base /tests
    return RedirectResponse(url=str(request.url_for("ui_list_tests")))


@router.post("/tests/run", response_class=HTMLResponse, name="ui_run_test")
async def ui_run_test(
    request: Request,
    test_names: List[str] = Form(...),
    suite_id: str = Form(...) # Make sure suite_id is submitted by the form
):
    test_results = []
    all_tests_passed = True
    error_occurred_during_run = False

    if not test_names:
        current_tests_list = list_tests_for_suite(suite_id) if suite_id else []
        return templates.TemplateResponse(
            "interactive_tests.html",
            {
                "request": request,
                "test_suites": TEST_SUITES_CONFIG,
                "current_suite_id": suite_id,
                "tests": current_tests_list,
                "selected_tests": [],
                "test_results": [],
                "overall_message": "No tests were selected to run.",
                "error_occurred": True,
            },
            status_code=400
        )

    for test_name in test_names:
        try:
            output, success = await asyncio.to_thread(run_unittest, test_name)
            test_results.append({"name": test_name, "output": output, "success": success})
            if not success:
                all_tests_passed = False
        except Exception as e:
            test_results.append({
                "name": test_name,
                "output": f"Failed to run test '{test_name}': {str(e)}",
                "success": False
            })
            all_tests_passed = False
            error_occurred_during_run = True

    current_tests_list = list_tests_for_suite(suite_id) if suite_id else []
    num_total = len(test_names)
    num_passed = sum(1 for r in test_results if r["success"])
    num_failed = num_total - num_passed

    overall_message = f"Executed {num_total} test(s) from suite '{get_test_suite_config_by_id(suite_id)['name'] if get_test_suite_config_by_id(suite_id) else suite_id}': {num_passed} passed, {num_failed} failed."
    if error_occurred_during_run:
        overall_message += " Some tests encountered errors during execution."

    return templates.TemplateResponse(
        "interactive_tests.html",
        {
            "request": request,
            "test_suites": TEST_SUITES_CONFIG,
            "current_suite_id": suite_id,
            "tests": current_tests_list,
            "selected_tests": test_names,  # Keep selected tests checked
            "test_results": test_results,
            "overall_message": overall_message,
            "error_occurred": not all_tests_passed or error_occurred_during_run,
        },
    )
