from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session as DbSession # Use DbSession for type hinting
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from datetime import datetime, timedelta
import secrets

from prompthelix.database import get_db
from prompthelix.api import crud
from prompthelix import schemas

from prompthelix.models.user_models import User as UserModel # For get_current_user return type
from prompthelix.utils import llm_utils
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.engine import PromptChromosome


# Import services (individual functions, not classes, based on previous service structure)
from prompthelix.services import user_service, performance_service
# PromptService is a class, so it's used via crud.py which instantiates it.

router = APIRouter()

# --- Authentication Setup ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: DbSession = Depends(get_db)) -> UserModel:
    db_session = user_service.get_session_by_token(db, session_token=token)
    if not db_session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if db_session.expires_at < datetime.utcnow():
        user_service.delete_session(db, session_token=token) # Clean up expired session
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = user_service.get_user(db, user_id=db_session.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found for session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user

# --- User Management Routes ---
@router.post("/users/", response_model=schemas.User, status_code=status.HTTP_201_CREATED, tags=["Users"], summary="Create a new user", description="Registers a new user in the system.")
def create_user_route(user_data: schemas.UserCreate, db: DbSession = Depends(get_db)):
    existing_user = user_service.get_user_by_username(db, username=user_data.username)
    if existing_user:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already registered")
    existing_email = user_service.get_user_by_email(db, email=user_data.email)
    if existing_email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    try:
        return user_service.create_user(db=db, user_create=user_data)
    except IntegrityError: # Should be caught by above checks, but as a safeguard
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username or email already exists")

@router.post("/auth/token", response_model=schemas.Token, tags=["Authentication"], summary="User login", description="Authenticates a user and returns an access token.")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: DbSession = Depends(get_db)):
    user = user_service.get_user_by_username(db, username=form_data.username)
    if not user or not user_service.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    from prompthelix.config import settings
    session = user_service.create_session(
        db,
        user_id=user.id,
        expires_delta_minutes=settings.DEFAULT_SESSION_EXPIRE_MINUTES,
    )
    return {"access_token": session.session_token, "token_type": "bearer"}

@router.post("/auth/logout", tags=["Authentication"], summary="User logout", description="Logs out the current user by invalidating their session token.")
async def logout(current_user: UserModel = Depends(get_current_user), token: str = Depends(oauth2_scheme), db: DbSession = Depends(get_db)):
    # The token is implicitly the one used by get_current_user
    deleted = user_service.delete_session(db=db, session_token=token)
    if deleted:
        return {"message": "Successfully logged out"}
    # If get_current_user passed, session must have been valid.
    # This path (deleted=False) should ideally not be reached if token was valid.
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not log out")


@router.get("/users/me", response_model=schemas.User, tags=["Users"], summary="Get current user", description="Retrieves the details of the currently authenticated user.")
async def read_users_me(current_user: UserModel = Depends(get_current_user)):
    return current_user

# --- Prompt Routes (Verified, using CRUD layer which delegates to PromptService) ---
@router.post("/api/prompts", response_model=schemas.Prompt, tags=["Prompts"], summary="Create a new prompt", description="Creates a new prompt for the authenticated user.")
def create_prompt_route(
    prompt: schemas.PromptCreate,
    db: DbSession = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    return crud.create_prompt(db=db, prompt=prompt, owner_id=current_user.id)

@router.get("/api/prompts", response_model=List[schemas.Prompt], tags=["Prompts"], summary="List all prompts", description="Retrieves a list of all prompts, with optional pagination.")
def read_prompts_route(skip: int = 0, limit: int = 100, db: DbSession = Depends(get_db)):
    prompts = crud.get_prompts(db, skip=skip, limit=limit)
    return prompts

@router.get("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Get a specific prompt", description="Retrieves a single prompt by its ID.")
def read_prompt_route(prompt_id: int, db: DbSession = Depends(get_db)):
    db_prompt = crud.get_prompt(db, prompt_id=prompt_id)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return db_prompt

@router.put("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Update a prompt", description="Updates an existing prompt by its ID. User must be the owner.")
def update_prompt_route(prompt_id: int, prompt_update_data: schemas.PromptUpdate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    db_prompt_existing = crud.get_prompt(db, prompt_id=prompt_id)
    if db_prompt_existing is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if db_prompt_existing.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prompt")
    db_prompt = crud.update_prompt(db, prompt_id=prompt_id, prompt_update=prompt_update_data)
    return db_prompt

@router.delete("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Delete a prompt", description="Deletes a prompt by its ID. User must be the owner.")
def delete_prompt_route(prompt_id: int, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    db_prompt_existing = crud.get_prompt(db, prompt_id=prompt_id)
    if db_prompt_existing is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if db_prompt_existing.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this prompt")
    db_prompt = crud.delete_prompt(db, prompt_id=prompt_id)
    return db_prompt

# --- PromptVersion Routes ---
@router.post("/api/prompts/{prompt_id}/versions", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Create a new prompt version", description="Creates a new version for a specific prompt. User must be owner of the parent prompt or have appropriate permissions.")
def create_prompt_version_route(prompt_id: int, version: schemas.PromptVersionCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check if current_user can add versions to this prompt
    db_prompt_check = crud.get_prompt(db, prompt_id=prompt_id) # Check prompt exists
    if db_prompt_check is None:
        raise HTTPException(status_code=404, detail="Prompt not found to associate version with")
    created_version = crud.create_prompt_version(db=db, version=version, prompt_id=prompt_id)
    if created_version is None:
        raise HTTPException(status_code=500, detail="Could not create prompt version")
    return created_version

@router.get("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Get a specific prompt version", description="Retrieves a single prompt version by its ID.")
def get_prompt_version_route(version_id: int, db: DbSession = Depends(get_db)):
    db_version = crud.get_prompt_version(db, prompt_version_id=version_id)
    if db_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return db_version

@router.get("/api/prompts/{prompt_id}/versions", response_model=List[schemas.PromptVersion], tags=["Prompt Versions"], summary="List versions for a prompt", description="Retrieves all versions associated with a specific prompt ID, with optional pagination.")
def get_versions_for_prompt_route(prompt_id: int, skip: int = 0, limit: int = 100, db: DbSession = Depends(get_db)):
    versions = crud.get_prompt_versions_for_prompt(db, prompt_id=prompt_id, skip=skip, limit=limit)
    return versions

@router.put("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Update a prompt version", description="Updates an existing prompt version by its ID. User must have appropriate permissions (e.g., owner of parent prompt).")
def update_prompt_version_route(version_id: int, version_update_data: schemas.PromptVersionUpdate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check ownership/permissions
    updated_version = crud.update_prompt_version(db, prompt_version_id=version_id, version_update=version_update_data)
    if updated_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return updated_version

@router.delete("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Delete a prompt version", description="Deletes a prompt version by its ID. User must have appropriate permissions (e.g., owner of parent prompt).")
def delete_prompt_version_route(version_id: int, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check ownership/permissions
    deleted_version = crud.delete_prompt_version(db, prompt_version_id=version_id)
    if deleted_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return deleted_version

# --- Performance Metrics Routes ---
@router.post("/api/performance_metrics/", response_model=schemas.PerformanceMetric, status_code=status.HTTP_201_CREATED, tags=["Performance Metrics"], summary="Record a performance metric", description="Records a new performance metric for a specific prompt version. User should have permissions to submit metrics for the version.")
def create_performance_metric_route(metric_data: schemas.PerformanceMetricCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Authorization: Ensure user can submit metrics for this prompt_version_id
    # Check if prompt version exists
    prompt_version = crud.get_prompt_version(db, prompt_version_id=metric_data.prompt_version_id)
    if not prompt_version:
        raise HTTPException(status_code=404, detail=f"PromptVersion with id {metric_data.prompt_version_id} not found.")
    return performance_service.record_performance_metric(db=db, metric_create=metric_data)

@router.get("/api/prompt_versions/{prompt_version_id}/performance_metrics/", response_model=List[schemas.PerformanceMetric], tags=["Performance Metrics"], summary="Get performance metrics for a version", description="Retrieves all performance metrics recorded for a specific prompt version.")
def get_metrics_for_version_route(prompt_version_id: int, db: DbSession = Depends(get_db)):
    # Check if prompt version exists
    prompt_version = crud.get_prompt_version(db, prompt_version_id=prompt_version_id)
    if not prompt_version:
        raise HTTPException(status_code=404, detail=f"PromptVersion with id {prompt_version_id} not found.")
    return performance_service.get_metrics_for_prompt_version(db=db, prompt_version_id=prompt_version_id)

# --- GA Experiment Route (Verified, using CRUD layer) ---
@router.post("/api/experiments/run-ga", response_model=schemas.PromptVersion, name="api_run_ga_experiment", tags=["Experiments"], summary="Run a Genetic Algorithm experiment", description="Executes a Genetic Algorithm to generate and optimize a prompt based on the provided parameters. The best resulting prompt is saved as a new version under a specified or new prompt.")
def run_ga_experiment_route(params: schemas.GAExperimentParams, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # ... (existing GA logic from file, ensuring it calls new crud methods)
    best_chromosome = main_ga_loop(
        task_desc=params.task_description,
        keywords=params.keywords,
        num_generations=params.num_generations,
        population_size=params.population_size,
        elitism_count=params.elitism_count,
        execution_mode=params.execution_mode,
        return_best=True
    )

    if not isinstance(best_chromosome, PromptChromosome):
        raise HTTPException(status_code=500, detail="GA did not return a valid prompt chromosome.")

    target_prompt = None
    if params.parent_prompt_id:
        target_prompt = crud.get_prompt(db, prompt_id=params.parent_prompt_id)
        if not target_prompt:
            raise HTTPException(status_code=404, detail=f"Parent prompt with id {params.parent_prompt_id} not found.")
    elif params.prompt_name:
        prompt_create_data = schemas.PromptCreate(name=params.prompt_name, description=params.prompt_description)
        target_prompt = crud.create_prompt(db, prompt=prompt_create_data, owner_id=current_user.id)
    else:
        default_name = f"GA Generated Prompt - {datetime.utcnow().isoformat()}"
        prompt_create_data = schemas.PromptCreate(name=default_name, description=params.prompt_description or "Generated by GA experiment")
        target_prompt = crud.create_prompt(db, prompt=prompt_create_data, owner_id=current_user.id)

    if not target_prompt:
        raise HTTPException(status_code=500, detail="Could not determine or create target prompt for GA result.")

    ga_params_for_version = params.model_dump(exclude={"parent_prompt_id", "prompt_name", "prompt_description"})
    version_create_data = schemas.PromptVersionCreate(
        content=best_chromosome.to_prompt_string(),
        parameters_used=ga_params_for_version,
        fitness_score=best_chromosome.fitness_score
    )
    created_version = crud.create_prompt_version(db, version=version_create_data, prompt_id=target_prompt.id)
    if not created_version:
        raise HTTPException(status_code=500, detail="Failed to save GA experiment result as a prompt version.")
    return created_version

# --- LLM Utility Routes (Verified, using CRUD layer for stats) ---
@router.post("/api/llm/test_prompt", response_model=schemas.LLMTestResponse, name="test_llm_prompt", tags=["LLM Utilities"], summary="Test a prompt with an LLM", description="Sends a given prompt text to a specified LLM service and returns the response. Increments usage statistics for the LLM service.")
async def test_llm_prompt_route(request_data: schemas.LLMTestRequest, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    try:
        response_text = llm_utils.call_llm_api(
            prompt=request_data.prompt_text, provider=request_data.llm_service, db=db
        )
        try:
            crud.increment_llm_statistic(db=db, service_name=request_data.llm_service)
        except Exception as e:
            print(f"Error incrementing LLM statistic for {request_data.llm_service}: {e}")
        return schemas.LLMTestResponse(llm_service=request_data.llm_service, response_text=response_text)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Unexpected error in test_llm_prompt_route: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@router.get("/api/llm/statistics", response_model=List[schemas.LLMStatistic], name="get_llm_statistics", tags=["LLM Utilities"], summary="Get LLM usage statistics", description="Retrieves statistics on the usage of different LLM services, such as call counts.")
async def get_llm_statistics_route(db: DbSession = Depends(get_db)):
    return crud.get_all_llm_statistics(db=db)

@router.get("/api/llm/available", response_model=List[str], name="get_available_llms", tags=["LLM Utilities"], summary="List available LLM services", description="Returns a list of LLM service names that are configured and available for use in the system.")
async def get_available_llms_route(db: DbSession = Depends(get_db)):
    try:
        return llm_utils.list_available_llms(db=db)
    except Exception as e:
        print(f"Error getting available LLMs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve available LLMs.")

# Note: APIKey routes are not in this file based on provided content.
# If they were, they'd be updated to use `crud.create_or_update_api_key(db=db, api_key_create=api_key_data)`
# and `crud.get_api_key(db=db, service_name=service_name)`.
# For example:
# @router.post("/api/settings/apikeys/", response_model=schemas.APIKeyDisplay, tags=["Settings"])
# def upsert_api_key_route(api_key_data: schemas.APIKeyCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
#     # crud.create_or_update_api_key now expects schemas.APIKeyCreate
#     db_apikey = crud.create_or_update_api_key(db=db, api_key_create=api_key_data)
#     # Return APIKeyDisplay to avoid exposing the key
#     return schemas.APIKeyDisplay(
#         id=db_apikey.id,
#         service_name=db_apikey.service_name,
#         api_key_hint=f"**********{db_apikey.api_key[-4:]}" if db_apikey.api_key else "Not Set",
#         is_set=bool(db_apikey.api_key)
#     )

# @router.get("/api/settings/apikeys/{service_name}", response_model=schemas.APIKeyDisplay, tags=["Settings"])
# def get_api_key_route(service_name: str, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
#     db_apikey = crud.get_api_key(db=db, service_name=service_name)
#     if not db_apikey:
#         raise HTTPException(status_code=404, detail="API Key not found for this service.")
#     return schemas.APIKeyDisplay(
#         id=db_apikey.id,
#         service_name=db_apikey.service_name,
#         api_key_hint=f"**********{db_apikey.api_key[-4:]}" if db_apikey.api_key else "Not Set",
#         is_set=bool(db_apikey.api_key)
#     )
