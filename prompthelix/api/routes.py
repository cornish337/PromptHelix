from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session as DbSession # Use DbSession for type hinting
from sqlalchemy.exc import IntegrityError
from typing import List, Optional
from pathlib import Path
import asyncio
import subprocess
from datetime import datetime, timedelta
import secrets
import uuid # Added for task_id generation
import logging # Added for logging in background task

from prompthelix.database import get_db, SessionLocal # Added SessionLocal for background task
from prompthelix.api import crud
from prompthelix import schemas
from prompthelix import globals as ph_globals
# Import GeneticAlgorithmRunner for type hinting if direct type checks are needed,
# otherwise, rely on the object having the methods.
# from prompthelix.experiment_runners.ga_runner import GeneticAlgorithmRunner


from prompthelix.models.user_models import User as UserModel # For get_current_user return type
from prompthelix.utils import llm_utils
from prompthelix.orchestrator import main_ga_loop
from prompthelix.genetics.chromosome import PromptChromosome # Updated import

from . import conversation_routes # Added for conversation logs
from .dependencies import get_current_user, oauth2_scheme

# Import services (individual functions, not classes, based on previous service structure)
from prompthelix.services import (
    user_service,
    performance_service,
    get_experiment_runs,
    get_experiment_run,
    get_chromosomes_for_run,
    get_generation_metrics_for_run,
)
from prompthelix.services.prompt_service import PromptService

prompt_service = PromptService()

router = APIRouter()

# Prefix conversation routes with /api/v1
router.include_router(conversation_routes.router, prefix="/api/v1") # Added for conversation logs

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
    print(f"Attempting login for username: {form_data.username}")
    user = user_service.get_user_by_username(db, username=form_data.username)
    if not user:
        print(f"User not found: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    print(f"User found: {form_data.username}. Attempting password verification.")
    password_verified = user_service.verify_password(form_data.password, user.hashed_password)
    print(f"Password verification result for {form_data.username}: {password_verified}")
    if not password_verified:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    from prompthelix.config import settings
    session_duration = settings.DEFAULT_SESSION_EXPIRE_MINUTES
    print(f"Authentication successful for {form_data.username}. Attempting session creation for user ID {user.id} with duration {session_duration} minutes.")
    session = user_service.create_session(
        db,
        user_id=user.id,
        expires_delta_minutes=session_duration,
    )
    print(f"Session created successfully for user ID {user.id}. Session token (first 8 chars): {session.session_token[:8]}...")
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
    return prompt_service.create_prompt(db=db, prompt_create=prompt, owner_id=current_user.id)

@router.get("/api/prompts", response_model=List[schemas.Prompt], tags=["Prompts"], summary="List all prompts", description="Retrieves a list of all prompts, with optional pagination.")
def read_prompts_route(skip: int = 0, limit: int = 100, db: DbSession = Depends(get_db)):
    prompts = prompt_service.get_prompts(db, skip=skip, limit=limit)
    return prompts

@router.get("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Get a specific prompt", description="Retrieves a single prompt by its ID.")
def read_prompt_route(prompt_id: int, db: DbSession = Depends(get_db)):
    db_prompt = prompt_service.get_prompt(db, prompt_id=prompt_id)
    if db_prompt is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    return db_prompt

@router.put("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Update a prompt", description="Updates an existing prompt by its ID. User must be the owner.")
def update_prompt_route(prompt_id: int, prompt_update_data: schemas.PromptUpdate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    db_prompt_existing = prompt_service.get_prompt(db, prompt_id=prompt_id)
    if db_prompt_existing is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if db_prompt_existing.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to modify this prompt")
    db_prompt = prompt_service.update_prompt(db, prompt_id=prompt_id, prompt_update=prompt_update_data)
    return db_prompt

@router.delete("/api/prompts/{prompt_id}", response_model=schemas.Prompt, tags=["Prompts"], summary="Delete a prompt", description="Deletes a prompt by its ID. User must be the owner.")
def delete_prompt_route(prompt_id: int, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    db_prompt_existing = prompt_service.get_prompt(db, prompt_id=prompt_id)
    if db_prompt_existing is None:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if db_prompt_existing.owner_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this prompt")
    db_prompt = prompt_service.delete_prompt(db, prompt_id=prompt_id)
    return db_prompt

# --- PromptVersion Routes ---
@router.post("/api/prompts/{prompt_id}/versions", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Create a new prompt version", description="Creates a new version for a specific prompt. User must be owner of the parent prompt or have appropriate permissions.")
def create_prompt_version_route(prompt_id: int, version: schemas.PromptVersionCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check if current_user can add versions to this prompt
    db_prompt_check = prompt_service.get_prompt(db, prompt_id=prompt_id)  # Check prompt exists
    if db_prompt_check is None:
        raise HTTPException(status_code=404, detail="Prompt not found to associate version with")
    created_version = prompt_service.create_prompt_version(db=db, prompt_id=prompt_id, version_create=version)
    if created_version is None:
        raise HTTPException(status_code=500, detail="Could not create prompt version")
    return created_version

@router.get("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Get a specific prompt version", description="Retrieves a single prompt version by its ID.")
def get_prompt_version_route(version_id: int, db: DbSession = Depends(get_db)):
    db_version = prompt_service.get_prompt_version(db, prompt_version_id=version_id)
    if db_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return db_version

@router.get("/api/prompts/{prompt_id}/versions", response_model=List[schemas.PromptVersion], tags=["Prompt Versions"], summary="List versions for a prompt", description="Retrieves all versions associated with a specific prompt ID, with optional pagination.")
def get_versions_for_prompt_route(prompt_id: int, skip: int = 0, limit: int = 100, db: DbSession = Depends(get_db)):
    versions = prompt_service.get_prompt_versions_for_prompt(db, prompt_id=prompt_id, skip=skip, limit=limit)
    return versions

@router.put("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Update a prompt version", description="Updates an existing prompt version by its ID. User must have appropriate permissions (e.g., owner of parent prompt).")
def update_prompt_version_route(version_id: int, version_update_data: schemas.PromptVersionUpdate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check ownership/permissions
    updated_version = prompt_service.update_prompt_version(db, prompt_version_id=version_id, version_update=version_update_data)
    if updated_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return updated_version

@router.delete("/api/prompt_versions/{version_id}", response_model=schemas.PromptVersion, tags=["Prompt Versions"], summary="Delete a prompt version", description="Deletes a prompt version by its ID. User must have appropriate permissions (e.g., owner of parent prompt).")
def delete_prompt_version_route(version_id: int, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Check ownership/permissions
    deleted_version = prompt_service.delete_prompt_version(db, prompt_version_id=version_id)
    if deleted_version is None:
        raise HTTPException(status_code=404, detail="Prompt version not found")
    return deleted_version

# --- Performance Metrics Routes ---
@router.post("/api/performance_metrics/", response_model=schemas.PerformanceMetric, status_code=status.HTTP_201_CREATED, tags=["Performance Metrics"], summary="Record a performance metric", description="Records a new performance metric for a specific prompt version. User should have permissions to submit metrics for the version.")
def create_performance_metric_route(metric_data: schemas.PerformanceMetricCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # TODO: Authorization: Ensure user can submit metrics for this prompt_version_id
    # Check if prompt version exists
    prompt_version = prompt_service.get_prompt_version(db, prompt_version_id=metric_data.prompt_version_id)
    if not prompt_version:
        raise HTTPException(status_code=404, detail=f"PromptVersion with id {metric_data.prompt_version_id} not found.")
    return performance_service.record_performance_metric(db=db, metric_create=metric_data)

@router.get("/api/prompt_versions/{prompt_version_id}/performance_metrics/", response_model=List[schemas.PerformanceMetric], tags=["Performance Metrics"], summary="Get performance metrics for a version", description="Retrieves all performance metrics recorded for a specific prompt version.")
def get_metrics_for_version_route(prompt_version_id: int, db: DbSession = Depends(get_db)):
    # Check if prompt version exists
    prompt_version = prompt_service.get_prompt_version(db, prompt_version_id=prompt_version_id)
    if not prompt_version:
        raise HTTPException(status_code=404, detail=f"PromptVersion with id {prompt_version_id} not found.")
    return performance_service.get_metrics_for_prompt_version(db=db, prompt_version_id=prompt_version_id)

# Logger for background tasks
background_task_logger = logging.getLogger("ga_background_task")
# Ensure the logger has a handler if not configured globally, e.g., by adding a StreamHandler
if not background_task_logger.hasHandlers():
    background_task_logger.addHandler(logging.StreamHandler()) # TODO: Configure proper logging for background tasks
    background_task_logger.setLevel(logging.INFO)


def run_ga_background_task(
    params: schemas.GAExperimentParams,
    current_user_id: int,
    task_id: str
):
    """
    Background task to run the GA experiment and save the results.
    """
    background_task_logger.info(f"Background task {task_id} started for GA experiment with params: {params.model_dump_json()}")
    db: DbSession = SessionLocal()
    try:
        # Note: main_ga_loop itself might set ph_globals.active_ga_runner
        # This needs to be managed if multiple background tasks run concurrently.
        # For now, BackgroundTasks from FastAPI run sequentially if not using a separate executor.
        # If they run concurrently (e.g. via thread pool), active_ga_runner needs protection or to be instance-based.
        best_chromosome = main_ga_loop(
            task_desc=params.task_description,
            keywords=params.keywords,
            num_generations=params.num_generations,
            population_size=params.population_size,
            elitism_count=params.elitism_count,
            execution_mode=params.execution_mode,
            # Pass other params from schemas.GAExperimentParams if main_ga_loop accepts them
            # e.g. initial_prompt_str, agent_settings_override, llm_settings_override, parallel_workers, population_path
            initial_prompt_str=params.initial_prompt_str,
            agent_settings_override=params.agent_settings_override,
            llm_settings_override=params.llm_settings_override,
            parallel_workers=params.parallel_workers,
            population_path=params.population_path,
            save_frequency_override=params.save_frequency, # Assuming GAExperimentParams has save_frequency
            return_best=True
        )

        if not isinstance(best_chromosome, PromptChromosome):
            background_task_logger.error(f"Task {task_id}: GA did not return a valid prompt chromosome. Result: {best_chromosome}")
            # Potentially update a DB record for the task_id with error status here
            return

        background_task_logger.info(f"Task {task_id}: GA completed. Best chromosome fitness: {best_chromosome.fitness_score}")

        # Logic to save the best_chromosome as a PromptVersion
        # This reuses the logic from the original synchronous route
        local_prompt_service = PromptService() # Create a new service instance for the background task

        target_prompt = None
        if params.parent_prompt_id:
            target_prompt = local_prompt_service.get_prompt(db, prompt_id=params.parent_prompt_id)
            if not target_prompt:
                background_task_logger.error(f"Task {task_id}: Parent prompt with id {params.parent_prompt_id} not found.")
                # Update task status to error
                return
        elif params.prompt_name:
            prompt_create_data = schemas.PromptCreate(name=params.prompt_name, description=params.prompt_description)
            target_prompt = local_prompt_service.create_prompt(db, prompt_create=prompt_create_data, owner_id=current_user_id)
        else:
            default_name = f"GA Generated Prompt - Task {task_id} - {datetime.utcnow().isoformat()}"
            prompt_create_data = schemas.PromptCreate(name=default_name, description=params.prompt_description or f"Generated by GA experiment task {task_id}")
            target_prompt = local_prompt_service.create_prompt(db, prompt_create=prompt_create_data, owner_id=current_user_id)

        if not target_prompt:
            background_task_logger.error(f"Task {task_id}: Could not determine or create target prompt for GA result.")
            # Update task status to error
            return

        ga_params_for_version = params.model_dump(exclude={"parent_prompt_id", "prompt_name", "prompt_description"})
        version_create_data = schemas.PromptVersionCreate(
            content=best_chromosome.to_prompt_string(),
            parameters_used=ga_params_for_version,
            fitness_score=best_chromosome.fitness_score
        )
        created_version = local_prompt_service.create_prompt_version(db, prompt_id=target_prompt.id, version_create=version_create_data)

        if not created_version:
            background_task_logger.error(f"Task {task_id}: Failed to save GA experiment result as a prompt version for prompt ID {target_prompt.id}.")
            # Update task status to error
        else:
            background_task_logger.info(f"Task {task_id}: Successfully saved GA result as PromptVersion ID {created_version.id} for Prompt ID {target_prompt.id}.")
            # Update task status to completed, store created_version.id

    except Exception as e:
        background_task_logger.exception(f"Task {task_id}: An error occurred during GA background task execution: {e}")
        # Update task status to error, store error message
    finally:
        db.close()
        background_task_logger.info(f"Background task {task_id} finished.")


# --- GA Experiment Route (Now uses BackgroundTasks) ---
@router.post("/api/experiments/run-ga", response_model=schemas.GARunResponse, name="api_run_ga_experiment", tags=["Experiments"], summary="Run a Genetic Algorithm experiment in the background", description="Starts a Genetic Algorithm to generate and optimize a prompt. The process runs in the background. The best resulting prompt will be saved as a new version under a specified or new prompt.")
def run_ga_experiment_route(
    params: schemas.GAExperimentParams,
    background_tasks: BackgroundTasks,
    # db: DbSession = Depends(get_db), # db session for main thread if needed for pre-checks
    current_user: UserModel = Depends(get_current_user)
):
    task_id = str(uuid.uuid4())
    background_task_logger.info(f"API: Received request to run GA experiment. Assigning task ID: {task_id}")

    # Optional: Perform any quick pre-checks here if needed before starting the background task
    # For example, validating parent_prompt_id if provided, using the 'db' from Depends(get_db)

    background_tasks.add_task(
        run_ga_background_task,
        params=params,
        current_user_id=current_user.id, # Pass user ID instead of the whole user object
        task_id=task_id
    )

    return schemas.GARunResponse(
        message="GA experiment started in background.",
        task_id=task_id,
        status_endpoint=f"/api/ga/status/{task_id}" # Example, actual status endpoint needs implementation
    )

# --- GA Control Routes ---
# TODO: GA Control routes (/pause, /resume, /cancel, /status) will need to be adapted
# to work with background tasks. ph_globals.active_ga_runner is problematic for multiple tasks.
# A dictionary mapping task_id to runner instances would be needed.
# The status endpoint should ideally query a database record for the task_id.

@router.post("/api/ga/pause", status_code=status.HTTP_200_OK, tags=["GA Control"], summary="Pause the running GA experiment")
def pause_ga_experiment():
    if ph_globals.active_ga_runner:
        try:
            # Check current status before pausing
            runner_status = ph_globals.active_ga_runner.get_status()
            if runner_status.get('status') == "RUNNING":
                ph_globals.active_ga_runner.pause()
                return {"message": "Genetic Algorithm experiment pause request sent."}
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"GA is not running, current status: {runner_status.get('status')}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error pausing GA: {str(e)}")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active Genetic Algorithm experiment found.")

@router.post("/api/ga/resume", status_code=status.HTTP_200_OK, tags=["GA Control"], summary="Resume a paused GA experiment")
def resume_ga_experiment():
    if ph_globals.active_ga_runner:
        try:
            runner_status = ph_globals.active_ga_runner.get_status()
            if runner_status.get('status') == "PAUSED":
                ph_globals.active_ga_runner.resume()
                return {"message": "Genetic Algorithm experiment resume request sent."}
            else:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"GA is not paused, current status: {runner_status.get('status')}")
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error resuming GA: {str(e)}")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active Genetic Algorithm experiment found.")

@router.post("/api/ga/cancel", status_code=status.HTTP_200_OK, tags=["GA Control"], summary="Cancel the running GA experiment")
def cancel_ga_experiment():
    if ph_globals.active_ga_runner:
        try:
            # Runner's 'run' method finally block will set ph_globals.active_ga_runner to None
            ph_globals.active_ga_runner.stop()
            return {"message": "Genetic Algorithm experiment cancel request sent."}
        except Exception as e:
            # This might catch errors if stop() fails, but active_ga_runner might still be set
            # The finally block in runner.run() is the primary mechanism for clearing it.
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error cancelling GA: {str(e)}")
    else:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No active Genetic Algorithm experiment found.")

@router.get("/api/ga/status", response_model=schemas.GAStatusResponse, tags=["GA Control"], summary="Get the status of the current GA experiment")
def get_ga_experiment_status():
    if ph_globals.active_ga_runner:
        try:
            status_data = ph_globals.active_ga_runner.get_status()
            # Ensure all fields required by GAStatusResponse are present in status_data
            # The get_status method in GeneticAlgorithmRunner should provide these.
            # If PopulationManager.get_ga_status() is called, ensure it includes:
            # is_paused, should_stop (added these to GAStatusResponse)
            # The runner's get_status was defined as:
            # pm_status = self.population_manager.get_ga_status()
            # status_report = pm_status.copy()
            # status_report.update(runner_status)
            # status_report['is_paused'] = self.population_manager.is_paused
            # status_report['should_stop'] = self.population_manager.should_stop
            # So, all fields should be present.
            return schemas.GAStatusResponse(**status_data)
        except Exception as e:
            # Consider adding logging here: logger.error(f"Error getting GA status: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error retrieving GA status: {str(e)}")
    else:
        # Return a default "IDLE" status if no runner is active
        # Ensure all required fields for GAStatusResponse are provided.
        # Based on GAStatusResponse, 'status', 'generation', 'population_size' are mandatory.
        return schemas.GAStatusResponse(
            status="IDLE",
            generation=0, # Default value for idle state
            population_size=0, # Default value for idle state
            best_fitness=None,
            fittest_individual_id=None,
            fittest_chromosome_string=None,
            agents_used=[],
            runner_current_generation=0,
            runner_target_generations=0,
            runner_population_manager_id=None,
            is_paused=False,
            should_stop=False
        )


# --- GA History Route ---
@router.get(
    "/api/ga/history",
    response_model=List[schemas.GAGenerationMetric],
    tags=["GA Control"],
    summary="Get GA fitness history",
)
def get_ga_history(
    run_id: int,
    skip: int = 0,
    limit: int = 100,
    db: DbSession = Depends(get_db),
):
    metrics = get_generation_metrics_for_run(db=db, run_id=run_id)
    return metrics[skip : skip + limit]


@router.get(
    "/api/experiments/runs",
    response_model=List[schemas.GAExperimentRun],
    tags=["Experiments"],
    summary="List GA experiment runs",
)
def list_ga_experiment_runs(
    skip: int = 0,
    limit: int = 100,
    db: DbSession = Depends(get_db),
):
    """Return a paginated list of recorded GA experiment runs."""
    return get_experiment_runs(db=db, skip=skip, limit=limit)


@router.get(
    "/api/experiments/runs/{run_id}/chromosomes",
    response_model=List[schemas.GAChromosome],
    tags=["Experiments"],
    summary="Get chromosomes for a GA run",
)
def list_chromosomes_for_run(
    run_id: int,
    skip: int = 0,
    limit: int = 100,
    db: DbSession = Depends(get_db),
):
    """Return chromosomes for the specified run, with optional pagination."""
    run = get_experiment_run(db=db, run_id=run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Experiment run not found")
    records = get_chromosomes_for_run(db=db, run_id=run_id)
    return records[skip : skip + limit]


# --- LLM Utility Routes (Verified, using CRUD layer for stats) ---

@router.post("/api/llm/test_prompt", response_model=schemas.LLMTestResponse, name="test_llm_prompt", tags=["LLM Utilities"], summary="Test a prompt with an LLM", description="Sends a given prompt text to a specified LLM service and returns the response. Increments usage statistics for the LLM service.")
#async def test_llm_prompt_route(request_data: schemas.LLMTestRequest, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
async def test_llm_prompt_route(request_data: schemas.LLMTestRequest, db: DbSession = Depends(get_db)):

    try:
        # llm_utils.call_llm_api is a synchronous function, so remove await
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

@router.post("/api/settings/apikeys/", response_model=schemas.APIKeyDisplay, tags=["Settings"])
def upsert_api_key_route(api_key_data: schemas.APIKeyCreate, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    # crud.create_or_update_api_key now expects schemas.APIKeyCreate
    db_apikey = crud.create_or_update_api_key(db=db, api_key_create=api_key_data)
    # Return APIKeyDisplay to avoid exposing the key
    return schemas.APIKeyDisplay(
        id=db_apikey.id,
        service_name=db_apikey.service_name,
        api_key_hint=f"**********{db_apikey.api_key[-4:]}" if db_apikey.api_key and len(db_apikey.api_key) >=4 else "Not Set", # Ensure api_key is not empty and long enough
        is_set=bool(db_apikey.api_key)
    )

@router.get("/api/settings/apikeys/{service_name}", response_model=schemas.APIKeyDisplay, tags=["Settings"])
def get_api_key_route(service_name: str, db: DbSession = Depends(get_db), current_user: UserModel = Depends(get_current_user)):
    db_apikey = crud.get_api_key(db=db, service_name=service_name)
    if not db_apikey:
        raise HTTPException(status_code=404, detail="API Key not found for this service.")
    return schemas.APIKeyDisplay(
        id=db_apikey.id,
        service_name=db_apikey.service_name,
        api_key_hint=f"**********{db_apikey.api_key[-4:]}" if db_apikey.api_key and len(db_apikey.api_key) >=4 else "Not Set", # Ensure api_key is not empty and long enough
        is_set=bool(db_apikey.api_key)
    )


@router.post("/api/interactive_tests/run", tags=["Tests"])
async def run_interactive_test(test_name: str):
    """Run an interactive pytest file and return its output."""
    root_dir = Path(__file__).resolve().parents[2]
    test_file = root_dir / "tests" / "interactive" / test_name
    if not test_file.is_file():
        raise HTTPException(status_code=404, detail="Test not found")
    try:
        result = await asyncio.to_thread(
            subprocess.run,
            ["pytest", str(test_file)],
            capture_output=True,
            text=True,
        )
        output = result.stdout + result.stderr
        return {"output": output, "returncode": result.returncode}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
