from sqlalchemy.orm import Session
from typing import Optional, List
from sqlalchemy import func
from typing import Optional
from prompthelix.models import Prompt, PromptVersion
from prompthelix.models.settings_models import APIKey
from prompthelix.schemas import PromptCreate, PromptVersionCreate
# Import models that are still directly used (APIKey, LLMUsageStatistic)
from prompthelix.models.statistics_models import LLMUsageStatistic
# Assuming schemas for APIKey will be created in a later step,
# for now, create_or_update_api_key will just take strings.
# We might need: from prompthelix import schemas


# Import Pydantic Schemas required for function signatures
from prompthelix import schemas # This will make all schemas available under schemas.<SchemaName>

# Import Services
from prompthelix.services import user_service, performance_service # user and perf services might not be used here yet
from prompthelix.services.prompt_service import PromptService

# Instantiate services that are used
prompt_service_instance = PromptService()

# --- Prompt CRUD Functions (delegating to PromptService) ---

def create_prompt(db: Session, prompt: schemas.PromptCreate) -> schemas.Prompt:
    # The service now returns a model instance, which FastAPI will convert using the response_model (schemas.Prompt)
    return prompt_service_instance.create_prompt(db=db, prompt_create=prompt)

def get_prompt(db: Session, prompt_id: int) -> Optional[schemas.Prompt]:
    return prompt_service_instance.get_prompt(db=db, prompt_id=prompt_id)

def get_prompts(db: Session, skip: int = 0, limit: int = 100) -> List[schemas.Prompt]:
    return prompt_service_instance.get_prompts(db=db, skip=skip, limit=limit)

def update_prompt(db: Session, prompt_id: int, prompt_update: schemas.PromptUpdate) -> Optional[schemas.Prompt]:
    # Ensure prompt_update is of type schemas.PromptUpdate as defined in the task
    return prompt_service_instance.update_prompt(db=db, prompt_id=prompt_id, prompt_update=prompt_update)

def delete_prompt(db: Session, prompt_id: int) -> Optional[schemas.Prompt]:
    return prompt_service_instance.delete_prompt(db=db, prompt_id=prompt_id)

# --- PromptVersion CRUD Functions (delegating to PromptService) ---

def create_prompt_version(db: Session, version: schemas.PromptVersionCreate, prompt_id: int) -> Optional[schemas.PromptVersion]:
    return prompt_service_instance.create_prompt_version(db=db, prompt_id=prompt_id, version_create=version)

def get_prompt_version(db: Session, prompt_version_id: int) -> Optional[schemas.PromptVersion]:
    return prompt_service_instance.get_prompt_version(db=db, prompt_version_id=prompt_version_id)

def get_prompt_versions_for_prompt(db: Session, prompt_id: int, skip: int = 0, limit: int = 100) -> List[schemas.PromptVersion]:
    return prompt_service_instance.get_prompt_versions_for_prompt(db=db, prompt_id=prompt_id, skip=skip, limit=limit)

def update_prompt_version(db: Session, prompt_version_id: int, version_update: schemas.PromptVersionUpdate) -> Optional[schemas.PromptVersion]:
    return prompt_service_instance.update_prompt_version(db=db, prompt_version_id=prompt_version_id, version_update=version_update)

def delete_prompt_version(db: Session, prompt_version_id: int) -> Optional[schemas.PromptVersion]:
    return prompt_service_instance.delete_prompt_version(db=db, prompt_version_id=prompt_version_id)


# --- APIKey Functions (kept as is for now, but using new schemas) ---

def get_api_key(db: Session, service_name: str) -> Optional[APIKey]: # Returns model
    return db.query(APIKey).filter(APIKey.service_name == service_name).first()

def create_or_update_api_key(db: Session, api_key_create: schemas.APIKeyCreate) -> APIKey: # Takes schema, returns model
    db_api_key = db.query(APIKey).filter(APIKey.service_name == api_key_create.service_name).first()
    if db_api_key:
        db_api_key.api_key = api_key_create.api_key
    else:
        db_api_key = APIKey(service_name=api_key_create.service_name, api_key=api_key_create.api_key)
        db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    return db_api_key


# --- LLMUsageStatistic Functions (kept as is for now, using new/updated schemas if applicable) ---

def get_llm_statistic(db: Session, service_name: str) -> Optional[LLMUsageStatistic]: # Returns model
    return db.query(LLMUsageStatistic).filter(LLMUsageStatistic.llm_service == service_name).first()

# For creating, let's assume we might pass a schema, or stick to service_name if simple
def create_llm_statistic(db: Session, stat_create: schemas.LLMUsageStatisticCreate) -> LLMUsageStatistic: # Takes schema
    """Creates a new LLM usage statistic entry."""
    # This function in the original crud.py created with count 0.
    # The schema LLMUsageStatisticCreate defaults request_count to 1.
    # For consistency with previous behavior, let's use the schema's default or allow override.
    # If the goal is to always start at 0 upon explicit "creation" vs "increment":
    # db_statistic = LLMUsageStatistic(llm_service=stat_create.llm_service, request_count=0)
    # However, services usually handle the creation logic. If this is direct DB access:
    db_statistic = LLMUsageStatistic(llm_service=stat_create.llm_service, request_count=stat_create.request_count)
    db.add(db_statistic)
    db.commit()
    db.refresh(db_statistic)
    return db_statistic

def increment_llm_statistic(db: Session, service_name: str) -> LLMUsageStatistic: # Returns model
    """Increments the request count for an LLM service. Creates the entry if it doesn't exist."""
    db_statistic = get_llm_statistic(db, service_name=service_name)
    if db_statistic:
        db_statistic.request_count += 1
    else:
        db_statistic = LLMUsageStatistic(llm_service=service_name, request_count=1) # Start with 1 on first increment
        db.add(db_statistic)
    db.commit()
    db.refresh(db_statistic)
    return db_statistic

def get_all_llm_statistics(db: Session) -> List[LLMUsageStatistic]: # Returns list of models
    return db.query(LLMUsageStatistic).order_by(LLMUsageStatistic.llm_service).all()

# Note: User and PerformanceMetric CRUD functions are not part of this file as per the task.
# They would typically be in their respective API route files, calling their services.
# If they were meant to be added here, the user_service and performance_service would be used.
