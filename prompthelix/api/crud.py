from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Optional
from prompthelix.models import Prompt, PromptVersion
from prompthelix.models.settings_models import APIKey
from prompthelix.schemas import PromptCreate, PromptVersionCreate
# Assuming schemas for APIKey will be created in a later step,
# for now, create_or_update_api_key will just take strings.
# We might need: from prompthelix import schemas

def create_prompt(db: Session, prompt: PromptCreate) -> Prompt:
    db_prompt = Prompt(name=prompt.name, description=prompt.description)
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

def get_prompt(db: Session, prompt_id: int) -> Prompt | None:
    return db.query(Prompt).filter(Prompt.id == prompt_id).first()

def get_prompts(db: Session, skip: int = 0, limit: int = 100) -> list[Prompt]:
    return db.query(Prompt).offset(skip).limit(limit).all()

def update_prompt(db: Session, prompt_id: int, prompt_update: PromptCreate) -> Prompt | None:
    db_prompt = get_prompt(db, prompt_id)
    if db_prompt:
        db_prompt.name = prompt_update.name
        db_prompt.description = prompt_update.description
        db.commit()
        db.refresh(db_prompt)
    return db_prompt

def delete_prompt(db: Session, prompt_id: int) -> Prompt | None:
    db_prompt = get_prompt(db, prompt_id)
    if db_prompt:
        db.delete(db_prompt)
        db.commit()
    return db_prompt

def create_prompt_version(db: Session, version: PromptVersionCreate, prompt_id: int) -> PromptVersion | None:
    db_prompt = get_prompt(db, prompt_id)
    if not db_prompt:
        return None

    # Determine the next version number
    last_version_number = db.query(func.max(PromptVersion.version_number)).filter(PromptVersion.prompt_id == prompt_id).scalar() or 0
    next_version_number = last_version_number + 1

    db_version = PromptVersion(
        prompt_id=prompt_id,
        content=version.content,
        parameters_used=version.parameters_used,
        fitness_score=version.fitness_score,
        version_number=next_version_number
    )
    db.add(db_version)
    db.commit()
    db.refresh(db_version)
    return db_version

# Functions for APIKey

def get_api_key(db: Session, service_name: str) -> Optional[APIKey]:
    return db.query(APIKey).filter(APIKey.service_name == service_name).first()

# No specific schema for api_key_data yet, using individual parameters
def create_or_update_api_key(db: Session, service_name: str, api_key_value: str) -> APIKey:
    db_api_key = db.query(APIKey).filter(APIKey.service_name == service_name).first()
    if db_api_key:
        db_api_key.api_key = api_key_value
    else:
        db_api_key = APIKey(service_name=service_name, api_key=api_key_value)
        db.add(db_api_key)
    db.commit()
    db.refresh(db_api_key)
    return db_api_key
