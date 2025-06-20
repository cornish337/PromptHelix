from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session as DbSession
from sqlalchemy import func

from prompthelix.models.prompt_models import Prompt, PromptVersion
from prompthelix.schemas import (
    PromptCreate,
    PromptUpdate,
    PromptVersionCreate,
    PromptVersionUpdate,
)


def prompt_version_to_dict(pv: PromptVersion) -> dict:
    """Serialize a PromptVersion for caching purposes."""
    return {
        "id": pv.id,
        "prompt_id": pv.prompt_id,
        "version_number": pv.version_number,
        "content": pv.content,
        "parameters_used": pv.parameters_used,
        "fitness_score": pv.fitness_score,
        "created_at": pv.created_at.isoformat() if pv.created_at else None,
    }


def dict_to_prompt_version(data: dict) -> PromptVersion:
    """Deserialize a PromptVersion object from cached data."""
    return PromptVersion(
        id=data.get("id"),
        prompt_id=data.get("prompt_id"),
        version_number=data.get("version_number"),
        content=data.get("content"),
        parameters_used=data.get("parameters_used"),
        fitness_score=data.get("fitness_score"),
        created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
    )


class PromptService:
    """CRUD operations for prompts and prompt versions."""

    def __init__(self, redis_client=None) -> None:
        self.redis = redis_client

    def create_prompt(self, db: DbSession, prompt_create: PromptCreate, owner_id: int) -> Prompt:
        db_prompt = Prompt(
            name=prompt_create.name,
            description=prompt_create.description,
            owner_id=owner_id,
        )
        db.add(db_prompt)
        db.commit()
        db.refresh(db_prompt)
        return db_prompt

    def get_prompt(self, db: DbSession, prompt_id: int) -> Optional[Prompt]:
        return db.query(Prompt).filter(Prompt.id == prompt_id).first()

    def get_prompts(self, db: DbSession, skip: int = 0, limit: int = 100) -> List[Prompt]:
        return db.query(Prompt).offset(skip).limit(limit).all()

    def update_prompt(self, db: DbSession, prompt_id: int, prompt_update: PromptUpdate) -> Optional[Prompt]:
        db_prompt = self.get_prompt(db, prompt_id)
        if not db_prompt:
            return None
        if prompt_update.name is not None:
            db_prompt.name = prompt_update.name
        if prompt_update.description is not None:
            db_prompt.description = prompt_update.description
        db.add(db_prompt)
        db.commit()
        db.refresh(db_prompt)
        return db_prompt

    def delete_prompt(self, db: DbSession, prompt_id: int) -> Optional[Prompt]:
        db_prompt = self.get_prompt(db, prompt_id)
        if db_prompt:
            db.delete(db_prompt)
            db.commit()
            return db_prompt
        return None

    # ----- Prompt Versions -----
    def create_prompt_version(self, db: DbSession, prompt_id: int, version_create: PromptVersionCreate) -> Optional[PromptVersion]:
        if not self.get_prompt(db, prompt_id):
            return None
        current_max = db.query(func.max(PromptVersion.version_number)).filter(PromptVersion.prompt_id == prompt_id).scalar()
        next_version = (current_max or 0) + 1
        db_version = PromptVersion(
            prompt_id=prompt_id,
            version_number=next_version,
            content=version_create.content,
            parameters_used=version_create.parameters_used,
            fitness_score=version_create.fitness_score,
        )
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        return db_version

    def get_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        return db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()

    def get_prompt_versions_for_prompt(self, db: DbSession, prompt_id: int, skip: int = 0, limit: int = 100) -> List[PromptVersion]:
        return (
            db.query(PromptVersion)
            .filter(PromptVersion.prompt_id == prompt_id)
            .order_by(PromptVersion.version_number)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_prompt_version(self, db: DbSession, prompt_version_id: int, version_update: PromptVersionUpdate) -> Optional[PromptVersion]:
        db_version = self.get_prompt_version(db, prompt_version_id)
        if not db_version:
            return None
        if version_update.content is not None:
            db_version.content = version_update.content
        if version_update.parameters_used is not None:
            db_version.parameters_used = version_update.parameters_used
        if version_update.fitness_score is not None:
            db_version.fitness_score = version_update.fitness_score
        db.add(db_version)
        db.commit()
        db.refresh(db_version)
        return db_version

    def delete_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        db_version = self.get_prompt_version(db, prompt_version_id)
        if db_version:
            db.delete(db_version)
            db.commit()
            return db_version
        return None
