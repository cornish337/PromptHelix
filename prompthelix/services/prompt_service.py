from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session as DbSession, selectinload
from sqlalchemy import func

from prompthelix.models.prompt_models import Prompt, PromptVersion
# Assuming schemas.py will exist and define these Pydantic models
from prompthelix.schemas import (
    PromptCreate,
    PromptUpdate,
    PromptVersionCreate,
    PromptVersionUpdate
)

class PromptService:
    def create_prompt(self, db: DbSession, prompt_create: PromptCreate) -> Prompt:
        """
        Creates a new prompt.
        """
        db_prompt = Prompt(
            name=prompt_create.name,
            description=prompt_create.description
        )
        db.add(db_prompt)
        db.commit()
        db.refresh(db_prompt)
        return db_prompt

    def get_prompt(self, db: DbSession, prompt_id: int) -> Optional[Prompt]:
        """
        Retrieves a prompt by its ID, including its versions.
        """
        return db.query(Prompt).options(selectinload(Prompt.versions)).filter(Prompt.id == prompt_id).first()

    def get_prompts(self, db: DbSession, skip: int = 0, limit: int = 100) -> List[Prompt]:
        """
        Retrieves a list of prompts with their versions.
        """
        return db.query(Prompt).options(selectinload(Prompt.versions)).offset(skip).limit(limit).all()

    def update_prompt(self, db: DbSession, prompt_id: int, prompt_update: PromptUpdate) -> Optional[Prompt]:
        """
        Updates a prompt's details (name, description).
        """
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
        """
        Deletes a prompt and its associated versions (due to cascade).
        Returns the deleted prompt object if found, otherwise None.
        """
        db_prompt = self.get_prompt(db, prompt_id)
        if db_prompt:
            db.delete(db_prompt)
            db.commit()
            return db_prompt
        return None

    def create_prompt_version(self, db: DbSession, prompt_id: int, version_create: PromptVersionCreate) -> Optional[PromptVersion]:
        """
        Creates a new version for a prompt.
        It automatically determines the next version_number.
        """
        db_prompt = self.get_prompt(db, prompt_id)
        if not db_prompt:
            return None # Prompt not found

        # Determine the next version number
        current_max_version = db.query(func.max(PromptVersion.version_number)).filter(PromptVersion.prompt_id == prompt_id).scalar()
        next_version_number = (current_max_version or 0) + 1

        db_prompt_version = PromptVersion(
            prompt_id=prompt_id,
            version_number=next_version_number,
            content=version_create.content,
            parameters_used=version_create.parameters_used,
            fitness_score=version_create.fitness_score
        )
        db.add(db_prompt_version)
        db.commit()
        db.refresh(db_prompt_version)
        return db_prompt_version

    def get_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        """
        Retrieves a specific prompt version.
        """
        return db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()

    def get_prompt_versions_for_prompt(self, db: DbSession, prompt_id: int, skip: int = 0, limit: int = 100) -> List[PromptVersion]:
        """
        Retrieves all versions for a given prompt.
        """
        return db.query(PromptVersion).filter(PromptVersion.prompt_id == prompt_id).order_by(PromptVersion.version_number).offset(skip).limit(limit).all()

    def update_prompt_version(self, db: DbSession, prompt_version_id: int, version_update: PromptVersionUpdate) -> Optional[PromptVersion]:
        """
        Updates a prompt version's content or other fields.
        """
        db_prompt_version = self.get_prompt_version(db, prompt_version_id)
        if not db_prompt_version:
            return None

        if version_update.content is not None:
            db_prompt_version.content = version_update.content
        if version_update.parameters_used is not None:
            db_prompt_version.parameters_used = version_update.parameters_used
        if version_update.fitness_score is not None:
            db_prompt_version.fitness_score = version_update.fitness_score

        db.add(db_prompt_version)
        db.commit()
        db.refresh(db_prompt_version)
        return db_prompt_version

    def delete_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        """
        Deletes a specific prompt version.
        Returns the deleted version object if found, otherwise None.
        """
        db_prompt_version = self.get_prompt_version(db, prompt_version_id)
        if db_prompt_version:
            db.delete(db_prompt_version)
            db.commit()
            return db_prompt_version
        return None

# Instantiate the service if you want to export an instance
# prompt_service = PromptService()
# Or allow users to instantiate it themselves. For now, we'll export the class.
