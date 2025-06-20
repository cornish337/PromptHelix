from typing import List, Optional, Dict, Any # Optional already here, good.
from redis import Redis # Added
import json # Added
from datetime import datetime # Added

from sqlalchemy.orm import Session as DbSession, selectinload
from sqlalchemy import func, desc, or_ # Added or_

from prompthelix.models.prompt_models import Prompt, PromptVersion
# Assuming schemas.py will exist and define these Pydantic models
from prompthelix.schemas import (
    PromptCreate,
    PromptUpdate,
    PromptVersionCreate,
    PromptVersionUpdate
)

# Helper functions for Caching (copied from prompt_service_db.py)
def prompt_version_to_dict(pv: PromptVersion) -> dict:
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
    """
    Service layer for managing prompts and prompt versions.

    This service handles the business logic for prompt operations, including
    interaction with the SQL database via SQLAlchemy for persistence,
    and utilizes Redis for caching frequently accessed prompt versions to
    enhance performance.

    An optional Redis client instance can be provided during instantiation
    to enable caching. If no client is provided, caching mechanisms will
    be bypassed. Database sessions (DbSession) are passed to each method
    that requires database interaction.
    """
    def __init__(self, redis_client: Optional[Redis] = None):
        """
        Initializes the PromptService.

        Args:
            redis_client: An optional instance of a Redis client. If provided,
                          it will be used for caching prompt versions.
        """
        self.redis = redis_client

    def create_prompt(
        self, db: DbSession, prompt_create: PromptCreate, owner_id: int
    ) -> Prompt:
        """
        Creates a new prompt in the database.

        Args:
            db: The SQLAlchemy database session.
            prompt_create: A schema object containing the data for the new prompt.
            owner_id: The ID of the user who owns this prompt.

        Returns:
            The newly created Prompt object.
        """
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
        """
        Retrieves a specific prompt by its ID from the database.
        This method loads the prompt along with all its associated versions.
        Currently, this method does not utilize caching.

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the prompt to retrieve.

        Returns:
            The Prompt object if found, otherwise None.
        """
        return db.query(Prompt).options(selectinload(Prompt.versions)).filter(Prompt.id == prompt_id).first()

    def get_prompts(self, db: DbSession, skip: int = 0, limit: int = 100) -> List[Prompt]:
        """
        Retrieves a list of prompts from the database with pagination.
        This method loads each prompt along with all its associated versions.
        Currently, this method does not utilize caching.

        Args:
            db: The SQLAlchemy database session.
            skip: The number of prompts to skip (for pagination).
            limit: The maximum number of prompts to return.

        Returns:
            A list of Prompt objects.
        """
        return db.query(Prompt).options(selectinload(Prompt.versions)).offset(skip).limit(limit).all()

    def search_prompts(self, db: DbSession, query: str, owner_id: Optional[int] = None, skip: int = 0, limit: int = 10) -> List[Prompt]:
        """
        Searches for prompts by a query string in their name or description.
        Results can be optionally filtered by owner_id and are paginated.
        This method loads matching prompts along with all their associated versions.
        Currently, this method does not utilize caching.

        Args:
            db: The SQLAlchemy database session.
            query: The search string to match against prompt names and descriptions (case-insensitive).
            owner_id: Optional ID of the owner to filter prompts by.
            skip: The number of matching prompts to skip (for pagination).
            limit: The maximum number of matching prompts to return.

        Returns:
            A list of Prompt objects matching the search criteria.
        """
        search_filter = or_(
            Prompt.name.ilike(f"%{query}%"),
            Prompt.description.ilike(f"%{query}%")
        )

        db_query = db.query(Prompt).options(selectinload(Prompt.versions)).filter(search_filter)

        if owner_id is not None:
            db_query = db_query.filter(Prompt.owner_id == owner_id)

        prompts = db_query.offset(skip).limit(limit).all()
        return prompts

    def update_prompt(self, db: DbSession, prompt_id: int, prompt_update: PromptUpdate) -> Optional[Prompt]:
        """
        Updates a prompt's details (name, description) in the database.
        This method does not directly interact with the cache, but modifications
        to prompt details do not affect version-specific caches.

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the prompt to update.
            prompt_update: A schema object containing the fields to update.

        Returns:
            The updated Prompt object if found and updated, otherwise None.
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
        Deletes a prompt and its associated versions from the database.
        Relies on database cascade for deleting associated PromptVersion records.
        Cache entries for deleted versions are not explicitly cleared here,
        as they will naturally expire or become stale. Future enhancements
        could involve more proactive cache clearing if necessary.

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the prompt to delete.

        Returns:
            The deleted Prompt object if found and deleted, otherwise None.
        """
        db_prompt = self.get_prompt(db, prompt_id)
        if db_prompt:
            db.delete(db_prompt)
            db.commit()
            return db_prompt
        return None

    def create_prompt_version(self, db: DbSession, prompt_id: int, version_create: PromptVersionCreate) -> Optional[PromptVersion]:
        """
        Creates a new version for an existing prompt.

        This method automatically determines the next version_number for the given prompt.
        After successfully creating the new version in the database, it performs
        cache invalidation for related entries in Redis if a Redis client is configured.
        Specifically, it invalidates:
        - The cache entry for the 'latest version' of this prompt (key: "prompt_latest_version:prompt_id:{prompt_id}").
        - The cache entry for the newly created version itself (key: "prompt_version:{new_version_id}").

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the parent prompt for which to create a new version.
            version_create: A schema object containing data for the new prompt version.

        Returns:
            The newly created PromptVersion object, or None if the parent prompt is not found.
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

        # Cache invalidation
        if self.redis:
            try:
                # Invalidate the "latest version" cache for this prompt
                latest_version_cache_key = f"prompt_latest_version:prompt_id:{prompt_id}" # prompt_id from args
                self.redis.delete(latest_version_cache_key)

                # Invalidate the cache for this specific new version
                version_cache_key = f"prompt_version:{db_prompt_version.id}"
                self.redis.delete(version_cache_key)
            except Exception as e_redis:
                # Log Redis error
                pass # Do not let cache invalidation errors interrupt the main flow

        return db_prompt_version

    def get_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        """
        Retrieves a specific prompt version by its ID, utilizing caching.

        If a Redis client is configured:
        1. It first attempts to fetch the prompt version from the cache using the key
           "prompt_version:{prompt_version_id}".
        2. If found in cache (cache hit), the deserialized PromptVersion object is returned.
        3. If not found (cache miss), it fetches the version from the database.
        4. If found in the database, the PromptVersion object is then serialized and stored
           in the cache with a 1-hour expiry before being returned.

        If no Redis client is configured, or if any Redis operation fails,
        it falls back to fetching directly from the database.

        Args:
            db: The SQLAlchemy database session.
            prompt_version_id: The ID of the prompt version to retrieve.

        Returns:
            The PromptVersion object if found (from cache or database), otherwise None.
        """
        cache_key = f"prompt_version:{prompt_version_id}"

        if self.redis:
            try:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    prompt_version_dict = json.loads(cached_data)
                    return dict_to_prompt_version(prompt_version_dict)
            except Exception as e:
                # Log Redis GET or JSON parsing error, e.g., print(f"Redis GET/parse error: {e}")
                pass # Fall through to database lookup

        # If not in cache, or Redis error, fetch from DB
        prompt_version_db = db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()

        if prompt_version_db and self.redis:
            try:
                prompt_version_dict = prompt_version_to_dict(prompt_version_db)
                json_string = json.dumps(prompt_version_dict)
                self.redis.set(cache_key, json_string, ex=3600) # Cache for 1 hour
            except Exception as e:
                # Log Redis SET or JSON conversion error, e.g., print(f"Redis SET/conversion error: {e}")
                pass # Do not fail the request if caching fails

        return prompt_version_db

    def get_latest_prompt_version(self, db: DbSession, prompt_id: int) -> Optional[PromptVersion]:
        """
        Retrieves the latest (highest version_number) prompt version for a given prompt_id,
        utilizing caching.

        If a Redis client is configured:
        1. It first attempts to fetch the prompt version from the cache using the key
           "prompt_latest_version:prompt_id:{prompt_id}".
        2. If found in cache (cache hit), the deserialized PromptVersion object is returned.
        3. If not found (cache miss), it fetches the latest version from the database.
        4. If found in the database, the PromptVersion object is then serialized and stored
           in the cache with a 1-hour expiry before being returned.

        If no Redis client is configured, or if any Redis operation fails,
        it falls back to fetching directly from the database.

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the parent prompt for which to retrieve the latest version.

        Returns:
            The latest PromptVersion object if found (from cache or database), otherwise None.
        """
        cache_key = f"prompt_latest_version:prompt_id:{prompt_id}"

        if self.redis:
            try:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    prompt_version_dict = json.loads(cached_data)
                    return dict_to_prompt_version(prompt_version_dict)
            except Exception as e:
                # Log Redis GET or JSON parsing error
                pass # Fall through to database lookup

        # If not in cache, or Redis error, fetch from DB
        prompt_version_db = db.query(PromptVersion) \
            .filter(PromptVersion.prompt_id == prompt_id) \
            .order_by(desc(PromptVersion.version_number)) \
            .first()

        if prompt_version_db and self.redis:
            try:
                prompt_version_dict = prompt_version_to_dict(prompt_version_db)
                json_string = json.dumps(prompt_version_dict)
                self.redis.set(cache_key, json_string, ex=3600) # Cache for 1 hour
            except Exception as e:
                # Log Redis SET or JSON conversion error
                pass # Do not fail the request if caching fails

        return prompt_version_db

    def get_prompt_versions_for_prompt(self, db: DbSession, prompt_id: int, skip: int = 0, limit: int = 100) -> List[PromptVersion]:
        """
        Retrieves all versions for a given prompt from the database, ordered by version number.
        This method supports pagination.
        Currently, this method does not utilize caching for the list of versions.

        Args:
            db: The SQLAlchemy database session.
            prompt_id: The ID of the parent prompt.
            skip: The number of versions to skip (for pagination).
            limit: The maximum number of versions to return.

        Returns:
            A list of PromptVersion objects.
        """
        return db.query(PromptVersion).filter(PromptVersion.prompt_id == prompt_id).order_by(PromptVersion.version_number).offset(skip).limit(limit).all()

    def update_prompt_version(self, db: DbSession, prompt_version_id: int, version_update: PromptVersionUpdate) -> Optional[PromptVersion]:
        """
        Updates a specific prompt version's content or other mutable fields in the database.

        Note: This method uses `get_prompt_version` internally, which means if the
        version was cached, the update will operate on a potentially detached object
        if the cached object is returned directly by `get_prompt_version` and then modified.
        However, the current implementation of `get_prompt_version` returns a new object
        reconstructed from cache, so this specific issue of modifying a cached instance
        is avoided. The standard practice is to fetch from DB for updates.
        For true robustness, cache invalidation for the specific version should occur here.
        (This is a potential enhancement not covered by the current subtask's scope).

        Args:
            db: The SQLAlchemy database session.
            prompt_version_id: The ID of the prompt version to update.
            version_update: A schema object containing the fields to update.

        Returns:
            The updated PromptVersion object if found and updated, otherwise None.
        """
        # For updates, it's often better to fetch directly from DB to avoid stale data issues with caching.
        # However, to use the existing get_prompt_version:
        db_prompt_version_obj = self.get_prompt_version(db, prompt_version_id)
        if not db_prompt_version_obj:
            return None

        # If the object came from cache, it's not session-bound. We need to get a session-bound object.
        # A simple way is to re-fetch from DB if we detect it's not in session or handle merging.
        # For simplicity, let's assume get_prompt_version (if it hit cache) returns a new, non-bound object.
        # We should query the actual DB object to update it.

        # Re-fetch from DB to ensure we're working with a session-managed instance for update
        db_prompt_version = db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()
        if not db_prompt_version:
             # Should not happen if get_prompt_version found it, but as a safeguard
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

        # Cache invalidation for the updated prompt version
        if self.redis:
            try:
                version_cache_key = f"prompt_version:{db_prompt_version.id}"
                self.redis.delete(version_cache_key)
                # Also invalidate latest_version if this version *was* the latest.
                # This requires checking if it was the latest, or just invalidating unconditionally.
                # For simplicity, let's assume unconditional invalidation of latest if it might have changed.
                latest_version_cache_key = f"prompt_latest_version:prompt_id:{db_prompt_version.prompt_id}"
                self.redis.delete(latest_version_cache_key)
            except Exception as e_redis:
                # Log Redis error
                pass # Do not let cache invalidation errors interrupt

        return db_prompt_version

    def delete_prompt_version(self, db: DbSession, prompt_version_id: int) -> Optional[PromptVersion]:
        """
        Deletes a specific prompt version from the database.
        Cache entries for the deleted version ("prompt_version:{id}") and
        potentially the "latest version" for its parent prompt are cleared if Redis is active.

        Args:
            db: The SQLAlchemy database session.
            prompt_version_id: The ID of the prompt version to delete.

        Returns:
            The deleted PromptVersion object if found and deleted, otherwise None.
        """
        # Fetch directly from DB for deletion to ensure it exists and to get prompt_id for cache invalidation
        db_prompt_version = db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()

        if db_prompt_version:
            prompt_id = db_prompt_version.prompt_id # Get prompt_id before deleting
            db.delete(db_prompt_version)
            db.commit()

            # Cache invalidation
            if self.redis:
                try:
                    version_cache_key = f"prompt_version:{prompt_version_id}"
                    self.redis.delete(version_cache_key)

                    latest_version_cache_key = f"prompt_latest_version:prompt_id:{prompt_id}"
                    self.redis.delete(latest_version_cache_key) # Invalidate latest, as the deleted one might have been the latest
                except Exception as e_redis:
                    # Log Redis error
                    pass
            return db_prompt_version
        return None

# Instantiate the service if you want to export an instance
# prompt_service = PromptService()
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
