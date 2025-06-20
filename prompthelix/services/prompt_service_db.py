import json # Added
from datetime import datetime # Added
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import Optional, List
from redis import Redis # Added for type hinting
from prompthelix.models.prompt_models import Prompt, PromptVersion

# Helper functions for Caching
def prompt_version_to_dict(pv: PromptVersion) -> dict:
    return {
        "id": pv.id,
        "prompt_id": pv.prompt_id,
        "version_number": pv.version_number,
        "content": pv.content,
        "parameters_used": pv.parameters_used,
        "fitness_score": pv.fitness_score,
        "created_at": pv.created_at.isoformat() if pv.created_at else None,
        # Assuming updated_at might also exist and need similar handling
        # "updated_at": pv.updated_at.isoformat() if pv.updated_at else None,
    }

def dict_to_prompt_version(data: dict) -> PromptVersion:
    # Create a new, non-session-bound PromptVersion object
    # Note: This object is not tracked by SQLAlchemy session.
    return PromptVersion(
        id=data.get("id"),
        prompt_id=data.get("prompt_id"),
        version_number=data.get("version_number"),
        content=data.get("content"),
        parameters_used=data.get("parameters_used"),
        fitness_score=data.get("fitness_score"),
        created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        # "updated_at": datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
    )


class PromptDbService:
    def __init__(self, db_session: Session, redis_client: Optional[Redis]): # Updated type hint
        self.db = db_session
        self.redis = redis_client # Ensure this is storing the passed client

    def create_prompt(self, name: str, owner_id: int, content: str, description: Optional[str] = None, parameters_used: Optional[dict] = None) -> PromptVersion:
        # Create new Prompt object
        new_prompt = Prompt(
            name=name,
            description=description,
            owner_id=owner_id
        )
        # self.db.add(new_prompt) # Not needed if using relationship correctly
        # self.db.flush() # Not strictly needed if cascade works as expected or if prompt is added to version via attribute

        # Create new PromptVersion object
        new_prompt_version = PromptVersion(
            prompt=new_prompt, # This should handle adding new_prompt to session if not already added due to cascade.
            version_number=1,
            content=content,
            parameters_used=parameters_used
        )
        self.db.add(new_prompt_version) # Adding version should also add prompt due to relationship and cascade.
                                      # Or add both explicitly: self.db.add_all([new_prompt, new_prompt_version])

        try:
            self.db.commit()
            self.db.refresh(new_prompt) # new_prompt might not be strictly needed by caller, but good to refresh.
            self.db.refresh(new_prompt_version)

            # Cache invalidation
            if self.redis:
                try:
                    latest_version_cache_key = f"prompt_latest_version:prompt_id:{new_prompt.id}"
                    self.redis.delete(latest_version_cache_key)
                    # print(f"Invalidated cache key: {latest_version_cache_key}") # For debugging

                    version_cache_key = f"prompt_version:{new_prompt_version.id}"
                    self.redis.delete(version_cache_key)
                    # print(f"Invalidated cache key: {version_cache_key}") # For debugging
                except Exception as e_redis:
                    # Log Redis error, e.g., print(f"Redis cache invalidation error in create_prompt: {e_redis}")
                    pass # Do not let cache invalidation errors interrupt the main flow

            return new_prompt_version
        except Exception as e:
            self.db.rollback()
            # Consider logging the exception e
            raise e


    def get_prompt_version(self, prompt_version_id: int) -> Optional[PromptVersion]:
        cache_key = f"prompt_version:{prompt_version_id}"

        if self.redis:
            try:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    # print(f"Cache hit for {cache_key}") # For debugging
                    prompt_version_dict = json.loads(cached_data)
                    return dict_to_prompt_version(prompt_version_dict)
            except Exception as e:
                # Log Redis GET or JSON parsing error, e.g., print(f"Redis GET/parse error: {e}")
                # Fall through to database lookup
                pass # Or log specifically: print(f"Cache read error for {cache_key}: {e}")

        # If not in cache, or Redis error, fetch from DB
        # print(f"Cache miss for {cache_key}, fetching from DB") # For debugging
        prompt_version_db = self.db.query(PromptVersion).filter(PromptVersion.id == prompt_version_id).first()

        if prompt_version_db and self.redis:
            try:
                prompt_version_dict = prompt_version_to_dict(prompt_version_db)
                json_string = json.dumps(prompt_version_dict)
                self.redis.set(cache_key, json_string, ex=3600) # Cache for 1 hour
                # print(f"Cached {cache_key} in Redis") # For debugging
            except Exception as e:
                # Log Redis SET or JSON conversion error, e.g., print(f"Redis SET/conversion error: {e}")
                # Do not fail the request if caching fails
                pass # Or log specifically: print(f"Cache write error for {cache_key}: {e}")

        return prompt_version_db

    def get_latest_prompt_version(self, prompt_id: int) -> Optional[PromptVersion]:
        cache_key = f"prompt_latest_version:prompt_id:{prompt_id}"

        if self.redis:
            try:
                cached_data = self.redis.get(cache_key)
                if cached_data:
                    # print(f"Cache hit for {cache_key}") # For debugging
                    prompt_version_dict = json.loads(cached_data)
                    return dict_to_prompt_version(prompt_version_dict)
            except Exception as e:
                # Log Redis GET or JSON parsing error
                # print(f"Cache read error for {cache_key}: {e}")
                pass # Fall through to database lookup

        # If not in cache, or Redis error, fetch from DB
        # print(f"Cache miss for {cache_key}, fetching from DB") # For debugging
        prompt_version_db = self.db.query(PromptVersion) \
            .filter(PromptVersion.prompt_id == prompt_id) \
            .order_by(desc(PromptVersion.version_number)) \
            .first()

        if prompt_version_db and self.redis:
            try:
                prompt_version_dict = prompt_version_to_dict(prompt_version_db)
                json_string = json.dumps(prompt_version_dict)
                self.redis.set(cache_key, json_string, ex=3600) # Cache for 1 hour
                # print(f"Cached {cache_key} in Redis") # For debugging
            except Exception as e:
                # Log Redis SET or JSON conversion error
                # print(f"Cache write error for {cache_key}: {e}")
                pass # Do not fail the request if caching fails

        return prompt_version_db

    def list_prompts(self, owner_id: Optional[int] = None) -> List[Prompt]:
        query = self.db.query(Prompt)
        if owner_id is not None:
            query = query.filter(Prompt.owner_id == owner_id)
        return query.all()

    def create_new_version_for_prompt(self, prompt_id: int, content: str, parameters_used: Optional[dict] = None) -> Optional[PromptVersion]:
        # First, check if the prompt exists
        prompt = self.db.query(Prompt).filter(Prompt.id == prompt_id).first()
        if not prompt:
            return None # Prompt not found

        # Determine the next version number
        latest_version_number = self.db.query(func.max(PromptVersion.version_number)) \
            .filter(PromptVersion.prompt_id == prompt_id) \
            .scalar_one_or_none()

        next_version_number = (latest_version_number or 0) + 1

        new_prompt_version = PromptVersion(
            prompt_id=prompt_id, # Link by foreign key
            version_number=next_version_number,
            content=content,
            parameters_used=parameters_used
        )
        self.db.add(new_prompt_version)
        try:
            self.db.commit()
            self.db.refresh(new_prompt_version)

            # Cache invalidation
            if self.redis:
                try:
                    # Invalidate the "latest version" cache for this prompt
                    latest_version_cache_key = f"prompt_latest_version:prompt_id:{new_prompt_version.prompt_id}"
                    self.redis.delete(latest_version_cache_key)
                    # print(f"Invalidated cache key: {latest_version_cache_key}") # For debugging

                    # Invalidate the cache for this specific version, if it somehow got cached by another call
                    version_cache_key = f"prompt_version:{new_prompt_version.id}"
                    self.redis.delete(version_cache_key)
                    # print(f"Invalidated cache key: {version_cache_key}") # For debugging
                except Exception as e_redis:
                    # Log Redis error, e.g., print(f"Redis cache invalidation error in create_new_version_for_prompt: {e_redis}")
                    pass # Do not let cache invalidation errors interrupt the main flow

            return new_prompt_version
        except Exception as e:
            self.db.rollback()
            # Consider logging the exception
            raise e

    # Placeholder for future methods
    pass
