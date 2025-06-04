from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any

class PromptVersionBase(BaseModel):
    content: str
    # Updated to Dict[str, Any] for more flexible parameter storage
    parameters_used: Optional[Dict[str, Any]] = None
    fitness_score: Optional[float] = None

class PromptVersionCreate(PromptVersionBase):
    pass

class PromptVersion(PromptVersionBase):
    id: int
    prompt_id: int
    version_number: int
    created_at: datetime

    class Config:
        from_attributes = True

class PromptBase(BaseModel):
    name: str
    description: Optional[str] = None

class PromptCreate(PromptBase):
    pass

class Prompt(PromptBase):
    id: int
    created_at: datetime
    versions: List[PromptVersion] = []

    class Config:
        from_attributes = True

# New Schemas for GA Experiment
class GAExperimentParams(BaseModel):
    task_description: str
    keywords: List[str] = Field(default_factory=list)
    num_generations: int = 10
    population_size: int = 20
    elitism_count: int = 2
    parent_prompt_id: Optional[int] = None
    prompt_name: Optional[str] = None
    prompt_description: Optional[str] = None

# GAExperimentResult will be represented by schemas.PromptVersion.
# If more fields are needed later, a dedicated model can be created.
