from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Dict, Any
from prompthelix.enums import ExecutionMode

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
    execution_mode: ExecutionMode = ExecutionMode.REAL
    keywords: List[str] = Field(default_factory=list)
    num_generations: int = 10
    population_size: int = 20
    elitism_count: int = 2
    parent_prompt_id: Optional[int] = None
    prompt_name: Optional[str] = None
    prompt_description: Optional[str] = None

# GAExperimentResult will be represented by schemas.PromptVersion.
# If more fields are needed later, a dedicated model can be created.

# Schemas for API Keys

class APIKeyBase(BaseModel):
    service_name: str = Field(..., description="The name of the LLM service (e.g., OPENAI, ANTHROPIC, GOOGLE)")
    api_key: str = Field(..., description="The API key for the service.")

class APIKeyCreate(APIKeyBase):
    pass # No additional fields needed for creation, inherits all from Base

class APIKeyDisplay(BaseModel):
    service_name: str
    api_key_hint: Optional[str] = Field(None, description="A non-sensitive hint of the API key (e.g., last 4 characters or 'Set')")
    is_set: bool = Field(..., description="True if the API key is set, False otherwise")

    class Config:
        from_attributes = True # If needed for ORM model conversion, though APIKeyDisplay is custom

# It might also be useful to have a schema for the APIKey model itself, for reading from DB
class APIKey(APIKeyBase):
    id: int

    class Config:
        from_attributes = True # For ORM mode

# New Schemas for LLM Testing and Statistics
class LLMTestRequest(BaseModel):
    llm_service: str = Field(..., description="The LLM service to use (e.g., OpenAI, Anthropic, Google)")
    prompt_text: str = Field(..., description="The text prompt to send to the LLM")

class LLMTestResponse(BaseModel):
    llm_service: str = Field(..., description="The LLM service used")
    response_text: str = Field(..., description="The text response from the LLM")

class LLMStatistic(BaseModel):
    llm_service: str = Field(..., description="The LLM service")
    request_count: int = Field(..., description="The number of requests made to this LLM service")

    class Config:
        from_attributes = True # For ORM mode if we map it directly to a model later
