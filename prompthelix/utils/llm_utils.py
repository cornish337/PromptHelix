from prompthelix.config import settings # Import the settings object
import logging
from sqlalchemy.orm import Session # Added for list_available_llms
from prompthelix.api import crud # Added for list_available_llms
from prompthelix.config import get_openai_api_key, get_anthropic_api_key, get_google_api_key
from typing import Optional # Added for Optional type hint

logger = logging.getLogger(__name__)

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", db: Session = None) -> str:
    api_key = get_openai_api_key(db) # Use the function from config.py

    if not api_key: # Simplified check, as get_..._api_key handles fallback
        logger.error("OpenAI API key not configured.")
        raise ValueError("OpenAI API key not configured. Please set it in the settings or environment.")

    logger.info(f"Simulating OpenAI API call with model {model} (key exists). Prompt: {prompt[:100]}...")
    return f"Simulated OpenAI API response for model {model}: '{prompt}'"

def call_claude_api(prompt: str, model: str = "claude-2", db: Session = None) -> str:
    api_key = get_anthropic_api_key(db) # Use the function from config.py

    if not api_key:
        logger.error("Anthropic (Claude) API key not configured.")
        raise ValueError("Anthropic (Claude) API key not configured. Please set it in the settings or environment.")

    logger.info(f"Simulating Claude API call with model {model} (key exists). Prompt: {prompt[:100]}...")
    return f"Simulated Claude API response for model {model}: '{prompt}'"

def call_google_api(prompt: str, model: str = "gemini-pro", db: Session = None) -> str:
    api_key = get_google_api_key(db) # Use the function from config.py

    if not api_key:
        logger.error("Google API key not configured.")
        raise ValueError("Google API key not configured. Please set it in the settings or environment.")

    logger.info(f"Simulating Google API call with model {model} (key exists). Prompt: {prompt[:100]}...")
    return f"Simulated Google API response for model {model}: '{prompt}'"

def call_llm_api(prompt: str, provider: str, model: Optional[str] = None, db: Session = None) -> str:
    logger.info(f"call_llm_api invoked with provider: {provider}, model: {model}")
    provider_lower = provider.lower()

    if provider_lower == "openai":
        model_to_use = model if model else "gpt-3.5-turbo"
        return call_openai_api(prompt, model=model_to_use, db=db)
    elif provider_lower == "anthropic":
        model_to_use = model if model else "claude-2"
        return call_claude_api(prompt, model=model_to_use, db=db)
    elif provider_lower == "google":
        model_to_use = model if model else "gemini-pro"
        return call_google_api(prompt, model=model_to_use, db=db)
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")

def list_available_llms(db: Session) -> list[str]:
    available_services = []
    # These are the services also considered in `ui_routes.SUPPORTED_LLM_SERVICES`
    # and `config.py` for API key retrieval.
    # The name string (e.g., "OPENAI") should match what `crud.get_api_key` expects.
    service_checks = [
        ("OPENAI", get_openai_api_key(db)),
        ("ANTHROPIC", get_anthropic_api_key(db)),
        ("GOOGLE", get_google_api_key(db)),
    ]

    for service_name, api_key_value in service_checks:
        if api_key_value: # If a key is found (either from DB or environment via config functions)
            available_services.append(service_name)

    logger.info(f"Available LLMs based on configured keys: {available_services}")
    return available_services
