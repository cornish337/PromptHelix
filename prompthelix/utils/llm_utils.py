from prompthelix.config import settings # Import the settings object
import logging

logger = logging.getLogger(__name__)

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Calls the OpenAI API with the given prompt and model.
    (This is a placeholder and needs actual implementation of the API call)
    """
    # API key is now accessed via settings.OPENAI_API_KEY
    # The settings object loads it from environment or DB
    api_key = settings.OPENAI_API_KEY
    if not api_key or api_key == "YOUR_OPENAI_KEY_HERE_CONFIG": # Check against placeholder in config if env var not set
        logger.error("OpenAI API key not configured in environment variables (OPENAI_API_KEY).")
        raise ValueError("OpenAI API key not configured.")

    # Actual OpenAI API call logic would go here using 'api_key'
    logger.info(f"Simulating OpenAI API call with model {model} (key starts with: {api_key[:5]}... if set). Prompt: {prompt[:100]}...")
    # Simulate API response for now
    if "requirements" in prompt:
        return "Parsed requirements: Feature X, Feature Y (Simulated OpenAI)"
    elif "template" in prompt:
        return "Selected template: Template A (Simulated OpenAI)"
    elif "genes" in prompt:
        return "Populated genes: Gene 1, Gene 2 (Simulated OpenAI)"
    return "OpenAI API response (Simulated)"

def call_claude_api(prompt: str, model: str = "claude-2") -> str:
    """
    Calls the Claude API with the given prompt and model.
    (This is a placeholder and needs actual implementation of the API call)
    """
    # API key is now accessed via settings.ANTHROPIC_API_KEY (assuming Claude is Anthropic)
    api_key = settings.ANTHROPIC_API_KEY
    if not api_key or api_key == "YOUR_CLAUDE_KEY_HERE_CONFIG": # Check against placeholder in config if env var not set
        logger.error("Anthropic (Claude) API key not configured in environment variables (ANTHROPIC_API_KEY).")
        raise ValueError("Anthropic (Claude) API key not configured.")

    # Actual Claude API call logic would go here using 'api_key'
    logger.info(f"Simulating Claude API call with model {model} (key starts with: {api_key[:5]}... if set). Prompt: {prompt[:100]}...")
    # Simulate API response for now
    if "requirements" in prompt:
        return "Parsed requirements: Feature X, Feature Y (Simulated Claude)"
    elif "template" in prompt:
        return "Selected template: Template A (Simulated Claude)"
    elif "genes" in prompt:
        return "Populated genes: Gene 1, Gene 2 (Simulated Claude)"
    return "Claude API response (Simulated)"

def call_llm_api(prompt: str, provider: str = "openai", model: str = None) -> str:
    """
    Calls the specified LLM API (OpenAI or Claude).
    """
    if provider == "openai":
        return call_openai_api(prompt, model=model if model else "gpt-3.5-turbo")
    elif provider == "claude":
        return call_claude_api(prompt, model=model if model else "claude-2")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
