from prompthelix.config import settings  # Import the settings object
import logging
from sqlalchemy.orm import Session  # Added for list_available_llms
from prompthelix.api import crud  # Added for list_available_llms
from prompthelix.config import (
    get_openai_api_key,
    get_anthropic_api_key,
    get_google_api_key,
)
from typing import Optional  # Added for Optional type hint

import openai
from openai import OpenAIError
import anthropic
from anthropic import Anthropic, AnthropicError
import google.generativeai as genai

logger = logging.getLogger(__name__)

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", db: Session = None) -> str:
    api_key = get_openai_api_key(db)

    if not api_key:
        logger.error("OpenAI API key not configured. Using simulated response.")
        return f"Simulated OpenAI API response for model {model}: '{prompt}'"

    client = openai.OpenAI(api_key=api_key)
    logger.info(f"Calling OpenAI API with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        return content
    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise

def call_claude_api(prompt: str, model: str = "claude-2", db: Session = None) -> str:
    api_key = get_anthropic_api_key(db)

    if not api_key:
        logger.error("Anthropic (Claude) API key not configured. Using simulated response.")
        return f"Simulated Claude API response for model {model}: '{prompt}'"

    client = Anthropic(api_key=api_key)
    logger.info(f"Calling Anthropic Claude API with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content:
            return response.content[0].text.strip()
        return ""
    except AnthropicError as e:
        logger.error(f"Anthropic API error: {e}")
        raise

def call_google_api(prompt: str, model: str = "gemini-pro", db: Session = None) -> str:
    api_key = get_google_api_key(db)

    if not api_key:
        logger.error("Google API key not configured. Using simulated response.")
        return f"Simulated Google API response for model {model}: '{prompt}'"

    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model)
    logger.info(f"Calling Google Generative AI with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = model_client.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Google Generative AI error: {e}")
        raise

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
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Backwards compatibility for older tests
def call_llm_api_directly(prompt: str, provider: str, model: Optional[str] = None, db: Session = None) -> str:
    return call_llm_api(prompt, provider, model, db)
