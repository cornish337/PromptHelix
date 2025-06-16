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
from openai import (
    OpenAIError,
    APIError as OpenAPIApiError,
    RateLimitError as OpenAIRateLimitError,
    AuthenticationError as OpenAIAuthenticationError,
    APIConnectionError as OpenAIAPIConnectionError,
)
try:  # pragma: no cover – support both old and new OpenAI SDKs
    from openai import InvalidRequestError as OpenAIInvalidRequestError
except ImportError:  # pragma: no cover
    from openai import BadRequestError as OpenAIInvalidRequestError


import anthropic
# Renamed Anthropic client import to avoid conflict with the error type
from anthropic import Anthropic as AnthropicClient, AnthropicError, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError, AuthenticationError as AnthropicAuthenticationError, APIStatusError as AnthropicAPIStatusError, APIConnectionError as AnthropicAPIConnectionError

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
# from google.generativeai.types import BlockedPromptException # Not directly caught for now

logger = logging.getLogger(__name__)

def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", db: Session = None) -> str:
    api_key = get_openai_api_key(db)

    if not api_key:
        logger.warning("OpenAI API key not configured.")
        return "API_KEY_MISSING_ERROR"

    client = openai.OpenAI(api_key=api_key)
    logger.info(f"Calling OpenAI API with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        return content
    except OpenAIRateLimitError as e:
        logger.error(f"OpenAI API RateLimitError: {e}")
        return "RATE_LIMIT_ERROR"
    except OpenAIAuthenticationError as e:
        logger.error(f"OpenAI API AuthenticationError: {e}")
        return "AUTHENTICATION_ERROR"
    except OpenAIAPIConnectionError as e:
        logger.error(f"OpenAI API ConnectionError: {e}")
        return "API_CONNECTION_ERROR"
    except OpenAIInvalidRequestError as e:
        logger.error(f"OpenAI API InvalidRequestError: {e}")
        return "INVALID_REQUEST_ERROR"
    except OpenAPIApiError as e:
        logger.error(f"OpenAI API APIError (general): {e}")
        return "API_ERROR"
    except OpenAIError as e:
        logger.error(f"OpenAI API generic OpenAIError: {e}")
        return "OPENAI_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True)
        return "UNEXPECTED_OPENAI_CALL_ERROR"

def call_claude_api(prompt: str, model: str = "claude-2", db: Session = None) -> str:
    api_key = get_anthropic_api_key(db)

    if not api_key:
        logger.warning("Anthropic API key not configured.")
        return "API_KEY_MISSING_ERROR"

    client = AnthropicClient(api_key=api_key) # Use aliased client
    logger.info(f"Calling Anthropic Claude API with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, 'text'): # Check if it's a TextBlock
                return first_block.text.strip()
            else:
                logger.warning(f"Anthropic API: First content block is not text. Type: {type(first_block)}. Block: {first_block}")
                return "MALFORMED_CLAUDE_RESPONSE_CONTENT"
        logger.warning(f"Anthropic API: Response content is empty or not in expected format. Content: {response.content}")
        return "EMPTY_CLAUDE_RESPONSE"
    except AnthropicRateLimitError as e:
        logger.error(f"Anthropic API RateLimitError: {e}")
        return "RATE_LIMIT_ERROR"
    except AnthropicAuthenticationError as e:
        logger.error(f"Anthropic API AuthenticationError: {e}")
        return "AUTHENTICATION_ERROR"
    except AnthropicAPIStatusError as e:
        logger.error(f"Anthropic API StatusError: {e} (Status Code: {e.status_code if hasattr(e, 'status_code') else 'N/A'})")
        return "API_STATUS_ERROR"
    except AnthropicAPIConnectionError as e:
        logger.error(f"Anthropic API ConnectionError: {e}")
        return "API_CONNECTION_ERROR"
    except AnthropicAPIError as e:
        logger.error(f"Anthropic API APIError (general): {e}")
        return "API_ERROR"
    except AnthropicError as e:
        logger.error(f"Anthropic API generic AnthropicError: {e}")
        return "ANTHROPIC_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during Anthropic API call: {e}", exc_info=True)
        return "UNEXPECTED_ANTHROPIC_CALL_ERROR"

def call_google_api(prompt: str, model: str = "gemini-pro", db: Session = None) -> str:
    api_key = get_google_api_key(db)

    if not api_key:
        logger.warning("Google API key not configured.")
        return "API_KEY_MISSING_ERROR"

    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model)
    logger.info(f"Calling Google Generative AI with model {model}. Prompt snippet: {prompt[:100]}...")
    try:
        response = model_client.generate_content(prompt)
        # Check response content and prompt feedback carefully
        if not response.parts: # No parts usually means blocked or empty
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logger.warning(f"Google API prompt blocked. Reason: {response.prompt_feedback.block_reason}")
                return "BLOCKED_PROMPT_ERROR"
            # Check finish reason if parts are empty and not blocked above
            if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'finish_reason'):
                finish_reason_name = response.candidates[0].finish_reason.name
                # Consider UNSPECIFIED and FINISH_REASON_UNSPECIFIED as non-error/ignorable stop reasons if parts are empty.
                # Only return specific error if it's a more active stop reason like SAFETY, RECITATION etc.
                if finish_reason_name not in ["STOP", "MAX_TOKENS", "UNSPECIFIED", "FINISH_REASON_UNSPECIFIED"]:
                    logger.warning(f"Google API generation finished due to: {finish_reason_name}")
                    return f"GENERATION_STOPPED_{finish_reason_name}"
            logger.warning(f"Google API returned empty response parts and no clear block/stop reason. Response: {response}")
            return "EMPTY_GOOGLE_RESPONSE"

        # If parts exist, try to get text
        # response.text might raise an exception if the prompt itself was blocked,
        # even if parts seems non-empty initially, or if all candidates are blocked.
        try:
          return response.text.strip()
        except ValueError as ve: # Often raised if all candidates are blocked / no valid content
            logger.warning(f"Google API: ValueError accessing response.text (often due to blocked content not caught by prompt_feedback): {ve}. Response: {response}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                return "BLOCKED_PROMPT_ERROR"
            return "EMPTY_GOOGLE_RESPONSE" # Or BLOCKED_CONTENT_ERROR

    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Google Generative AI ResourceExhausted error (likely rate limit/quota): {e}")
        return "RATE_LIMIT_ERROR"
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Google Generative AI PermissionDenied error: {e}")
        return "AUTHENTICATION_ERROR"
    except google_exceptions.InvalidArgument as e:
        logger.error(f"Google Generative AI InvalidArgument error: {e}")
        return "INVALID_ARGUMENT_ERROR"
    except google_exceptions.InternalServerError as e:
        logger.error(f"Google Generative AI InternalServerError: {e}")
        return "API_SERVER_ERROR"
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google Generative AI APIError (general): {e}")
        return "API_ERROR"
    # except genai.types.BlockedPromptException as e: # If direct catch is needed
    #     logger.error(f"Google Generative AI BlockedPromptException: {e}")
    #     return "BLOCKED_PROMPT_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during Google API call: {e}", exc_info=True)
        if "google.generativeai" in str(type(e)) or "google.api_core" in str(type(e)):
            logger.error(f"Google API/SDK specific library error: {e}")
            return "GOOGLE_SDK_ERROR"
        return "UNEXPECTED_GOOGLE_CALL_ERROR"

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
        return "UNSUPPORTED_PROVIDER_ERROR"


# Backwards compatibility for older tests
def call_llm_api_directly(prompt: str, provider: str, model: Optional[str] = None, db: Session = None) -> str:
    return call_llm_api(prompt, provider, model, db)
