import datetime
from prompthelix.config import settings  # Import the settings object
import logging
from sqlalchemy.orm import Session  # Used for DB API key lookup if available
from prompthelix.api import crud  # For potential DB access to API keys
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
    AsyncOpenAI,  # Added AsyncOpenAI
)
import time # For latency measurement
from prompthelix.metrics import ( # For LLM metrics
    LLM_CALL_LATENCY_SECONDS,
    LLM_INPUT_TOKENS_TOTAL,
    LLM_OUTPUT_TOKENS_TOTAL,
    LLM_CALLS_TOTAL
)

try:  # pragma: no cover – support both old and new OpenAI SDKs
    from openai import InvalidRequestError as OpenAIInvalidRequestError
except ImportError:  # pragma: no cover
    from openai import BadRequestError as OpenAIInvalidRequestError

import anthropic
# Renamed Anthropic client import to avoid conflict with the error type
from anthropic import AnthropicError, APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError, \
    AuthenticationError as AnthropicAuthenticationError, APIStatusError as AnthropicAPIStatusError, \
    APIConnectionError as AnthropicAPIConnectionError
from anthropic import AsyncAnthropic as AsyncAnthropicClient  # Added AsyncAnthropicClient

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# from google.generativeai.types import BlockedPromptException # Not directly caught for now

logger = logging.getLogger(__name__)
LOG_FILE_PATH = "llm_api_calls.log"


def list_available_llms(db: Session | None = None) -> list[str]:
    """Return the list of LLM service names with configured API keys."""
    services = []
    # Try loading from DB if CRUD helper available
    if hasattr(crud, "get_api_key"):
        for name in ["OPENAI", "ANTHROPIC", "GOOGLE"]:
            if crud.get_api_key(db, service_name=name):
                services.append(name)
    else:
        if get_openai_api_key(db):
            services.append("OPENAI")
        if get_anthropic_api_key(db):
            services.append("ANTHROPIC")
        if get_google_api_key(db):
            services.append("GOOGLE")
    return services


async def call_openai_api(prompt: str, model: str = "gpt-3.5-turbo", db: Session = None) -> str:  # Changed to async def
    api_key = get_openai_api_key(db)

    if not api_key:
        logger.warning("OpenAI API key not configured.")
        return "API_KEY_MISSING_ERROR"

    client = AsyncOpenAI(api_key=api_key)  # Changed to AsyncOpenAI
    logger.info(f"Calling AsyncOpenAI API with model {model}. Prompt snippet: {prompt[:100]}...")

    processed_prompt = prompt
    start_time = time.time()
    status = "success"
    input_tokens = 0
    output_tokens = 0

    try:
        # Attempt to parse the prompt if it looks like a JSON list of strings
        if prompt.strip().startswith('[') and prompt.strip().endswith(']'):
            import json
            try:
                prompt_parts = json.loads(prompt)
                if isinstance(prompt_parts, list) and all(isinstance(part, str) for part in prompt_parts):
                    processed_prompt = "\n".join(prompt_parts)
                    logger.info(
                        f"Successfully parsed and joined JSON string array prompt. New prompt snippet: {processed_prompt[:100]}...")
                else:
                    logger.warning(
                        "Prompt looked like JSON array but failed to parse into list of strings. Using original prompt.")
            except json.JSONDecodeError:
                logger.warning("Prompt looked like JSON array but failed to decode. Using original prompt.")
    except Exception as e:
        logger.error(f"Error during prompt processing: {e}. Using original prompt.", exc_info=True)

    try:
        response = await client.chat.completions.acreate(  # Changed to acreate and added await
            model=model,
            messages=[{"role": "user", "content": processed_prompt}],
        )
        content = response.choices[0].message.content.strip()
        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
        return content
    except OpenAIRateLimitError as e:
        logger.error(f"AsyncOpenAI API RateLimitError: {e}")
        status = "error_rate_limit"
        return "RATE_LIMIT_ERROR"
    except OpenAIAuthenticationError as e:
        logger.error(f"OpenAI API AuthenticationError: {e}")
        status = "error_auth"
        return "AUTHENTICATION_ERROR"
    except OpenAIAPIConnectionError as e:
        logger.error(f"OpenAI API ConnectionError: {e}")
        status = "error_connection"
        return "API_CONNECTION_ERROR"
    except OpenAIInvalidRequestError as e:
        logger.error(f"OpenAI API InvalidRequestError: {e}")
        status = "error_invalid_request"
        return "INVALID_REQUEST_ERROR"
    except OpenAPIApiError as e:
        logger.error(f"OpenAI API APIError (general): {e}")
        status = "error_api"
        return "API_ERROR"
    except OpenAIError as e:
        logger.error(f"OpenAI API generic OpenAIError: {e}")
        status = "error_openai"
        return "OPENAI_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during OpenAI API call: {type(e).__name__} - {e}", exc_info=True)
        status = "error_unexpected"
        return "UNEXPECTED_OPENAI_CALL_ERROR"
    finally:
        latency = time.time() - start_time
        LLM_CALL_LATENCY_SECONDS.labels(llm_provider="openai", llm_model=model).observe(latency)
        LLM_CALLS_TOTAL.labels(llm_provider="openai", llm_model=model, status=status).inc()
        if input_tokens > 0:
            LLM_INPUT_TOKENS_TOTAL.labels(llm_provider="openai", llm_model=model).inc(input_tokens)
        if output_tokens > 0:
            LLM_OUTPUT_TOKENS_TOTAL.labels(llm_provider="openai", llm_model=model).inc(output_tokens)


async def call_claude_api(prompt: str, model: str = "claude-2", db: Session = None) -> str:  # Changed to async def
    api_key = get_anthropic_api_key(db)

    if not api_key:
        logger.warning("Anthropic API key not configured.")
        return "API_KEY_MISSING_ERROR"

    client = AsyncAnthropicClient(api_key=api_key)  # Changed to AsyncAnthropicClient
    logger.info(f"Calling AsyncAnthropic Claude API with model {model}. Prompt snippet: {prompt[:100]}...")
    start_time = time.time()
    status = "success"
    input_tokens = 0
    output_tokens = 0

    try:
        response = await client.messages.acreate(  # Changed to acreate and added await
            model=model,
            max_tokens=1024,  # Assuming acreate supports max_tokens directly, or it's part of model config
            messages=[{"role": "user", "content": prompt}],
        )
        if response.usage:
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

        if response.content and isinstance(response.content, list) and len(response.content) > 0:
            first_block = response.content[0]
            if hasattr(first_block, 'text'):  # Check if it's a TextBlock
                return first_block.text.strip()
            else:
                logger.warning(
                    f"Anthropic API: First content block is not text. Type: {type(first_block)}. Block: {first_block}")
                status = "error_malformed_response"
                return "MALFORMED_CLAUDE_RESPONSE_CONTENT"
        logger.warning(
            f"Anthropic API: Response content is empty or not in expected format. Content: {response.content}")
        status = "error_empty_response"
        return "EMPTY_CLAUDE_RESPONSE"
    except AnthropicRateLimitError as e:
        logger.error(f"Anthropic API RateLimitError: {e}")
        status = "error_rate_limit"
        return "RATE_LIMIT_ERROR"
    except AnthropicAuthenticationError as e:
        logger.error(f"Anthropic API AuthenticationError: {e}")
        status = "error_auth"
        return "AUTHENTICATION_ERROR"
    except AnthropicAPIStatusError as e:
        logger.error(
            f"Anthropic API StatusError: {e} (Status Code: {e.status_code if hasattr(e, 'status_code') else 'N/A'})")
        status = "error_api_status"
        return "API_STATUS_ERROR"
    except AnthropicAPIConnectionError as e:
        logger.error(f"Anthropic API ConnectionError: {e}")
        status = "error_connection"
        return "API_CONNECTION_ERROR"
    except AnthropicAPIError as e:
        logger.error(f"Anthropic API APIError (general): {e}")
        status = "error_api"
        return "API_ERROR"
    except AnthropicError as e:
        logger.error(f"Anthropic API generic AnthropicError: {e}")
        status = "error_anthropic"
        return "ANTHROPIC_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during Anthropic API call: {e}", exc_info=True)
        status = "error_unexpected"
        return "UNEXPECTED_ANTHROPIC_CALL_ERROR"
    finally:
        latency = time.time() - start_time
        LLM_CALL_LATENCY_SECONDS.labels(llm_provider="anthropic", llm_model=model).observe(latency)
        LLM_CALLS_TOTAL.labels(llm_provider="anthropic", llm_model=model, status=status).inc()
        if input_tokens > 0:
            LLM_INPUT_TOKENS_TOTAL.labels(llm_provider="anthropic", llm_model=model).inc(input_tokens)
        if output_tokens > 0:
            LLM_OUTPUT_TOKENS_TOTAL.labels(llm_provider="anthropic", llm_model=model).inc(output_tokens)


async def call_google_api(prompt: str, model: str = "gemini-pro", db: Session = None) -> str:  # Changed to async def
    api_key = get_google_api_key(db)

    if not api_key:
        logger.warning("Google API key not configured.")
        return "API_KEY_MISSING_ERROR"

    genai.configure(api_key=api_key)  # This is synchronous, should be fine to call once.
    model_client = genai.GenerativeModel(model)
    logger.info(f"Calling Async Google Generative AI with model {model}. Prompt snippet: {prompt[:100]}...")
    start_time = time.time()
    status = "success"
    input_tokens = 0
    output_tokens = 0 # Google API makes this harder to get without another call for response.text

    try:
        # Get input token count before the main call
        try:
            count_response = await model_client.count_tokens_async(prompt)
            input_tokens = count_response.total_tokens
        except Exception as e_count:
            logger.warning(f"Google API: Could not count input tokens for model {model}. Error: {e_count}")

        response = await model_client.generate_content_async(prompt)  # Changed to generate_content_async

        # Check for usage metadata if available (newer Gemini versions might include it)
        if hasattr(response, 'usage_metadata'):
            if response.usage_metadata: # Ensure it's not None
                # Fields are typically prompt_token_count and candidates_token_count
                if hasattr(response.usage_metadata, 'prompt_token_count'):
                    input_tokens = response.usage_metadata.prompt_token_count # Override if available
                if hasattr(response.usage_metadata, 'candidates_token_count'): # This is for output
                    output_tokens = response.usage_metadata.candidates_token_count
                elif hasattr(response.usage_metadata, 'total_token_count') and input_tokens > 0: # Fallback if only total is given
                    output_tokens = response.usage_metadata.total_token_count - input_tokens


        # Check response content and prompt feedback carefully
        if not response.parts:  # No parts usually means blocked or empty
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                logger.warning(f"Google API prompt blocked. Reason: {response.prompt_feedback.block_reason}")
                status = "error_blocked_prompt"
                return "BLOCKED_PROMPT_ERROR"
            # Check finish reason if parts are empty and not blocked above
            if hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0],
                                                                                   'finish_reason'):
                finish_reason_name = response.candidates[0].finish_reason.name
                # Consider UNSPECIFIED and FINISH_REASON_UNSPECIFIED as non-error/ignorable stop reasons if parts are empty.
                # Only return specific error if it's a more active stop reason like SAFETY, RECITATION etc.
                if finish_reason_name not in ["STOP", "MAX_TOKENS", "UNSPECIFIED", "FINISH_REASON_UNSPECIFIED"]:
                    logger.warning(f"Google API generation finished due to: {finish_reason_name}")
                    status = f"error_stopped_{finish_reason_name.lower()}"
                    return f"GENERATION_STOPPED_{finish_reason_name}"
            logger.warning(
                f"Google API returned empty response parts and no clear block/stop reason. Response: {response}")
            status = "error_empty_response"
            return "EMPTY_GOOGLE_RESPONSE"

        # If parts exist, try to get text
        # response.text might raise an exception if the prompt itself was blocked,
        # even if parts seems non-empty initially, or if all candidates are blocked.
        try:
            return response.text.strip()
        except ValueError as ve:  # Often raised if all candidates are blocked / no valid content
            logger.warning(
                f"Google API: ValueError accessing response.text (often due to blocked content not caught by prompt_feedback): {ve}. Response: {response}")
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
                status = "error_blocked_prompt_value_error"
                return "BLOCKED_PROMPT_ERROR"
            status = "error_empty_response_value_error"
            return "EMPTY_GOOGLE_RESPONSE"  # Or BLOCKED_CONTENT_ERROR

    except google_exceptions.ResourceExhausted as e:
        logger.error(f"Google Generative AI ResourceExhausted error (likely rate limit/quota): {e}")
        status = "error_rate_limit"
        return "RATE_LIMIT_ERROR"
    except google_exceptions.PermissionDenied as e:
        logger.error(f"Google Generative AI PermissionDenied error: {e}")
        status = "error_auth"
        return "AUTHENTICATION_ERROR"
    except google_exceptions.InvalidArgument as e:
        logger.error(f"Google Generative AI InvalidArgument error: {e}")
        status = "error_invalid_argument"
        return "INVALID_ARGUMENT_ERROR"
    except google_exceptions.InternalServerError as e:
        logger.error(f"Google Generative AI InternalServerError: {e}")
        status = "error_server"
        return "API_SERVER_ERROR"
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google Generative AI APIError (general): {e}")
        status = "error_api"
        return "API_ERROR"
    # except genai.types.BlockedPromptException as e: # If direct catch is needed
    #     logger.error(f"Google Generative AI BlockedPromptException: {e}")
    #     status = "error_blocked_prompt_exception"
    #     return "BLOCKED_PROMPT_ERROR"
    except Exception as e:
        logger.error(f"An unexpected error occurred during Google API call: {e}", exc_info=True)
        if "google.generativeai" in str(type(e)) or "google.api_core" in str(type(e)):
            logger.error(f"Google API/SDK specific library error: {e}")
            status = "error_google_sdk"
            return "GOOGLE_SDK_ERROR"
        status = "error_unexpected"
        return "UNEXPECTED_GOOGLE_CALL_ERROR"
    finally:
        latency = time.time() - start_time
        LLM_CALL_LATENCY_SECONDS.labels(llm_provider="google", llm_model=model).observe(latency)
        LLM_CALLS_TOTAL.labels(llm_provider="google", llm_model=model, status=status).inc()
        if input_tokens > 0:
            LLM_INPUT_TOKENS_TOTAL.labels(llm_provider="google", llm_model=model).inc(input_tokens)
        if output_tokens > 0: # Will be 0 if not available from API
            LLM_OUTPUT_TOKENS_TOTAL.labels(llm_provider="google", llm_model=model).inc(output_tokens)


async def call_llm_api(prompt: str, provider: str, model: Optional[str] = None,
                       db: Session = None) -> str:  # Changed to async def
    # This wrapper function does not need to change much, as the metric updates
    # are now handled within each specific provider's function (call_openai_api, etc.)
    # The existing logging within this function can remain as is.

    logger.info(f"Async call_llm_api invoked with provider: {provider}, model: {model}")
    provider_lower = provider.lower()
    timestamp = datetime.datetime.now().isoformat()

    determined_model = ""
    if provider_lower == "openai":
        determined_model = model if model else "gpt-3.5-turbo"
    elif provider_lower == "anthropic":
        determined_model = model if model else "claude-2"
    elif provider_lower == "google":
        determined_model = model if model else "gemini-pro"
    else:
        # For unsupported providers, or if model is None
        determined_model = model if model else "unknown_model"

    api_path = f"{provider_lower}/{determined_model}"

    # Ensure prompt is a string, even if it's lengthy.
    # For logging, newlines in prompt should be escaped or handled if they break log format.
    # Python's f-string handles multiline strings in `prompt` correctly for the variable itself.
    # When writing to file, newlines in `prompt` will be preserved.
    log_entry_start = f"Timestamp: {timestamp}\nAPI Path: {api_path}\nPrompt: {prompt}\n"

    result = ""
    error_message_for_log = None  # Use a different variable name to avoid confusion with API result

    try:
        if provider_lower == "openai":
            result = await call_openai_api(prompt, model=determined_model, db=db)  # Added await
            if result in ["API_KEY_MISSING_ERROR", "RATE_LIMIT_ERROR", "AUTHENTICATION_ERROR", "API_CONNECTION_ERROR",
                          "INVALID_REQUEST_ERROR", "API_ERROR", "OPENAI_ERROR", "UNEXPECTED_OPENAI_CALL_ERROR"]:
                error_message_for_log = result
        elif provider_lower == "anthropic":
            result = await call_claude_api(prompt, model=determined_model, db=db)  # Added await
            if result in ["API_KEY_MISSING_ERROR", "RATE_LIMIT_ERROR", "AUTHENTICATION_ERROR", "API_STATUS_ERROR",
                          "API_CONNECTION_ERROR", "API_ERROR", "ANTHROPIC_ERROR", "UNEXPECTED_ANTHROPIC_CALL_ERROR",
                          "MALFORMED_CLAUDE_RESPONSE_CONTENT", "EMPTY_CLAUDE_RESPONSE"]:
                error_message_for_log = result
        elif provider_lower == "google":
            result = await call_google_api(prompt, model=determined_model, db=db)  # Added await
            google_error_conditions = [
                "API_KEY_MISSING_ERROR", "RATE_LIMIT_ERROR", "AUTHENTICATION_ERROR",
                "INVALID_ARGUMENT_ERROR", "API_SERVER_ERROR", "API_ERROR",
                "BLOCKED_PROMPT_ERROR", "EMPTY_GOOGLE_RESPONSE", "GOOGLE_SDK_ERROR",
                "UNEXPECTED_GOOGLE_CALL_ERROR"
            ]
            if result in google_error_conditions or (
                    isinstance(result, str) and result.startswith("GENERATION_STOPPED_")):
                error_message_for_log = result
        else:
            logger.error(f"Unsupported LLM provider: {provider}")
            result = "UNSUPPORTED_PROVIDER_ERROR"
            error_message_for_log = result

    except Exception as e:
        # This is a fallback for unexpected errors during the call_llm_api orchestration itself
        # or if a sub-call raises an exception instead of returning an error string.
        logger.error(f"Unexpected error in call_llm_api orchestration or sub-call: {e}", exc_info=True)
        # Try to form a descriptive error string if possible
        error_str = str(e)
        # Check for common error types that might not be caught by string checks
        if isinstance(e, (OpenAIError, AnthropicError, google_exceptions.GoogleAPIError)):
            # This provides a more specific error if the sub-functions unexpectedly raised.
            error_str = f"{type(e).__name__}: {str(e)}"

        result = f"UNEXPECTED_CALL_LLM_API_ERROR"  # Generic result for the caller
        error_message_for_log = f"UNEXPECTED_CALL_LLM_API_ERROR: {error_str}"

    # Perform logging
    try:
        with open(LOG_FILE_PATH, "a") as f:
            f.write(log_entry_start)  # This includes the prompt
            if error_message_for_log:
                f.write(f"Error: {error_message_for_log}\n")
            f.write("---\n")  # Separator for entries
    except Exception as e:
        logger.error(f"Failed to write to LLM API call log: {e}", exc_info=True)
        # Do not propagate logging errors to the caller of call_llm_api

    return result


# Backwards compatibility for older tests
async def call_llm_api_directly(prompt: str, provider: str, model: Optional[str] = None,
                                db: Session = None) -> str:  # Changed to async def
    return await call_llm_api(prompt, provider, model, db)  # Added await
