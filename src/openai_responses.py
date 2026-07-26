import logging
import random
from time import sleep
from typing import Any, Optional

import openai
from openai import AzureOpenAI


logger = logging.getLogger(__name__)

LEGACY_OPENAI_TEXT_MODEL_MAP = {
    "gpt-3.5-turbo": "gpt-5-mini",
    "gpt-4": "gpt-5",
    "gpt-4o": "gpt-5",
    "gpt-4o-mini": "gpt-5-mini",
    "gpt-5.2-mini": "gpt-5.2",
    "gpt-5.2-nano": "gpt-5.2",
    "gpt-5.3-instant": "gpt-5.3-chat-latest",
    "gpt-5.4-thinking": "gpt-5.4",
}

SUPPORTED_OPENAI_TEXT_MODELS = (
    "gpt-5",
    "gpt-5-mini",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5-nano",
    "gpt-5.2",
    "gpt-5.3-chat-latest",
    "gpt-5.4",
)

DEFAULT_TEXT_MODEL = "gpt-5-mini"

# Errors that are worth retrying: rate limits, connection issues, timeouts, 5xx.
TRANSIENT_OPENAI_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


def normalize_text_model(model_name: Optional[str], default: str = DEFAULT_TEXT_MODEL) -> str:
    candidate = (model_name or default).strip()
    if not candidate:
        raise ValueError("OpenAI text model name must be a non-empty string")
    lowered = candidate.lower()
    normalized = LEGACY_OPENAI_TEXT_MODEL_MAP.get(lowered, lowered)
    if normalized in SUPPORTED_OPENAI_TEXT_MODELS:
        return normalized
    return candidate


def provider_for_client(client: Any) -> str:
    return "azure-openai" if isinstance(client, AzureOpenAI) else "openai"


def _usage_field(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, dict):
        return container.get(key)
    return getattr(container, key, None)


def extract_usage_tokens(
    response: Any,
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None, None, None, None
    input_details = _usage_field(usage, "input_tokens_details") or _usage_field(
        usage, "prompt_tokens_details"
    )
    return (
        _usage_field(usage, "input_tokens") or _usage_field(usage, "prompt_tokens"),
        _usage_field(usage, "output_tokens")
        or _usage_field(usage, "completion_tokens"),
        _usage_field(usage, "total_tokens"),
        _usage_field(input_details, "cached_tokens"),
    )


def get_response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()

    collected: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            content_text = getattr(content, "text", None)
            if content_text:
                collected.append(str(content_text))
    return "\n".join(collected).strip()


def create_response_text(
    client: Any,
    *,
    model: str,
    input_value: Any,
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_attempts: int = 3,
) -> tuple[str, Any]:
    """Create a Responses API completion with retry on transient failures.

    Rate limits, connection errors, timeouts, and 5xx responses are retried
    with exponential backoff (on top of the SDK's own low-level retries).
    Non-transient errors (e.g. BadRequestError) propagate to the caller.
    """
    request: dict[str, Any] = {
        "model": model,
        "input": input_value,
    }
    if instructions:
        request["instructions"] = instructions
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    last_error: Exception | None = None
    for attempt in range(max(1, max_attempts)):
        try:
            response = client.responses.create(**request)
            return get_response_text(response), response
        except TRANSIENT_OPENAI_ERRORS as e:
            last_error = e
            backoff = min(30.0, 2.0 ** attempt) + random.random()
            logger.warning(
                "Transient OpenAI error on %s (attempt %d/%d): %s. Retrying in %.1fs",
                model,
                attempt + 1,
                max_attempts,
                e,
                backoff,
            )
            sleep(backoff)

    assert last_error is not None
    raise last_error
