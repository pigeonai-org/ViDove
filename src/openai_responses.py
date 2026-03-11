from typing import Any, Optional

from openai import AzureOpenAI


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
    "gpt-5-nano",
    "gpt-5.2",
    "gpt-5.3-chat-latest",
    "gpt-5.4",
)

DEFAULT_TEXT_MODEL = "gpt-5-mini"


def normalize_text_model(model_name: Optional[str], default: str = DEFAULT_TEXT_MODEL) -> str:
    candidate = (model_name or default).strip()
    lowered = candidate.lower()
    normalized = LEGACY_OPENAI_TEXT_MODEL_MAP.get(lowered, lowered)
    if normalized in SUPPORTED_OPENAI_TEXT_MODELS:
        return normalized
    raise NotImplementedError(f"Unsupported OpenAI text model: {model_name}")


def provider_for_client(client: Any) -> str:
    return "azure-openai" if isinstance(client, AzureOpenAI) else "openai"


def extract_usage_tokens(response: Any) -> tuple[Optional[int], Optional[int], Optional[int]]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
        getattr(usage, "total_tokens", None),
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
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> tuple[str, Any]:
    request: dict[str, Any] = {
        "model": model,
        "input": input_value,
    }
    if instructions:
        request["instructions"] = instructions
    if temperature is not None:
        request["temperature"] = temperature
    if max_output_tokens is not None:
        request["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**request)
    return get_response_text(response), response
