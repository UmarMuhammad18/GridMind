"""
Thin provider layer so the agent can talk to OpenAI, Anthropic, Groq, or a
local OpenAI-compatible server (e.g. Ollama) through one interface.

Transient errors (rate limits, 5xx, network) are retried with backoff by the
provider SDKs themselves via `api_retries`.
"""

from dataclasses import dataclass
from typing import Any, Protocol

# JSON schema every provider is asked to follow.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "explanation": {"type": "string"},
        "note": {"type": "string"},
    },
    "required": ["action", "explanation", "note"],
    "additionalProperties": False,
}

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {"model": "gpt-4o-mini", "temperature": 0.2},
    "anthropic": {"model": "claude-opus-5-5", "effort": "low"},
    "groq": {"model": "llama-3.3-70b-versatile", "temperature": 0.2,
             "base_url": "https://api.groq.com/openai/v1"},
    "ollama": {"model": "llama3.1", "temperature": 0.2,
               "base_url": "http://localhost:11434/v1"},
}

API_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq": "GROQ_API_KEY",
}


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> LLMResponse: ...


class OpenAIClient:
    """OpenAI Chat Completions; also used for Ollama and other compatible servers."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None,
                 temperature: float | None = None, api_retries: int = 3,
                 provider: str = "openai") -> None:
        from openai import OpenAI

        self.name = f"{provider}:{model}"
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url, max_retries=api_retries)

    def complete(self, system: str, user: str) -> LLMResponse:
        import openai

        kwargs: dict[str, Any] = {"response_format": {"type": "json_object"}}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except openai.APIError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        usage = response.usage
        return LLMResponse(
            text=response.choices[0].message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )


class AnthropicClient:
    """Claude via the Messages API, with output constrained to RESPONSE_SCHEMA."""

    def __init__(self, model: str, api_key: str | None = None, effort: str | None = None,
                 api_retries: int = 3, max_tokens: int = 16000, fallbacks: bool = True) -> None:
        from anthropic import Anthropic

        self.name = f"anthropic:{model}"
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.fallbacks = fallbacks
        self._client = Anthropic(api_key=api_key, max_retries=api_retries)

    def complete(self, system: str, user: str) -> LLMResponse:
        import anthropic

        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}
        }
        if self.effort:
            output_config["effort"] = self.effort

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": output_config,
        }
        try:
            if self.fallbacks:
                # If a safety classifier declines, the API retries on a fallback model.
                response = self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs
                )
            else:
                response = self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        text = ""
        if response.stop_reason != "refusal":
            text = "".join(b.text for b in response.content if b.type == "text")
        return LLMResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )


def make_client(config: dict[str, Any]) -> LLMClient:
    provider = config["provider"]
    api_retries = config.get("api_retries", 3)

    if provider == "anthropic":
        return AnthropicClient(
            model=config["model"],
            api_key=config.get("api_key"),
            effort=config.get("effort"),
            api_retries=api_retries,
            fallbacks=config.get("fallbacks", True),
        )
    if provider in ("openai", "groq", "ollama"):
        return OpenAIClient(
            model=config["model"],
            # Ollama ignores the key, but the OpenAI SDK requires one.
            api_key=config.get("api_key") or ("ollama" if provider == "ollama" else None),
            base_url=config.get("base_url"),
            temperature=config.get("temperature"),
            api_retries=api_retries,
            provider=provider,
        )
    raise ValueError(f"Unknown provider '{provider}'. Choose from: {', '.join(PROVIDER_DEFAULTS)}")
