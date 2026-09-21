"""
LLM provider abstraction.

Currently supports:
  - openai  (official OpenAI API)
  - groq    (Groq – fast open models, OpenAI-compatible)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from openai import OpenAI


class BaseProvider(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.15, max_tokens: int = 350) -> str:
        ...


class OpenAIProvider(BaseProvider):
    def __init__(self, model: str | None = None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        super().__init__(model or os.getenv("GRIDMIND_MODEL", "gpt-4o-mini"))
        self.client = OpenAI(api_key=api_key)

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.15, max_tokens: int = 350) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class GroqProvider(BaseProvider):
    """Groq uses an OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        super().__init__(model or os.getenv("GRIDMIND_MODEL", "llama-3.3-70b-versatile"))
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.15, max_tokens: int = 350) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def get_provider(provider: str | None = None, model: str | None = None) -> BaseProvider:
    """
    Factory. Priority:
      1. Explicit provider argument
      2. GRIDMIND_PROVIDER env var
      3. Fallback: openai if key exists, else groq
    """
    name = (provider or os.getenv("GRIDMIND_PROVIDER") or "").lower().strip()

    if not name:
        if os.getenv("OPENAI_API_KEY"):
            name = "openai"
        elif os.getenv("GROQ_API_KEY"):
            name = "groq"
        else:
            raise RuntimeError(
                "No LLM provider configured. "
                "Set OPENAI_API_KEY or GROQ_API_KEY (see .env.example)."
            )

    if name == "openai":
        return OpenAIProvider(model)
    if name == "groq":
        return GroqProvider(model)

    raise ValueError(f"Unknown provider '{name}'. Supported: openai, groq")
