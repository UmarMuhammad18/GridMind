"""
Configuration loading. Later sources win:

  built-in defaults < config.json < environment (.env) < command-line flags

Switching provider at a later level without naming a model resets the model
(and other provider-specific settings) to that provider's defaults.
"""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agent.llm_clients import API_KEY_ENV, PROVIDER_DEFAULTS

DEFAULTS: dict[str, Any] = {
    "provider": "openai",
    "prompt": "default",
    "max_retries": 2,
    "api_retries": 3,
    "history_window": 8,
    "pricing": None,
}

PROVIDER_SPECIFIC = ("model", "temperature", "effort", "base_url", "api_key", "pricing")
PLACEHOLDER_KEYS = {"", "YOUR_API_KEY_HERE"}


def load_config(path: str | Path = "config.json",
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    load_dotenv()

    layers: list[dict[str, Any]] = []
    path = Path(path)
    if path.is_file():
        layers.append(json.loads(path.read_text(encoding="utf-8")))

    env_layer = {
        "provider": os.getenv("GRIDMIND_PROVIDER"),
        "model": os.getenv("GRIDMIND_MODEL"),
    }
    layers.append(env_layer)
    layers.append(overrides or {})

    config = dict(DEFAULTS)
    for layer in layers:
        layer = {k: v for k, v in layer.items() if v is not None}
        if "provider" in layer and layer["provider"] != config["provider"] and "model" not in layer:
            for key in PROVIDER_SPECIFIC:
                config.pop(key, None)
        config.update(layer)

    provider = config["provider"]
    for key, value in PROVIDER_DEFAULTS.get(provider, {}).items():
        config.setdefault(key, value)

    if config.get("api_key") in PLACEHOLDER_KEYS:
        config["api_key"] = None
    if not config.get("api_key") and provider in API_KEY_ENV:
        config["api_key"] = os.getenv(API_KEY_ENV[provider])

    return config
