from typing import Any

from agent.base import Agent
from agent.baselines import BFSAgent, RandomAgent
from agent.llm_agent import LLMAgent
from agent.llm_clients import API_KEY_ENV, make_client

AGENT_KINDS = ("llm", "random", "bfs")


def make_agent(kind: str, config: dict[str, Any] | None = None,
               seed: int | None = None) -> Agent:
    if kind == "random":
        return RandomAgent(seed)
    if kind == "bfs":
        return BFSAgent()
    if kind == "llm":
        config = config or {}
        provider = config.get("provider")
        if provider in API_KEY_ENV and not config.get("api_key"):
            raise RuntimeError(
                f"No API key for {provider}. Set {API_KEY_ENV[provider]} "
                "(in your shell or a .env file) or add api_key to config.json."
            )
        return LLMAgent(
            make_client(config),
            prompt=config.get("prompt", "default"),
            max_retries=config.get("max_retries", 2),
            history_window=config.get("history_window", 8),
        )
    raise ValueError(f"Unknown agent '{kind}'. Choose from: {', '.join(AGENT_KINDS)}")
