from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Decision:
    action: str
    explanation: str = ""
    valid: bool = True  # False when the agent failed to produce a usable action


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def cost(self, pricing: dict[str, float] | None) -> float | None:
        """Cost in dollars, given {"input_per_mtok": ..., "output_per_mtok": ...}."""
        if not pricing:
            return None
        return (
            self.input_tokens * pricing.get("input_per_mtok", 0.0)
            + self.output_tokens * pricing.get("output_per_mtok", 0.0)
        ) / 1_000_000


class Agent(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(
        self,
        observation: dict[str, Any],
        actions: list[str],
        history: list[dict[str, Any]],
    ) -> Decision: ...
