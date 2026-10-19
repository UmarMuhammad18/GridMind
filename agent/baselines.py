"""Non-LLM reference agents: a lower bound (random) and an upper bound (optimal BFS)."""

import random
from typing import Any

from agent.base import Decision
from world.solver import solve_from_observation


class RandomAgent:
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.rng = random.Random(self.seed)

    def act(self, observation: dict[str, Any], actions: list[str],
            history: list[dict[str, Any]]) -> Decision:
        return Decision(self.rng.choice(actions), "Random choice.")


class BFSAgent:
    """Replans the shortest path from the current observation every step."""

    name = "bfs"

    def reset(self) -> None:
        pass

    def act(self, observation: dict[str, Any], actions: list[str],
            history: list[dict[str, Any]]) -> Decision:
        plan = solve_from_observation(observation)
        if not plan:
            return Decision("DESCRIBE", "No path to the goal exists.")
        return Decision(plan[0], f"Shortest path: {len(plan)} actions remaining.")
