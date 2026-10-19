from dataclasses import dataclass

from world.levels import load_level
from world.solver import find_cell, solve

DEFAULT_GOAL = "Find the key, then go to the door, open it, and reach the goal tile behind it."


@dataclass(frozen=True)
class Task:
    name: str
    grid: tuple[str, ...]
    goal: str = DEFAULT_GOAL
    max_steps: int = 40
    optimal_steps: int | None = None


def make_task(level: str = "key_door", seed: int | None = None,
              max_steps: int | None = None) -> Task:
    """
    Build a task from a level. Unless given, the step limit is generous:
    at least 40 steps, or three times the optimal solution length.
    """
    grid = load_level(level, seed)
    start = find_cell(grid, "A")
    plan = solve([row.replace("A", ".") for row in grid], start)
    optimal = len(plan) if plan is not None else None

    if max_steps is None:
        max_steps = max(40, 3 * optimal) if optimal else 40

    name = f"random-{seed}" if level == "random" else level
    return Task(name=name, grid=tuple(grid), max_steps=max_steps, optimal_steps=optimal)


KEY_DOOR_TASK = make_task("key_door")
