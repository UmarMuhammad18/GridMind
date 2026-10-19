"""Loading levels from text files and generating random ones."""

import random
from pathlib import Path

from world.solver import find_cell, solve

LEVELS_DIR = Path(__file__).resolve().parent.parent / "levels"
ALLOWED_CELLS = set("#.AKDG")


class LevelError(ValueError):
    pass


def list_levels() -> list[str]:
    return sorted(p.stem for p in LEVELS_DIR.glob("*.txt"))


def load_level(name_or_path: str, seed: int | None = None) -> list[str]:
    """
    Load a level by name ("key_door"), by file path, or generate one ("random").
    """
    if name_or_path == "random":
        return generate_level(seed)

    path = Path(name_or_path)
    if not path.suffix:
        path = LEVELS_DIR / f"{name_or_path}.txt"
    if not path.is_file():
        raise LevelError(
            f"Unknown level '{name_or_path}'. Available: {', '.join(list_levels())}, random"
        )

    grid = [line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()]
    grid = [line for line in grid if line]
    validate_level(grid)
    return grid


def validate_level(grid: list[str]) -> None:
    """Raise LevelError unless the grid is a well-formed, solvable level."""
    if not grid:
        raise LevelError("Level is empty.")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise LevelError("Level rows must all have the same width.")

    cells = "".join(grid)
    bad = set(cells) - ALLOWED_CELLS
    if bad:
        raise LevelError(f"Unknown cell symbols: {''.join(sorted(bad))}")

    for symbol, name, allowed in (("A", "agent", {1}), ("G", "goal", {1}),
                                  ("K", "key", {0, 1}), ("D", "door", {0, 1})):
        if cells.count(symbol) not in allowed:
            raise LevelError(f"Level must contain {'exactly' if allowed == {1} else 'at most'} "
                             f"one {name} ('{symbol}').")

    edges = grid[0] + grid[-1] + "".join(row[0] + row[-1] for row in grid)
    if set(edges) != {"#"}:
        raise LevelError("Level must be surrounded by walls.")

    start = find_cell(grid, "A")
    clean = [row.replace("A", ".") for row in grid]
    if solve(clean, start) is None:
        raise LevelError("Level is not solvable.")


def generate_level(
    seed: int | None = None, width: int = 11, height: int = 7, wall_density: float = 0.15
) -> list[str]:
    """
    Generate a solvable two-room level: the agent and key on one side of a wall,
    the goal on the other, reachable only through a locked door.
    """
    if width < 7 or height < 4:
        raise LevelError("Random levels must be at least 7x4.")
    rng = random.Random(seed)

    for _ in range(1000):
        grid = [["#"] * width]
        grid += [["#"] + ["."] * (width - 2) + ["#"] for _ in range(height - 2)]
        grid += [["#"] * width]

        wall_x = rng.randint(3, width - 4)
        for y in range(1, height - 1):
            grid[y][wall_x] = "#"
        grid[rng.randint(1, height - 2)][wall_x] = "D"

        rows = range(1, height - 1)
        left = [(x, y) for x in range(1, wall_x) for y in rows]
        right = [(x, y) for x in range(wall_x + 1, width - 1) for y in rows]
        for x, y in left + right:
            if rng.random() < wall_density:
                grid[y][x] = "#"

        free_left = [(x, y) for x, y in left if grid[y][x] == "."]
        free_right = [(x, y) for x, y in right if grid[y][x] == "."]
        if len(free_left) < 2 or not free_right:
            continue

        (ax, ay), (kx, ky) = rng.sample(free_left, 2)
        gx, gy = rng.choice(free_right)
        grid[ky][kx] = "K"
        grid[gy][gx] = "G"

        lines = ["".join(row) for row in grid]
        if solve(lines, (ax, ay)) is not None:
            row = lines[ay]
            lines[ay] = row[:ax] + "A" + row[ax + 1 :]
            return lines

    raise LevelError("Could not generate a solvable level; try a lower wall density.")
