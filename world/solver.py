"""Breadth-first search over GridWorld states, used to validate levels and by the BFS baseline."""

from collections import deque
from typing import Any

MOVES: dict[str, tuple[int, int]] = {
    "MOVE_UP": (0, -1),
    "MOVE_DOWN": (0, 1),
    "MOVE_LEFT": (-1, 0),
    "MOVE_RIGHT": (1, 0),
}

# (x, y, has_key, key_on_grid, door_closed)
State = tuple[int, int, bool, bool, bool]


def find_cell(grid: list[str], target: str) -> tuple[int, int] | None:
    for y, row in enumerate(grid):
        x = row.find(target)
        if x != -1:
            return (x, y)
    return None


def solve(
    grid: list[str], start: tuple[int, int], has_key: bool = False
) -> list[str] | None:
    """
    Return the shortest action sequence from `start` to the goal tile, or None.

    `grid` must not contain the agent ('A'); the key and door are taken to be
    present if their symbols appear in it.
    """
    key = find_cell(grid, "K")
    door = find_cell(grid, "D")
    start_state: State = (start[0], start[1], has_key, key is not None, door is not None)

    parents: dict[State, tuple[State, str] | None] = {start_state: None}
    queue = deque([start_state])
    while queue:
        state = queue.popleft()
        x, y = state[0], state[1]
        if grid[y][x] == "G":
            return _reconstruct(parents, state)
        for action, nxt in _successors(grid, state, key, door):
            if nxt not in parents:
                parents[nxt] = (state, action)
                queue.append(nxt)
    return None


def solve_from_observation(observation: dict[str, Any]) -> list[str] | None:
    """Plan from an environment observation (the grid there has 'A' drawn in)."""
    x, y = observation["agent_position"]
    grid = list(observation["grid"])
    row = grid[y]
    grid[y] = row[:x] + observation["standing_on"] + row[x + 1 :]
    return solve(grid, (x, y), observation["inventory"]["has_key"])


def _successors(grid, state: State, key, door):
    x, y, has_key, key_on_grid, door_closed = state
    for action, (dx, dy) in MOVES.items():
        nx, ny = x + dx, y + dy
        if not (0 <= ny < len(grid) and 0 <= nx < len(grid[ny])):
            continue
        if grid[ny][nx] == "#" or (door_closed and (nx, ny) == door):
            continue
        yield action, (nx, ny, has_key, key_on_grid, door_closed)

    if key_on_grid and (x, y) == key:
        yield "PICK_UP_KEY", (x, y, True, False, door_closed)

    if has_key and door_closed and abs(door[0] - x) + abs(door[1] - y) == 1:
        yield "OPEN_DOOR", (x, y, has_key, key_on_grid, False)


def _reconstruct(parents, state: State) -> list[str]:
    actions = []
    while parents[state] is not None:
        state, action = parents[state]
        actions.append(action)
    return actions[::-1]
