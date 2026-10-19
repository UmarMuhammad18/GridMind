from typing import Any

from world.solver import MOVES
from world.tasks import Task, make_task

ACTIONS: list[str] = [
    "MOVE_UP",
    "MOVE_DOWN",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "PICK_UP_KEY",
    "OPEN_DOOR",
    "DESCRIBE",
]

CELL_NAMES = {"#": "wall", ".": "empty", "K": "key", "D": "locked door", "G": "goal"}

STEP_COST = -0.01
BUMP_PENALTY = -0.15
FAILED_ACTION_PENALTY = -0.1
INVALID_ACTION_PENALTY = -0.2
KEY_REWARD = 1.5
DOOR_REWARD = 2.5
GOAL_REWARD = 8.0

# ANSI colors for terminal rendering
COLORS = {
    "#": "\033[90m",   # gray walls
    ".": "\033[37m",   # white empty
    "A": "\033[92m",   # green agent
    "K": "\033[93m",   # yellow key
    "D": "\033[91m",   # red door
    "G": "\033[96m",   # cyan goal
    "?": "\033[90m",   # fog
}
RESET_COLOR = "\033[0m"


class GridWorld:
    """
    2D grid world with a Gymnasium-style API.

    Legend:
      '#' = wall
      '.' = empty
      'A' = agent (only drawn in observations; never stored in the grid)
      'K' = key
      'D' = locked door
      'G' = goal tile behind the door

      '?' = out of view (only with partial observability)

    Coordinates are (x, y): x is the column, y is the row, and y grows downward.

    With partial_observability, the agent only sees cells within `view_radius`
    (Manhattan distance); everything else is shown as '?'.
    """

    def __init__(self, task: Task | None = None, partial_observability: bool = False,
                 view_radius: int = 2) -> None:
        self.task = task or make_task("key_door")
        self.action_space: list[str] = list(ACTIONS)
        self.partial_observability = partial_observability
        self.view_radius = view_radius

        self.grid: list[list[str]] = []
        self.agent_pos: tuple[int, int] = (0, 0)
        self.inventory: dict[str, bool] = {"has_key": False}
        self.step_count = 0
        self.door_open = False
        self.done = True

    @property
    def max_steps(self) -> int:
        return self.task.max_steps

    @property
    def goal_description(self) -> str:
        return self.task.goal

    def reset(self, task: Task | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if task is not None:
            self.task = task

        self.grid = [list(row) for row in self.task.grid]
        self.inventory = {"has_key": False}
        self.step_count = 0
        self.door_open = False
        self.done = False

        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "A":
                    self.agent_pos = (x, y)
                    row[x] = "."

        return self._build_observation(), {"task": self.task.name}

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Returns (observation, reward, terminated, truncated, info)."""
        if self.done:
            raise RuntimeError("Episode has ended; call reset() first.")

        self.step_count += 1
        reward = 0.0
        info: dict[str, Any] = {}

        if action not in self.action_space:
            info["error"] = f"Invalid action: {action}"
            reward += INVALID_ACTION_PENALTY

        elif action in MOVES:
            moved, move_info = self._handle_move(action)
            info.update(move_info)
            reward += STEP_COST if moved else BUMP_PENALTY

        elif action == "PICK_UP_KEY":
            picked, pick_info = self._handle_pick_up()
            info.update(pick_info)
            reward += KEY_REWARD if picked else FAILED_ACTION_PENALTY

        elif action == "OPEN_DOOR":
            opened, open_info = self._handle_open_door()
            info.update(open_info)
            reward += DOOR_REWARD if opened else BUMP_PENALTY

        elif action == "DESCRIBE":
            info["description"] = self._describe_surroundings()

        terminated = self._current_cell() == "G"
        truncated = not terminated and self.step_count >= self.max_steps

        if terminated:
            reward += GOAL_REWARD
            info["success"] = True
            info["message"] = "Agent reached the goal tile."
        elif truncated:
            info["success"] = False
            info["message"] = "Max steps reached."

        self.done = terminated or truncated
        return self._build_observation(), reward, terminated, truncated, info

    def render_text(self, use_color: bool = False) -> str:
        rows = self.visible_grid()
        if use_color:
            rows = ["".join(f"{COLORS.get(c, '')}{c}{RESET_COLOR}" for c in row) for row in rows]
        lines = ["Grid world:", *rows]
        lines.append(f"Inventory: {self.inventory}")
        lines.append(f"Door open: {self.door_open}")
        lines.append(f"Goal: {self.goal_description}")
        lines.append(f"Step: {self.step_count}/{self.max_steps}")
        if self.partial_observability:
            lines.append(f"Vision: radius {self.view_radius} (fog of war)")
        return "\n".join(lines)

    def render(self, use_color: bool = True) -> None:
        print(self.render_text(use_color))

    def visible_grid(self) -> list[str]:
        """The grid as the agent sees it: agent drawn in, fog applied if enabled."""
        ax, ay = self.agent_pos
        rows = []
        for y, row in enumerate(self.grid):
            cells = []
            for x, cell in enumerate(row):
                if (x, y) == (ax, ay):
                    cells.append("A")
                elif self.partial_observability and abs(x - ax) + abs(y - ay) > self.view_radius:
                    cells.append("?")
                else:
                    cells.append(cell)
            rows.append("".join(cells))
        return rows

    # ----------------- Internal helpers ----------------- #

    def _build_observation(self) -> dict[str, Any]:
        return {
            "agent_position": list(self.agent_pos),
            "standing_on": self._current_cell(),
            "grid": self.visible_grid(),
            "inventory": self.inventory.copy(),
            "door_open": self.door_open,
            "partial_observability": self.partial_observability,
            "goal": self.goal_description,
            "step": self.step_count,
            "max_steps": self.max_steps,
        }

    def _current_cell(self) -> str:
        x, y = self.agent_pos
        return self.grid[y][x]

    def _handle_move(self, action: str) -> tuple[bool, dict[str, Any]]:
        dx, dy = MOVES[action]
        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if not self._in_bounds(nx, ny):
            return False, {"error": "Move out of bounds."}

        target = self.grid[ny][nx]
        if target == "#":
            return False, {"error": "Bumped into a wall."}
        if target == "D":
            return False, {"error": "Door is closed. You must open it first."}

        self.agent_pos = (nx, ny)
        return True, {}

    def _handle_pick_up(self) -> tuple[bool, dict[str, Any]]:
        x, y = self.agent_pos
        if self.grid[y][x] == "K":
            self.inventory["has_key"] = True
            self.grid[y][x] = "."
            return True, {"message": "Picked up the key."}
        return False, {"error": "Nothing to pick up here."}

    def _handle_open_door(self) -> tuple[bool, dict[str, Any]]:
        if not self.inventory.get("has_key", False):
            return False, {"error": "You do not have the key."}

        door_pos = self._find_cell("D")
        if door_pos is None:
            return False, {"error": "No locked door to open."}

        dx = abs(door_pos[0] - self.agent_pos[0])
        dy = abs(door_pos[1] - self.agent_pos[1])
        if dx + dy != 1:
            return False, {"error": "You must stand next to the door to open it."}

        x, y = door_pos
        self.grid[y][x] = "."
        self.door_open = True
        return True, {"message": "Door opened."}

    def _describe_surroundings(self) -> str:
        x, y = self.agent_pos
        parts = [f"You are at ({x}, {y}) standing on {CELL_NAMES[self._current_cell()]}."]
        for action, (dx, dy) in MOVES.items():
            nx, ny = x + dx, y + dy
            cell = self.grid[ny][nx] if self._in_bounds(nx, ny) else "#"
            parts.append(f"{action.removeprefix('MOVE_').lower()}: {CELL_NAMES[cell]}")
        return " ".join(parts[:1]) + " Neighbours - " + ", ".join(parts[1:]) + "."

    def _find_cell(self, target: str) -> tuple[int, int] | None:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == target:
                    return (x, y)
        return None

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])
