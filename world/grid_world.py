from typing import Any, Dict, List, Optional, Tuple


class GridWorld:
    """
    Simple 2D grid world for LLM agents.

    Legend:
      '#' = wall
      '.' = empty
      'A' = agent (starting position)
      'K' = key
      'D' = locked door
      'G' = goal tile
    """

    DEFAULT_GRID = [
        list("########"),
        list("#A..K..#"),
        list("#..##..#"),
        list("#..D.G.#"),
        list("########"),
    ]

    # ANSI colors for terminal rendering
    COLORS = {
        "#": "\033[90m",   # bright black / gray walls
        ".": "\033[37m",   # white empty
        "A": "\033[92m",   # green agent
        "K": "\033[93m",   # yellow key
        "D": "\033[91m",   # red door
        "G": "\033[96m",   # cyan goal
        "?": "\033[90m",   # fog
        "reset": "\033[0m",
    }

    def __init__(self, partial_observability: bool = False, view_radius: int = 2) -> None:
        self.grid: List[List[str]] = []
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.inventory: Dict[str, bool] = {"has_key": False}
        self.goal_description: str = ""
        self.step_count: int = 0
        self.max_steps: int = 40
        self.door_open: bool = False
        self.notes: List[str] = []  # agent memory

        self.partial_observability = partial_observability
        self.view_radius = view_radius

        self.action_space: List[str] = [
            "MOVE_UP",
            "MOVE_DOWN",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "PICK_UP_KEY",
            "OPEN_DOOR",
            "WRITE_NOTE",
            "DESCRIBE",
        ]

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        base = task.get("grid") or self.DEFAULT_GRID
        self.grid = [row[:] for row in base]

        self.inventory = {"has_key": False}
        self.door_open = False
        self.notes = []
        self.goal_description = task.get(
            "goal", "Find the key, open the door, and reach the goal."
        )
        self.max_steps = int(task.get("max_steps", 40))
        self.step_count = 0

        found = False
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "A":
                    self.agent_pos = (x, y)
                    self.grid[y][x] = "."
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError("No agent start position 'A' found in grid")

        return self._build_observation()

    def step(self, action: str, note_text: str | None = None):
        self.step_count += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if action not in self.action_space:
            info["error"] = f"Invalid action: {action}"
            reward -= 0.2
            return self._build_observation(), reward, done, info

        if action.startswith("MOVE_"):
            moved, move_info = self._handle_move(action)
            info.update(move_info)
            reward -= 0.01 if moved else 0.15

        elif action == "PICK_UP_KEY":
            picked, pick_info = self._handle_pick_up()
            info.update(pick_info)
            reward += 1.5 if picked else -0.1

        elif action == "OPEN_DOOR":
            opened, open_info = self._handle_open_door()
            info.update(open_info)
            if opened:
                reward += 2.5
                self.door_open = True
            else:
                reward -= 0.15

        elif action == "WRITE_NOTE":
            text = (note_text or "").strip()
            if text:
                # Keep notes short and limited
                note = text[:120]
                self.notes.append(note)
                # Keep only the last 8 notes
                self.notes = self.notes[-8:]
                info["message"] = f"Note saved: {note}"
                reward += 0.05
            else:
                info["error"] = "WRITE_NOTE requires non-empty text."
                reward -= 0.05

        elif action == "DESCRIBE":
            info["description"] = self._describe()

        # Success check
        x, y = self.agent_pos
        if self.grid[y][x] == "G":
            reward += 8.0
            done = True
            info["success"] = True
            info["message"] = "Agent reached the goal tile."

        if self.step_count >= self.max_steps and not done:
            done = True
            info["success"] = False
            info["message"] = "Max steps reached without completing the goal."

        return self._build_observation(), reward, done, info

    def render(self, use_color: bool = True) -> None:
        display = self._visible_grid() if self.partial_observability else [row[:] for row in self.grid]
        ax, ay = self.agent_pos

        # Place agent on the visible grid
        if 0 <= ay < len(display) and 0 <= ax < len(display[0]):
            display[ay][ax] = "A"

        print("Grid:")
        for row in display:
            if use_color:
                colored = "".join(
                    f"{self.COLORS.get(c, '')}{c}{self.COLORS['reset']}" for c in row
                )
                print(colored)
            else:
                print("".join(row))

        print(f"Inventory : {self.inventory}")
        print(f"Door open : {self.door_open}")
        print(f"Notes     : {self.notes[-3:] if self.notes else '[]'}")
        print(f"Goal      : {self.goal_description}")
        print(f"Step      : {self.step_count}/{self.max_steps}")
        if self.partial_observability:
            print(f"Vision    : radius {self.view_radius} (fog-of-war)")

    # ------------------------------------------------------------------
    # Observation & visibility
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        if self.partial_observability:
            visible = ["".join(row) for row in self._visible_grid()]
        else:
            visible = ["".join(row) for row in self.grid]

        return {
            "agent_position": list(self.agent_pos),
            "grid": visible,
            "inventory": self.inventory.copy(),
            "door_open": self.door_open,
            "notes": self.notes.copy(),
            "goal": self.goal_description,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "partial_observability": self.partial_observability,
            "available_actions": self.action_space,
        }

    def _visible_grid(self) -> List[List[str]]:
        """Return a grid where cells outside the view radius are replaced by '?'."""
        h = len(self.grid)
        w = len(self.grid[0]) if h else 0
        ax, ay = self.agent_pos
        visible = []

        for y in range(h):
            row = []
            for x in range(w):
                dist = abs(x - ax) + abs(y - ay)  # Manhattan
                if dist <= self.view_radius:
                    row.append(self.grid[y][x])
                else:
                    row.append("?")
            visible.append(row)
        return visible

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handle_move(self, action: str) -> Tuple[bool, Dict[str, Any]]:
        dx, dy = 0, 0
        if action == "MOVE_UP":
            dy = -1
        elif action == "MOVE_DOWN":
            dy = 1
        elif action == "MOVE_LEFT":
            dx = -1
        elif action == "MOVE_RIGHT":
            dx = 1

        x, y = self.agent_pos
        nx, ny = x + dx, y + dy

        if not self._in_bounds(nx, ny):
            return False, {"error": "Move out of bounds."}

        target = self.grid[ny][nx]
        if target == "#":
            return False, {"error": "Bumped into a wall."}
        if target == "D" and not self.door_open:
            return False, {"error": "Door is locked. Use OPEN_DOOR first."}

        self.agent_pos = (nx, ny)
        return True, {}

    def _handle_pick_up(self) -> Tuple[bool, Dict[str, Any]]:
        x, y = self.agent_pos
        if self.grid[y][x] == "K":
            self.inventory["has_key"] = True
            self.grid[y][x] = "."
            return True, {"message": "Picked up the key."}
        return False, {"error": "No key here."}

    def _handle_open_door(self) -> Tuple[bool, Dict[str, Any]]:
        if not self.inventory.get("has_key", False):
            return False, {"error": "You do not have the key."}

        door_pos = self._find_cell("D")
        if door_pos is None:
            if self.door_open:
                return False, {"error": "Door is already open."}
            return False, {"error": "No door found."}

        dx = abs(door_pos[0] - self.agent_pos[0])
        dy = abs(door_pos[1] - self.agent_pos[1])
        if dx + dy != 1:
            return False, {"error": "You must stand next to the door."}

        x, y = door_pos
        self.grid[y][x] = "."
        self.door_open = True
        return True, {"message": "Door unlocked and opened."}

    def _describe(self) -> str:
        x, y = self.agent_pos
        nearby = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx, ny = x + dx, y + dy
                if self._in_bounds(nx, ny):
                    cell = self.grid[ny][nx]
                    if cell not in (".", "A"):
                        nearby.append(f"{cell} at ({nx},{ny})")
        inv = "has key" if self.inventory.get("has_key") else "no key"
        notes_preview = self.notes[-2:] if self.notes else []
        return (
            f"Standing at {self.agent_pos}. Inventory: {inv}. "
            f"Nearby: {nearby or 'nothing special'}. Recent notes: {notes_preview}"
        )

    def _find_cell(self, target: str) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == target:
                    return (x, y)
        return None

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])
