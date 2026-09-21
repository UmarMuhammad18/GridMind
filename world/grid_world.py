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

    def __init__(self) -> None:
        self.grid: List[List[str]] = []
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.inventory: Dict[str, bool] = {"has_key": False}
        self.goal_description: str = ""
        self.step_count: int = 0
        self.max_steps: int = 40
        self.door_open: bool = False

        self.action_space: List[str] = [
            "MOVE_UP",
            "MOVE_DOWN",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "PICK_UP_KEY",
            "OPEN_DOOR",
            "DESCRIBE",
        ]

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Use custom grid from task if provided, otherwise default
        base = task.get("grid") or self.DEFAULT_GRID
        self.grid = [row[:] for row in base]

        self.inventory = {"has_key": False}
        self.door_open = False
        self.goal_description = task.get(
            "goal", "Find the key, open the door, and reach the goal."
        )
        self.max_steps = int(task.get("max_steps", 40))
        self.step_count = 0

        # Locate agent start
        found = False
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "A":
                    self.agent_pos = (x, y)
                    self.grid[y][x] = "."  # agent is tracked separately
                    found = True
                    break
            if found:
                break

        if not found:
            raise ValueError("No agent start position 'A' found in grid")

        return self._build_observation()

    def step(self, action: str):
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

        elif action == "DESCRIBE":
            info["description"] = self._describe()

        # Success: standing on goal
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

    def render(self) -> None:
        display = [row[:] for row in self.grid]
        ax, ay = self.agent_pos
        display[ay][ax] = "A"

        print("Grid:")
        for row in display:
            print("".join(row))
        print(f"Inventory : {self.inventory}")
        print(f"Door open : {self.door_open}")
        print(f"Goal      : {self.goal_description}")
        print(f"Step      : {self.step_count}/{self.max_steps}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, Any]:
        return {
            "agent_position": list(self.agent_pos),
            "grid": ["".join(row) for row in self.grid],
            "inventory": self.inventory.copy(),
            "door_open": self.door_open,
            "goal": self.goal_description,
            "step": self.step_count,
            "max_steps": self.max_steps,
            "available_actions": self.action_space,
        }

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
            # Already opened earlier
            if self.door_open:
                return False, {"error": "Door is already open."}
            return False, {"error": "No door found."}

        dx = abs(door_pos[0] - self.agent_pos[0])
        dy = abs(door_pos[1] - self.agent_pos[1])
        if dx + dy != 1:
            return False, {"error": "You must stand next to the door."}

        x, y = door_pos
        self.grid[y][x] = "."  # open it
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
        return f"Standing at {self.agent_pos}. Inventory: {inv}. Nearby: {nearby or 'nothing special'}."

    def _find_cell(self, target: str) -> Optional[Tuple[int, int]]:
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == target:
                    return (x, y)
        return None

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])
