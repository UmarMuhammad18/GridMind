from typing import Dict, List, Tuple, Any


class GridWorld:
    """
    Simple 2D grid world.

    Legend:
      '#' = wall
      '.' = empty
      'A' = agent
      'K' = key
      'D' = locked door
      'G' = goal tile behind the door
    """

    def __init__(self) -> None:
        # Fixed grid layout for simplicity
        self._base_grid = [
            list("########"),
            list("#A..K..#"),
            list("#..##..#"),
            list("#..D.G.#"),
            list("########"),
        ]

        self.grid: List[List[str]] = []
        self.agent_pos: Tuple[int, int] = (0, 0)
        self.inventory: Dict[str, bool] = {"has_key": False}
        self.goal_description: str = ""
        self.step_count: int = 0
        self.max_steps: int = 0

        self.action_space: List[str] = [
            "MOVE_UP",
            "MOVE_DOWN",
            "MOVE_LEFT",
            "MOVE_RIGHT",
            "PICK_UP",
            "OPEN_DOOR",
            "DESCRIBE_ENV",
        ]

    def reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.grid = [row[:] for row in self._base_grid]
        self.inventory = {"has_key": False}
        self.goal_description = task.get("goal", "Find the key and open the door.")
        self.max_steps = task.get("max_steps", 40)
        self.step_count = 0

        # Find agent initial position
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == "A":
                    self.agent_pos = (x, y)

        return self._build_observation()

    def step(self, action: str):
        self.step_count += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if action not in self.action_space:
            info["error"] = f"Invalid action: {action}"
            return self._build_observation(), reward, done, info

        if action.startswith("MOVE_"):
            moved, move_info = self._handle_move(action)
            info.update(move_info)
            if not moved:
                reward -= 0.1  # small penalty for bumping into walls
            else:
                reward -= 0.01  # small step cost

        elif action == "PICK_UP":
            picked, pick_info = self._handle_pick_up()
            info.update(pick_info)
            if picked:
                reward += 1.0
            else:
                reward -= 0.05

        elif action == "OPEN_DOOR":
            opened, open_info = self._handle_open_door()
            info.update(open_info)
            if opened:
                reward += 2.0
            else:
                reward -= 0.1

        elif action == "DESCRIBE_ENV":
            # No direct environment change; reward neutral
            info["description"] = "Agent attempted to describe the environment."

        # Check success condition: agent on 'G' and door opened
        x, y = self.agent_pos
        if self.grid[y][x] == "G":
            reward += 5.0
            done = True
            info["success"] = True
            info["message"] = "Agent reached the goal tile."

        if self.step_count >= self.max_steps and not done:
            done = True
            info["success"] = False
            info["message"] = "Max steps reached."

        return self._build_observation(), reward, done, info

    def render(self) -> None:
        """Prints the grid with the agent position."""
        display_grid = [row[:] for row in self.grid]
        ax, ay = self.agent_pos
        display_grid[ay][ax] = "A"

        print("Grid world:")
        for row in display_grid:
            print("".join(row))
        print(f"Inventory: {self.inventory}")
        print(f"Goal: {self.goal_description}")
        print(f"Step: {self.step_count}/{self.max_steps}")

    # ----------------- Internal helpers ----------------- #

    def _build_observation(self) -> Dict[str, Any]:
        return {
            "agent_position": list(self.agent_pos),
            "grid": ["".join(row) for row in self.grid],
            "inventory": self.inventory.copy(),
            "goal": self.goal_description,
            "step": self.step_count,
            "max_steps": self.max_steps,
        }

    def _handle_move(self, action: str):
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
        if target == "D":
            return False, {"error": "Door is closed. You must open it first."}

        # Move agent
        self.agent_pos = (nx, ny)
        return True, {}

    def _handle_pick_up(self):
        x, y = self.agent_pos
        cell = self.grid[y][x]
        if cell == "K":
            self.inventory["has_key"] = True
            self.grid[y][x] = "."  # remove key from grid
            return True, {"message": "Picked up the key."}
        return False, {"error": "Nothing to pick up here."}

    def _handle_open_door(self):
        if not self.inventory.get("has_key", False):
            return False, {"error": "You do not have the key."}

        # Door is at a fixed position in this simple world
        door_pos = self._find_cell("D")
        if door_pos is None:
            return False, {"error": "No door found to open."}

        dx = abs(door_pos[0] - self.agent_pos[0])
        dy = abs(door_pos[1] - self.agent_pos[1])
        if dx + dy != 1:
            return False, {"error": "You must stand next to the door to open it."}

        # Open the door: replace 'D' with '.'
        x, y = door_pos
        self.grid[y][x] = "."
        return True, {"message": "Door opened."}

    def _find_cell(self, target: str):
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if cell == target:
                    return (x, y)
        return None

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= y < len(self.grid) and 0 <= x < len(self.grid[0])
