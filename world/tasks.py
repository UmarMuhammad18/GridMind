from typing import Any, Dict, List


# Each task defines a name, goal text, max steps, and an optional custom grid.
# If no grid is provided, the default layout from GridWorld is used.

KEY_DOOR_TASK: Dict[str, Any] = {
    "name": "key_door",
    "goal": "Find the key, open the locked door, and reach the goal tile behind it.",
    "max_steps": 40,
}

# Slightly larger / different layout
MAZE_TASK: Dict[str, Any] = {
    "name": "maze",
    "goal": "Navigate the maze, pick up the key, open the door, and reach the goal.",
    "max_steps": 60,
    "grid": [
        list("##########"),
        list("#A.......#"),
        list("#.##.###.#"),
        list("#.#...#K.#"),
        list("#.###.#..#"),
        list("#....D..G#"),
        list("##########"),
    ],
}

# Key is farther, door is closer to start (tests planning)
LONG_KEY_TASK: Dict[str, Any] = {
    "name": "long_key",
    "goal": "The key is far away. Retrieve it, return to open the door, then reach the goal.",
    "max_steps": 80,
    "grid": [
        list("############"),
        list("#A..D.G....#"),
        list("#.########.#"),
        list("#..........#"),
        list("#.########.#"),
        list("#.........K#"),
        list("############"),
    ],
}

ALL_TASKS: Dict[str, Dict[str, Any]] = {
    "key_door": KEY_DOOR_TASK,
    "maze": MAZE_TASK,
    "long_key": LONG_KEY_TASK,
}


def get_task(name: str = "key_door") -> Dict[str, Any]:
    """Return a task by name. Falls back to key_door."""
    return ALL_TASKS.get(name, KEY_DOOR_TASK).copy()


def list_tasks() -> List[str]:
    return list(ALL_TASKS.keys())
