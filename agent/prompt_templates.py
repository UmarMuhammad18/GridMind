import json
from typing import Any

_BASE_PROMPT = """
You are an agent acting inside a 2D grid world environment.

You must achieve the given goal by choosing exactly ONE action per step
from the allowed action space.

Grid legend:
  '#' wall, '.' empty, 'A' you, 'K' key, 'D' locked door, 'G' goal.
Coordinates are [x, y]: x is the column, y is the row (row 0 is the top).
MOVE_UP decreases y, MOVE_DOWN increases y, MOVE_LEFT decreases x, MOVE_RIGHT increases x.
"standing_on" tells you what is under you, since 'A' hides it on the grid.
When "partial_observability" is true, '?' cells are out of view: explore to find
the key, door and goal, and use your notes to remember where things are.

Rules:
- To take the key, stand on its tile and use PICK_UP_KEY.
- To open the door, hold the key and stand directly next to the door
  (up, down, left or right, not diagonal), then use OPEN_DOOR.
- Walls and closed doors cannot be walked through.
- The episode ends as soon as you step onto the goal tile.
{extra}
You MUST respond with a single JSON object of the form:

{{
  "action": "<ONE_OF_THE_ALLOWED_ACTIONS>",
  "explanation": "<short reasoning in one or two sentences>",
  "note": "<a short fact worth remembering for later steps, or an empty string>"
}}

Only choose actions from the provided list. Do not include any extra keys.
""".strip()

_PLANNER_EXTRA = """
Before choosing, work out the coordinates of yourself, the key, the door and the
goal, then plan the shortest route for the current sub-goal (key, then door, then
goal). Put that plan briefly in your explanation and take its first action.
"""

SYSTEM_PROMPTS: dict[str, str] = {
    "default": _BASE_PROMPT.format(extra=""),
    "planner": _BASE_PROMPT.format(extra=_PLANNER_EXTRA),
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["default"]


def build_user_prompt(
    observation: dict[str, Any],
    actions: list[str],
    history: list[dict[str, Any]],
    notes: list[str] | None = None,
    window: int = 8,
) -> str:
    """
    Build the user prompt from the current observation, action space, recent
    history, and the agent's own notes.
    """
    recent = []
    for step in history[-window:]:
        info = step["info"]
        recent.append({
            "step": step["step"],
            "from": step["position"],
            "action": step["action"],
            "to": step["position_after"],
            "result": info.get("message") or info.get("error") or info.get("description", ""),
        })

    position = observation["agent_position"]
    visits = sum(1 for step in history if step["position"] == position)

    sections = [
        f"Observation (JSON):\n{json.dumps(observation, indent=2)}",
        f"Available actions:\n{json.dumps(actions)}",
        f"Recent history (last {len(recent)} of {len(history)} steps):\n"
        f"{json.dumps(recent, indent=2)}",
    ]
    if visits:
        sections.append(
            f"You have been at position {position} {visits} time(s) before. "
            "If you are going in circles, try a different route."
        )
    if notes:
        sections.append("Your notes so far:\n" + "\n".join(f"- {n}" for n in notes))
    sections.append(f"Goal:\n{observation.get('goal')}")

    return "\n\n".join(sections)
