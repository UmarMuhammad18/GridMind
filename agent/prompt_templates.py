import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are an autonomous agent inside a 2D grid world.

Your job is to achieve the given goal by choosing exactly ONE action per step
from the allowed action list.

You MUST reply with a single valid JSON object and nothing else:

{
  "action": "<ONE_OF_THE_ALLOWED_ACTIONS>",
  "explanation": "<short reasoning in 1-2 sentences>",
  "note": "<optional short note to remember, only when using WRITE_NOTE>"
}

Rules:
- Only use actions from the provided list.
- Never invent new actions.
- Prefer efficient paths toward the goal.
- If you see a key ('K'), pick it up when standing on it (PICK_UP_KEY).
- You must open the door ('D') with OPEN_DOOR while standing next to it (and holding the key) before you can walk through.
- Use WRITE_NOTE to remember important locations or facts (e.g. "Key is at (4,1)"). Notes persist and appear in future observations.
- Reach the goal tile ('G') to finish.
- When partial_observability is true, cells shown as '?' are unknown / out of view.
""".strip()


def build_user_prompt(
    observation: Dict[str, Any],
    actions: List[str],
    history: List[Dict[str, Any]],
) -> str:
    last_steps = history[-6:]
    history_summary = [
        {
            "step": s["step"],
            "action": s["action"],
            "reward": s["reward"],
            "info": s.get("info", {}),
        }
        for s in last_steps
    ]

    prompt = f"""
Current observation:
{json.dumps(observation, indent=2)}

Allowed actions:
{json.dumps(actions)}

Recent history (last {len(history_summary)} steps):
{json.dumps(history_summary, indent=2)}

Choose the single best next action. If you want to remember something, use WRITE_NOTE and put the text in the "note" field.
""".strip()

    return prompt
