import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are an autonomous agent inside a 2D grid world.

Your job is to achieve the given goal by choosing exactly ONE action per step
from the allowed action list.

You MUST reply with a single valid JSON object and nothing else:

{
  "action": "<ONE_OF_THE_ALLOWED_ACTIONS>",
  "explanation": "<short reasoning in 1-2 sentences>"
}

Rules:
- Only use actions from the provided list.
- Never invent new actions.
- Prefer efficient paths toward the goal.
- If you see a key ('K'), pick it up when standing on it.
- You must open the door ('D') with OPEN_DOOR while standing next to it (and holding the key) before you can walk through.
- Reach the goal tile ('G') to finish.
""".strip()


def build_user_prompt(
    observation: Dict[str, Any],
    actions: List[str],
    history: List[Dict[str, Any]],
) -> str:
    # Keep only the last few steps to control token usage
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

Choose the single best next action and explain briefly.
""".strip()

    return prompt
