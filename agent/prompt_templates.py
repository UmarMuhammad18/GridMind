import json
from typing import Any, Dict, List


SYSTEM_PROMPT = """
You are an agent acting inside a 2D grid world environment.

You must achieve the given goal by choosing exactly ONE action per step
from the allowed action space.

You MUST respond with a single JSON object of the form:

{
  "action": "<ONE_OF_THE_ALLOWED_ACTIONS>",
  "explanation": "<short reasoning in one or two sentences>"
}

Rules:
- Only choose actions from the provided list.
- Do not invent new actions.
- Do not include any extra keys in the JSON.
- The JSON must be valid and parseable.
""".strip()


def build_user_prompt(
    observation: Dict[str, Any],
    actions: List[str],
    history: List[Dict[str, Any]],
) -> str:
    """
    Build the user prompt given the current observation, action space, and history.
    History is summarized to keep the prompt compact.
    """
    # Summarize last few steps
    last_steps = history[-5:]
    history_summary = []
    for step in last_steps:
        history_summary.append(
            {
                "step": step["step"],
                "action": step["action"],
                "reward": step["reward"],
                "done": step["done"],
                "info": step["info"],
            }
        )

    prompt = f"""
Observation (JSON):
{json.dumps(observation, indent=2)}

Available actions:
{json.dumps(actions)}

Recent history (last {len(history_summary)} steps):
{json.dumps(history_summary, indent=2)}

Goal:
{observation.get("goal")}
""".strip()

    return prompt
