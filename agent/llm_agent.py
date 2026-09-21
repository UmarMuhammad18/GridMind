import json
import os
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from agent.prompt_templates import SYSTEM_PROMPT, build_user_prompt


class LLMAgent:
    """
    LLM-powered agent that receives structured observations and returns
    a valid action + short explanation.
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key, or export it."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("GRIDMIND_MODEL", "gpt-4o-mini")

    def choose_action(
        self,
        observation: Dict[str, Any],
        actions: List[str],
        history: List[Dict[str, Any]],
    ) -> Tuple[str, str]:
        user_prompt = build_user_prompt(observation, actions, history)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.15,
            max_tokens=300,
        )

        content = response.choices[0].message.content or ""
        action, explanation = self._parse_response(content, actions)
        return action, explanation

    def _parse_response(
        self, content: str, actions: List[str]
    ) -> Tuple[str, str]:
        """
        Extract JSON even if the model wraps it in markdown fences.
        Falls back safely when parsing fails.
        """
        cleaned = content.strip()

        # Remove ```json ... ``` or ``` ... ```
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
        if fence:
            cleaned = fence.group(1).strip()

        try:
            data = json.loads(cleaned)
            action = str(data.get("action", "")).strip()
            explanation = str(data.get("explanation", "")).strip()
        except Exception:
            action = self._fallback_extract_action(content, actions)
            explanation = "Could not parse JSON — used fallback extraction."

        if action not in actions:
            # Last resort: safe no-op style action
            action = "DESCRIBE"
            explanation = (
                (explanation + " ") if explanation else ""
            ) + "(Invalid action received; defaulted to DESCRIBE.)"

        return action, explanation

    @staticmethod
    def _fallback_extract_action(text: str, actions: List[str]) -> str:
        upper = text.upper()
        for act in actions:
            if act in upper:
                return act
        return "DESCRIBE"
