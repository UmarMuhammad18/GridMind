import json
import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI

from agent.prompt_templates import SYSTEM_PROMPT, build_user_prompt


class LLMAgent:
    """
    Simple LLM-based agent that:
    - Receives an observation and action space
    - Builds a prompt
    - Calls an LLM
    - Parses a JSON response with { "action": ..., "explanation": ... }
    """

    def __init__(self, model: str | None = None) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable is not set. "
                "Set it before running the agent."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model or "gpt-4o-mini"

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
            temperature=0.2,
        )

        content = response.choices[0].message.content
        action, explanation = self._parse_response(content, actions)
        return action, explanation

    def _parse_response(
        self, content: str, actions: List[str]
    ) -> Tuple[str, str]:
        """
        Try to parse the model's response as JSON and extract the action.
        If parsing fails or the action is invalid, fall back to a safe default.
        """
        try:
            data = json.loads(content)
            action = data.get("action", "").strip()
            explanation = data.get("explanation", "").strip()
        except Exception:
            # Fallback: try to extract an action name by scanning
            action = self._fallback_extract_action(content, actions)
            explanation = "Fallback parsing: could not parse JSON response."

        if action not in actions:
            # If invalid, choose a safe default
            action = "DESCRIBE_ENV"
            explanation = (
                explanation
                + " (Chosen DESCRIBE_ENV as a safe default due to invalid action.)"
            )

        return action, explanation

    @staticmethod
    def _fallback_extract_action(text: str, actions: List[str]) -> str:
        upper_text = text.upper()
        for act in actions:
            if act in upper_text:
                return act
        return "DESCRIBE_ENV"
