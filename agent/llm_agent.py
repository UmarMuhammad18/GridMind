import json
from typing import Any

from agent.base import Decision, Usage
from agent.llm_clients import LLMClient
from agent.prompt_templates import SYSTEM_PROMPTS, build_user_prompt

FALLBACK_ACTION = "DESCRIBE_ENV"


class ResponseError(ValueError):
    pass


class LLMAgent:
    """
    LLM-based agent that:
    - Builds a prompt from the observation, recent history and its own notes
    - Calls an LLM through a provider-neutral client
    - Parses a JSON response with { "action", "explanation", "note" }
    - Re-asks the model when the reply is invalid, then falls back to a safe action
    """

    def __init__(
        self,
        client: LLMClient,
        prompt: str = "default",
        max_retries: int = 2,
        history_window: int = 8,
        max_notes: int = 10,
    ) -> None:
        if prompt not in SYSTEM_PROMPTS:
            raise ValueError(f"Unknown prompt '{prompt}'. Choose from: {', '.join(SYSTEM_PROMPTS)}")
        self.client = client
        self.prompt = prompt
        self.system_prompt = SYSTEM_PROMPTS[prompt]
        self.max_retries = max_retries
        self.history_window = history_window
        self.max_notes = max_notes
        self.name = f"llm[{client.name}|{prompt}]"
        self.reset()

    def reset(self) -> None:
        self.notes: list[str] = []
        self.usage = Usage()

    def act(
        self,
        observation: dict[str, Any],
        actions: list[str],
        history: list[dict[str, Any]],
    ) -> Decision:
        user_prompt = build_user_prompt(
            observation, actions, history, self.notes, self.history_window
        )

        prompt = user_prompt
        error = ""
        for _ in range(self.max_retries + 1):
            response = self.client.complete(self.system_prompt, prompt)
            self.usage.calls += 1
            self.usage.input_tokens += response.input_tokens
            self.usage.output_tokens += response.output_tokens

            try:
                action, explanation, note = parse_response(response.text, actions)
            except ResponseError as exc:
                error = str(exc)
                prompt = (
                    f"{user_prompt}\n\nYour previous reply was invalid: {error}\n"
                    "Reply again with only the JSON object."
                )
                continue

            if note:
                self.notes = (self.notes + [note])[-self.max_notes :]
            return Decision(action, explanation)

        return Decision(
            FALLBACK_ACTION,
            f"Chose {FALLBACK_ACTION} as a safe default: {error}",
            valid=False,
        )


def parse_response(text: str, actions: list[str]) -> tuple[str, str, str]:
    """Parse and validate a model reply. Raises ResponseError if unusable."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ResponseError(f"response was not valid JSON ({exc.msg}).") from exc
    if not isinstance(data, dict):
        raise ResponseError("response must be a JSON object.")

    action = str(data.get("action", "")).strip().upper()
    if action not in actions:
        raise ResponseError(f"'{action}' is not one of the allowed actions {actions}.")

    explanation = str(data.get("explanation", "")).strip()
    note = str(data.get("note") or "").strip()
    return action, explanation, note
