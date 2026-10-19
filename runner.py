"""Runs a single episode of an agent in a GridWorld and records what happened."""

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.base import Agent
from world.grid_world import GridWorld


@dataclass
class EpisodeResult:
    agent: str
    level: str
    success: bool
    steps: int
    total_reward: float
    invalid_actions: int
    optimal_steps: int | None = None
    seed: int | None = None
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    log_path: str | None = None

    def summary(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("history")
        return data


def run_episode(
    env: GridWorld,
    agent: Agent,
    seed: int | None = None,
    pricing: dict[str, float] | None = None,
    verbose: bool = True,
    log_dir: str | Path | None = "logs",
) -> EpisodeResult:
    agent.reset()
    observation, _ = env.reset()
    history: list[dict[str, Any]] = []
    total_reward = 0.0
    invalid = 0
    success = False

    while True:
        if verbose:
            print(f"\n=== STEP {env.step_count} ===")
            env.render()

        decision = agent.act(observation, env.action_space, history)
        if verbose:
            print(f"Agent chose action: {decision.action}")
            if decision.explanation:
                print(f"Reasoning: {decision.explanation}")

        position = list(env.agent_pos)
        new_obs, reward, terminated, truncated, info = env.step(decision.action)
        total_reward += reward
        if not decision.valid or "Invalid action" in info.get("error", ""):
            invalid += 1

        history.append({
            "step": len(history),
            "position": position,
            "action": decision.action,
            "explanation": decision.explanation,
            "valid": decision.valid,
            "reward": reward,
            "position_after": new_obs["agent_position"],
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
            "observation": observation,
        })
        observation = new_obs

        if terminated or truncated:
            success = bool(info.get("success"))
            if verbose:
                print("\nEpisode finished.")
                env.render()
                print(f"Final info: {info}")
            break

    usage = getattr(agent, "usage", None)
    result = EpisodeResult(
        agent=agent.name,
        level=env.task.name,
        success=success,
        steps=len(history),
        total_reward=round(total_reward, 4),
        invalid_actions=invalid,
        optimal_steps=env.task.optimal_steps,
        seed=seed,
        llm_calls=usage.calls if usage else 0,
        input_tokens=usage.input_tokens if usage else 0,
        output_tokens=usage.output_tokens if usage else 0,
        cost=usage.cost(pricing) if usage else None,
        history=history,
    )

    if log_dir is not None:
        result.log_path = str(save_logs(result, observation, log_dir))
        if verbose:
            print(f"\nLog saved to: {result.log_path}")
    return result


def save_logs(result: EpisodeResult, final_observation: dict[str, Any],
              log_dir: str | Path) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{result.agent}_{result.level}").strip("-")
    json_path = log_dir / f"run_{timestamp}_{slug}.json"
    txt_path = json_path.with_suffix(".txt")

    payload = {
        "meta": result.summary() | {"created": timestamp},
        "steps": result.history,
        "final_observation": final_observation,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with txt_path.open("w", encoding="utf-8") as f:
        for step in result.history:
            f.write(f"STEP {step['step']}\n")
            f.write(f"Position: {step['position']} -> {step['position_after']}\n")
            f.write(f"Action: {step['action']}\n")
            f.write(f"Explanation: {step['explanation']}\n")
            f.write(f"Reward: {step['reward']}\n")
            f.write(f"Info: {step['info']}\n")
            f.write(f"Observation: {json.dumps(step['observation'])}\n\n")

    return json_path
