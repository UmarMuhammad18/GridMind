#!/usr/bin/env python3
"""
GridMind — run an LLM agent inside a 2D grid world.
"""

import argparse
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

from agent.llm_agent import LLMAgent
from world.grid_world import GridWorld
from world.tasks import get_task, list_tasks


def run_episode(
    task_name: str = "key_door",
    max_steps: int | None = None,
    model: str | None = None,
    partial: bool = False,
    view_radius: int = 2,
    log_dir: str = "logs",
) -> dict:
    load_dotenv()

    os.makedirs(log_dir, exist_ok=True)

    task = get_task(task_name)
    if max_steps is not None:
        task["max_steps"] = max_steps

    env = GridWorld(partial_observability=partial, view_radius=view_radius)
    agent = LLMAgent(model=model)

    observation = env.reset(task)
    history = []

    print(f"\n=== GridMind | task={task['name']} | model={agent.model} ===")
    if partial:
        print(f"Partial observability ON (radius={view_radius})")
    print(f"Goal: {task['goal']}\n")

    for step in range(task["max_steps"]):
        print(f"\n=== STEP {step} ===")
        env.render(use_color=True)

        action, explanation, note = agent.choose_action(
            observation=observation,
            actions=env.action_space,
            history=history,
        )

        print(f"Action     : {action}")
        if explanation:
            print(f"Reasoning  : {explanation}")
        if note:
            print(f"Note       : {note}")

        new_obs, reward, done, info = env.step(action, note_text=note)

        record = {
            "step": step,
            "action": action,
            "explanation": explanation,
            "note": note,
            "reward": reward,
            "done": done,
            "info": info,
            "observation": observation,
        }
        history.append(record)
        observation = new_obs

        if done:
            print("\n--- Episode finished ---")
            env.render(use_color=True)
            print(f"Result: {info}")
            break

    # Persist logs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"run_{task_name}_{ts}.json")
    txt_path = os.path.join(log_dir, f"run_{task_name}_{ts}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for s in history:
            f.write(f"STEP {s['step']}\n")
            f.write(f"Action: {s['action']}\n")
            f.write(f"Explanation: {s['explanation']}\n")
            if s.get("note"):
                f.write(f"Note: {s['note']}\n")
            f.write(f"Reward: {s['reward']}\n")
            f.write(f"Info: {s['info']}\n\n")

    print(f"\nLogs saved:\n  {json_path}\n  {txt_path}")

    success = any(h.get("info", {}).get("success") for h in history)
    return {"success": success, "steps": len(history), "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description="GridMind LLM Agent")
    parser.add_argument(
        "--task",
        default="key_door",
        choices=list_tasks(),
        help="Which task / map to run",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--model", default=None, help="OpenAI model name")
    parser.add_argument(
        "--partial",
        action="store_true",
        help="Enable partial observability (fog-of-war)",
    )
    parser.add_argument(
        "--view-radius",
        type=int,
        default=2,
        help="Vision radius when --partial is set (Manhattan distance)",
    )
    parser.add_argument("--list-tasks", action="store_true")
    args = parser.parse_args()

    if args.list_tasks:
        print("Available tasks:")
        for name in list_tasks():
            t = get_task(name)
            print(f"  {name:12}  {t['goal'][:70]}...")
        return

    run_episode(
        task_name=args.task,
        max_steps=args.max_steps,
        model=args.model,
        partial=args.partial,
        view_radius=args.view_radius,
    )


if __name__ == "__main__":
    main()
