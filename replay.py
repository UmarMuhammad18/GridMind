"""
Replay a saved episode in the terminal.

  python replay.py logs/run_20260101_120000_000000_bfs_key_door.json
  python replay.py <log> --step        # press Enter to advance
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


def load_log(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "steps" not in data:
        raise ValueError(f"{path} is not a GridMind episode log.")
    return data


def format_step(step: dict[str, Any]) -> str:
    obs = step["observation"]
    info = step["info"]
    result = info.get("message") or info.get("error") or info.get("description", "")
    lines = [
        f"=== STEP {step['step']} ===",
        *obs["grid"],
        f"Inventory: {obs['inventory']}",
        f"Action: {step['action']}" + ("" if step.get("valid", True) else "  (invalid reply)"),
    ]
    if step.get("explanation"):
        lines.append(f"Reasoning: {step['explanation']}")
    lines.append(f"Reward: {step['reward']:+.2f}" + (f"  - {result}" if result else ""))
    return "\n".join(lines)


def format_summary(data: dict[str, Any]) -> str:
    meta = data.get("meta", {})
    lines = ["=== FINAL ===", *data.get("final_observation", {}).get("grid", [])]
    if meta:
        lines.append(
            f"{'SUCCESS' if meta.get('success') else 'FAILED'}: {meta.get('agent')} on "
            f"{meta.get('level')} in {meta.get('steps')} steps"
            + (f" (optimal {meta['optimal_steps']})" if meta.get("optimal_steps") else "")
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("log", help="Path to a run_*.json log.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds between steps.")
    parser.add_argument("--step", action="store_true", help="Wait for Enter between steps.")
    parser.add_argument("--visual", action="store_true", help="Also replay in a Pygame window.")
    args = parser.parse_args(argv)

    try:
        data = load_log(args.log)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    viz = None
    if args.visual:
        from main import open_visualizer

        viz = open_visualizer(f"GridMind replay - {Path(args.log).name}")

    try:
        for step in data["steps"]:
            print(format_step(step) + "\n")
            if viz and not _draw(viz, step["observation"]):
                break
            if args.step:
                input("[Enter] ")
            elif args.delay:
                time.sleep(args.delay)
        print(format_summary(data))
        if viz and "final_observation" in data:
            _draw(viz, data["final_observation"], status="SUCCESS"
                  if data.get("meta", {}).get("success") else "FAILED")
            time.sleep(1.5)
    finally:
        if viz:
            viz.close()
    return 0


def _draw(viz: Any, observation: dict[str, Any], status: str = "") -> bool:
    return viz.draw(
        grid=observation["grid"],
        agent_pos=tuple(observation["agent_position"]),
        inventory=observation["inventory"],
        door_open=observation.get("door_open", False),
        notes=[],
        step=observation["step"],
        max_steps=observation["max_steps"],
        status=status,
    )


if __name__ == "__main__":
    sys.exit(main())
