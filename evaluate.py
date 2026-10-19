"""
Benchmark agents across levels and print a comparison table.

Examples:
  python evaluate.py --agents bfs random --levels key_door maze random --episodes 10
  python evaluate.py --agents llm --models openai:gpt-4o-mini anthropic:claude-opus-5-5
  python evaluate.py --agents llm --prompts default planner
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from agent.factory import AGENT_KINDS, make_agent
from agent.llm_clients import LLMError
from agent.prompt_templates import SYSTEM_PROMPTS
from config import load_config
from main import add_env_arguments, add_llm_arguments
from runner import EpisodeResult, run_episode
from world.grid_world import GridWorld
from world.levels import LevelError, list_levels
from world.tasks import make_task


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--agents", nargs="+", choices=AGENT_KINDS, default=["bfs", "random"])
    parser.add_argument("--levels", nargs="+", default=["key_door"],
                        help=f"Any of {', '.join(list_levels())}, 'random', or .txt paths.")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per agent and level.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Episode i uses seed + i (random levels and random agent).")
    parser.add_argument("--models", nargs="+", metavar="PROVIDER:MODEL",
                        help="Compare several LLMs, e.g. openai:gpt-4o-mini.")
    parser.add_argument("--prompts", nargs="+", choices=list(SYSTEM_PROMPTS),
                        help="Compare several system prompts.")
    parser.add_argument("--save-logs", action="store_true", help="Write a log per episode.")
    parser.add_argument("--out-dir", default="logs", help="Where to write the summary JSON.")
    add_env_arguments(parser)
    add_llm_arguments(parser)
    return parser.parse_args(argv)


def agent_variants(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Expand --agents/--models/--prompts into (kind, config) pairs."""
    variants = []
    for kind in args.agents:
        if kind != "llm":
            variants.append((kind, {}))
            continue
        models = args.models or [None]
        prompts = args.prompts or [None]
        for spec in models:
            provider, model = args.provider, args.model
            if spec:
                provider, _, model = spec.partition(":")
                model = model or None
            for prompt in prompts:
                overrides = {"provider": provider, "model": model, "prompt": prompt}
                variants.append((kind, load_config(args.config, overrides)))
    return variants


def summarise(results: list[EpisodeResult]) -> dict[str, Any]:
    wins = [r for r in results if r.success]
    total_steps = sum(r.steps for r in results)
    efficiency = [r.optimal_steps / r.steps for r in wins if r.optimal_steps]
    costs = [r.cost for r in results if r.cost is not None]
    return {
        "agent": results[0].agent,
        "level": results[0].level if len({r.level for r in results}) == 1 else "random",
        "episodes": len(results),
        "success_rate": len(wins) / len(results),
        "avg_steps_success": mean(r.steps for r in wins) if wins else None,
        "efficiency": mean(efficiency) if efficiency else None,
        "invalid_rate": sum(r.invalid_actions for r in results) / total_steps,
        "avg_reward": mean(r.total_reward for r in results),
        "tokens": sum(r.input_tokens + r.output_tokens for r in results),
        "cost": sum(costs) if costs else None,
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    def fmt(value: Any, kind: str) -> str:
        if value is None:
            return "-"
        spec = {"pct": "{:.0%}", "f1": "{:.1f}", "f2": "{:.2f}", "usd": "${:.4f}"}
        return spec.get(kind, "{}").format(value)

    columns = [("agent", "Agent", ""), ("level", "Level", ""), ("episodes", "N", ""),
               ("success_rate", "Success", "pct"), ("avg_steps_success", "Avg steps", "f1"),
               ("efficiency", "Efficiency", "pct"), ("invalid_rate", "Invalid", "pct"),
               ("avg_reward", "Reward", "f2"), ("tokens", "Tokens", ""), ("cost", "Cost", "usd")]
    table = [[title for _, title, _ in columns]]
    table += [[fmt(row[key], kind) for key, _, kind in columns] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows, episodes = [], []

    try:
        for kind, config in agent_variants(args):
            for level in args.levels:
                results = []
                for i in range(args.episodes):
                    seed = args.seed + i
                    task = make_task(level, seed=seed)
                    agent = make_agent(kind, config, seed=seed)
                    env = GridWorld(task, partial_observability=args.partial,
                                    view_radius=args.view_radius)
                    result = run_episode(
                        env, agent, seed=seed, pricing=config.get("pricing"),
                        verbose=False, log_dir=args.out_dir if args.save_logs else None,
                    )
                    results.append(result)
                    episodes.append(result.summary())
                    print(f"  {result.agent} {result.level} seed={seed}: "
                          f"{'success' if result.success else 'fail'} in {result.steps} steps",
                          file=sys.stderr)
                rows.append(summarise(results))
    except (LevelError, LLMError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    print(format_table(rows))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"eval_{timestamp}.json"
    out_path.write_text(json.dumps({"args": vars(args), "summary": rows, "episodes": episodes},
                                   indent=2), encoding="utf-8")
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
