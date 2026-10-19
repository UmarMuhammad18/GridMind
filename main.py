"""Run a single GridMind episode."""

import argparse
import sys

from agent.factory import AGENT_KINDS, make_agent
from agent.llm_clients import PROVIDER_DEFAULTS, LLMError
from agent.prompt_templates import SYSTEM_PROMPTS
from config import load_config
from runner import run_episode
from world.grid_world import GridWorld
from world.levels import LevelError, list_levels
from world.tasks import list_tasks, make_task


def add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.json", help="Path to a config file.")
    parser.add_argument("--provider", choices=list(PROVIDER_DEFAULTS))
    parser.add_argument("--model", help="Model name for the chosen provider.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=AGENT_KINDS, default="llm")
    parser.add_argument("--level", "--task", dest="level", default="key_door",
                        help=f"One of {', '.join(list_levels())}, 'random', or a .txt path.")
    parser.add_argument("--list-tasks", "--list-levels", action="store_true",
                        help="List the bundled levels and exit.")
    parser.add_argument("--seed", type=int, help="Seed for random levels and the random agent.")
    parser.add_argument("--max-steps", type=int, help="Override the task's step limit.")
    parser.add_argument("--prompt", choices=list(SYSTEM_PROMPTS))
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--quiet", action="store_true", help="Only print the final result.")
    parser.add_argument("--visual", action="store_true", help="Show a Pygame window.")
    add_env_arguments(parser)
    add_llm_arguments(parser)
    return parser.parse_args(argv)


def add_env_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--partial", action="store_true",
                        help="Enable fog of war (partial observability).")
    parser.add_argument("--view-radius", type=int, default=2,
                        help="How far the agent can see with --partial.")


def open_visualizer(title: str):
    try:
        from world.visualizer import Visualizer

        return Visualizer(title=title)
    except Exception as exc:  # pygame missing or no display available
        print(f"[warn] Could not start visualizer: {exc}. Falling back to terminal only.",
              file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_tasks:
        for name in list_tasks():
            print(name)
        return 0

    config = load_config(args.config, {
        "provider": args.provider, "model": args.model, "prompt": args.prompt,
    })

    viz = None
    try:
        task = make_task(args.level, seed=args.seed, max_steps=args.max_steps)
        agent = make_agent(args.agent, config, seed=args.seed)
        env = GridWorld(task, partial_observability=args.partial, view_radius=args.view_radius)
        if args.visual:
            viz = open_visualizer(f"GridMind - {task.name}")
        result = run_episode(env, agent, seed=args.seed,
                             pricing=config.get("pricing"), verbose=not args.quiet,
                             log_dir=args.log_dir, visualizer=viz)
    except (LevelError, LLMError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if viz:
            viz.close()

    outcome = "SUCCESS" if result.success else "FAILED"
    line = f"\n{outcome}: {result.agent} on {result.level} in {result.steps} steps"
    if result.optimal_steps:
        line += f" (optimal {result.optimal_steps})"
    if result.llm_calls:
        line += f", {result.input_tokens + result.output_tokens} tokens"
    if result.cost is not None:
        line += f", ${result.cost:.4f}"
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
