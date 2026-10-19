# GridMind

> A virtual world for LLM-driven agents.

Built by **Umar Muhammad**

GridMind is a lightweight, fully observable 2D grid-world environment and benchmark for exploring how Large Language Models operate as autonomous agents.

The project places an LLM inside a virtual environment where it must perceive the world, reason about its current state, and choose actions to achieve a goal. Inspired by concepts from reinforcement learning, robotics, and agent-based systems, GridMind provides a clean agent-environment harness for experimenting with autonomous decision-making, and an evaluation suite for comparing models and prompts against fixed baselines.

---

## Features

* 2D grid world with a Gymnasium-style API, fully observable or with fog of war
* Interactive objects (keys, locked doors, goals)
* Levels loaded from text files, plus seeded random level generation
* Every level is checked for solvability, and the goal can only be reached through the door
* Pluggable LLM providers: OpenAI, Anthropic (Claude), Groq, and local models via Ollama
* Enforced JSON output, automatic re-asking on invalid replies, and API retries
* Agent memory: recent moves with positions, loop detection, and self-written notes
* Baseline agents: random (lower bound) and BFS-optimal (upper bound)
* Evaluation script reporting success rate, efficiency, invalid actions, tokens and cost
* Colored terminal output and an optional Pygame window, live or for replays
* Episode logs with a replay viewer
* Test suite and CI

---

## How It Works

The agent is placed inside a virtual environment containing:

* Walls
* Open spaces
* Keys
* Locked doors
* Goal tiles

At every step:

1. The environment generates an observation.
2. The observation is sent to the agent (an LLM or a baseline).
3. The agent reasons about the current state.
4. The agent returns a valid action.
5. The environment executes the action.
6. The process repeats until the goal is reached or the step limit runs out.

Example objective:

> Find the key, then go to the door, open it, and reach the goal tile behind it.

---

## Architecture

```text
+-----------------------------------------------------------+
|                         GridMind                          |
+-----------------------------------------------------------+

+-------------------+        +------------------------------+
|   GridWorld Env   | <----> |            Agent             |
|  - Grid state     |        |  - LLMAgent (any provider)   |
|  - Rules & logic  |        |  - RandomAgent / BFSAgent    |
|  - Rewards        |        |  - Returns next action       |
+-------------------+        +------------------------------+
          ^                                |
          |                                v
+-------------------+        +------------------------------+
|  Levels & Tasks   |        |     Runner / Evaluation      |
|  - Text levels    |        |  - Runs episodes             |
|  - Random levels  |        |  - Logs & metrics            |
|  - BFS solver     |        |  - Replay                    |
+-------------------+        +------------------------------+
```

The architecture intentionally maintains a clean boundary between:

* Environment simulation (`world/`)
* Agent reasoning (`agent/`)
* Level and task definition (`levels/`, `world/levels.py`, `world/tasks.py`)
* Episode orchestration and evaluation (`runner.py`, `main.py`, `evaluate.py`)

---

## Observation Format

The agent receives structured JSON observations:

```json
{
  "agent_position": [1, 1],
  "standing_on": ".",
  "grid": [
    "########",
    "#A..K..#",
    "#......#",
    "###D####",
    "#....G.#",
    "########"
  ],
  "inventory": { "has_key": false },
  "door_open": false,
  "partial_observability": false,
  "goal": "Find the key, then go to the door, open it, and reach the goal tile behind it.",
  "step": 0,
  "max_steps": 40
}
```

Coordinates are `[x, y]`: `x` is the column, `y` is the row, and `y` grows downward. `standing_on` reports the tile under the agent, since `A` hides it on the grid.

### Fog of war

With `--partial`, the agent only sees tiles within `--view-radius` steps (Manhattan distance, default 2); everything else is shown as `?`. The agent has to explore, and its notes become its map. The BFS baseline assumes it can see the whole grid, so it isn't a meaningful upper bound in this mode.

---

## Action Space

```text
MOVE_UP
MOVE_DOWN
MOVE_LEFT
MOVE_RIGHT
PICK_UP_KEY
OPEN_DOOR
DESCRIBE
```

* `PICK_UP_KEY` works while standing on the key.
* `OPEN_DOOR` works while holding the key and standing next to the door (not diagonally).
* `DESCRIBE` returns a text description of the neighbouring tiles.

The LLM must reply with a JSON object:

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "Moving toward the key.",
  "note": "Key at [4, 1]."
}
```

`note` is optional memory: notes are carried into later prompts. Invalid replies are sent back to the model with the error, and if it still fails the agent falls back to `DESCRIBE`. Each fallback counts as an invalid action in the metrics.

### Rewards

| Event                       | Reward |
| --------------------------- | ------ |
| Move                        | -0.01  |
| Bump into wall / closed door, failed `OPEN_DOOR` | -0.15 |
| Failed `PICK_UP_KEY`        | -0.1   |
| Invalid action              | -0.2   |
| Pick up key                 | +1.5   |
| Open door                   | +2.5   |
| Reach goal                  | +8.0   |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/UmarMuhammad18/GridMind.git
cd GridMind
```

Install dependencies (Python 3.10+):

```bash
pip install -r requirements.txt          # or requirements-dev.txt for tests and linting
```

Add your API key to a `.env` file (see `.env.example`):

```bash
OPENAI_API_KEY=your_api_key
# or
ANTHROPIC_API_KEY=your_api_key
# or
GROQ_API_KEY=your_api_key
```

Optionally copy `config.example.json` to `config.json` to set the provider, model, prompt, retries and pricing. Settings are applied in this order, later ones winning:

```text
defaults < config.json < environment (.env) < command-line flags
```

| Provider    | Default model     | Notes                                                         |
| ----------- | ----------------- | ------------------------------------------------------------- |
| `openai`    | `gpt-4o-mini`     | JSON mode enforced                                            |
| `anthropic` | `claude-opus-5-5` | JSON schema enforced, `effort` set to `low` (configurable)    |
| `groq`      | `llama-3.3-70b-versatile` | Fast hosted open models, JSON mode enforced           |
| `ollama`    | `llama3.1`        | Local, via `http://localhost:11434/v1`; no API key needed     |

Costs are only reported when `pricing` (dollars per million input and output tokens) is set in `config.json`. Check your provider's current prices.

---

## Running GridMind

Run one episode:

```bash
python main.py                                         # LLM agent on the default level
python main.py --provider anthropic                    # use Claude
python main.py --level maze --prompt planner           # different level and prompt
python main.py --level random --seed 42                # generated level
python main.py --agent bfs                             # optimal baseline, no API key needed
python main.py --level path/to/my_level.txt            # your own level
python main.py --partial --view-radius 2               # fog of war
python main.py --visual                                # Pygame window
python main.py --list-tasks                            # list levels
```

`--task` works as an alias for `--level`. The Pygame window needs `pygame-ce` (included in `requirements.txt`); if it can't start, GridMind falls back to the terminal.

Each run writes a JSON and text log to `logs/`. Replay a run in the terminal:

```bash
python replay.py logs/run_<timestamp>_<agent>_<level>.json
python replay.py logs/run_<...>.json --step            # press Enter to advance
python replay.py logs/run_<...>.json --visual          # replay in the Pygame window
```

---

## Evaluation

`evaluate.py` runs several episodes per agent and level and prints a comparison table:

```bash
python evaluate.py --agents bfs random --levels key_door maze random --episodes 10
python evaluate.py --agents llm bfs --levels key_door two_rooms --episodes 5
python evaluate.py --agents llm --models openai:gpt-4o-mini anthropic:claude-opus-5-5
python evaluate.py --agents llm --prompts default planner
python evaluate.py --agents llm random --partial       # benchmark under fog of war
```

```text
Agent   Level     N  Success  Avg steps  Efficiency  Invalid  Reward  Tokens  Cost
------  --------  -  -------  ---------  ----------  -------  ------  ------  ----
bfs     key_door  5  100%     11.0       100%        0%       11.91   0       -
bfs     maze      5  100%     23.0       100%        0%       11.79   0       -
bfs     random    5  100%     14.4       100%        0%       11.88   0       -
random  key_door  5  0%       -          -           0%       -2.03   0       -
random  maze      5  0%       -          -           0%       -5.25   0       -
random  random    5  0%       -          -           0%       -2.14   0       -
```

* **Success**: share of episodes that reached the goal.
* **Avg steps**: mean steps over successful episodes.
* **Efficiency**: optimal steps ÷ steps taken, averaged over successes (100% = as good as BFS).
* **Invalid**: share of steps where the agent produced no usable action.

Episode `i` uses seed `--seed + i`, so random levels are reproducible. A full summary is saved to `logs/eval_<timestamp>.json`; add `--save-logs` to keep every episode's log for replay.

---

## Levels

Levels are plain text files in `levels/`:

```text
########
#A..K..#
#......#
###D####
#....G.#
########
```

| Symbol | Meaning     |
| ------ | ----------- |
| `#`    | Wall        |
| `.`    | Empty       |
| `A`    | Agent start |
| `K`    | Key         |
| `D`    | Locked door |
| `G`    | Goal        |

A level must be rectangular, surrounded by walls, have exactly one `A` and one `G`, at most one `K` and one `D`, and be solvable. These rules are checked on load. The step limit is 40 or three times the optimal path length, whichever is larger.

Bundled levels: `key_door` (default), `two_rooms`, `maze`, `long_key` (the door is next to the start but the key is far away). Use `random` for a generated two-room level.

---

## Project Structure

```text
GridMind/
│
├── main.py                 # Run one episode
├── evaluate.py             # Benchmark agents/models/prompts
├── replay.py               # Replay a saved episode
├── runner.py               # Episode loop, metrics and logging
├── config.py               # Config loading (file, .env, CLI)
├── config.example.json
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
│
├── agent/
│   ├── base.py             # Agent protocol, Decision, Usage
│   ├── llm_agent.py        # LLM agent: prompting, parsing, retries, notes
│   ├── llm_clients.py      # OpenAI / Anthropic / Groq / Ollama clients
│   ├── prompt_templates.py # System prompts and user prompt builder
│   ├── baselines.py        # RandomAgent, BFSAgent
│   └── factory.py
│
├── world/
│   ├── grid_world.py       # The environment (incl. fog of war, colored rendering)
│   ├── visualizer.py       # Optional Pygame window
│   ├── levels.py           # Level loading, validation, generation
│   ├── solver.py           # BFS solver
│   └── tasks.py
│
├── levels/                 # Level files
├── tests/                  # pytest suite
├── examples/
│   └── sample_run.md
└── logs/                   # Episode logs (git-ignored)
```

---

## Development

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The tests use the baselines and a fake LLM client, so they need no API key. CI runs lint and tests on every push and pull request.

---

## Design Principles

### Structured Observations

The environment exposes state in a machine-readable format, making debugging and experimentation straightforward.

### Discrete Action Space

Restricting actions keeps the agent grounded and prevents invalid outputs.

### Strict JSON Interface

Every action must conform to a predefined schema, which is enforced by the provider where possible and validated by the agent.

### Baselines Before Claims

Every LLM result is reported next to a random agent and an optimal BFS agent, so success rates and efficiency can be read in context.

### Separation of Concerns

* `world/` handles simulation and environment logic
* `agent/` handles reasoning and decision-making
* `runner.py`, `main.py` and `evaluate.py` orchestrate execution

This modular design makes it easy to swap environments, prompts, models, or tasks.

---

## Future Improvements

* Multi-room environments with several keys and doors
* Additional object types and interactions
* An exploring baseline for fog-of-war levels
* Multi-agent simulations
* Web-based replay viewer
* Reinforcement learning integrations

---

## Tech Stack

| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Language      | Python 3.10+                         |
| LLM Interface | OpenAI, Anthropic, Groq, Ollama      |
| Visualization | ANSI terminal colors, Pygame (pygame-ce) |
| Configuration | JSON + `.env`                        |
| Testing       | pytest, ruff, GitHub Actions         |
| Environment   | Custom grid world (Gymnasium-style)  |

---

## Author

**Umar Muhammad**

Computer Science Student — University of West London

Aspiring Software Engineer, AI Engineer, Systems Builder, and 2d Game Dev.

---

## License

MIT License

Feel free to use, modify, and build upon GridMind.
