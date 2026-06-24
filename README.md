# GridMind

> A virtual world for LLM-driven agents.

Built by **Umar Muhammad**

GridMind is a lightweight, fully observable 2D grid-world environment designed to explore how Large Language Models can operate as autonomous agents.

The project places an LLM inside a virtual environment where it must perceive the world, reason about its current state, and choose actions to achieve a goal. Inspired by concepts from reinforcement learning, robotics, and agent-based systems, GridMind provides a clean agent-environment harness for experimenting with autonomous decision-making.

---

## Features

* Fully observable 2D grid world
* Interactive objects (keys, locked doors, goals)
* Structured JSON observations
* Goal-driven task system
* LLM-powered reasoning loop
* Step-by-step decision logging
* Modular architecture with clear separation of concerns
* Strict action schema for reliable execution

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
2. The observation is sent to the LLM.
3. The LLM reasons about the current state.
4. The LLM returns a valid action.
5. The environment executes the action.
6. The process repeats until the task is completed or the step limit is reached.

Example objective:

> Find the key, unlock the door, and reach the goal tile.

---

## Architecture

```text
+-----------------------------------------------------------+
|                         GridMind                          |
+-----------------------------------------------------------+

+-------------------+        +------------------------------+
|   GridWorld Env   | <----> |         LLM Agent            |
|  - Grid state     |        |  - Receives observations     |
|  - Rules & logic  |        |  - Performs reasoning        |
|  - Rewards        |        |  - Returns next action       |
+-------------------+        +------------------------------+

          ^                                |
          |                                v

+-------------------+        +------------------------------+
|    Task Module    |        |        Control Loop          |
|  - Goal text      |        |  - Runs episodes            |
|  - Step limits    |        |  - Logs decisions           |
+-------------------+        +------------------------------+
```

The architecture intentionally maintains a clean boundary between:

* Environment simulation
* Agent reasoning
* Task definition
* Episode orchestration

This mirrors patterns commonly found in reinforcement learning and robotics systems.

---

## Observation Format

The agent receives structured JSON observations containing:

```json
{
  "position": [1, 2],
  "inventory": {
    "has_key": false
  },
  "goal": "Find the key, unlock the door, and reach the goal.",
  "step": 3,
  "grid": [
    ["#", "#", "#"],
    ["#", "A", "."],
    ["#", "K", "G"]
  ]
}
```

Providing structured observations makes the agent's reasoning process transparent, deterministic, and easy to debug.

---

## Action Space

GridMind uses a constrained action space to keep agent behavior predictable and machine-readable.

Supported actions:

```text
MOVE_UP
MOVE_DOWN
MOVE_LEFT
MOVE_RIGHT
PICK_UP_KEY
OPEN_DOOR
DESCRIBE
```

The agent must always return a valid JSON response:

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "Moving toward the key."
}
```

This prevents ambiguous outputs and ensures reliable environment execution.

---

## Example Episode

```text
=== STEP 0 ===

########
#A..K..#
#..##..#
#..D.G.#
########

Inventory: { has_key: false }

Goal:
Find the key, then open the door and reach the goal tile.

Agent Action:
MOVE_RIGHT

Reasoning:
Moving toward the key is the next logical step.
```

Full example runs can be found in:

```text
examples/sample_run.md
```

---

## Project Structure

```text
GridMind/
│
├── main.py
├── requirements.txt
├── config.example.json
├── .gitignore
│
├── agent/
│   ├── llm_agent.py
│   ├── prompt_templates.py
│   └── __init__.py
│
├── world/
│   ├── grid_world.py
│   ├── tasks.py
│   └── __init__.py
│
├── logs/
│
└── examples/
    └── sample_run.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/GridMind.git
cd GridMind
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Or create a configuration file based on:

```text
config.example.json
```

---

## Running GridMind

Start an episode:

```bash
python main.py
```

The agent will:

* Observe the environment
* Reason about the current state
* Select actions
* Interact with objects
* Attempt to complete the task

Logs are automatically saved to:

```text
logs/
```

---

## Design Principles

### Structured Observations

The environment exposes state in a machine-readable format, making debugging and experimentation straightforward.

### Discrete Action Space

Restricting actions keeps the agent grounded and prevents invalid outputs.

### Strict JSON Interface

Every action must conform to a predefined schema, ensuring reliable communication between the agent and the environment.

### Separation of Concerns

* `world/` handles simulation and environment logic
* `agent/` handles reasoning and decision-making
* `main.py` orchestrates execution

This modular design makes it easy to swap environments, prompts, models, or tasks.

---

## Future Improvements

* Partial observability (fog of war)
* Memory-augmented agents
* Multi-room environments
* Dynamic task generation
* Additional object types and interactions
* Multi-agent simulations
* Web-based visualizer
* Pygame renderer
* Reinforcement learning integrations

---

## Tech Stack

| Component     | Technology              |
| ------------- | ----------------------- |
| Language      | Python                  |
| LLM Interface | OpenAI API              |
| Configuration | JSON                    |
| Logging       | Python Standard Library |
| Environment   | Custom Grid World       |

---

## Author

**Umar Muhammad**

Computer Science Student — University of West London

Aspiring Software Engineer, AI Engineer, Systems Builder, and 2d Game Dev.

---

## License

MIT License

Feel free to use, modify, and build upon GridMind.
