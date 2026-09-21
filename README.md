# GridMind

> A lightweight 2D grid world for LLM-driven agents.

GridMind places a Large Language Model inside a fully observable grid environment. The agent receives structured JSON observations, reasons about the state, and returns a single valid action each step until it completes the goal or hits the step limit.

Built by **Umar Muhammad**.

---

## Features

- Fully observable 2D grid world
- Multiple built-in tasks / maps
- Structured JSON observations + strict action schema
- LLM agent with robust JSON parsing (handles markdown fences)
- Step-by-step decision logging (JSON + text)
- CLI interface
- Unit tests for the environment

---

## Quick Start

```bash
git clone https://github.com/UmarMuhammad18/GridMind.git
cd GridMind

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

Run the default task:

```bash
python main.py
```

List available tasks:

```bash
python main.py --list-tasks
```

Run a specific task with a different model:

```bash
python main.py --task maze --model gpt-4o-mini --max-steps 50
```

---

## Available Tasks

| Name       | Description                                      |
|------------|--------------------------------------------------|
| `key_door` | Classic: find key → open door → reach goal      |
| `maze`     | Larger maze layout                               |
| `long_key` | Key is far away; requires longer planning        |

---

## Action Space

```text
MOVE_UP | MOVE_DOWN | MOVE_LEFT | MOVE_RIGHT
PICK_UP_KEY
OPEN_DOOR
DESCRIBE
```

The agent must always reply with:

```json
{
  "action": "MOVE_RIGHT",
  "explanation": "Moving toward the key."
}
```

---

## Project Structure

```text
GridMind/
├── main.py                 # CLI entry point
├── requirements.txt
├── config.example.json
├── .env.example
├── agent/
│   ├── llm_agent.py
│   └── prompt_templates.py
├── world/
│   ├── grid_world.py
│   └── tasks.py
├── tests/
│   └── test_grid_world.py
├── examples/
│   └── sample_run.md
└── logs/                   # created at runtime
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## How It Works

1. Environment builds a structured observation (grid, position, inventory, goal…).
2. Observation + recent history are sent to the LLM.
3. LLM returns a single JSON action.
4. Environment executes the action and returns the next observation + reward.
5. Loop continues until success or max steps.

---

## Future Improvements

- [ ] Partial observability (fog of war)
- [ ] Persistent memory / notes for the agent
- [ ] Pygame or web visualizer
- [ ] Procedural / multi-room maps
- [ ] Support for additional LLM providers
- [ ] Multi-agent scenarios

---

## License

MIT
