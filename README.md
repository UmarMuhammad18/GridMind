# GridMind

> A lightweight 2D grid world for LLM-driven agents.

GridMind places a Large Language Model inside a grid environment. The agent receives structured JSON observations, can keep short notes (memory), optionally sees only a limited radius (fog-of-war), and must return a single valid action each step until it completes the goal or hits the step limit.

Built by **Umar Muhammad**.

---

## Features

- Fully observable **or** partial observability (fog-of-war)
- Agent **memory** via `WRITE_NOTE` (persistent short notes)
- Multiple built-in tasks / maps
- Structured JSON observations + strict action schema
- Robust LLM JSON parsing (handles markdown fences)
- Colored terminal renderer
- CLI interface
- Unit tests

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

Run with fog-of-war (partial observability):

```bash
python main.py --task maze --partial --view-radius 2
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
WRITE_NOTE          # save a short memory note
DESCRIBE
```

Example agent reply:

```json
{
  "action": "WRITE_NOTE",
  "explanation": "I should remember where the key is.",
  "note": "Key seen at (4,1)"
}
```

---

## Partial Observability

When `--partial` is enabled, the agent only sees cells within a Manhattan distance of `--view-radius` (default 2). Everything else appears as `?`.

This makes the task significantly harder and encourages the use of `WRITE_NOTE`.

---

## Project Structure

```text
GridMind/
├── main.py
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
└── logs/                   # created at runtime
```

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Future Improvements

- [ ] Pygame or web visualizer
- [ ] Procedural / multi-room maps
- [ ] Support for additional LLM providers (Anthropic, Groq, local)
- [ ] Multi-agent scenarios
- [ ] Stronger memory (vector notes / summaries)

---

## License

MIT
