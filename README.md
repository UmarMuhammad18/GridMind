# GridMind

> A lightweight 2D grid world for LLM-driven agents.

GridMind places a Large Language Model inside a grid environment. The agent receives structured observations, can keep short notes (memory), optionally sees only a limited radius (fog-of-war), and must return a single valid action each step until it completes the goal or hits the step limit.

**Supports OpenAI and Groq.** Optional live Pygame visualizer.

Built by **Umar Muhammad**.

---

## Features

| Feature                    | Description                                      |
|---------------------------|--------------------------------------------------|
| Multiple tasks / maps     | `key_door`, `maze`, `long_key`                   |
| Agent memory              | `WRITE_NOTE` – persistent short notes            |
| Partial observability     | Fog-of-war with configurable view radius         |
| Multi-provider LLMs       | OpenAI + Groq (easy to extend)                   |
| Live visualizer           | Optional Pygame window (`--visual`)              |
| Colored terminal output   | Clear, readable ASCII rendering                  |
| Strict action schema      | Reliable JSON interface for the LLM              |
| Logging                   | Full JSON + text episode logs                    |
| Tests                     | Unit tests for the environment                   |

---

## Quick Start

```bash
git clone https://github.com/UmarMuhammad18/GridMind.git
cd GridMind

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Add at least one key:
#   OPENAI_API_KEY=...
#   or GROQ_API_KEY=...
```

Run the default task:

```bash
python main.py
```

---

## Usage Examples

```bash
# List tasks
python main.py --list-tasks

# Specific task + model
python main.py --task maze --provider openai --model gpt-4o-mini

# Fast inference with Groq
python main.py --task key_door --provider groq

# Fog-of-war (partial observability)
python main.py --task maze --partial --view-radius 2

# Live Pygame visualizer
python main.py --task maze --visual

# Combine everything
python main.py --task long_key --partial --visual --provider groq
```

---

## Available Tasks

| Name       | Description                                      |
|------------|--------------------------------------------------|
| `key_door` | Classic: find key → open door → reach goal       |
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

## Architecture

```text
GridMind/
├── main.py                 # CLI entry point
├── agent/
│   ├── llm_agent.py        # High-level agent loop
│   ├── providers.py        # OpenAI / Groq abstraction
│   └── prompt_templates.py
├── world/
│   ├── grid_world.py       # Environment + fog-of-war + memory
│   ├── tasks.py            # Task definitions
│   └── visualizer.py       # Optional Pygame renderer
├── tests/
└── logs/                   # Episode logs (auto-created)
```

The environment and the agent are cleanly separated. You can swap the LLM provider or add new tasks without touching the core loop.

---

## Configuration

| Variable            | Description                          | Default              |
|---------------------|--------------------------------------|----------------------|
| `OPENAI_API_KEY`    | OpenAI API key                       | —                    |
| `GROQ_API_KEY`      | Groq API key                         | —                    |
| `GRIDMIND_PROVIDER` | `openai` or `groq`                   | auto-detect          |
| `GRIDMIND_MODEL`    | Model name override                  | provider default     |

---

## Running Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

---

## Roadmap

- [x] Multiple tasks
- [x] Agent memory (`WRITE_NOTE`)
- [x] Partial observability
- [x] Multi-provider support (OpenAI + Groq)
- [x] Pygame visualizer
- [ ] Procedural / multi-room maps
- [ ] Anthropic + local model support
- [ ] Multi-agent scenarios
- [ ] Web-based visualizer

---

## License

MIT
