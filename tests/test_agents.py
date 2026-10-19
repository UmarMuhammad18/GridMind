import json

import pytest

from agent.baselines import BFSAgent, RandomAgent
from agent.factory import make_agent
from agent.llm_agent import LLMAgent, ResponseError, parse_response
from agent.llm_clients import LLMResponse
from runner import run_episode
from world.grid_world import ACTIONS, GridWorld
from world.levels import list_levels
from world.tasks import make_task


class FakeClient:
    """Returns canned replies in order and records the prompts it was sent."""

    name = "fake:test"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append(user)
        return LLMResponse(self.replies.pop(0), input_tokens=100, output_tokens=20)


def reply(action, explanation="ok", note=""):
    return json.dumps({"action": action, "explanation": explanation, "note": note})


@pytest.mark.parametrize("level", list_levels() + ["random"])
def test_bfs_agent_solves_optimally(level):
    task = make_task(level, seed=1)
    result = run_episode(GridWorld(task), BFSAgent(), verbose=False, log_dir=None)
    assert result.success
    assert result.steps == task.optimal_steps


def test_random_agent_is_seeded():
    obs, _ = GridWorld().reset()
    first = [RandomAgent(3).act(obs, ACTIONS, []).action for _ in range(1)]
    a, b = RandomAgent(3), RandomAgent(3)
    assert [a.act(obs, ACTIONS, []).action for _ in range(20)] == \
           [b.act(obs, ACTIONS, []).action for _ in range(20)]
    assert first[0] in ACTIONS


def test_parse_response_accepts_valid_json():
    assert parse_response(reply("move_up", "go", "n"), ACTIONS) == ("MOVE_UP", "go", "n")


def test_parse_response_strips_code_fences():
    text = "```json\n" + reply("PICK_UP_KEY") + "\n```"
    assert parse_response(text, ACTIONS)[0] == "PICK_UP_KEY"


@pytest.mark.parametrize("text", ["not json", "[1, 2]", reply("FLY")])
def test_parse_response_rejects_bad_replies(text):
    with pytest.raises(ResponseError):
        parse_response(text, ACTIONS)


def test_llm_agent_returns_action_and_tracks_usage():
    agent = LLMAgent(FakeClient([reply("MOVE_RIGHT")]))
    obs, _ = GridWorld().reset()
    decision = agent.act(obs, ACTIONS, [])
    assert decision.action == "MOVE_RIGHT" and decision.valid
    assert (agent.usage.calls, agent.usage.input_tokens, agent.usage.output_tokens) == (1, 100, 20)


def test_llm_agent_retries_invalid_reply_with_feedback():
    client = FakeClient(["garbage", reply("MOVE_DOWN")])
    agent = LLMAgent(client, max_retries=2)
    obs, _ = GridWorld().reset()
    decision = agent.act(obs, ACTIONS, [])
    assert decision.action == "MOVE_DOWN" and decision.valid
    assert "previous reply was invalid" in client.prompts[1]


def test_llm_agent_falls_back_after_retries():
    agent = LLMAgent(FakeClient(["garbage"] * 3), max_retries=2)
    obs, _ = GridWorld().reset()
    decision = agent.act(obs, ACTIONS, [])
    assert decision.action == "DESCRIBE"
    assert not decision.valid
    assert agent.usage.calls == 3


def test_llm_agent_keeps_notes_between_steps():
    client = FakeClient([reply("MOVE_RIGHT", note="key is at (4,1)"), reply("MOVE_RIGHT")])
    agent = LLMAgent(client)
    obs, _ = GridWorld().reset()
    agent.act(obs, ACTIONS, [])
    agent.act(obs, ACTIONS, [])
    assert "key is at (4,1)" in client.prompts[1]


def test_llm_agent_completes_episode_with_scripted_replies():
    task = make_task("key_door")
    from world.solver import solve
    start = next((r.index("A"), y) for y, r in enumerate(task.grid) if "A" in r)
    plan = solve([r.replace("A", ".") for r in task.grid], start)

    agent = LLMAgent(FakeClient([reply(a) for a in plan]))
    result = run_episode(GridWorld(task), agent, pricing={"input_per_mtok": 1.0,
                                                           "output_per_mtok": 2.0},
                         verbose=False, log_dir=None)
    assert result.success
    assert result.llm_calls == len(plan)
    assert result.cost == pytest.approx(len(plan) * (100 * 1.0 + 20 * 2.0) / 1_000_000)


def test_unknown_prompt_rejected():
    with pytest.raises(ValueError):
        LLMAgent(FakeClient([]), prompt="nope")


def test_make_agent_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_agent("llm", {"provider": "openai", "model": "gpt-4o-mini", "api_key": None})
