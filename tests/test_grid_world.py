"""Basic unit tests for the GridWorld environment."""

import pytest

from world.grid_world import GridWorld
from world.tasks import KEY_DOOR_TASK, MAZE_TASK, get_task


@pytest.fixture
def env():
    world = GridWorld()
    world.reset(KEY_DOOR_TASK)
    return world


def test_reset_places_agent(env):
    obs = env._build_observation()
    assert "agent_position" in obs
    assert obs["inventory"]["has_key"] is False
    assert obs["door_open"] is False
    assert obs["notes"] == []


def test_move_into_wall_fails(env):
    obs, reward, done, info = env.step("MOVE_LEFT")
    assert "error" in info
    assert reward < 0


def test_pick_up_key(env):
    key_pos = env._find_cell("K")
    assert key_pos is not None
    env.agent_pos = key_pos

    obs, reward, done, info = env.step("PICK_UP_KEY")
    assert env.inventory["has_key"] is True
    assert reward > 0
    assert "message" in info


def test_cannot_open_door_without_key(env):
    door_pos = env._find_cell("D")
    env.agent_pos = (door_pos[0], door_pos[1] - 1)

    obs, reward, done, info = env.step("OPEN_DOOR")
    assert env.door_open is False
    assert "error" in info


def test_open_door_with_key(env):
    env.inventory["has_key"] = True
    door_pos = env._find_cell("D")
    env.agent_pos = (door_pos[0], door_pos[1] - 1)

    obs, reward, done, info = env.step("OPEN_DOOR")
    assert env.door_open is True
    assert reward > 0


def test_reach_goal(env):
    goal_pos = env._find_cell("G")
    env.agent_pos = goal_pos

    obs, reward, done, info = env.step("DESCRIBE")
    assert done is True
    assert info.get("success") is True
    assert reward >= 8.0


def test_write_note(env):
    obs, reward, done, info = env.step("WRITE_NOTE", note_text="Key is somewhere east")
    assert len(env.notes) == 1
    assert "Key is somewhere east" in env.notes[0]
    assert "message" in info


def test_notes_are_capped(env):
    for i in range(12):
        env.step("WRITE_NOTE", note_text=f"note {i}")
    assert len(env.notes) <= 8


def test_partial_observability():
    world = GridWorld(partial_observability=True, view_radius=1)
    obs = world.reset(KEY_DOOR_TASK)
    # Far cells should be hidden
    grid_str = "".join(obs["grid"])
    assert "?" in grid_str
    assert obs["partial_observability"] is True


def test_custom_map_loads():
    world = GridWorld()
    obs = world.reset(MAZE_TASK)
    assert obs["max_steps"] == 60
    assert len(obs["grid"][0]) == 10


def test_get_task_fallback():
    t = get_task("nonexistent")
    assert t["name"] == "key_door"
