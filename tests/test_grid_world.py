import pytest

from world.grid_world import GridWorld
from world.tasks import Task, make_task

# Agent at (1,1), key at (2,1), door at (2,2), goal at (2,3) sealed behind it.
SMALL = ("#####",
         "#AK.#",
         "##D##",
         "#.G.#",
         "#####")


def make_env(grid=SMALL, max_steps=40):
    env = GridWorld(Task(name="small", grid=grid, max_steps=max_steps))
    env.reset()
    return env


def play(env, actions):
    out = None
    for action in actions:
        out = env.step(action)
    return out


def test_reset_returns_observation_and_info():
    env = GridWorld(Task(name="small", grid=SMALL))
    obs, info = env.reset()
    assert obs["agent_position"] == [1, 1]
    assert obs["inventory"] == {"has_key": False}
    assert info == {"task": "small"}


def test_observation_draws_agent_at_current_position_only():
    env = make_env()
    obs, *_ = env.step("MOVE_RIGHT")
    assert obs["agent_position"] == [2, 1]
    assert obs["grid"][1] == "#.A.#"  # no stale 'A' left at the start tile
    assert "".join(obs["grid"]).count("A") == 1


def test_standing_on_reveals_hidden_tile():
    env = make_env()
    obs, *_ = env.step("MOVE_RIGHT")
    assert obs["standing_on"] == "K"


def test_wall_bump_does_not_move_and_is_penalised():
    env = make_env()
    obs, reward, *_, info = env.step("MOVE_UP")
    assert obs["agent_position"] == [1, 1]
    assert reward < 0
    assert info["error"] == "Bumped into a wall."


def test_closed_door_blocks_movement():
    env = make_env()
    obs, _, _, _, info = play(env, ["MOVE_RIGHT", "MOVE_DOWN"])
    assert obs["agent_position"] == [2, 1]
    assert "Door is closed" in info["error"]


def test_pick_up_key():
    env = make_env()
    obs, reward, _, _, info = play(env, ["MOVE_RIGHT", "PICK_UP_KEY"])
    assert obs["inventory"]["has_key"]
    assert obs["standing_on"] == "."
    assert reward == 1.5
    assert info["message"] == "Picked up the key."


def test_pick_up_with_nothing_there_fails():
    env = make_env()
    _, reward, _, _, info = env.step("PICK_UP_KEY")
    assert reward < 0
    assert info["error"] == "Nothing to pick up here."


def test_open_door_without_key_fails():
    env = make_env()
    _, _, _, _, info = play(env, ["MOVE_RIGHT", "OPEN_DOOR"])
    assert info["error"] == "You do not have the key."
    assert "D" in "".join(env.render_text())


def test_open_door_must_be_adjacent():
    grid = ("######",
            "#A.K.#",
            "###D##",
            "#..G.#",
            "######")
    env = make_env(grid)
    _, _, _, _, info = play(env, ["MOVE_RIGHT", "MOVE_RIGHT", "PICK_UP_KEY", "MOVE_RIGHT",
                                  "MOVE_RIGHT"])  # second move bumps the wall
    assert env.agent_pos == (4, 1)
    _, _, _, _, info = env.step("OPEN_DOOR")
    assert info["error"] == "You must stand next to the door to open it."


def test_full_solution_reaches_goal():
    env = make_env()
    obs, reward, terminated, truncated, info = play(
        env, ["MOVE_RIGHT", "PICK_UP_KEY", "OPEN_DOOR", "MOVE_DOWN", "MOVE_DOWN"]
    )
    assert terminated and not truncated
    assert info["success"] is True
    assert reward == pytest.approx(-0.01 + 8.0)
    assert obs["agent_position"] == [2, 3]


def test_goal_unreachable_without_door_on_bundled_level():
    """Regression: the original map let the agent walk around the door to the goal."""
    from world.solver import solve

    task = make_task("key_door")
    grid = [row.replace("A", ".").replace("K", ".") for row in task.grid]
    start = next((r.index("A"), y) for y, r in enumerate(task.grid) if "A" in r)
    assert solve(grid, start) is None


def test_truncates_at_max_steps():
    env = make_env(max_steps=3)
    *_, terminated, truncated, info = play(env, ["MOVE_UP"] * 3)
    assert truncated and not terminated
    assert info["success"] is False


def test_step_after_episode_end_raises():
    env = make_env(max_steps=1)
    env.step("MOVE_UP")
    with pytest.raises(RuntimeError):
        env.step("MOVE_UP")


def test_invalid_action_is_penalised():
    env = make_env()
    _, reward, _, _, info = env.step("FLY")
    assert reward < 0
    assert info["error"] == "Invalid action: FLY"


def test_describe_env_lists_neighbours():
    env = make_env()
    _, reward, _, _, info = env.step("DESCRIBE")
    assert reward == 0
    assert "right: key" in info["description"]
    assert "up: wall" in info["description"]


def test_reset_restores_state():
    env = make_env()
    play(env, ["MOVE_RIGHT", "PICK_UP_KEY"])
    obs, _ = env.reset()
    assert obs["inventory"]["has_key"] is False
    assert "K" in "".join(obs["grid"])
    assert obs["step"] == 0


def test_door_open_flag_in_observation():
    env = make_env()
    obs, *_ = play(env, ["MOVE_RIGHT", "PICK_UP_KEY"])
    assert obs["door_open"] is False
    obs, *_ = env.step("OPEN_DOOR")
    assert obs["door_open"] is True


def test_partial_observability_hides_far_cells():
    env = GridWorld(make_task("key_door"), partial_observability=True, view_radius=1)
    obs, _ = env.reset()
    grid = obs["grid"]
    assert obs["partial_observability"] is True
    assert grid[1][1] == "A"
    assert grid[1][2] == "."            # distance 1: visible
    assert grid[1][4] == "?"            # key at distance 3: hidden
    assert "G" not in "".join(grid)


def test_partial_observability_reveals_cells_as_agent_moves():
    env = GridWorld(make_task("key_door"), partial_observability=True, view_radius=1)
    env.reset()
    obs, *_ = play(env, ["MOVE_RIGHT", "MOVE_RIGHT"])
    assert obs["grid"][1][4] == "K"


def test_full_observability_has_no_fog():
    obs, _ = GridWorld().reset()
    assert "?" not in "".join(obs["grid"])


def test_colored_render_uses_ansi_codes():
    env = make_env()
    assert "\033[" in env.render_text(use_color=True)
    assert "\033[" not in env.render_text()
