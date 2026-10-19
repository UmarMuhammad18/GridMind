import pytest

from world.levels import LevelError, generate_level, list_levels, load_level, validate_level
from world.solver import solve
from world.tasks import make_task


@pytest.mark.parametrize("name", list_levels())
def test_bundled_levels_are_valid(name):
    grid = load_level(name)
    validate_level(grid)
    assert make_task(name).optimal_steps > 0


@pytest.mark.parametrize("name", list_levels())
def test_bundled_levels_require_the_door(name):
    grid = load_level(name)
    start = next((row.index("A"), y) for y, row in enumerate(grid) if "A" in row)
    no_key = [row.replace("A", ".").replace("K", ".") for row in grid]
    assert solve(no_key, start) is None


@pytest.mark.parametrize("seed", range(50))
def test_random_levels_are_valid(seed):
    validate_level(generate_level(seed))


def test_random_levels_are_deterministic_per_seed():
    assert generate_level(7) == generate_level(7)
    assert generate_level(7) != generate_level(8)


def test_task_max_steps_scales_with_level_size():
    task = make_task("maze")
    assert task.max_steps >= 3 * task.optimal_steps


@pytest.mark.parametrize(
    "grid, message",
    [
        (["####", "#AG#", "###"], "same width"),
        (["#####", "#A.G#", "#####", "#X..#"], "Unknown cell"),
        (["#####", "#...#", "#..G#", "#####"], "agent"),
        (["#####", "#A.G#", "#.#.G", "#####"], "goal"),
        (["#####", "A..G#", "#####"], "surrounded"),
        (["#####", "#A#G#", "#####"], "not solvable"),
        (["######", "#A#.G#", "##D###", "######"], "not solvable"),  # door but no key
    ],
)
def test_invalid_levels_are_rejected(grid, message):
    with pytest.raises(LevelError, match=message):
        validate_level(grid)


def test_unknown_level_name():
    with pytest.raises(LevelError, match="Unknown level"):
        load_level("does_not_exist")


def test_load_level_from_path(tmp_path):
    path = tmp_path / "custom.txt"
    path.write_text("#####\n#A.G#\n#####\n", encoding="utf-8")
    assert load_level(str(path)) == ["#####", "#A.G#", "#####"]
