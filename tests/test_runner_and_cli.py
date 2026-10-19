import json

import evaluate
import main
import replay
from agent.baselines import BFSAgent
from config import load_config
from runner import run_episode
from world.grid_world import GridWorld


def test_run_episode_writes_logs(tmp_path):
    result = run_episode(GridWorld(), BFSAgent(), verbose=False, log_dir=tmp_path)
    data = json.loads(open(result.log_path, encoding="utf-8").read())
    assert data["meta"]["success"] is True
    assert len(data["steps"]) == result.steps
    assert data["steps"][0]["position"] == [1, 1]
    assert (tmp_path / result.log_path.replace(".json", ".txt")).exists()


def test_replay_prints_episode(tmp_path, capsys):
    result = run_episode(GridWorld(), BFSAgent(), verbose=False, log_dir=tmp_path)
    assert replay.main([result.log_path, "--delay", "0"]) == 0
    out = capsys.readouterr().out
    assert "=== STEP 0 ===" in out
    assert "SUCCESS: bfs on key_door in 11 steps" in out


def test_main_cli_runs_bfs(tmp_path, capsys):
    assert main.main(["--agent", "bfs", "--quiet", "--log-dir", str(tmp_path)]) == 0
    assert "SUCCESS" in capsys.readouterr().out


def test_main_cli_reports_bad_level(capsys):
    assert main.main(["--agent", "bfs", "--level", "nope", "--log-dir", "unused"]) == 1
    assert "Unknown level" in capsys.readouterr().err


def test_evaluate_cli_writes_summary(tmp_path, capsys):
    code = evaluate.main(["--agents", "bfs", "random", "--levels", "key_door", "random",
                          "--episodes", "2", "--out-dir", str(tmp_path)])
    assert code == 0
    summary = json.loads(next(tmp_path.glob("eval_*.json")).read_text(encoding="utf-8"))
    rows = {(r["agent"], r["level"]): r for r in summary["summary"]}
    assert rows[("bfs", "key_door")]["success_rate"] == 1.0
    assert rows[("bfs", "random")]["efficiency"] == 1.0
    assert "Success" in capsys.readouterr().out


def test_config_layering(tmp_path, monkeypatch):
    for var in ("GRIDMIND_PROVIDER", "GRIDMIND_MODEL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)  # keep load_dotenv from picking up a real .env
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "openai", "model": "gpt-4o",
                                "api_key": "YOUR_API_KEY_HERE"}), encoding="utf-8")

    config = load_config(path)
    assert config["model"] == "gpt-4o"
    assert config["api_key"] is None  # placeholder ignored

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    switched = load_config(path, {"provider": "anthropic"})
    assert switched["model"] == "claude-opus-5-5"  # file's OpenAI model not carried over
    assert switched["api_key"] == "sk-test"
    assert "temperature" not in switched
