import json
import os
from datetime import datetime

from world.grid_world import GridWorld
from world.tasks import KEY_DOOR_TASK
from agent.llm_agent import LLMAgent


def run_episode(max_steps: int = 40, log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)

    env = GridWorld()
    agent = LLMAgent()

    observation = env.reset(KEY_DOOR_TASK)
    history = []

    for step in range(max_steps):
        print(f"\n=== STEP {step} ===")
        env.render()

        action, explanation = agent.choose_action(
            observation=observation,
            actions=env.action_space,
            history=history,
        )

        print(f"Agent chose action: {action}")
        if explanation:
            print(f"Reasoning: {explanation}")

        new_obs, reward, done, info = env.step(action)

        step_record = {
            "step": step,
            "action": action,
            "explanation": explanation,
            "reward": reward,
            "done": done,
            "info": info,
            "observation": observation,
        }
        history.append(step_record)

        observation = new_obs

        if done:
            print("\nEpisode finished.")
            env.render()
            print(f"Final info: {info}")
            break

    # Save logs
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(log_dir, f"run_{timestamp}.json")
    txt_path = os.path.join(log_dir, f"run_{timestamp}.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        for step in history:
            f.write(f"STEP {step['step']}\n")
            f.write(f"Action: {step['action']}\n")
            f.write(f"Explanation: {step['explanation']}\n")
            f.write(f"Reward: {step['reward']}\n")
            f.write(f"Done: {step['done']}\n")
            f.write(f"Info: {step['info']}\n")
            f.write(f"Observation: {json.dumps(step['observation'])}\n")
            f.write("\n")

    print(f"\nLogs saved to:\n- {json_path}\n- {txt_path}")


if __name__ == "__main__":
    run_episode()
