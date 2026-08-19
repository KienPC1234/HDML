from __future__ import annotations

import torch
import numpy as np
from hdml.evaluation.unitree_a1_maze_env import UnitreeA1MazeEnv
from hdml.models.hdml_model import HDMLModel
from hdml.utils.config import HDMLConfig


def evaluate_dynamic_goals():
    cfg = HDMLConfig.from_yaml("configs/unitree_a1_maze_unsupervised.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HDMLModel.from_config(cfg.model).to(device)
    ckpt = torch.load("checkpoints/unitree_a1_maze/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = np.load("data/unitree_a1_maze_trajectories.npz")
    st_mean = dataset["state_mean"]
    st_std = dataset["state_std"]

    test_goals = [
        ("North-East Chamber", (2.8, 2.8), (-2.8, -2.8)),
        ("North-West Corner", (-2.8, 2.8), (2.8, -2.8)),
        ("South-East Chamber", (2.8, -2.8), (-2.8, 2.8)),
    ]

    print("=======================================================================")
    print("Zero-Shot HDML Generalization on Dynamic Random Goal Pillar Locations:")
    print("=======================================================================")

    results = []

    for name, goal, start in test_goals:
        env = UnitreeA1MazeEnv(max_episode_steps=900, goal_pos=goal, start_pos=start, sensor_noise=0.02)
        obs, info = env.reset(seed=100, options={"goal": goal})

        history_states = []
        history_actions = []
        history_rtgs = []
        history_timesteps = []
        current_rtg = 1.0
        hx = None
        solved = False

        for t in range(900):
            norm_obs = (obs - st_mean) / st_std
            scaled_rtg = current_rtg / 1.0

            history_states.append(norm_obs)
            history_rtgs.append(scaled_rtg)
            history_timesteps.append(t)
            if len(history_actions) == 0:
                history_actions.append(np.zeros(cfg.model.action_dim, dtype=np.float32))

            ctx_len = min(len(history_states), cfg.training.context_length)
            ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
            ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
            ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
            ctx_time = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

            t_states = torch.from_numpy(ctx_states).unsqueeze(0).to(device)
            t_actions = torch.from_numpy(ctx_actions).unsqueeze(0).to(device)
            t_rtgs = torch.from_numpy(ctx_rtgs).unsqueeze(0).to(device)
            t_timesteps = torch.from_numpy(ctx_time).unsqueeze(0).to(device)

            with torch.inference_mode():
                act_tensor, hx, _ = model.get_action(
                    states=t_states,
                    rtgs=t_rtgs,
                    actions=t_actions,
                    timesteps=t_timesteps,
                    hx=hx,
                )

            act = act_tensor[0].cpu().numpy().astype(np.float32)
            act = np.clip(act, -1.0, 1.0)

            obs, r, term, trunc, step_info = env.step(act)
            history_actions.append(act)
            current_rtg -= float(r)

            if step_info.get("goal_reached", False):
                solved = True
                print(f"[PASSED] {name}: Solved at step {t:03d}! Final pos: [{step_info['pos'][0]:+.2f}, {step_info['pos'][1]:+.2f}] | Dist: {step_info['dist_to_goal']:.2f}m")
                results.append((name, True, t, step_info["dist_to_goal"]))
                break
            if term or trunc:
                print(
                    f"[ENDED] {name}: Step {t:03d} | Pos: [{step_info['pos'][0]:+.2f}, {step_info['pos'][1]:+.2f}] | "
                    f"Term: {term}, Trunc: {trunc} | Height: {step_info['height']:.3f} | "
                    f"Roll: {step_info['roll']:.3f} | Pitch: {step_info['pitch']:.3f} | Dist: {step_info['dist_to_goal']:.2f}m"
                )
                results.append((name, False, t, step_info["dist_to_goal"]))
                break

        env.close()

    print("\n-----------------------------------------------------------------------")
    print(f"Summary: {sum(r[1] for r in results)}/{len(results)} dynamic goals reached successfully.")
    print("-----------------------------------------------------------------------\n")


if __name__ == "__main__":
    evaluate_dynamic_goals()
