from __future__ import annotations

import logging
from typing import Any
import gymnasium as gym
import numpy as np
import torch

from hdml.models.hdml_model import HDMLModel
from hdml.evaluation.perturbations import SensorNoisePerturbation, ForceImpulsePerturbation
from hdml.utils.metrics import compute_action_smoothness, compute_action_rate_of_change

logger = logging.getLogger(__name__)


class HDMLEvaluator:
    """Evaluates HDML policy in closed-loop MuJoCo continuous control simulations."""

    def __init__(
        self,
        model: HDMLModel,
        env_name: str = "Ant-v5",
        context_length: int = 20,
        target_return: float = 5000.0,
        scale_return: float = 1000.0,
        state_mean: np.ndarray | None = None,
        state_std: np.ndarray | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()

        self.env_name = env_name
        self.context_length = context_length
        self.target_return = target_return
        self.scale_return = scale_return

        self.state_mean = state_mean
        self.state_std = state_std

    def evaluate_episode(
        self,
        env: gym.Env[Any, Any],
        seed: int = 42,
        sensor_noise: SensorNoisePerturbation | None = None,
        force_perturb: ForceImpulsePerturbation | None = None,
        max_steps: int = 1000,
        macro_interval: int = 1,
    ) -> dict[str, Any]:
        """Run a single closed-loop evaluation episode with optional hierarchical decoupling.

        Args:
            env: Active Gymnasium simulation environment.
            seed: Environment reset seed.
            sensor_noise: Optional sensor noise generator.
            force_perturb: Optional force impulse generator.
            max_steps: Maximum allowable episode steps.
            macro_interval: Number of micro-steps per Mamba macro-planning invocation (1 = synchronous).

        Returns:
            Dictionary containing episodic return, length, actions, and metrics.
        """
        obs, _ = env.reset(seed=seed)
        obs_dim = env.observation_space.shape[0]  # type: ignore
        act_dim = env.action_space.shape[0]        # type: ignore

        # Prepare state normalization
        st_mean = self.state_mean if self.state_mean is not None else np.zeros(obs_dim, dtype=np.float32)
        st_std = self.state_std if self.state_std is not None else np.ones(obs_dim, dtype=np.float32)

        # Rolling history buffers for context window
        history_states: list[np.ndarray] = []
        history_actions: list[np.ndarray] = []
        history_rtgs: list[float] = []
        history_timesteps: list[int] = []

        ep_actions: list[np.ndarray] = []
        ep_rewards: list[float] = []

        target_rtg = self.target_return
        cfc_hx = None
        current_subgoal: torch.Tensor | None = None

        for t in range(max_steps):
            raw_obs = np.asarray(obs, dtype=np.float32)
            if sensor_noise is not None:
                raw_obs = sensor_noise.apply(raw_obs)

            norm_obs = (raw_obs - st_mean) / st_std
            scaled_rtg = target_rtg / self.scale_return

            history_states.append(norm_obs)
            history_rtgs.append(scaled_rtg)
            history_timesteps.append(t)
            if len(history_actions) == 0:
                history_actions.append(np.zeros(act_dim, dtype=np.float32))            # Trigger Mamba Macro-Planner at macro intervals (or on first step)
            if t % macro_interval == 0 or current_subgoal is None:
                ctx_len = min(len(history_states), self.context_length)
                ctx_states = np.array(history_states[-ctx_len:], dtype=np.float32)
                ctx_actions = np.array(history_actions[-ctx_len:], dtype=np.float32)
                ctx_rtgs = np.array(history_rtgs[-ctx_len:], dtype=np.float32).reshape(-1, 1)
                ctx_timesteps = np.array(history_timesteps[-ctx_len:], dtype=np.int64)

                if ctx_len < self.context_length:
                    pad_k = self.context_length - ctx_len
                    pad_states = np.zeros((pad_k, obs_dim), dtype=np.float32)
                    pad_actions = np.zeros((pad_k, act_dim), dtype=np.float32)
                    pad_rtgs = np.zeros((pad_k, 1), dtype=np.float32)
                    pad_time = np.zeros((pad_k,), dtype=np.int64)

                    inp_states = np.vstack([pad_states, ctx_states])
                    inp_actions = np.vstack([pad_actions, ctx_actions])
                    inp_rtgs = np.vstack([pad_rtgs, ctx_rtgs])
                    inp_timesteps = np.concatenate([pad_time, ctx_timesteps])
                else:
                    inp_states = ctx_states
                    inp_actions = ctx_actions
                    inp_rtgs = ctx_rtgs
                    inp_timesteps = ctx_timesteps

                t_states = torch.from_numpy(inp_states).unsqueeze(0).to(self.device)
                t_actions = torch.from_numpy(inp_actions).unsqueeze(0).to(self.device)
                t_rtgs = torch.from_numpy(inp_rtgs).unsqueeze(0).to(self.device)
                t_timesteps = torch.from_numpy(inp_timesteps).unsqueeze(0).to(self.device)

                with torch.inference_mode():
                    action_tensor, cfc_hx, current_subgoal = self.model.get_action(
                        states=t_states,
                        rtgs=t_rtgs,
                        actions=t_actions,
                        timesteps=t_timesteps,
                        hx=cfc_hx,
                    )
            else:
                # Fast Micro-Actuation: Direct Liquid ODE Execution without Mamba
                t_prop_current = torch.from_numpy(norm_obs).unsqueeze(0).to(self.device)
                with torch.inference_mode():
                    action_tensor, cfc_hx = self.model.liquid_head(
                        subgoals=current_subgoal,
                        current_prop=t_prop_current,
                        hx=cfc_hx,
                    )

            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Apply force perturbation if active
            exec_action = action
            if force_perturb is not None:
                exec_action = force_perturb.apply_action_perturbation(action)

            next_obs, reward, terminated, truncated, _ = env.step(exec_action)

            ep_actions.append(action)
            ep_rewards.append(float(reward))
            target_rtg -= float(reward)

            # Update action history with the actual action taken (the input action at
            # the final context position is thus a_{t-1}, matching the training
            # causal action-input convention).
            history_actions.append(action)

            obs = next_obs
            done = terminated or truncated
            if done:
                break

        actions_arr = np.array(ep_actions, dtype=np.float32)
        total_return = float(sum(ep_rewards))
        smoothness = compute_action_smoothness(actions_arr)
        rate_of_change = compute_action_rate_of_change(actions_arr)

        return {
            "episode_return": total_return,
            "episode_length": len(ep_rewards),
            "action_smoothness": smoothness,
            "action_rate_of_change": rate_of_change,
        }

    def evaluate_benchmark(
        self,
        num_episodes: int = 10,
        with_perturbations: bool = False,
        macro_interval: int = 1,
        seed: int = 42,
    ) -> dict[str, float]:
        """Evaluate HDML agent across multiple benchmark episodes.

        Args:
            num_episodes: Number of evaluation episodes.
            with_perturbations: If True, applies sensor noise & force impulses.
            macro_interval: Micro-steps per Mamba macro-planning invocation.
            seed: Base seed.

        Returns:
            Dictionary with mean and std for returns, lengths, and smoothness metrics.
        """
        env = gym.make(self.env_name)
        sensor_noise = SensorNoisePerturbation(noise_std=0.05, seed=seed) if with_perturbations else None
        force_perturb = ForceImpulsePerturbation(impulse_prob=0.05, force_magnitude=0.5, seed=seed) if with_perturbations else None

        returns: list[float] = []
        lengths: list[int] = []
        smoothnesses: list[float] = []
        rates_of_change: list[float] = []

        logger.info(
            f"Evaluating HDML on {self.env_name} for {num_episodes} episodes "
            f"(Perturbations={'ON' if with_perturbations else 'OFF'}, Macro Interval={macro_interval})..."
        )

        for ep in range(num_episodes):
            ep_result = self.evaluate_episode(
                env=env,
                seed=seed + ep,
                sensor_noise=sensor_noise,
                force_perturb=force_perturb,
                macro_interval=macro_interval,
            )
            returns.append(ep_result["episode_return"])
            lengths.append(ep_result["episode_length"])
            smoothnesses.append(ep_result["action_smoothness"])
            rates_of_change.append(ep_result["action_rate_of_change"])

        env.close()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        results = {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "mean_smoothness": float(np.mean(smoothnesses)),
            "mean_rate_of_change": float(np.mean(rates_of_change)),
        }

        logger.info(
            f"Evaluation Summary [{self.env_name}]: Return = {results['mean_return']:.2f} +/- {results['std_return']:.2f} "
            f"| Length = {results['mean_length']:.1f} | Smoothness (Jerk) = {results['mean_smoothness']:.4f}"
        )
        return results
