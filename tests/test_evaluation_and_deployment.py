"""Tests for Evaluation, PACE Controller, Metrics, and ONNX Deployment."""

from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from hdml.evaluation.evaluator import HDMLEvaluator
from hdml.evaluation.pace_controller import PACEController
from hdml.utils.metrics import get_d4rl_normalized_score
from hdml.utils.rliable_metrics import compute_iqm, stratified_bootstrap_ci, compute_probability_of_improvement
from hdml.deployment.onnx_exporter import export_hdml_to_onnx, infer_native_mamba3


def test_pace_controller() -> None:
    pace = PACEController(threshold=0.5, norm_p=2)
    # Set a plan
    action_chunk = torch.randn(4, 6)
    predicted_states = torch.zeros(4, 10)
    pace.set_new_plan(action_chunk, predicted_states)

    # First action under threshold
    act, trunc = pace.get_next_action(torch.zeros(10))
    assert act is not None
    assert not trunc

    # Reset
    pace.reset()
    assert pace.active_chunk is None
    assert pace.statistics()["events"] == []


def test_pace_prediction_is_checked_at_its_own_horizon() -> None:
    """s_{t+k} must not be compared with s_{t+1}."""
    pace = PACEController(threshold=0.5)
    action_chunk = torch.zeros(5, 3)
    # Prediction made at step 5 for step 10 (k=5).
    predicted = torch.tensor([[1.0, 0.0, 0.0]])
    pace.set_new_plan(
        action_chunk,
        predicted,
        prediction_due_steps=[10],
        horizon=20,
        current_step=5,
    )

    for step in (5, 6, 9):
        needs, reason, deviation = pace.should_replan(torch.zeros(3), current_step=step)
        assert not needs
        assert reason == "none"
        assert deviation is None

    needs, reason, deviation = pace.should_replan(torch.zeros(3), current_step=10)
    assert needs
    assert reason == "deviation"
    assert deviation is not None and abs(deviation - 1.0) < 1e-6
    stats = pace.statistics()
    assert stats["deviation_triggers"] == 1
    assert stats["events"][0]["step"] == 10
    assert stats["events"][0]["reason"] == "deviation"


def test_pace_distinguishes_expiry_from_deviation() -> None:
    pace = PACEController(threshold=0.5)
    pace.set_new_plan(
        torch.zeros(5, 3),
        predicted_states=None,
        horizon=3,
        current_step=7,
    )
    assert pace.should_replan(torch.zeros(3), current_step=7)[0] is False
    assert pace.should_replan(torch.zeros(3), current_step=9)[0] is False

    needs, reason, deviation = pace.should_replan(torch.zeros(3), current_step=10)
    assert needs and reason == "expired"
    assert deviation is None
    stats = pace.statistics()
    assert stats["expiry_triggers"] == 1
    assert stats["deviation_triggers"] == 0
    assert stats["events"][0] == {
        "step": 10,
        "reason": "expired",
        "deviation": None,
        "threshold": 0.5,
        "plan_start_step": 7,
        "horizon": 3,
    }


class _ConstantObsEnv:
    """Minimal gym-like environment used to exercise evaluator integration."""

    def __init__(self, prop_dim: int = 3, action_dim: int = 3) -> None:
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(prop_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        self._obs = np.zeros(prop_dim, dtype=np.float32)
        self.steps = 0

    def reset(self, seed: int | None = None):
        self.steps = 0
        return self._obs.copy(), {}

    def step(self, action):
        self.steps += 1
        return self._obs.copy(), 0.0, False, False, {}


class _CountingPolicy(nn.Module):
    """Deterministic fake policy that records macro/micro call counts."""

    def __init__(self, prop_dim: int = 3, action_dim: int = 3) -> None:
        super().__init__()
        self.prop_dim = int(prop_dim)
        self.action_dim = int(action_dim)
        self.d_subgoal = 4
        self.d_model = 8
        self.chunk_size = 5
        self.get_action_calls = 0
        self.act_from_subgoal_calls = 0
        self.prediction_offset = 10.0

    def get_action(self, states, rtgs, actions, timesteps, hx=None):
        self.get_action_calls += 1
        batch = states.shape[0]
        current = states[:, -1, :]
        predicted = current + self.prediction_offset
        return (
            torch.zeros(batch, self.action_dim, device=states.device),
            torch.zeros(batch, 4, device=states.device),
            {
                "subgoal": torch.zeros(batch, self.d_subgoal, device=states.device),
                "latent": torch.zeros(batch, self.d_model, device=states.device),
                "next_state": predicted,
                "chunk": torch.zeros(
                    batch, self.chunk_size, self.action_dim, device=states.device
                ),
            },
        )

    def act_from_subgoal(self, subgoal, current_prop, rtg, hx=None, latent=None):
        self.act_from_subgoal_calls += 1
        return (
            torch.zeros(current_prop.shape[0], self.action_dim, device=current_prop.device),
            hx,
        )


def test_pace_evaluator_integration_does_not_replan_every_step() -> None:
    """PACE must trigger on the due prediction without forcing per-step backbone calls."""
    policy = _CountingPolicy()
    env = _ConstantObsEnv()
    evaluator = HDMLEvaluator(
        model=policy,
        env_name="dummy",
        context_length=20,
        target_return=0.0,
        scale_return=1.0,
        device="cpu",
    )
    pace = PACEController(threshold=0.5)
    result = evaluator.evaluate_episode(
        env=env,
        seed=0,
        max_steps=12,
        macro_interval=10,
        pace_controller=pace,
    )

    # Macro plans at t=0, t=5 (prediction due), t=10 (next prediction due).
    assert policy.get_action_calls == 3
    assert policy.act_from_subgoal_calls == 9
    assert result["pace"]["deviation_triggers"] == 2
    assert result["pace"]["expiry_triggers"] == 0
    assert [event["step"] for event in result["pace"]["events"]] == [5, 10]


def test_rliable_metrics() -> None:
    scores = np.array([1.0, 1.2, 0.8, 1.5, 0.9, 1.1, 1.3])
    iqm = compute_iqm(scores)
    assert 0.8 <= iqm <= 1.5

    point_est, ci_low, ci_high = stratified_bootstrap_ci(scores, num_bootstraps=100)
    assert np.isfinite(point_est) and np.isfinite(ci_low) and np.isfinite(ci_high)
    assert ci_low <= point_est <= ci_high

    scores_b = scores + 0.2
    p_pt, p_lo, p_hi = compute_probability_of_improvement(scores_b.reshape(1, -1), scores.reshape(1, -1))
    assert all(np.isfinite(v) for v in (p_pt, p_lo, p_hi))
    assert 0.0 <= p_lo <= p_pt <= p_hi <= 1.0


def test_onnx_export(tmp_path) -> None:
    from hdml.models.hdml_model import HDMLModel
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, cfc_units=16,
        use_native_mamba3=False,
    )
    onnx_file = tmp_path / "model.onnx"
    export_hdml_to_onnx(model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10)
    assert onnx_file.exists()


def test_onnx_export_native_mamba3(tmp_path) -> None:
    import torch
    from hdml.models.hdml_model import HDMLModel
    if not torch.cuda.is_available():
        pytest.skip("native Mamba-3 ONNX export requires CUDA")
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2, cfc_units=16,
        use_native_mamba3=True, mamba3_headdim=32, d_state=16,
    ).cuda()
    onnx_file = tmp_path / "model_native.onnx"
    export_hdml_to_onnx(model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10)
    assert onnx_file.exists()
    assert onnx_file.stat().st_size > 0


def test_onnx_export_action_only_and_without_cfc(tmp_path) -> None:
    from hdml.models.hdml_model import HDMLModel
    model = HDMLModel(
        prop_dim=17, action_dim=6, d_model=64, num_mamba_layers=2,
        use_cfc=False, action_policy="gaussian", use_critic=False,
        use_native_mamba3=False,
    )
    onnx_file = tmp_path / "model_action_only.onnx"
    res = export_hdml_to_onnx(
        model, str(onnx_file), prop_dim=17, action_dim=6, context_length=10,
        action_only=True, verify=True,
    )
    assert onnx_file.exists()
    assert res["action_only"] is True
    assert res["max_numerical_diff"] < 1e-3


def test_d4rl_normalization_matches_env_family_token() -> None:
    assert get_d4rl_normalized_score("Ant-v4", 0.0) != 0.0
    # "antmaze" must not inherit Ant's reference bounds.
    assert get_d4rl_normalized_score("antmaze-medium-v2", 0.0) == 0.0
    assert get_d4rl_normalized_score("HalfCheetah-v5", 0.0) != 0.0


def test_infer_native_mamba3_from_state_dict() -> None:
    native = {"mamba_backbone.layers.0.mamba.B_norm.weight": torch.zeros(1)}
    legacy = {"mamba_backbone.layers.0.mamba.x_proj.weight": torch.zeros(1)}
    other = {"fusion.prop_encoder.0.weight": torch.zeros(1)}
    assert infer_native_mamba3(native) is True
    assert infer_native_mamba3(legacy) is False
    assert infer_native_mamba3(other) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires a GPU")
def test_cuda_graph_matches_eager_for_short_windows_and_real_timesteps() -> None:
    """Graph replay must use the true timesteps and never zero-pad short windows."""
    from hdml.deployment.graph_runner import HDMLGraphRunner
    from hdml.models.hdml_model import HDMLModel

    torch.manual_seed(0)
    model = HDMLModel(
        prop_dim=8,
        action_dim=3,
        d_model=32,
        num_mamba_layers=1,
        d_subgoal=8,
        chunk_size=3,
        use_native_mamba3=True,
        use_cfc=True,
        cfc_units=8,
        use_critic=False,
        num_flow_steps=2,
    ).cuda().eval()
    model.deterministic = True
    runner = HDMLGraphRunner(
        model, prop_dim=8, action_dim=3, context_length=6, device="cuda", precapture=True
    )

    rng = np.random.default_rng(11)
    for seq_len in (1, 2, 4, 6):
        states = rng.normal(size=(seq_len, 8)).astype(np.float32)
        actions = rng.normal(size=(seq_len, 3)).astype(np.float32)
        rtgs = rng.normal(size=(seq_len, 1)).astype(np.float32)
        timesteps = (100 + np.arange(seq_len)).astype(np.int64)
        runner.reset()
        with torch.inference_mode():
            eager, eager_hx, _ = model.get_action(
                states=torch.from_numpy(states).unsqueeze(0).cuda(),
                rtgs=torch.from_numpy(rtgs).unsqueeze(0).cuda(),
                actions=torch.from_numpy(actions).unsqueeze(0).cuda(),
                timesteps=torch.from_numpy(timesteps).unsqueeze(0).cuda(),
                hx=None,
            )
        graph = runner(states, rtgs, actions, timesteps)
        np.testing.assert_allclose(graph, eager[0].cpu().numpy(), atol=1e-5, rtol=1e-5)
        assert runner.get_hx() is not None and eager_hx is not None
        np.testing.assert_allclose(
            runner.get_hx(), eager_hx.cpu().numpy(), atol=1e-5, rtol=1e-5
        )

    # Stateful rollout parity: same window repeated while CfC state is carried.
    runner.reset()
    hx_eager: torch.Tensor | None = None
    actions_seen: list[np.ndarray] = []
    states = rng.normal(size=(6, 8)).astype(np.float32)
    actions = rng.normal(size=(6, 3)).astype(np.float32)
    rtgs = rng.normal(size=(6, 1)).astype(np.float32)
    timesteps = (200 + np.arange(6)).astype(np.int64)
    for _ in range(4):
        with torch.inference_mode():
            eager, hx_eager, _ = model.get_action(
                states=torch.from_numpy(states).unsqueeze(0).cuda(),
                rtgs=torch.from_numpy(rtgs).unsqueeze(0).cuda(),
                actions=torch.from_numpy(actions).unsqueeze(0).cuda(),
                timesteps=torch.from_numpy(timesteps).unsqueeze(0).cuda(),
                hx=hx_eager,
            )
        graph = runner(states, rtgs, actions, timesteps)
        np.testing.assert_allclose(graph, eager[0].cpu().numpy(), atol=1e-5, rtol=1e-5)
        assert hx_eager is not None and runner.get_hx() is not None
        np.testing.assert_allclose(
            runner.get_hx(), hx_eager.cpu().numpy(), atol=1e-5, rtol=1e-5
        )
        actions_seen.append(graph)
    assert not np.allclose(actions_seen[0], actions_seen[-1]), (
        "carried CfC state had no effect on the graph rollout"
    )
    runner.reset()
    assert np.allclose(runner.get_hx(), 0.0)

    # Distinct absolute timesteps must change the output and still match eager.
    # Both sides start from the same non-zero CfC state, so this comparison
    # isolates the timestep change instead of mixing in carried-state drift.
    assert runner.cfc_state_size is not None
    hx0 = torch.randn(1, runner.cfc_state_size, device="cuda")
    states = rng.normal(size=(4, 8)).astype(np.float32)
    actions = rng.normal(size=(4, 3)).astype(np.float32)
    rtgs = np.zeros((4, 1), np.float32)
    timesteps_a = np.arange(4, dtype=np.int64)
    timesteps_b = (1000 + np.arange(4)).astype(np.int64)
    graph_a = runner(states, rtgs, actions, timesteps_a, hx=hx0)
    graph_b = runner(states, rtgs, actions, timesteps_b, hx=hx0)
    assert not np.allclose(graph_a, graph_b), "runner ignored the supplied timesteps"

    for timesteps in (timesteps_a, timesteps_b):
        with torch.inference_mode():
            eager, _, _ = model.get_action(
                states=torch.from_numpy(states).unsqueeze(0).cuda(),
                rtgs=torch.from_numpy(rtgs).unsqueeze(0).cuda(),
                actions=torch.from_numpy(actions).unsqueeze(0).cuda(),
                timesteps=torch.from_numpy(timesteps).unsqueeze(0).cuda(),
                hx=hx0,
            )
        graph = runner(states, rtgs, actions, timesteps, hx=hx0)
        np.testing.assert_allclose(graph, eager[0].cpu().numpy(), atol=1e-5, rtol=1e-5)

