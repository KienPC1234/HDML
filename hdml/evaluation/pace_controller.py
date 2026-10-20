from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class PACEController:
    """Phase-Aware Chunk Execution (PACE) plan monitor.

    Operating mode in this repository
    ----------------------------------
    The evaluator uses the *subgoal-hold* micro tier: a macro plan freezes a
    subgoal, and every micro step regenerates a fresh action with
    :meth:`HDMLModel.act_from_subgoal`. The action chunk returned by the flow
    policy is therefore **not** executed action-by-action; only its first
    action is used. ``chunk_size`` (how many actions the flow head emits) and
    the macro plan lifetime (``macro_interval`` control steps) are separate
    quantities and may differ.

    PACE watches the forward-dynamics prediction produced by the macro tier.
    That head is trained to predict ``s_{t+k}`` with ``k = chunk_size`` (see
    ``hdml/data/dataset.py``), so its output is scheduled for control step
    ``t + k`` and is never compared against ``s_{t+1}``. A plan can expire
    before that prediction is due; pending predictions are kept until their
    own due step so a short macro interval does not silently disable PACE.

    Every trigger records an absolute control step and a reason
    (``"deviation"`` or ``"expired"``). Normal plan expiry is therefore
    distinguishable from an early replan caused by model error.
    """

    def __init__(self, threshold: float = 0.5, norm_p: int = 2) -> None:
        if threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if norm_p < 1:
            raise ValueError("norm_p must be >= 1")
        self.threshold = float(threshold)
        self.norm_p = int(norm_p)

        # Active plan (kept for backwards-compatible get_next_action callers).
        self.active_chunk: torch.Tensor | None = None
        self.predicted_states: torch.Tensor | None = None
        self.prediction_due_steps: list[int] | None = None
        self.horizon: int = 0
        self.plan_start_step: int = 0
        self.step_idx: int = 0

        # Predictions can outlive their plan when chunk_size > macro_interval.
        self._pending_predictions: list[tuple[int, torch.Tensor]] = []

        # Instrumentation.
        self.trigger_events: list[dict[str, Any]] = []
        self.deviation_triggers: int = 0
        self.expiry_triggers: int = 0
        self.last_trigger_reason: str | None = None
        self.last_deviation: float | None = None

    # ------------------------------------------------------------------
    # Plan lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the current plan, pending predictions, and episode counters."""
        self._clear_plan(clear_pending=True)
        self.trigger_events = []
        self.deviation_triggers = 0
        self.expiry_triggers = 0
        self.last_trigger_reason = None
        self.last_deviation = None

    def _clear_plan(self, *, clear_pending: bool) -> None:
        self.active_chunk = None
        self.predicted_states = None
        self.prediction_due_steps = None
        self.horizon = 0
        self.plan_start_step = 0
        self.step_idx = 0
        if clear_pending:
            self._pending_predictions = []

    def set_new_plan(
        self,
        action_chunk: torch.Tensor | None = None,
        predicted_states: torch.Tensor | None = None,
        *,
        prediction_due_steps: list[int] | None = None,
        horizon: int | None = None,
        current_step: int = 0,
    ) -> None:
        """Register a fresh macro plan.

        Args:
            action_chunk: ``(chunk_size, action_dim)`` reference chunk. It is
                kept for diagnostics/backwards compatibility; the evaluator's
                subgoal-hold mode does not execute it directly.
            predicted_states: ``(K, state_dim)`` forward-dynamics predictions.
                Each row is paired with one absolute control step.
            prediction_due_steps: absolute control steps at which each row of
                ``predicted_states`` becomes comparable to the real state. When
                omitted, legacy behaviour is used and rows are treated as
                one-step-ahead predictions (``current_step + 1, ...``).
            horizon: number of control steps the macro plan is valid. Defaults
                to ``len(action_chunk)`` when a chunk is provided, else 1.
            current_step: absolute control step at which the plan starts.
        """
        if action_chunk is not None and action_chunk.ndim != 2:
            raise ValueError(
                f"action_chunk must be (chunk_size, action_dim), got {tuple(action_chunk.shape)}"
            )
        if horizon is None:
            horizon = int(action_chunk.shape[0]) if action_chunk is not None else 1
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        if predicted_states is not None:
            predicted_states = torch.atleast_2d(predicted_states)
            if prediction_due_steps is None:
                prediction_due_steps = [
                    int(current_step) + i + 1 for i in range(predicted_states.shape[0])
                ]
            if len(prediction_due_steps) != predicted_states.shape[0]:
                raise ValueError(
                    "prediction_due_steps must have one entry per predicted state "
                    f"({len(prediction_due_steps)} != {predicted_states.shape[0]})"
                )
            self._pending_predictions.extend(
                (int(due), predicted_states[i].detach())
                for i, due in enumerate(prediction_due_steps)
            )

        self.active_chunk = action_chunk
        self.predicted_states = predicted_states
        self.prediction_due_steps = (
            [int(v) for v in prediction_due_steps] if prediction_due_steps is not None else None
        )
        self.horizon = int(horizon)
        self.plan_start_step = int(current_step)
        self.step_idx = 0

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------
    def should_replan(
        self,
        current_state: torch.Tensor,
        current_step: int | None = None,
    ) -> tuple[bool, str, float | None]:
        """Decide whether the current macro plan must be refreshed.

        Args:
            current_state: observed (normalized) state, shape ``(state_dim,)``.
            current_step: absolute control step. Defaults to the next step of
                the active plan.

        Returns:
            ``(needs_replanning, reason, deviation)``. ``reason`` is one of
            ``"none"``, ``"deviation"``, ``"expired"``, or ``"no_plan"``.
        """
        if current_step is None:
            current_step = self.plan_start_step + self.step_idx
        current_step = int(current_step)

        if self.active_chunk is None and not self._pending_predictions:
            return True, "no_plan", None

        state = current_state.detach().reshape(-1)

        # Check every prediction whose due step has arrived. Late checks (for
        # example after a skipped call) are still evaluated and then removed.
        due = [item for item in self._pending_predictions if item[0] <= current_step]
        self._pending_predictions = [
            item for item in self._pending_predictions if item[0] > current_step
        ]
        max_deviation: float | None = None
        for due_step, expected_state in due:
            if due_step != current_step:
                logger.warning(
                    "PACE skipped due prediction at step %d while observing step %d",
                    due_step,
                    current_step,
                )
            expected = expected_state.reshape(-1).to(device=state.device, dtype=state.dtype)
            if expected.shape != state.shape:
                raise ValueError(
                    "predicted state shape does not match current state: "
                    f"{tuple(expected.shape)} != {tuple(state.shape)}"
                )
            deviation = float(torch.norm(state - expected, p=self.norm_p).item())
            max_deviation = deviation if max_deviation is None else max(max_deviation, deviation)

        if max_deviation is not None and max_deviation > self.threshold:
            self.last_deviation = max_deviation
            self._record_trigger("deviation", current_step, max_deviation)
            self._clear_plan(clear_pending=True)
            return True, "deviation", max_deviation

        if (
            self.active_chunk is not None
            and current_step - self.plan_start_step >= self.horizon
        ):
            self._record_trigger("expired", current_step, max_deviation)
            self._clear_plan(clear_pending=False)
            return True, "expired", max_deviation

        return False, "none", max_deviation

    def get_next_action(
        self,
        current_state: torch.Tensor,
        current_step: int | None = None,
    ) -> tuple[torch.Tensor | None, bool]:
        """Legacy chunk-execution API.

        Args:
            current_state: observed state, shape ``(state_dim,)``.
            current_step: absolute control step; defaults to the plan cursor.

        Returns:
            ``(action, needs_replanning)``. ``action`` is ``None`` when the
            plan has expired or deviated. The evaluator uses
            :meth:`should_replan` plus the subgoal-hold micro tier instead, so
            this method is kept for compatibility and for callers that really
            execute a stored chunk.
        """
        needs_replanning, _reason, _deviation = self.should_replan(
            current_state, current_step=current_step
        )
        if needs_replanning:
            return None, True
        assert self.active_chunk is not None
        action = self.active_chunk[self.step_idx]
        self.step_idx += 1
        return action, self.step_idx >= self.horizon

    def statistics(self) -> dict[str, Any]:
        """Return episode-level trigger counts and the absolute-step event log."""
        return {
            "threshold": self.threshold,
            "deviation_triggers": self.deviation_triggers,
            "expiry_triggers": self.expiry_triggers,
            "total_triggers": self.deviation_triggers + self.expiry_triggers,
            "events": [dict(event) for event in self.trigger_events],
        }

    def _record_trigger(self, reason: str, step: int, deviation: float | None) -> None:
        if reason == "deviation":
            self.deviation_triggers += 1
        else:
            self.expiry_triggers += 1
        self.last_trigger_reason = reason
        event = {
            "step": int(step),
            "reason": reason,
            "deviation": None if deviation is None else float(deviation),
            "threshold": self.threshold,
            "plan_start_step": int(self.plan_start_step),
            "horizon": int(self.horizon),
        }
        self.trigger_events.append(event)
        logger.info(
            "PACE trigger step=%d reason=%s deviation=%s threshold=%.4f "
            "plan_start=%d horizon=%d",
            step,
            reason,
            "n/a" if deviation is None else f"{deviation:.4f}",
            self.threshold,
            self.plan_start_step,
            self.horizon,
        )
