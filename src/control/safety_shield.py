"""
Phase 8: Hard Safety Shield for Glucose-Insulin Controllers.

Safety is a mandatory invariant, not a reward optimization objective.
The safety shield executes unconditionally before EVERY candidate action is applied to the patient simulation.

Mechanistic forward simulation over a configurable horizon (default: 60 min).
If predicted minimum glucose < hypo_threshold_mgdL (default: 80 mg/dL buffer, hard floor 70 mg/dL):
  - Clamps the candidate action to the maximum admissible safe rate, OR
  - Vetoes the action and enforces insulin suspension (u = 0) if no positive dose is safe.

Maintains explicit counters for:
  - total candidate actions evaluated
  - shield activations
  - safety clamps
  - safety vetoes
  - invalid actions rejected
  - prediction failures
  - episodes with intervention
  - hypoglycemic observations (< 70 mg/dL)
  - simulated hypoglycemic episodes
  - safety invariant violations
"""
from __future__ import annotations

import logging
from typing import Any, Sequence, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class SafetyShield:
    """
    Hard mechanistic forward-simulation safety shield.
    """

    def __init__(
        self,
        bergman_model=None,
        hypo_threshold_mgdL: float = 80.0,
        hard_floor_mgdL: float = 70.0,
        horizon_minutes: float = 60.0,
        dt_min: float = 1.0,
        max_iob_units: float = 10.0,
        **kwargs,
    ) -> None:
        if bergman_model is None:
            from src.models.bergman import BergmanModel
            bergman_model = BergmanModel()
        thresh = kwargs.get("hypo_threshold_mgdl", hypo_threshold_mgdL)
        self.bergman = bergman_model
        self.hypo_threshold = float(thresh)
        self.hard_floor = float(hard_floor_mgdL)
        self.horizon = float(horizon_minutes)
        self.dt = float(dt_min)
        self.max_iob = float(max_iob_units)

        # Instrumentation counters
        self.counters = {
            "total_evaluated": 0,
            "activations": 0,
            "clamps": 0,
            "vetoes": 0,
            "invalid_rejected": 0,
            "prediction_failures": 0,
            "episodes_with_intervention": 0,
            "hypo_observations": 0,
            "hypo_episodes": 0,
            "invariant_violations": 0,
        }
        self._current_episode_has_intervention = False

    def reset_episode(self) -> None:
        """Call at the beginning of each evaluation/training episode."""
        if self._current_episode_has_intervention:
            self.counters["episodes_with_intervention"] += 1
        self._current_episode_has_intervention = False

    def evaluate_action(
        self,
        current_g: float,
        current_x: float,
        current_i: float,
        proposed_u_mU_per_min: float,
        iob_units: float = 0.0,
    ) -> Tuple[float, bool]:
        """
        Evaluate and sanitize candidate action.

        Returns:
            (executed_action, was_intervened)
        """
        state = [float(current_g), float(current_x), float(current_i)]
        return self.__call__(proposed_u_mU_per_min, state, iob_units=iob_units)

    def __call__(
        self,
        u_proposed: float,
        current_state: Sequence[float],
        iob_units: float = 0.0,
    ) -> Tuple[float, bool]:
        """
        Main safety shield interception entrypoint.
        """
        self.counters["total_evaluated"] += 1

        # 1. Check validity of proposed action
        if not np.isfinite(u_proposed) or u_proposed < 0.0:
            self.counters["invalid_rejected"] += 1
            self.counters["activations"] += 1
            self.counters["vetoes"] += 1
            self._current_episode_has_intervention = True
            logger.warning("[SafetyShield] Invalid proposed action (%s). Vetoed to 0.0 mU/min.", u_proposed)
            return 0.0, True

        u_prop = float(u_proposed)
        state = [float(v) for v in current_state]
        curr_g = state[0]

        # Record observation tracking
        if curr_g < self.hard_floor:
            self.counters["hypo_observations"] += 1

        # 2. Check IOB constraint limit
        if iob_units > self.max_iob and u_prop > 0.0:
            self.counters["activations"] += 1
            self.counters["vetoes"] += 1
            self._current_episode_has_intervention = True
            logger.warning("[SafetyShield] IOB limit exceeded (%.2f U > %.2f U). Vetoed to 0.", iob_units, self.max_iob)
            return 0.0, True

        # 3. Forward simulation over safety horizon
        try:
            min_g_proposed = self.predict_min_glucose(u_prop, state)
        except Exception as e:
            self.counters["prediction_failures"] += 1
            self.counters["activations"] += 1
            self.counters["vetoes"] += 1
            self._current_episode_has_intervention = True
            logger.warning("[SafetyShield] Prediction failure: %s. Safe fallback u=0.", e)
            return 0.0, True

        # 4. If proposed action is predicted safe, allow unchanged
        if min_g_proposed >= self.hypo_threshold:
            return u_prop, False

        # 5. Otherwise, the proposed action threatens hypoglycemia — intervene!
        self.counters["activations"] += 1
        self._current_episode_has_intervention = True

        # Attempt to find maximal safe clamped action via bisection
        u_clamped = self._find_max_safe_action(state, u_prop)

        if u_clamped > 0.0:
            self.counters["clamps"] += 1
            logger.info(
                "[SafetyShield] CLAMP: Proposed %.2f mU/min clamped to safe rate %.2f mU/min (min predicted G=%.1f mg/dL).",
                u_prop, u_clamped, self.predict_min_glucose(u_clamped, state)
            )
            return u_clamped, True
        else:
            self.counters["vetoes"] += 1
            logger.warning(
                "[SafetyShield] VETO: Proposed %.2f mU/min vetoed to 0.0 mU/min (predicts min G=%.1f < %.1f mg/dL).",
                u_prop, min_g_proposed, self.hypo_threshold
            )
            return 0.0, True

    def _find_max_safe_action(self, initial_state: list[float], u_max: float) -> float:
        """Find the largest u in [0, u_max] that keeps min predicted G >= hypo_threshold."""
        # Check if u=0 is safe
        if self.predict_min_glucose(0.0, initial_state) < self.hypo_threshold:
            return 0.0

        low = 0.0
        high = u_max
        best_safe = 0.0

        for _ in range(8):  # 8 bisection steps -> precision < 0.5%
            mid = 0.5 * (low + high)
            if self.predict_min_glucose(mid, initial_state) >= self.hypo_threshold:
                best_safe = mid
                low = mid
            else:
                high = mid

        return float(best_safe)

    def is_safe(self, u: float, current_state: Sequence[float]) -> bool:
        """Return True if action is predicted safe without intervention."""
        u_safe, intervened = self.__call__(u, current_state)
        return not intervened

    def predict_min_glucose(self, u: float, current_state: Sequence[float]) -> float:
        """Return the minimum predicted glucose over the safety horizon."""
        t_eval = np.arange(0.0, self.horizon + 1e-9, self.dt)
        u_fn = lambda t: float(u)
        ra_fn = lambda t: 0.0  # Conservative estimate: no unexpected meal rescue assumed
        _, G_sim, _, _ = self.bergman.simulate(
            (0.0, self.horizon),
            list(current_state),
            u_fn=u_fn,
            ra_fn=ra_fn,
            t_eval=t_eval,
        )
        return float(np.min(G_sim))

    @property
    def veto_count(self) -> int:
        return self.counters["vetoes"]

    @property
    def clamp_count(self) -> int:
        return self.counters["clamps"]

    @property
    def activation_count(self) -> int:
        return self.counters["activations"]
