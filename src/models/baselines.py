"""
Baseline models for glucose forecasting (AGENTS.md Rule 2).

Rule 2: ALWAYS implement and report:
  1. PersistenceBaseline  — glucose stays where it is (trivial oracle)
  2. MechanisticOnlyBaseline — Bergman open-loop (no EKF, no Neural ODE)
before evaluating any ML model.

All baselines expose a .predict(initial_state, horizon_min, ...) interface
compatible with BergmanModel and DigitalTwinModel.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PersistenceBaseline:
    """
    Naive persistence forecast: glucose at t+h = glucose at t (last known value).

    This is the zero-hypothesis baseline. Any model that does not beat it on
    30-min horizon is not worth clinical consideration.
    """

    def predict(
        self,
        initial_state: tuple | list | float | np.ndarray,
        horizon_min: float = 30.0,
        u_fn: Optional[Callable] = None,
        ra_fn: Optional[Callable] = None,
        dt_min: float = 5.0,
        horizon_steps: Optional[int] = None,
        *args,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if isinstance(initial_state, (list, tuple, np.ndarray)):
            G0 = float(initial_state[-1] if len(initial_state) > 0 else 120.0)
        else:
            G0 = float(initial_state)

        if horizon_steps is not None:
            n_steps = horizon_steps
            G = np.full(n_steps, fill_value=G0)
            return G

        t = np.arange(0.0, horizon_min + 1e-9, dt_min)
        G = np.full_like(t, fill_value=G0)
        return t, G


class MechanisticOnlyBaseline:
    """
    Open-loop Bergman simulation: no state estimation, no learned residual.

    Uses the current (possibly pre-fitted) BergmanModel parameters to forecast
    glucose given a piecewise-constant insulin input and optional meal appearance.

    This establishes how much mechanistic physics alone contributes before
    adding EKF state correction or Neural ODE residuals.
    """

    def __init__(self, bergman_model=None, params=None) -> None:
        if bergman_model is None:
            from src.models.bergman import BergmanModel
            bergman_model = BergmanModel()
        self.bergman = bergman_model

    def predict(
        self,
        initial_state: tuple | list | float | np.ndarray,
        basal_mU_per_min: Optional[Sequence] = None,
        bolus_mU: Optional[Sequence] = None,
        horizon_min: float = 30.0,
        horizon_steps: Optional[int] = None,
        u_fn: Optional[Callable] = None,
        ra_fn: Optional[Callable] = None,
        dt_min: float = 5.0,
        *args,
        **kwargs,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if horizon_steps is not None:
            horizon_min = horizon_steps * dt_min

        state = initial_state
        if isinstance(initial_state, (list, tuple, np.ndarray)):
            if len(initial_state) == 1 or isinstance(initial_state[0], (int, float, np.floating)):
                G0 = float(initial_state[-1] if hasattr(initial_state, '__len__') else initial_state)
                state = [G0, 0.0, self.bergman.Ib]

        if u_fn is None:
            basal = self.bergman.steady_state_basal_insulin()
            u_fn = lambda t: basal

        t, G = self.bergman.predict(
            initial_state=state,
            horizon_min=horizon_min,
            u_fn=u_fn,
            ra_fn=ra_fn,
            dt_min=dt_min,
        )

        if horizon_steps is not None:
            return G[:horizon_steps]

        return t, G
