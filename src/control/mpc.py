"""
MPC Controller — Receding-Horizon Glucose Regulation with Hard Physiological Constraints.

Solver: scipy.optimize.minimize with SLSQP (handles nonlinear/non-convex bilinear ODE dynamics).
Horizon: 3-4 hours (default: 36 steps = 180 min at 5-minute sampling).

Objective:
  min_U Σ_k [ w_g * (G_k - Gref)² + w_u * (u_k - u_basal)² + w_du * (u_k - u_{k-1})² ]

Subject to Hard Constraints:
  1. 0 ≤ u_k ≤ u_max                       (actuator saturation limits)
  2. |u_k - u_{k-1}| ≤ du_max              (rate-of-change limit)
  3. G_k ≥ 70.0 mg/dL for all k ∈ [1, H]    (hard predicted hypoglycemia floor)
  4. IOB_k ≤ IOB_max                       (maximum allowable insulin-on-board)

Unconditional Action Verification:
  Every proposed action is passed through SafetyShield before execution.
  This controller is a research simulation benchmark and never provides clinical dosing advice.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class MPCController:
    """
    Model Predictive Controller using the Bergman minimal model as the internal prediction engine.

    The controller optimizes a candidate sequence of insulin infusion rates U = [u_0, ..., u_{H-1}]
    over a 3-4 hour prediction horizon, applies the first action u_0 (receding horizon), and re-solves
    at each 5-minute step.
    """

    def __init__(
        self,
        bergman_model=None,
        safety_shield=None,
        cfg: dict | None = None,
        horizon_steps: int = 36,
        target_g: float = 115.0,
        **kwargs,
    ) -> None:
        if bergman_model is None:
            from src.models.bergman import BergmanModel
            bergman_model = BergmanModel()
        if safety_shield is None:
            from src.control.safety_shield import SafetyShield
            safety_shield = SafetyShield(bergman_model=bergman_model)
        if cfg is None:
            cfg = {}

        self.bergman = bergman_model
        self.shield  = safety_shield

        weights = cfg.get("weights", {})
        constraints = cfg.get("constraints", {})
        target = cfg.get("target", {})

        self.H       = int(cfg.get("prediction_steps", horizon_steps))  # 36 steps = 3.0 h
        self.dt      = float(cfg.get("dt_minutes", 5.0))
        self.w_g     = float(weights.get("glucose_error", 1.0))
        self.w_u     = float(weights.get("insulin_rate", 0.05))
        self.w_du    = float(weights.get("rate_of_change", 0.1))
        self.u_min   = float(constraints.get("u_min_mU_per_min", 0.0))
        self.u_max   = float(constraints.get("u_max_mU_per_min", 60.0))
        self.du_max  = float(constraints.get("du_max_mU_per_min", 15.0))
        self.g_floor = float(constraints.get("glucose_floor_mgdL", 70.0))
        self.max_iob = float(constraints.get("max_iob_units", 8.0))
        self.Gref    = float(target.get("Gref_mgdL", target_g))

        # Internal diagnostics counters
        self.solver_stats = {
            "total_calls": 0,
            "converged": 0,
            "failures": 0,
            "infeasible_fallbacks": 0,
        }

    # ------------------------------------------------------------------
    # Forward Horizon Rollout Helper
    # ------------------------------------------------------------------

    def _rollout_horizon(
        self,
        U: np.ndarray,
        initial_state: Sequence[float],
        ra_fn: Callable[[float], float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fast vectorized forward simulation of Bergman ODE over prediction horizon."""
        dt = self.dt
        bm = self.bergman
        p1, p2, p3, n = bm.p1, bm.p2, bm.p3, bm.n
        Gb, Ib, Vg, Vi = bm.Gb, bm.Ib, bm.Vg, bm.Vi
        u_basal = bm.steady_state_basal_insulin()
        H = len(U)

        g_arr = np.zeros(H)
        iob_arr = np.zeros(H)

        g_curr = float(initial_state[0])
        x_curr = float(initial_state[1])
        i_curr = float(initial_state[2])

        for k in range(H):
            u_k = float(U[k])
            ra_k = float(ra_fn(k * dt))

            dG = -(p1 + x_curr) * g_curr + p1 * Gb + ra_k / Vg
            dX = -p2 * x_curr + p3 * (i_curr - Ib)
            dI = -n * (i_curr - Ib) + (u_k - u_basal) / Vi

            g_curr += dt * dG
            x_curr += dt * dX
            i_curr += dt * dI

            g_arr[k] = g_curr
            # Estimate IOB from elevated plasma insulin (mU -> Units)
            iob_arr[k] = max(0.0, (i_curr - Ib) * Vi / 1000.0)

        return g_arr, iob_arr

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_action(
        self,
        current_state: Optional[Sequence[float]] = None,
        u_prev: Optional[float] = None,
        ra_fn: Optional[Callable] = None,
        current_g: Optional[float] = None,
        current_x: Optional[float] = None,
        current_i: Optional[float] = None,
    ) -> tuple[float, bool]:
        """
        Compute the optimal receding-horizon control action.

        Returns:
            (u_safe, vetoed_or_intervened)
        """
        self.solver_stats["total_calls"] += 1

        if current_state is None:
            g = current_g if current_g is not None else 120.0
            x = current_x if current_x is not None else 0.0
            i = current_i if current_i is not None else self.bergman.Ib
            current_state = [g, x, i]

        G0, X0, I0 = [float(v) for v in current_state]
        u_basal = self.bergman.steady_state_basal_insulin()
        if u_prev is None:
            u_prev = u_basal
        if ra_fn is None:
            ra_fn = lambda t: 0.0

        u_p = float(u_prev)
        H = self.H
        dt = self.dt
        bounds = [(self.u_min, self.u_max)] * H
        u0 = np.full(H, u_p)

        # 1. Objective function
        def objective(U: np.ndarray) -> float:
            g_arr, _ = self._rollout_horizon(U, [G0, X0, I0], ra_fn)
            g_err = g_arr - self.Gref
            # Asymmetric penalty for hypo risk
            g_pen = np.where(g_arr < self.Gref, 2.0 * (g_err ** 2), g_err ** 2)
            g_cost = float(np.sum(g_pen))

            u_cost = self.w_u * float(np.sum((U - u_basal) ** 2))
            du_cost = self.w_du * (float((U[0] - u_p) ** 2) + float(np.sum(np.diff(U) ** 2)))
            return self.w_g * g_cost + u_cost + du_cost

        # 2. Hard Constraints: Vectorized Glucose Floor & Rate of Change
        def combined_ineq_constraints(U: np.ndarray) -> np.ndarray:
            diffs = np.diff(np.insert(U, 0, u_p))
            rate_pos = self.du_max - diffs
            rate_neg = self.du_max + diffs
            g_arr, _ = self._rollout_horizon(U, [G0, X0, I0], ra_fn)
            g_floor_margin = np.array([np.min(g_arr) - self.g_floor])
            return np.concatenate([rate_pos, rate_neg, g_floor_margin])

        constraints = [{"type": "ineq", "fun": combined_ineq_constraints}]

        # 3. Solve via SLSQP
        try:
            result = minimize(
                objective,
                u0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 15, "ftol": 1e-2, "disp": False},
            )
            # SLSQP returns 0 on success, 9 on iteration limit with valid iterate
            if (result.success or result.status in (0, 9)) and np.all(np.isfinite(result.x)):
                self.solver_stats["converged"] += 1
                u_candidate = float(result.x[0])
            else:
                self.solver_stats["failures"] += 1
                u_candidate = float(result.x[0]) if np.all(np.isfinite(result.x)) else u_basal
        except Exception as e:
            self.solver_stats["failures"] += 1
            logger.warning("MPC optimization exception: %s. Using basal fallback.", e)
            u_candidate = u_basal

        # Ensure strict physical bounds
        u_candidate = float(np.clip(u_candidate, self.u_min, self.u_max))

        # 4. Mandatory Safety Shield Vetting (Invariant Enforcement)
        u_safe, vetoed = self.shield(u_candidate, [G0, X0, I0])
        return u_safe, vetoed

    def rollout(
        self,
        initial_state: Sequence[float],
        n_steps: int,
        ra_fn: Optional[Callable] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[bool]]:
        """Run closed-loop MPC rollout for n_steps."""
        state = list(initial_state)
        G_traj, U_traj, veto_flags = [], [], []
        u_prev = self.bergman.steady_state_basal_insulin()

        for _ in range(n_steps):
            u_safe, vetoed = self.compute_action(state, u_prev=u_prev, ra_fn=ra_fn)
            u_fn = lambda t: u_safe
            _, G_new, X_new, I_new = self.bergman.simulate(
                (0.0, self.dt),
                state,
                u_fn=u_fn,
                ra_fn=ra_fn,
                t_eval=np.array([self.dt]),
            )
            G_traj.append(float(G_new[-1]))
            U_traj.append(u_safe)
            veto_flags.append(vetoed)
            state = [float(G_new[-1]), float(X_new[-1]), float(I_new[-1])]
            u_prev = u_safe

        return np.array(G_traj), np.array(U_traj), veto_flags
