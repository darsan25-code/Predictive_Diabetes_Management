"""
Mechanistic Bergman Minimal Model & Parameter Estimation.

Reference: Bergman RN et al. (1981) J Clin Invest 68:1456–1467.

State vector: [G (mg/dL), X (1/min), I (mU/L)]
  G — plasma glucose concentration
  X — remote (interstitial) insulin action
  I — plasma insulin concentration

Inputs:
  u(t)  — total insulin infusion rate (mU/min)
  Ra(t) — rate of glucose appearance from meals (mg/min)

Parameters:
  p1 — insulin-independent glucose effectiveness (1/min)
  p2 — rate of remote insulin action decay (1/min)
  p3 — insulin sensitivity coefficient (L/(mU min^2))
  n  — insulin clearance rate (1/min)
  Gb — basal glucose concentration (mg/dL)
  Ib — basal insulin concentration (mU/L)
  Vg — glucose distribution volume (dL)
  Vi — insulin distribution volume (L)
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence, Optional, Any
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize

logger = logging.getLogger(__name__)

DEFAULT_BERGMAN_PARAMS: dict[str, float] = {
    "p1": 0.028,
    "p2": 0.028,
    "p3": 5.0e-5,
    "n":  0.15,
    "Gb": 100.0,
    "Ib": 10.0,
    "Vg": 117.0,
    "Vi": 12.0,
}

_PARAM_KEYS = ("p1", "p2", "p3", "n")

_PARAM_LOG_BOUNDS = {
    "p1": (np.log(1e-4), np.log(0.5)),
    "p2": (np.log(1e-4), np.log(0.5)),
    "p3": (np.log(1e-8), np.log(1e-2)),
    "n":  (np.log(1e-3), np.log(2.0)),
}


def bergman_rhs(
    t: float,
    state: list[float] | np.ndarray,
    params: dict[str, float],
    u_fn: Callable[[float], float],
    ra_fn: Callable[[float], float],
) -> list[float]:
    """
    Standard 3-state Bergman Minimal Model ODE right-hand side.

    dG/dt = -(p1 + X)*G + p1*Gb + Ra/Vg
    dX/dt = -p2*X + p3*(I - Ib)
    dI/dt = -n*(I - Ib) + (u - u_basal)/Vi
    """
    G = float(np.clip(state[0], 0.0, 1000.0))
    X = float(np.clip(state[1], -0.5, 0.5))
    I = float(np.clip(state[2], 0.0, 500.0))
    p1 = float(params["p1"])
    p2 = float(params["p2"])
    p3 = float(params["p3"])
    n  = float(params["n"])
    Gb = float(params["Gb"])
    Ib = float(params["Ib"])
    Vg = float(params["Vg"])
    Vi = float(params["Vi"])

    Ra = float(ra_fn(t))
    u  = float(u_fn(t))
    u_basal = n * Ib * Vi

    dG = float(np.clip(-(p1 + X) * G + p1 * Gb + Ra / Vg, -500.0, 500.0))
    dX = float(np.clip(-p2 * X + p3 * (I - Ib), -50.0, 50.0))
    dI = float(np.clip(-n * (I - Ib) + (u - u_basal) / Vi, -500.0, 500.0))
    return [dG, dX, dI]


def simulate_bergman(
    t_eval: np.ndarray,
    params: dict[str, float] | None = None,
    u_fn: Optional[Callable[[float], float]] = None,
    ra_fn: Optional[Callable[[float], float]] = None,
    initial_state: Optional[Sequence[float]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the Bergman model ODEs over time array t_eval.

    Returns:
        (G, X, I) tuple of 1D numpy arrays.
    """
    p = dict(DEFAULT_BERGMAN_PARAMS)
    if params is not None:
        p.update(params)

    if u_fn is None:
        u_basal = p["n"] * p["Ib"] * p["Vi"]
        u_fn = lambda t: u_basal
    if ra_fn is None:
        ra_fn = lambda t: 0.0

    if initial_state is None:
        y0 = [p["Gb"], 0.0, p["Ib"]]
    else:
        y0 = list(initial_state)

    t_eval = np.asarray(t_eval, dtype=float)
    t_span = (float(t_eval[0]), float(t_eval[-1]) + 1e-6)

    try:
        sol = solve_ivp(
            bergman_rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            args=(p, u_fn, ra_fn),
            method="RK45",
            rtol=1e-3,
            atol=1e-5,
        )
        if not sol.success or sol.y.shape[1] != len(t_eval):
            # Fallback to LSODA if RK45 encounters stiffness
            sol = solve_ivp(
                bergman_rhs,
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                args=(p, u_fn, ra_fn),
                method="LSODA",
                rtol=1e-3,
                atol=1e-5,
            )
    except Exception:
        return np.full_like(t_eval, y0[0]), np.full_like(t_eval, y0[1]), np.full_like(t_eval, y0[2])

    if not sol.success or sol.y.shape[1] != len(t_eval):
        if hasattr(sol, "y") and sol.y.shape[1] == len(t_eval):
            return np.clip(sol.y[0], 0.0, 1000.0), sol.y[1], sol.y[2]
        return np.full_like(t_eval, y0[0]), np.full_like(t_eval, y0[1]), np.full_like(t_eval, y0[2])

    return np.clip(sol.y[0], 0.0, 1000.0), sol.y[1], sol.y[2]


def fit_parameters(
    time_min: np.ndarray,
    glucose_mgdL: np.ndarray,
    u_fn: Callable[[float], float],
    ra_fn: Optional[Callable[[float], float]] = None,
    initial_params: Optional[dict[str, float]] = None,
    pop_params: Optional[dict[str, float]] = None,
    l2_reg: float = 0.0,
    n_starts: int = 5,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Fit patient-specific Bergman parameters [p1, p2, p3, n] via scipy.optimize.least_squares.

    Features:
      - Log-parameterization: theta = log(p), ensuring p = exp(theta) > 0 strictly.
      - Bounded multi-start optimization to prevent local minima traps.
      - Optional ridge regularization toward pop_params vector.

    Returns:
        (fitted_params, metrics_dict)
    """
    t_arr = np.asarray(time_min, dtype=float)
    g_obs = np.asarray(glucose_mgdL, dtype=float)
    if ra_fn is None:
        ra_fn = lambda t: 0.0

    base_p = dict(DEFAULT_BERGMAN_PARAMS)
    if initial_params is not None:
        base_p.update(initial_params)

    pop_p = dict(base_p)
    if pop_params is not None:
        pop_p.update(pop_params)

    initial_state = [g_obs[0], 0.0, base_p["Ib"]]

    # Evaluate baseline RMSE before optimization
    G_init, _, _ = simulate_bergman(t_arr, base_p, u_fn, ra_fn, initial_state)
    rmse_init = float(np.sqrt(np.mean((G_init - g_obs) ** 2)))

    log_pop = np.array([np.log(pop_p[k]) for k in _PARAM_KEYS])

    def calc_residuals(log_p_vec: np.ndarray) -> np.ndarray:
        p_curr = dict(base_p)
        for idx, k in enumerate(_PARAM_KEYS):
            p_curr[k] = float(np.exp(log_p_vec[idx]))

        try:
            G_sim, _, _ = simulate_bergman(t_arr, p_curr, u_fn, ra_fn, initial_state)
            if G_sim.shape != g_obs.shape or np.any(~np.isfinite(G_sim)):
                data_res = np.full_like(g_obs, 500.0)
            else:
                data_res = np.clip(G_sim - g_obs, -500.0, 500.0)
        except Exception:
            data_res = np.full_like(g_obs, 500.0)

        if l2_reg > 0:
            reg_res = np.sqrt(l2_reg) * (log_p_vec - log_pop)
            return np.concatenate([data_res, reg_res])
        return data_res

    bounds_lo = np.array([_PARAM_LOG_BOUNDS[k][0] for k in _PARAM_KEYS])
    bounds_hi = np.array([_PARAM_LOG_BOUNDS[k][1] for k in _PARAM_KEYS])

    log_init = np.array([np.log(base_p[k]) for k in _PARAM_KEYS])
    log_init = np.clip(log_init, bounds_lo, bounds_hi)

    rng = np.random.default_rng(seed)
    candidates = [log_init]
    for _ in range(n_starts - 1):
        sample = rng.uniform(bounds_lo, bounds_hi)
        candidates.append(sample)

    best_res = None
    best_cost = float("inf")

    for x0 in candidates:
        res = least_squares(
            calc_residuals,
            x0,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
            max_nfev=150,
        )
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    fitted_p = dict(base_p)
    if best_res is not None:
        for idx, k in enumerate(_PARAM_KEYS):
            fitted_p[k] = float(np.exp(best_res.x[idx]))

    G_fit, _, _ = simulate_bergman(t_arr, fitted_p, u_fn, ra_fn, initial_state)
    rmse_fit = float(np.sqrt(np.mean((G_fit - g_obs) ** 2)))

    info = {
        "rmse_initial": rmse_init,
        "rmse_fitted": rmse_fit,
        "rmse_improvement": rmse_init - rmse_fit,
        "least_squares_cost": best_cost,
        "success": best_res.success if best_res is not None else False,
    }
    return fitted_p, info


def fit_population(
    patient_datasets: list[tuple[np.ndarray, np.ndarray, Callable[[float], float], Optional[Callable[[float], float]]]],
    initial_params: Optional[dict[str, float]] = None,
    l2_reg: float = 0.0,
    n_starts: int = 5,
    seed: int = 42,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Fit one shared Bergman parameter set across multiple patients.

    Args:
        patient_datasets: List of tuples (time_min, glucose_mgdL, u_fn, ra_fn).

    Returns:
        (shared_params, info_dict)
    """
    base_p = dict(DEFAULT_BERGMAN_PARAMS)
    if initial_params is not None:
        base_p.update(initial_params)

    log_init = np.array([np.log(base_p[k]) for k in _PARAM_KEYS])
    bounds_lo = np.array([_PARAM_LOG_BOUNDS[k][0] for k in _PARAM_KEYS])
    bounds_hi = np.array([_PARAM_LOG_BOUNDS[k][1] for k in _PARAM_KEYS])

    def calc_combined_residuals(log_p_vec: np.ndarray) -> np.ndarray:
        p_curr = dict(base_p)
        for idx, k in enumerate(_PARAM_KEYS):
            p_curr[k] = float(np.exp(log_p_vec[idx]))

        res_list = []
        for t_arr, g_obs, u_fn, ra_fn in patient_datasets:
            r_fn = ra_fn if ra_fn is not None else (lambda t: 0.0)
            initial_state = [g_obs[0], 0.0, base_p["Ib"]]
            try:
                G_sim, _, _ = simulate_bergman(t_arr, p_curr, u_fn, r_fn, initial_state)
                if G_sim.shape != g_obs.shape or np.any(~np.isfinite(G_sim)):
                    res_list.append(np.full_like(g_obs, 500.0))
                else:
                    res_list.append(np.clip(G_sim - g_obs, -500.0, 500.0))
            except Exception:
                res_list.append(np.full_like(g_obs, 500.0))

        combined = np.concatenate(res_list)
        if l2_reg > 0:
            reg_res = np.sqrt(l2_reg) * (log_p_vec - log_init)
            combined = np.concatenate([combined, reg_res])
        return combined

    rng = np.random.default_rng(seed)
    candidates = [log_init]
    for _ in range(n_starts - 1):
        candidates.append(rng.uniform(bounds_lo, bounds_hi))

    best_res = None
    best_cost = float("inf")

    for x0 in candidates:
        res = least_squares(
            calc_combined_residuals,
            x0,
            bounds=(bounds_lo, bounds_hi),
            method="trf",
            ftol=1e-4,
            xtol=1e-4,
            max_nfev=150,
        )
        if res.cost < best_cost:
            best_cost = res.cost
            best_res = res

    shared_p = dict(base_p)
    if best_res is not None:
        for idx, k in enumerate(_PARAM_KEYS):
            shared_p[k] = float(np.exp(best_res.x[idx]))

    info = {
        "least_squares_cost": best_cost,
        "success": best_res.success if best_res is not None else False,
        "num_patients": len(patient_datasets),
    }
    return shared_p, info


class BergmanModel:
    """
    Object-oriented wrapper around Bergman Minimal Model ODEs & fitting.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        p_dict = cfg.get("parameters", {})
        self.params = dict(DEFAULT_BERGMAN_PARAMS)
        self.params.update(p_dict)

        unc = cfg.get("uncertainty", {})
        self.n_bootstrap = int(unc.get("n_bootstrap", 10))
        self.ci_level    = float(unc.get("ci_level", 0.95))
        self._param_cis: Optional[dict] = None

    @property
    def p1(self) -> float:
        return float(self.params["p1"])
    @p1.setter
    def p1(self, val: float) -> None:
        self.params["p1"] = float(val)

    @property
    def p2(self) -> float:
        return float(self.params["p2"])
    @p2.setter
    def p2(self, val: float) -> None:
        self.params["p2"] = float(val)

    @property
    def p3(self) -> float:
        return float(self.params["p3"])
    @p3.setter
    def p3(self, val: float) -> None:
        self.params["p3"] = float(val)

    @property
    def n(self) -> float:
        return float(self.params["n"])
    @n.setter
    def n(self, val: float) -> None:
        self.params["n"] = float(val)

    @property
    def Gb(self) -> float:
        return float(self.params["Gb"])
    @Gb.setter
    def Gb(self, val: float) -> None:
        self.params["Gb"] = float(val)

    @property
    def Ib(self) -> float:
        return float(self.params["Ib"])
    @Ib.setter
    def Ib(self, val: float) -> None:
        self.params["Ib"] = float(val)

    @property
    def Vg(self) -> float:
        return float(self.params["Vg"])
    @Vg.setter
    def Vg(self, val: float) -> None:
        self.params["Vg"] = float(val)

    @property
    def Vi(self) -> float:
        return float(self.params["Vi"])
    @Vi.setter
    def Vi(self, val: float) -> None:
        self.params["Vi"] = float(val)

    def simulate(
        self,
        t_span: Any,
        initial_state: Any = None,
        u_fn: Any = None,
        ra_fn: Any = None,
        t_eval: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple:
        """
        Simulate Bergman dynamics. Supports both time array and t_span inputs.
        """
        if isinstance(t_span, (np.ndarray, list, tuple)) and len(t_span) > 2:
            t_array = np.asarray(t_span, dtype=float)
            basal_arr = np.asarray(initial_state, dtype=float) if initial_state is not None else np.full_like(t_array, 15.0)
            bolus_arr = np.asarray(u_fn, dtype=float) if u_fn is not None else np.zeros_like(t_array)
            meal_arr  = np.asarray(ra_fn, dtype=float) if ra_fn is not None else np.zeros_like(t_array)

            u_interp  = lambda t: float(np.interp(t, t_array, basal_arr + (bolus_arr / 5.0)))
            ra_interp = lambda t: float(np.interp(t, t_array, meal_arr * 10.0))

            G, X, I = simulate_bergman(t_array, self.params, u_interp, ra_interp, [self.Gb, 0.0, self.Ib])
            return G, X, I

        t_eval_arr = t_eval if t_eval is not None else np.arange(float(t_span[0]), float(t_span[1]) + 1e-6, 1.0)
        G, X, I = simulate_bergman(t_eval_arr, self.params, u_fn, ra_fn, initial_state)
        return t_eval_arr, G, X, I

    def steady_state_basal_insulin(self) -> float:
        return self.n * self.Ib * self.Vi

    def predict(
        self,
        initial_state: Sequence,
        horizon_min: float,
        u_fn: Callable,
        ra_fn: Optional[Callable] = None,
        dt_min: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Forecast glucose for `horizon_min` minutes at `dt_min` intervals.

        Returns:
            (t_forecast_min, G_forecast_mgdL)
        """
        t_eval = np.arange(0.0, horizon_min + 1e-9, dt_min)
        G, _, _ = simulate_bergman(t_eval, self.params, u_fn, ra_fn, list(initial_state))
        return t_eval, G

    def fit(
        self,
        time_min: np.ndarray,
        glucose_mgdL: np.ndarray,
        u_fn: Callable,
        ra_fn: Optional[Callable] = None,
        verbose: bool = False,
    ) -> None:
        fitted, info = fit_parameters(time_min, glucose_mgdL, u_fn, ra_fn, initial_params=self.params)
        self.params.update(fitted)
        if verbose:
            logger.info("Bergman fit complete: RMSE %.3f -> %.3f", info["rmse_initial"], info["rmse_fitted"])

    def fit_with_ci(
        self,
        time_min: np.ndarray,
        glucose_mgdL: np.ndarray,
        basal_mU_per_min: Any = None,
        bolus_mU: Any = None,
        meal_CHO_g: Any = None,
        n_bootstrap: int = 10,
        **kwargs,
    ) -> dict:
        t_arr = np.asarray(time_min, dtype=float)
        g_arr = np.asarray(glucose_mgdL, dtype=float)

        if callable(basal_mU_per_min):
            u_fn = basal_mU_per_min
            ra_fn = kwargs.get("ra_fn", lambda t: 0.0)
        else:
            b_arr = np.asarray(basal_mU_per_min, dtype=float) if basal_mU_per_min is not None else np.full_like(t_arr, 15.0)
            bo_arr = np.asarray(bolus_mU, dtype=float) if bolus_mU is not None else np.zeros_like(t_arr)
            m_arr = np.asarray(meal_CHO_g, dtype=float) if meal_CHO_g is not None else np.zeros_like(t_arr)
            u_fn = lambda t: float(np.interp(t, t_arr, b_arr + (bo_arr / 5.0)))
            ra_fn = lambda t: float(np.interp(t, t_arr, m_arr * 10.0))

        fitted, info = fit_parameters(t_arr, g_arr, u_fn, ra_fn, initial_params=self.params)
        self.params.update(fitted)

        # Bootstrap CIs
        self._compute_bootstrap_cis(t_arr, g_arr, u_fn, ra_fn, n_bootstrap)

        return {
            "fitted_params": dict(self.params),
            "confidence_intervals_95": {k: (v[0], v[2]) for k, v in self._param_cis.items()},
        }

    def _compute_bootstrap_cis(
        self, t_arr: np.ndarray, g_arr: np.ndarray, u_fn: Callable, ra_fn: Callable, n_boot: int
    ) -> None:
        initial_state = [g_arr[0], 0.0, self.Ib]
        G_fit, _, _ = simulate_bergman(t_arr, self.params, u_fn, ra_fn, initial_state)
        residuals = g_arr - G_fit
        rng = np.random.default_rng(0)

        boot_params = []
        for _ in range(n_boot):
            g_boot = G_fit + rng.choice(residuals, size=len(residuals), replace=True)
            p_boot, _ = fit_parameters(t_arr, g_boot, u_fn, ra_fn, initial_params=self.params, n_starts=1)
            boot_params.append([p_boot[k] for k in _PARAM_KEYS])

        arr = np.array(boot_params)
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)

        self._param_cis = {
            k: (float(lo[i]), float(self.params[k]), float(hi[i]))
            for i, k in enumerate(_PARAM_KEYS)
        }

    @classmethod
    def from_defaults(cls) -> BergmanModel:
        return cls()
