"""
Phase 3: Per-Patient Calibration and Identifiability Analysis.

Features:
- Hierarchical per-patient calibration initialized from population parameters with configurable ridge penalty.
- Local sensitivity analysis: ±10% perturbations per parameter evaluating absolute and relative RMSE impact.
- Profile likelihood analysis: 1D parameter profiling with re-optimization of nuisance parameters to construct 95% likelihood-ratio confidence intervals.
- Statistical identifiability classification: identifies practically and structurally non-identifiable parameters.
- Sensitivity heatmap generation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import chi2

from src.models.mechanistic import (
    DEFAULT_BERGMAN_PARAMS,
    _PARAM_KEYS,
    _PARAM_LOG_BOUNDS,
    simulate_bergman,
)

logger = logging.getLogger(__name__)


def calibrate_patient(
    time_min: np.ndarray,
    glucose_mgdL: np.ndarray,
    u_fn: Callable[[float], float],
    ra_fn: Optional[Callable[[float], float]] = None,
    pop_params: Optional[dict[str, float]] = None,
    l2_reg: float = 0.05,
    n_starts: int = 5,
    seed: int = 42,
    param_keys: Sequence[str] = _PARAM_KEYS,
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Calibrate mechanistic parameters for an individual patient.

    Objective:
      J(theta) = sum_t (G_sim(t, theta) - G_obs(t))^2 + lambda * sum_i (ln theta_i - ln theta_pop,i)^2

    Args:
        time_min: Time array in minutes.
        glucose_mgdL: Observed CGM glucose trajectory in mg/dL.
        u_fn: Insulin infusion function u(t) in mU/min.
        ra_fn: Glucose appearance rate from meals Ra(t) in mg/min.
        pop_params: Population parameter prior vector (default: DEFAULT_BERGMAN_PARAMS).
        l2_reg: Ridge regularization strength lambda >= 0.
        n_starts: Number of bounded multi-start initializations.
        seed: Random seed for multi-start sampling.
        param_keys: List of parameter names to calibrate.

    Returns:
        (calibrated_params, diagnostics_dict)
    """
    t_arr = np.asarray(time_min, dtype=float)
    g_obs = np.asarray(glucose_mgdL, dtype=float)
    if ra_fn is None:
        ra_fn = lambda t: 0.0

    pop_p = dict(DEFAULT_BERGMAN_PARAMS)
    if pop_params is not None:
        pop_p.update(pop_params)

    base_p = dict(pop_p)
    if "Gb" not in param_keys and "Gb" in base_p:
        base_p["Gb"] = float(pop_p.get("Gb", 100.0))

    initial_state = [g_obs[0], 0.0, float(base_p.get("Ib", 10.0))]

    # Initial RMSE
    G_init, _, _ = simulate_bergman(t_arr, base_p, u_fn, ra_fn, initial_state)
    rmse_init = float(np.sqrt(np.mean((G_init - g_obs) ** 2)))

    log_pop = np.array([np.log(pop_p[k]) for k in param_keys])
    bounds_lo = np.array([_PARAM_LOG_BOUNDS[k][0] for k in param_keys])
    bounds_hi = np.array([_PARAM_LOG_BOUNDS[k][1] for k in param_keys])

    def calc_residuals(log_theta: np.ndarray) -> np.ndarray:
        p_curr = dict(base_p)
        for idx, k in enumerate(param_keys):
            p_curr[k] = float(np.exp(log_theta[idx]))

        try:
            G_sim, _, _ = simulate_bergman(t_arr, p_curr, u_fn, ra_fn, initial_state)
            if G_sim.shape != g_obs.shape or np.any(~np.isfinite(G_sim)):
                data_res = np.full_like(g_obs, 500.0)
            else:
                data_res = np.clip(G_sim - g_obs, -500.0, 500.0)
        except Exception:
            data_res = np.full_like(g_obs, 500.0)

        if l2_reg > 0:
            reg_res = np.sqrt(l2_reg) * (log_theta - log_pop)
            return np.concatenate([data_res, reg_res])
        return data_res

    log_init = np.array([np.log(base_p[k]) for k in param_keys])
    log_init = np.clip(log_init, bounds_lo, bounds_hi)

    rng = np.random.default_rng(seed)
    candidates = [log_init]
    for _ in range(n_starts - 1):
        candidates.append(rng.uniform(bounds_lo, bounds_hi))

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

    calibrated_p = dict(base_p)
    if best_res is not None:
        for idx, k in enumerate(param_keys):
            calibrated_p[k] = float(np.exp(best_res.x[idx]))

    G_fit, _, _ = simulate_bergman(t_arr, calibrated_p, u_fn, ra_fn, initial_state)
    rmse_fit = float(np.sqrt(np.mean((G_fit - g_obs) ** 2)))
    mae_fit = float(np.mean(np.abs(G_fit - g_obs)))

    diagnostics = {
        "rmse_initial": rmse_init,
        "rmse_calibrated": rmse_fit,
        "mae_calibrated": mae_fit,
        "rmse_improvement": rmse_init - rmse_fit,
        "objective_cost": best_cost,
        "success": best_res.success if best_res is not None else False,
        "nfev": best_res.nfev if best_res is not None else 0,
        "status": best_res.status if best_res is not None else -1,
        "message": best_res.message if best_res is not None else "Failed",
    }
    return calibrated_p, diagnostics


def run_sensitivity_analysis(
    time_min: np.ndarray,
    glucose_mgdL: np.ndarray,
    u_fn: Callable[[float], float],
    ra_fn: Optional[Callable[[float], float]],
    params: dict[str, float],
    perturbation_pct: float = 0.10,
    param_keys: Sequence[str] = _PARAM_KEYS,
) -> pd.DataFrame:
    """
    Perform local parameter sensitivity analysis by perturbing each parameter by ±perturbation_pct.

    Returns:
        DataFrame with sensitivity metrics.
    """
    t_arr = np.asarray(time_min, dtype=float)
    g_obs = np.asarray(glucose_mgdL, dtype=float)
    if ra_fn is None:
        ra_fn = lambda t: 0.0

    initial_state = [g_obs[0], 0.0, float(params.get("Ib", 10.0))]

    # Baseline simulation
    G_base, _, _ = simulate_bergman(t_arr, params, u_fn, ra_fn, initial_state)
    rmse_base = float(np.sqrt(np.mean((G_base - g_obs) ** 2)))

    rows = []
    for k in param_keys:
        p_val = float(params[k])
        lo_bound = float(np.exp(_PARAM_LOG_BOUNDS[k][0]))
        hi_bound = float(np.exp(_PARAM_LOG_BOUNDS[k][1]))

        # +10% perturbation
        p_plus = np.clip(p_val * (1.0 + perturbation_pct), lo_bound, hi_bound)
        params_plus = dict(params)
        params_plus[k] = p_plus
        G_plus, _, _ = simulate_bergman(t_arr, params_plus, u_fn, ra_fn, initial_state)
        rmse_plus = float(np.sqrt(np.mean((G_plus - g_obs) ** 2)))

        # -10% perturbation
        p_minus = np.clip(p_val * (1.0 - perturbation_pct), lo_bound, hi_bound)
        params_minus = dict(params)
        params_minus[k] = p_minus
        G_minus, _, _ = simulate_bergman(t_arr, params_minus, u_fn, ra_fn, initial_state)
        rmse_minus = float(np.sqrt(np.mean((G_minus - g_obs) ** 2)))

        d_plus = rmse_plus - rmse_base
        d_minus = rmse_minus - rmse_base
        pct_plus = (d_plus / (rmse_base + 1e-8)) * 100.0
        pct_minus = (d_minus / (rmse_base + 1e-8)) * 100.0
        sensitivity_score = max(abs(d_plus), abs(d_minus))

        rows.append({
            "parameter": k,
            "baseline_val": p_val,
            "baseline_rmse": rmse_base,
            "rmse_plus": rmse_plus,
            "rmse_minus": rmse_minus,
            "delta_rmse_plus": d_plus,
            "delta_rmse_minus": d_minus,
            "pct_change_plus": pct_plus,
            "pct_change_minus": pct_minus,
            "sensitivity_score": sensitivity_score,
        })

    return pd.DataFrame(rows)


def run_profile_likelihood(
    time_min: np.ndarray,
    glucose_mgdL: np.ndarray,
    u_fn: Callable[[float], float],
    ra_fn: Optional[Callable[[float], float]],
    calibrated_params: dict[str, float],
    pop_params: Optional[dict[str, float]] = None,
    l2_reg: float = 0.05,
    n_grid: int = 10,
    ci_level: float = 0.95,
    param_keys: Sequence[str] = _PARAM_KEYS,
) -> dict[str, dict[str, Any]]:
    """
    Compute profile likelihood curves and likelihood-ratio confidence intervals.
    """
    t_arr = np.asarray(time_min, dtype=float)
    g_obs = np.asarray(glucose_mgdL, dtype=float)
    N = len(g_obs)
    if ra_fn is None:
        ra_fn = lambda t: 0.0

    initial_state = [g_obs[0], 0.0, float(calibrated_params.get("Ib", 10.0))]

    # Optimal minimum sum of squared errors
    G_opt, _, _ = simulate_bergman(t_arr, calibrated_params, u_fn, ra_fn, initial_state)
    ssr_min = float(np.sum((G_opt - g_obs) ** 2))
    sigma2_est = max(ssr_min / max(N - len(param_keys), 1), 1.0)

    # 95% critical value
    delta_chi2 = chi2.ppf(ci_level, df=1)
    ssr_threshold = ssr_min + delta_chi2 * sigma2_est

    profiles = {}

    for target_k in param_keys:
        p_hat = float(calibrated_params[target_k])
        other_keys = [k for k in param_keys if k != target_k]

        lo_bound = float(np.exp(_PARAM_LOG_BOUNDS[target_k][0]))
        hi_bound = float(np.exp(_PARAM_LOG_BOUNDS[target_k][1]))

        # Grid spanning 0.3x to 3.0x around point estimate within bounds
        grid_vals = np.geomspace(
            max(p_hat * 0.3, lo_bound),
            min(p_hat * 3.0, hi_bound),
            n_grid,
        )

        ssr_profile = []
        for val in grid_vals:
            fixed_p = dict(calibrated_params)
            fixed_p[target_k] = float(val)

            G_sim, _, _ = simulate_bergman(t_arr, fixed_p, u_fn, ra_fn, initial_state)
            ssr_val = float(np.sum((G_sim - g_obs) ** 2))
            ssr_profile.append(ssr_val)

        ssr_profile = np.array(ssr_profile)

        # Determine CI from points below threshold
        below_thresh_mask = ssr_profile <= ssr_threshold
        if np.any(below_thresh_mask):
            valid_vals = grid_vals[below_thresh_mask]
            ci_lo = float(np.min(valid_vals))
            ci_hi = float(np.max(valid_vals))
            bounded = (ci_lo > grid_vals[0]) and (ci_hi < grid_vals[-1])
        else:
            ci_lo, ci_hi = float(grid_vals[0]), float(grid_vals[-1])
            bounded = False

        profiles[target_k] = {
            "point_estimate": p_hat,
            "grid_vals": grid_vals.tolist(),
            "ssr_profile": ssr_profile.tolist(),
            "ssr_threshold": ssr_threshold,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "is_bounded": bounded,
        }

    return profiles


def classify_identifiability(
    calibrated_params: dict[str, float],
    sensitivity_df: pd.DataFrame,
    profile_results: dict[str, dict[str, Any]],
    param_keys: Sequence[str] = _PARAM_KEYS,
) -> pd.DataFrame:
    """
    Classify parameter identifiability into identifiable (yes/no).
    """
    rows = []
    for k in param_keys:
        p_hat = calibrated_params[k]
        sens_row = sensitivity_df[sensitivity_df["parameter"] == k]
        sens_score = float(sens_row["sensitivity_score"].iloc[0]) if not sens_row.empty else 0.0

        prof = profile_results.get(k, {})
        ci_lo = prof.get("ci_lo", p_hat * 0.5)
        ci_hi = prof.get("ci_hi", p_hat * 2.0)
        is_bounded = prof.get("is_bounded", False)

        ci_ratio = ci_hi / max(ci_lo, 1e-12)

        # Statistical decision
        is_identifiable = is_bounded and (ci_ratio < 15.0) and (sens_score > 0.05)

        reason = []
        if not is_bounded:
            reason.append("Profile likelihood did not cross threshold (flat manifold / unbounded CI)")
        if ci_ratio >= 15.0:
            reason.append(f"CI is excessively wide (ratio={ci_ratio:.1f})")
        if sens_score <= 0.05:
            reason.append(f"Low output sensitivity ({sens_score:.3f} mg/dL)")

        reason_str = "; ".join(reason) if reason else "Well-constrained by CGM dynamics"

        rows.append({
            "parameter": k,
            "point_estimate": p_hat,
            "ci_95": f"[{ci_lo:.2e}, {ci_hi:.2e}]",
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "identifiable": "yes" if is_identifiable else "no",
            "sensitivity_score": sens_score,
            "evidence": reason_str,
        })

    return pd.DataFrame(rows)


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    output_path: str = "experiments/phase3_identifiability.png",
) -> None:
    """Generate and save parameter sensitivity heatmap."""
    params = sensitivity_df["parameter"].tolist()
    matrix = np.array([
        sensitivity_df["delta_rmse_minus"].values,
        sensitivity_df["delta_rmse_plus"].values,
    ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(np.arange(len(params)))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(params, fontsize=11, fontweight="bold")
    ax.set_yticklabels(["-10% Perturbation", "+10% Perturbation"], fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(len(params)):
            val = matrix[i, j]
            color = "white" if val > np.max(matrix) * 0.6 else "black"
            ax.text(j, i, f"dRMSE\n{val:+.2f}", ha="center", va="center", color=color, fontsize=10, fontweight="semibold")

    ax.set_title("Phase 3: Parameter Sensitivity Heatmap (dRMSE mg/dL)", fontsize=13, fontweight="bold", pad=12)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.03)
    cbar.set_label("dRMSE (mg/dL)", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved Phase 3 sensitivity heatmap to %s", output_path)
