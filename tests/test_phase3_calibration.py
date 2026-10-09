"""
Automated unit tests for Phase 3: Per-Patient Calibration and Identifiability Analysis.
"""
import pytest
import numpy as np

from src.models.mechanistic import DEFAULT_BERGMAN_PARAMS, simulate_bergman
from src.models.calibration import (
    calibrate_patient,
    run_sensitivity_analysis,
    run_profile_likelihood,
    classify_identifiability,
)


def test_calibrate_patient_improves_fit():
    """Per-patient calibration initialized from population parameters improves RMSE."""
    t_eval = np.arange(0, 180, 5)
    true_params = dict(DEFAULT_BERGMAN_PARAMS)
    true_params["p1"] = 0.038
    true_params["p2"] = 0.019
    true_params["p3"] = 4.0e-5
    true_params["n"]  = 0.12

    u_fn = lambda t: 15.0 + (10.0 if 30 <= t <= 70 else 0.0)
    ra_fn = lambda t: 180.0 * np.exp(-0.02 * (t - 30)) if t >= 30 else 0.0

    G_true, _, _ = simulate_bergman(t_eval, true_params, u_fn, ra_fn)
    rng = np.random.default_rng(42)
    G_obs = G_true + rng.normal(0.0, 3.0, size=len(G_true))

    # Population prior
    pop_params = dict(DEFAULT_BERGMAN_PARAMS)

    calibrated_p, diag = calibrate_patient(
        t_eval, G_obs, u_fn, ra_fn,
        pop_params=pop_params,
        l2_reg=0.01,
        n_starts=2,
    )

    assert diag["success"] is True
    assert diag["rmse_calibrated"] < diag["rmse_initial"]
    for k in ("p1", "p2", "p3", "n"):
        assert calibrated_p[k] > 0.0


def test_ridge_regularization_shrinkage():
    """Higher ridge regularization lambda pulls estimates closer to the population vector."""
    t_eval = np.arange(0, 180, 5)
    true_params = dict(DEFAULT_BERGMAN_PARAMS)
    true_params["p1"] = 0.050
    true_params["p2"] = 0.035

    u_fn = lambda t: 15.0 + (10.0 if 30 <= t <= 70 else 0.0)
    ra_fn = lambda t: 150.0 * np.exp(-0.02 * (t - 30)) if t >= 30 else 0.0
    G_true, _, _ = simulate_bergman(t_eval, true_params, u_fn, ra_fn)

    pop_params = dict(DEFAULT_BERGMAN_PARAMS)  # p1 = 0.028

    p_low_reg, _ = calibrate_patient(t_eval, G_true, u_fn, ra_fn, pop_params=pop_params, l2_reg=0.0, n_starts=2)
    p_high_reg, _ = calibrate_patient(t_eval, G_true, u_fn, ra_fn, pop_params=pop_params, l2_reg=1.0, n_starts=2)

    # High reg estimate should be closer to pop_params["p1"] (0.028) than unregularized estimate
    dist_low = abs(p_low_reg["p1"] - pop_params["p1"])
    dist_high = abs(p_high_reg["p1"] - pop_params["p1"])
    assert dist_high < dist_low


def test_sensitivity_and_profile_likelihood():
    """Sensitivity and profile likelihood correctly compute deltas and confidence intervals."""
    t_eval = np.arange(0, 120, 5)
    params = dict(DEFAULT_BERGMAN_PARAMS)
    u_fn = lambda t: 15.0
    ra_fn = lambda t: 100.0 * np.exp(-0.02 * t)
    G_obs, _, _ = simulate_bergman(t_eval, params, u_fn, ra_fn)

    sens_df = run_sensitivity_analysis(t_eval, G_obs, u_fn, ra_fn, params, perturbation_pct=0.10)
    assert len(sens_df) == 4
    assert "delta_rmse_plus" in sens_df.columns

    prof_res = run_profile_likelihood(t_eval, G_obs, u_fn, ra_fn, params, n_grid=6)
    assert "p1" in prof_res
    assert prof_res["p1"]["ci_lo"] <= prof_res["p1"]["point_estimate"] <= prof_res["p1"]["ci_hi"]

    ident_df = classify_identifiability(params, sens_df, prof_res)
    assert len(ident_df) == 4
    assert "identifiable" in ident_df.columns
