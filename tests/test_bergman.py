"""
Tests for Phase 2: Bergman minimal model ODE and parameter fitting.
"""
import pytest
import numpy as np

from src.models.mechanistic import (
    BergmanModel,
    bergman_rhs,
    simulate_bergman,
    fit_parameters,
    fit_population,
    DEFAULT_BERGMAN_PARAMS,
)
from src.data.loaders import SyntheticLoader


def test_steady_state():
    """Integrating from steady state with no external inputs returns steady state."""
    params = dict(DEFAULT_BERGMAN_PARAMS)
    t_eval = np.arange(0, 300, 5)  # 5 hours
    u_basal = params["n"] * params["Ib"] * params["Vi"]
    u_fn = lambda t: u_basal
    ra_fn = lambda t: 0.0

    G, X, I = simulate_bergman(t_eval, params, u_fn, ra_fn)

    # Assert G stays at Gb (92.0), X stays at 0.0, I stays at Ib (11.0)
    np.testing.assert_allclose(G, params["Gb"], atol=1e-3)
    np.testing.assert_allclose(X, 0.0, atol=1e-3)
    np.testing.assert_allclose(I, params["Ib"], atol=1e-3)


def test_params_positivity():
    """Params are provably positive after optimization via log-parameterization."""
    # Generate synthetic noisy data with perturbed params
    t_eval = np.arange(0, 180, 5)
    true_params = dict(DEFAULT_BERGMAN_PARAMS)
    true_params["p1"] = 0.035
    true_params["p2"] = 0.015
    true_params["p3"] = 2.0e-5
    true_params["n"]  = 0.12

    u_fn = lambda t: 15.0 if t < 60 else 25.0
    ra_fn = lambda t: 200.0 * np.exp(-0.02 * (t - 30)) if t >= 30 else 0.0

    G_true, _, _ = simulate_bergman(t_eval, true_params, u_fn, ra_fn)
    rng = np.random.default_rng(42)
    G_obs = G_true + rng.normal(0.0, 4.0, size=len(G_true))

    # Fit parameters starting from bad initial guess
    bad_init = dict(DEFAULT_BERGMAN_PARAMS)
    bad_init["p1"] = 0.01
    bad_init["p2"] = 0.08
    bad_init["p3"] = 1.0e-6
    bad_init["n"]  = 0.02

    fitted_params, info = fit_parameters(
        t_eval, G_obs, u_fn, ra_fn, initial_params=bad_init, n_starts=3
    )

    for k in ("p1", "p2", "p3", "n", "Gb", "Ib", "Vg", "Vi"):
        assert fitted_params[k] > 0.0, f"Parameter {k} must be strictly positive!"


def test_fit_improves_rmse():
    """Fit improves RMSE vs an unmodified initial guess on synthetic data."""
    t_eval = np.arange(0, 240, 5)
    true_params = dict(DEFAULT_BERGMAN_PARAMS)
    true_params["p1"] = 0.040
    true_params["p2"] = 0.018

    u_fn = lambda t: 15.0 + (5.0 if 30 <= t <= 90 else 0.0)
    ra_fn = lambda t: 150.0 * np.exp(-0.03 * (t - 40)) if t >= 40 else 0.0

    G_true, _, _ = simulate_bergman(t_eval, true_params, u_fn, ra_fn)
    rng = np.random.default_rng(123)
    G_obs = G_true + rng.normal(0.0, 3.0, size=len(G_true))

    # Bad initial guess
    init_params = dict(DEFAULT_BERGMAN_PARAMS)
    init_params["p1"] = 0.015
    init_params["p2"] = 0.050

    fitted_params, info = fit_parameters(
        t_eval, G_obs, u_fn, ra_fn, initial_params=init_params, n_starts=3
    )

    assert info["rmse_fitted"] < info["rmse_initial"]
    assert info["rmse_improvement"] > 0.0


def test_fit_population():
    """fit_population finds a shared parameter vector across multiple patient datasets."""
    # Use short traces (2h) and minimal starts for fast test execution
    loader = SyntheticLoader({"n_patients": 2, "duration_hours": 2, "seed": 42})
    df_raw = loader.load()

    patient_datasets = []
    for pid, group in df_raw.groupby("patient_id"):
        t_arr = (group["timestamp"] - group["timestamp"].iloc[0]).dt.total_seconds().values / 60.0
        g_arr = group["glucose_mgdL"].values
        u_arr = group["insulin_mU_per_min"].values
        cho_arr = group["meal_cho_g"].values

        u_fn = lambda t, t_arr=t_arr, u_arr=u_arr: float(np.interp(t, t_arr, u_arr))
        ra_fn = lambda t, t_arr=t_arr, cho_arr=cho_arr: float(np.interp(t, t_arr, cho_arr * 10.0))
        patient_datasets.append((t_arr, g_arr, u_fn, ra_fn))

    shared_params, info = fit_population(patient_datasets, n_starts=1)
    for k in ("p1", "p2", "p3", "n"):
        assert shared_params[k] > 0.0
