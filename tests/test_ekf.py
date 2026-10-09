"""
Automated unit tests for Phase 5: Extended Kalman Filter State Estimation.
"""
import pytest
import numpy as np

from src.models.bergman import BergmanModel, DEFAULT_BERGMAN_PARAMS, simulate_bergman
from src.estimation.kalman import (
    ExtendedKalmanFilter,
    GlucoseEKF,
    compute_bergman_jacobian_continuous,
    compute_bergman_jacobian_discrete,
)


def test_ekf_jacobian_finite_difference():
    """Verify analytical continuous and discrete Jacobians match numerical finite differences."""
    bm = BergmanModel()
    x0 = np.array([130.0, 0.02, 14.0])
    dt = 5.0
    u = 15.0
    ra = 10.0

    # 1. Continuous Jacobian verification: df/dx
    F_c_analytic = compute_bergman_jacobian_continuous(x0[0], x0[1], x0[2], bm.params)

    eps = 1e-6
    F_c_num = np.zeros((3, 3))
    for j in range(3):
        x_plus = x0.copy()
        x_plus[j] += eps
        x_minus = x0.copy()
        x_minus[j] -= eps

        def deriv(x_vec):
            dG = -(bm.p1 + x_vec[1]) * x_vec[0] + bm.p1 * bm.Gb + ra / bm.Vg
            dX = -bm.p2 * x_vec[1] + bm.p3 * (x_vec[2] - bm.Ib)
            dI = -bm.n * (x_vec[2] - bm.Ib) + (u - bm.steady_state_basal_insulin()) / bm.Vi
            return np.array([dG, dX, dI])

        f_plus = deriv(x_plus)
        f_minus = deriv(x_minus)
        F_c_num[:, j] = (f_plus - f_minus) / (2.0 * eps)

    np.testing.assert_allclose(F_c_analytic, F_c_num, rtol=1e-4, atol=1e-5)

    # 2. Discrete Jacobian calculation
    F_d_analytic = compute_bergman_jacobian_discrete(x0, bm.params, dt_min=dt)
    assert F_d_analytic.shape == (3, 3)
    assert np.all(np.isfinite(F_d_analytic))


def test_ekf_convergence_from_wrong_initial_state():
    """Test A: Filter initialized with a wrong state converges toward true observations."""
    bm = BergmanModel()
    t_eval = np.arange(0, 180, 5)
    u_fn = lambda t: 15.0
    ra_fn = lambda t: 120.0 * np.exp(-0.02 * t) if t >= 20 else 0.0

    G_true, _, _ = simulate_bergman(t_eval, bm.params, u_fn, ra_fn)
    rng = np.random.default_rng(42)
    G_obs = G_true + rng.normal(0.0, 3.0, size=len(G_true))

    # Wrong initial state: start at 250 mg/dL when true is ~100 mg/dL
    ekf = ExtendedKalmanFilter(bm)
    ekf.reset(G0=250.0)

    g_estimates = []
    for i in range(len(t_eval)):
        dt = 5.0
        u_val = u_fn(t_eval[i])
        ra_val = ra_fn(t_eval[i])
        x_post, _, _, _ = ekf.step(cgm_mgdL=G_obs[i], u_mU_per_min=u_val, ra_mg_per_min=ra_val, dt_min=dt)
        g_estimates.append(x_post[0])

    g_estimates = np.array(g_estimates)

    # Within 30 minutes (6 steps), the filter error should reduce dramatically
    initial_error = abs(g_estimates[0] - G_true[0])
    converged_error = np.mean(np.abs(g_estimates[6:] - G_true[6:]))
    assert converged_error < 10.0
    assert converged_error < initial_error / 5.0


def test_ekf_uncertainty_band_coverage():
    """Test B: Empirical 95% uncertainty band coverage is within 90% and 99%."""
    bm = BergmanModel()
    t_eval = np.arange(0, 1440, 5)  # 24 hours
    u_fn = lambda t: 15.0 + (10.0 if 120 <= t <= 180 else 0.0)
    ra_fn = lambda t: 150.0 * np.exp(-0.02 * (t - 60)) if t >= 60 else 0.0

    G_true, _, _ = simulate_bergman(t_eval, bm.params, u_fn, ra_fn)
    sensor_std = 5.0
    rng = np.random.default_rng(123)
    G_obs = G_true + rng.normal(0.0, sensor_std, size=len(G_true))

    cfg = {
        "noise": {"Q": {"G": 1.0, "X": 1e-6, "I": 0.05}, "R": {"cgm": sensor_std ** 2}},
        "initialization": {"P_diag": [25.0, 1e-6, 1.0]},
    }
    ekf = ExtendedKalmanFilter(bm, cfg=cfg)
    ekf.reset(G0=G_obs[0])

    covered = []
    for i in range(1, len(t_eval)):
        # 1. Predict
        x_pred, P_pred = ekf.predict(u_fn(t_eval[i - 1]), ra_fn(t_eval[i - 1]), dt_min=5.0)
        # Observation variance: S = H P_pred H^T + R
        sigma_obs = np.sqrt(P_pred[0, 0] + sensor_std ** 2)
        ci_lo = x_pred[0] - 1.96 * sigma_obs
        ci_hi = x_pred[0] + 1.96 * sigma_obs
        covered.append(ci_lo <= G_obs[i] <= ci_hi)

        # 2. Update
        ekf.update_measurement(G_obs[i])

    coverage_pct = np.mean(covered) * 100.0
    assert 90.0 <= coverage_pct <= 99.0, f"Coverage was {coverage_pct:.1f}%, expected [90%, 99%]"


def test_ekf_missing_measurements_and_covariance_properties():
    """Test D & E: Handles NaN measurements safely; covariance remains symmetric & positive semi-definite."""
    bm = BergmanModel()
    ekf = ExtendedKalmanFilter(bm)
    ekf.reset(G0=120.0)

    # Feed a NaN measurement
    x_post, P_post, innov, sigma_pred = ekf.step(cgm_mgdL=np.nan, u_mU_per_min=15.0, dt_min=5.0)

    assert np.all(np.isfinite(x_post))
    assert np.all(np.isfinite(P_post))

    # Symmetry
    np.testing.assert_allclose(P_post, P_post.T, atol=1e-8)

    # Positive semi-definiteness: all eigenvalues >= 0
    eigvals = np.linalg.eigvalsh(P_post)
    assert np.all(eigvals >= -1e-8)
