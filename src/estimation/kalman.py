"""
Phase 5: Extended Kalman Filter (EKF) State Estimation for Glucose-Insulin Dynamics.

State:
  Standard 3-state: [G (mg/dL), X (1/min), I (mU/L)]
  Augmented 4-state: [G, X, I, p3] where p3 is the slow-drifting insulin sensitivity parameter.

Observation:
  CGM glucose reading (mg/dL) with observation matrix H = [1, 0, 0] or [1, 0, 0, 0].

Features:
- Analytical and symbolic Jacobian derivations of the nonlinear Bergman dynamics.
- Joseph form numerically stabilized covariance updates: P_post = (I - KH) P_pred (I - KH)^T + K R K^T.
- Handling of missing CGM observations (forward propagation without measurement update).
- Handling of irregular delta_t sampling intervals.
- Validation-based tuning of process noise Q and measurement noise R.
- 95% predictive uncertainty interval evaluation and empirical coverage calculation.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence
import numpy as np

logger = logging.getLogger(__name__)


def compute_bergman_jacobian_continuous(
    G: float,
    X: float,
    I: float,
    params: dict[str, float],
    augmented: bool = False,
) -> np.ndarray:
    """
    Compute the continuous-time Jacobian matrix F_c = df/dx analytically.

    Continuous ODEs:
      dG/dt = -(p1 + X)*G + p1*Gb + Ra/Vg
      dX/dt = -p2*X + p3*(I - Ib)
      dI/dt = -n*(I - Ib) + (u - u_basal)/Vi

    Partial derivatives:
      d(dG)/dG = -(p1 + X)
      d(dG)/dX = -G
      d(dG)/dI = 0

      d(dX)/dG = 0
      d(dX)/dX = -p2
      d(dX)/dI = p3

      d(dI)/dG = 0
      d(dI)/dX = 0
      d(dI)/dI = -n

    For augmented state [G, X, I, p3]:
      d(dX)/dp3 = I - Ib
      d(dp3)/d[G,X,I,p3] = 0 (slow drift / random walk)
    """
    p1 = float(params["p1"])
    p2 = float(params["p2"])
    p3 = float(params["p3"])
    n  = float(params["n"])
    Ib = float(params["Ib"])

    if not augmented:
        F_c = np.array([
            [-(p1 + X), -G,   0.0],
            [0.0,       -p2,  p3 ],
            [0.0,        0.0, -n ],
        ], dtype=float)
    else:
        F_c = np.array([
            [-(p1 + X), -G,   0.0, 0.0     ],
            [0.0,       -p2,  p3,  (I - Ib)],
            [0.0,        0.0, -n,  0.0     ],
            [0.0,        0.0, 0.0, 0.0     ],
        ], dtype=float)

    return F_c


def compute_bergman_jacobian_discrete(
    x: np.ndarray,
    params: dict[str, float],
    dt_min: float,
    augmented: bool = False,
) -> np.ndarray:
    """
    First-order / second-order discrete-time state transition Jacobian:
      F_d = I + F_c * dt + (F_c * dt)^2 / 2
    """
    G, X, I = float(x[0]), float(x[1]), float(x[2])
    dim = 4 if augmented else 3
    F_c = compute_bergman_jacobian_continuous(G, X, I, params, augmented=augmented)
    
    # Taylor expansion of matrix exponential exp(F_c * dt)
    M = F_c * dt_min
    F_d = np.eye(dim) + M + 0.5 * (M @ M)
    return F_d


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for T1D physiological state estimation.
    """

    def __init__(
        self,
        bergman_model=None,
        cfg: dict | None = None,
        augmented: bool = False,
    ) -> None:
        if bergman_model is None:
            from src.models.bergman import BergmanModel
            bergman_model = BergmanModel()
        if cfg is None:
            cfg = {}

        self._bm = bergman_model
        self.augmented = augmented
        self.dim_x = 4 if augmented else 3
        self.dim_z = 1

        noise = cfg.get("noise", {})
        q_cfg = noise.get("Q", {})
        r_cfg = noise.get("R", {})
        p_init = cfg.get("initialization", {}).get("P_diag", [400.0, 1.0e-6, 1.0])

        if not augmented:
            self._Q = np.diag([
                float(q_cfg.get("G", 2.0)),
                float(q_cfg.get("X", 1e-6)),
                float(q_cfg.get("I", 0.05)),
            ])
            self._P0 = np.diag([float(v) for v in p_init[:3]])
        else:
            self._Q = np.diag([
                float(q_cfg.get("G", 2.0)),
                float(q_cfg.get("X", 1e-6)),
                float(q_cfg.get("I", 0.05)),
                float(q_cfg.get("p3", 1e-12)),
            ])
            p_init_aug = list(p_init) + [1e-10] if len(p_init) == 3 else p_init
            self._P0 = np.diag([float(v) for v in p_init_aug[:4]])

        self._R = np.array([[float(r_cfg.get("cgm", 25.0))]])

        self.x = np.zeros(self.dim_x, dtype=float)
        self.P = self._P0.copy()
        self._initialised = False

    @property
    def Q(self) -> np.ndarray:
        return self._Q

    @Q.setter
    def Q(self, val: np.ndarray) -> None:
        self._Q = np.asarray(val, dtype=float)

    @property
    def R(self) -> np.ndarray:
        return self._R

    @R.setter
    def R(self, val: np.ndarray) -> None:
        self._R = np.asarray(val, dtype=float)

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    def reset(
        self,
        G0: float = 120.0,
        X0: float = 0.0,
        I0: Optional[float] = None,
        p3_0: Optional[float] = None,
    ) -> None:
        """Reset state estimate and covariance."""
        if I0 is None:
            I0 = float(self._bm.Ib)
        if not self.augmented:
            self.x = np.array([float(G0), float(X0), float(I0)], dtype=float)
        else:
            p3_val = float(self._bm.p3) if p3_0 is None else float(p3_0)
            self.x = np.array([float(G0), float(X0), float(I0), p3_val], dtype=float)

        self.P = self._P0.copy()
        self._initialised = True

    def predict(
        self,
        u_mU_per_min: float,
        ra_mg_per_min: float = 0.0,
        dt_min: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        EKF Prediction step: propagate state and covariance forward.
        """
        if not self._initialised:
            self.reset()

        params = dict(self._bm.params)
        if self.augmented:
            params["p3"] = float(self.x[3])

        # 1. State propagation via Runge-Kutta 4th order / Euler integration
        x_pred = self._integrate_state(self.x, u_mU_per_min, ra_mg_per_min, dt_min, params)

        # 2. Covariance propagation using discrete Jacobian
        F_d = compute_bergman_jacobian_discrete(self.x, params, dt_min, augmented=self.augmented)
        P_pred = F_d @ self.P @ F_d.T + self._Q * dt_min

        # Ensure covariance symmetry and positive semi-definiteness
        P_pred = 0.5 * (P_pred + P_pred.T)

        self.x = x_pred
        self.P = P_pred
        return self.x.copy(), self.P.copy()

    def update_measurement(
        self,
        cgm_mgdL: Optional[float],
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        EKF Measurement update step using Joseph form for numerical stability.

        Returns:
            (x_post, P_post, innovation, predictive_sigma)
        """
        H = np.zeros((1, self.dim_x), dtype=float)
        H[0, 0] = 1.0  # CGM observes glucose G directly

        # Predictive measurement variance S = H P H^T + R
        S = H @ self.P @ H.T + self._R
        sigma_pred = float(np.sqrt(max(S[0, 0], 1e-6)))

        if cgm_mgdL is None or not np.isfinite(cgm_mgdL):
            # Missing or invalid measurement — skip update, keep prediction
            return self.x.copy(), self.P.copy(), 0.0, sigma_pred

        z = float(cgm_mgdL)
        y = z - float(self.x[0])  # Innovation

        # Kalman Gain: K = P H^T S^-1
        K = (self.P @ H.T) / S[0, 0]

        # Posterior state: x_post = x_pred + K * y
        x_post = self.x + K.flatten() * y

        # Clip state to physiologically valid ranges
        x_post[0] = np.clip(x_post[0], 20.0, 600.0)  # Glucose (mg/dL)
        x_post[1] = max(x_post[1], 0.0)               # Remote insulin X >= 0
        x_post[2] = max(x_post[2], 0.0)               # Insulin I >= 0
        if self.augmented:
            x_post[3] = np.clip(x_post[3], 1e-8, 1e-2)

        # Joseph Form Covariance Update: P_post = (I - K H) P_pred (I - K H)^T + K R K^T
        I_KH = np.eye(self.dim_x) - K @ H
        P_post = I_KH @ self.P @ I_KH.T + K @ self._R @ K.T
        P_post = 0.5 * (P_post + P_post.T)

        self.x = x_post
        self.P = P_post
        return self.x.copy(), self.P.copy(), y, sigma_pred

    def step(
        self,
        cgm_mgdL: Optional[float],
        u_mU_per_min: float,
        ra_mg_per_min: float = 0.0,
        dt_min: float = 5.0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """
        Execute full EKF cycle: predict forward, then apply measurement update.
        """
        if not self._initialised:
            G_init = float(cgm_mgdL) if (cgm_mgdL is not None and np.isfinite(cgm_mgdL)) else 120.0
            self.reset(G0=G_init)

        # 1. Prediction step
        self.predict(u_mU_per_min, ra_mg_per_min, dt_min)

        # 2. Correction step
        return self.update_measurement(cgm_mgdL)

    def _integrate_state(
        self,
        x: np.ndarray,
        u: float,
        Ra: float,
        dt: float,
        params: dict[str, float],
    ) -> np.ndarray:
        """RK4 numerical integration for nonlinear state transition."""
        def rhs(s: np.ndarray) -> np.ndarray:
            G, X, I = s[0], s[1], s[2]
            p1, p2 = float(params["p1"]), float(params["p2"])
            p3     = float(s[3]) if self.augmented else float(params["p3"])
            n, Gb  = float(params["n"]),  float(params["Gb"])
            Ib, Vg = float(params["Ib"]), float(params["Vg"])
            Vi     = float(params["Vi"])
            u_basal = n * Ib * Vi

            dG = -(p1 + X) * G + p1 * Gb + Ra / Vg
            dX = -p2 * X + p3 * (I - Ib)
            dI = -n * (I - Ib) + (u - u_basal) / Vi
            if not self.augmented:
                return np.array([dG, dX, dI])
            return np.array([dG, dX, dI, 0.0])  # p3 parameter drift is 0 mean

        k1 = rhs(x)
        k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2)
        k4 = rhs(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x_next


# Backwards-compatible alias for existing imports
GlucoseEKF = ExtendedKalmanFilter


def tune_ekf_hyperparameters(
    val_df: Any,
    bergman_model=None,
    candidate_q_g: Sequence[float] = (0.5, 2.0, 5.0, 10.0),
    candidate_r: Sequence[float] = (9.0, 16.0, 25.0, 36.0),
) -> dict[str, Any]:
    """
    Tune process noise Q_G and measurement noise R on validation data.

    Evaluation criterion: Minimum validation RMSE while maintaining uncertainty calibration (coverage ~ 95%).
    """
    if bergman_model is None:
        from src.models.bergman import BergmanModel
        bergman_model = BergmanModel()

    best_score = float("inf")
    best_config = {"q_g": 2.0, "r_cgm": 25.0}

    # Extract validation traces
    patients = val_df["patient_id"].unique()

    for q_g in candidate_q_g:
        for r_cgm in candidate_r:
            cfg = {
                "noise": {"Q": {"G": q_g, "X": 1e-6, "I": 0.05}, "R": {"cgm": r_cgm}},
                "initialization": {"P_diag": [400.0, 1e-6, 1.0]},
            }
            ekf = ExtendedKalmanFilter(bergman_model, cfg=cfg)

            sq_errors = []
            covered_flags = []

            for pid in patients:
                pdf = val_df[val_df["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
                if len(pdf) < 10:
                    continue
                t0 = pdf["timestamp"].iloc[0]
                t_arr = (pdf["timestamp"] - t0).dt.total_seconds().values / 60.0
                g_arr = pdf["glucose_mgdL"].values
                u_arr = pdf["insulin_mU_per_min"].values

                ekf.reset(G0=g_arr[0])
                for idx in range(1, len(pdf)):
                    dt = t_arr[idx] - t_arr[idx - 1]
                    if dt <= 0:
                        continue
                    # 1. Prior predict
                    x_pred, P_pred = ekf.predict(
                        u_mU_per_min=u_arr[idx - 1],
                        dt_min=dt,
                    )
                    sigma_pred = float(np.sqrt(P_pred[0, 0] + r_cgm))
                    g_pred = x_pred[0]
                    g_true = g_arr[idx]
                    sq_errors.append((g_pred - g_true) ** 2)

                    # Check 95% uncertainty interval
                    ci_lo = g_pred - 1.96 * sigma_pred
                    ci_hi = g_pred + 1.96 * sigma_pred
                    covered_flags.append(ci_lo <= g_true <= ci_hi)

                    # 2. Measurement update
                    ekf.update_measurement(cgm_mgdL=g_true)

            if not sq_errors:
                continue

            rmse = float(np.sqrt(np.mean(sq_errors)))
            cov_pct = float(np.mean(covered_flags)) * 100.0

            # Penalize coverage deviations from 95%
            cov_penalty = abs(cov_pct - 95.0) * 0.5
            total_score = rmse + cov_penalty

            if total_score < best_score:
                best_score = total_score
                best_config = {
                    "q_g": q_g,
                    "r_cgm": r_cgm,
                    "rmse": rmse,
                    "coverage_pct": cov_pct,
                }

    return best_config
