"""
Extended Kalman Filter for glucose state estimation.
Re-exported from src.estimation.kalman for backwards compatibility.
"""
from src.estimation.kalman import (
    ExtendedKalmanFilter,
    GlucoseEKF,
    compute_bergman_jacobian_continuous,
    compute_bergman_jacobian_discrete,
    tune_ekf_hyperparameters,
)

__all__ = [
    "ExtendedKalmanFilter",
    "GlucoseEKF",
    "compute_bergman_jacobian_continuous",
    "compute_bergman_jacobian_discrete",
    "tune_ekf_hyperparameters",
]
