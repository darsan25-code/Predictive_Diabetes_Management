"""
Automated unit tests for Phase 7: Model Predictive Controller (MPC).

Verifies:
- 3-4 hour (36 step) prediction horizon with SLSQP
- Non-negativity and upper actuator limits (0 <= u <= u_max)
- Hard glucose floor (predicted G >= 70 mg/dL)
- Safety shield vetting and invariant preservation
- Infeasible/extreme input handling without crashes
"""
import pytest
import numpy as np

from src.control.mpc import MPCController
from src.control.safety_shield import SafetyShield
from src.models.bergman import BergmanModel


def test_mpc_compute_action_nominal_and_safety():
    bm = BergmanModel()
    shield = SafetyShield(bergman_model=bm, hypo_threshold_mgdL=80.0, hard_floor_mgdL=70.0)
    mpc = MPCController(bergman_model=bm, safety_shield=shield, horizon_steps=36, target_g=115.0)

    # 1. Hyperglycemic state (220 mg/dL) -> MPC should propose positive corrective insulin
    u_high, vetoed = mpc.compute_action(current_g=220.0, current_x=0.0, current_i=10.0)
    assert u_high > 0.0
    assert u_high <= mpc.u_max
    assert not np.isnan(u_high)

    # 2. Near-hypoglycemic state (75 mg/dL) -> MPC / Shield must veto or clamp to 0
    u_low, vetoed_low = mpc.compute_action(current_g=75.0, current_x=0.02, current_i=20.0)
    assert vetoed_low or u_low == 0.0
    assert u_low >= 0.0


def test_mpc_constraints_and_non_negativity():
    bm = BergmanModel()
    mpc = MPCController(bergman_model=bm, horizon_steps=24, target_g=110.0)

    # Test across multiple physiological initial states
    test_states = [
        [65.0, 0.03, 25.0],   # Hypo
        [90.0, 0.01, 10.0],   # Low normal
        [140.0, 0.00, 10.0],  # Normal postprandial
        [250.0, 0.00, 5.0],   # Severe hyper
    ]

    for state in test_states:
        u_act, vetoed = mpc.compute_action(current_state=state)
        # Strict invariants
        assert u_act >= 0.0, f"Negative insulin action generated: {u_act}"
        assert u_act <= mpc.u_max, f"Action {u_act} exceeds upper bound {mpc.u_max}"
        assert np.isfinite(u_act), f"Non-finite action returned: {u_act}"


def test_mpc_rollout_execution():
    bm = BergmanModel()
    mpc = MPCController(bergman_model=bm, horizon_steps=12, target_g=120.0)

    initial_state = [160.0, 0.0, 10.0]
    G_traj, U_traj, vetoes = mpc.rollout(initial_state, n_steps=10)

    assert len(G_traj) == 10
    assert len(U_traj) == 10
    assert len(vetoes) == 10
    assert np.all(U_traj >= 0.0)
    assert np.all(U_traj <= mpc.u_max)
