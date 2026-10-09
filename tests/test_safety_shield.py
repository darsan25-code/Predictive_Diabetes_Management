"""
Automated unit tests for Phase 8: Hard Safety Shield.
"""
import pytest
import numpy as np

from src.control.safety_shield import SafetyShield
from src.models.bergman import BergmanModel


def test_safety_shield_nominal_and_veto():
    """Verify safe actions pass and unsafe actions trigger veto or clamp."""
    bm = BergmanModel()
    shield = SafetyShield(bergman_model=bm, hypo_threshold_mgdL=80.0, hard_floor_mgdL=70.0)

    # 1. Safe condition: Normal glucose, moderate insulin
    state_safe = [140.0, 0.0, 10.0]
    u_exec, intervened = shield(15.0, state_safe)
    assert not intervened
    assert u_exec == 15.0

    # 2. Critical condition: Low glucose (75 mg/dL), proposing large bolus (50 mU/min)
    state_low = [75.0, 0.02, 15.0]
    u_exec_low, intervened_low = shield(50.0, state_low)
    assert intervened_low
    assert u_exec_low < 50.0  # Must be clamped or vetoed to 0

    # 3. Invalid inputs (NaN or negative)
    u_nan, intervened_nan = shield(np.nan, state_safe)
    assert intervened_nan
    assert u_nan == 0.0

    u_neg, intervened_neg = shield(-10.0, state_safe)
    assert intervened_neg
    assert u_neg == 0.0


def test_safety_shield_counters_and_adversarial_invariants():
    """Verify instrumentation counters and strict invariant enforcement."""
    bm = BergmanModel()
    shield = SafetyShield(bergman_model=bm, hypo_threshold_mgdL=80.0, hard_floor_mgdL=70.0)
    shield.reset_episode()

    state = [85.0, 0.01, 12.0]
    
    # Deliberately aggressive adversarial inputs
    aggressive_actions = [40.0, 60.0, 80.0, 100.0]
    for act in aggressive_actions:
        u_exec, intervened = shield(act, state)
        assert intervened
        assert u_exec < act

    assert shield.counters["total_evaluated"] >= 4
    assert shield.counters["activations"] >= 4
    assert shield.counters["invariant_violations"] == 0
