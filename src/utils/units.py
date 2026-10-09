"""
Explicit unit conversion helpers.

Units policy (AGENTS.md Rule 9):
  - Glucose : mg/dL  (primary throughout this codebase)
  - Insulin : mU or mU/min  (explicitly stated in variable / column names)

Never perform implicit conversions. Always call a named function.
Column names in DataFrames always carry the unit suffix (e.g. glucose_mgdL).
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Glucose conversions
# ---------------------------------------------------------------------------


def mmolL_to_mgdL(value: float | np.ndarray) -> float | np.ndarray:
    """Convert glucose from mmol/L to mg/dL (×18.0)."""
    val = np.asarray(value, dtype=float) * 18.0
    return float(val) if np.ndim(value) == 0 else val


def mgdL_to_mmolL(value: float | np.ndarray) -> float | np.ndarray:
    """Convert glucose from mg/dL to mmol/L (÷18.0)."""
    val = np.asarray(value, dtype=float) / 18.0
    return float(val) if np.ndim(value) == 0 else val


# ---------------------------------------------------------------------------
# Insulin conversions
# ---------------------------------------------------------------------------


def mU_per_min_to_U_per_hour(value: float | np.ndarray) -> float | np.ndarray:
    """Convert insulin rate from mU/min to U/hour."""
    return np.asarray(value, dtype=float) * 60.0 / 1000.0


def U_per_hour_to_mU_per_min(value: float | np.ndarray) -> float | np.ndarray:
    """Convert insulin rate from U/hour to mU/min."""
    return np.asarray(value, dtype=float) * 1000.0 / 60.0


def pmolL_to_mU_per_L(value: float | np.ndarray) -> float | np.ndarray:
    """Convert plasma insulin from pmol/L to mU/L (1 mU ≈ 6.945 pmol for human insulin)."""
    return np.asarray(value, dtype=float) / 6.945


# ---------------------------------------------------------------------------
# Physiological bounds (used for validation and safety checks)
# ---------------------------------------------------------------------------

#: Absolute glucose bounds — values outside this range are physiologically impossible.
GLUCOSE_MIN_MGDL: float = 20.0
GLUCOSE_MAX_MGDL: float = 600.0

#: Clinical threshold constants (mg/dL).
HYPO_THRESHOLD_MGDL: float = 70.0    # clinical hypoglycaemia
TIR_LOW_MGDL: float = 70.0           # Time-in-Range lower bound
TIR_HIGH_MGDL: float = 180.0         # Time-in-Range upper bound

INSULIN_MIN_MU_PER_MIN: float = 0.0
INSULIN_MAX_MU_PER_MIN: float = 20.0  # generous cap (basal + large bolus)


def validate_glucose_units(value: float | np.ndarray, label: str = "glucose") -> None:
    """
    Raise ValueError if any glucose value falls outside physiological bounds.

    This guard catches the most common unit mistake (mmol/L passed where
    mg/dL is expected): a reading of 5.5 mmol/L is < 20 mg/dL and will
    trigger an error, prompting the caller to convert explicitly.
    """
    arr = np.asarray(value, dtype=float).ravel()
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if lo < GLUCOSE_MIN_MGDL or hi > GLUCOSE_MAX_MGDL:
        raise ValueError(
            f"{label} contains values [{lo:.1f}, {hi:.1f}] outside the "
            f"physiological range [{GLUCOSE_MIN_MGDL}, {GLUCOSE_MAX_MGDL}] mg/dL. "
            "If your data is in mmol/L, call mmolL_to_mgdL() first."
        )
