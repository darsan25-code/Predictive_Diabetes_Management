"""
Evaluation metrics for glucose forecasting and control.

AGENTS.md Rule 3: Never report RMSE alone — always include Clarke EGA + TIR.
AGENTS.md Rule 4: Time-below-range (<70 mg/dL) is the PRIMARY safety metric.

All glucose values must be in mg/dL (Rule 9).
compute_metrics() always returns ALL required metrics; callers must not cherry-pick.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

from src.utils.units import TIR_LOW_MGDL, TIR_HIGH_MGDL, HYPO_THRESHOLD_MGDL

# ---------------------------------------------------------------------------
# Clarke Error Grid Analysis
# ---------------------------------------------------------------------------


def assign_clarke_zone(ref: float, pred: float) -> str:
    """
    Assign Clarke Error Grid zone (A–E) for a single (reference, prediction) pair.
    Both values in mg/dL.

    Reference: Clarke WL et al., Diabetes Care 2005;28(10):2413–2417.

    Zone A — clinically accurate
    Zone B — clinically acceptable (>20% error but no clinically incorrect action)
    Zone C — would lead to unnecessary treatment of a benign reading
    Zone D — failure to detect/treat a clinically significant glucose excursion
    Zone E — would result in erroneous treatment (dangerous reversal)
    """
    ref  = max(float(ref),  1.0)   # guard against division by zero
    pred = float(pred)

    # ── Zone A (accurate) ────────────────────────────────────────────────
    if ref <= 70.0 and pred <= 70.0:
        return "A"
    if abs(pred - ref) / ref <= 0.20:
        return "A"

    # ── Zone E (dangerous reversal of treatment) ──────────────────────────
    # Patient is hypo; meter reads hyper → would give insulin → fatal
    if ref < 70.0 and pred > 180.0:
        return "E"
    # Patient is hyper; meter reads hypo → would give glucose → dangerous
    if ref > 180.0 and pred < 70.0:
        return "E"

    # ── Zone D (failure to detect, requires action) ───────────────────────
    # Reference in-range but prediction is hypoglycaemic — fails to detect hypo
    if 70.0 <= ref <= 180.0 and pred < 70.0:
        return "D"
    # Reference in-range but prediction is hyperglycaemic — fails to detect hyper
    if 70.0 <= ref <= 180.0 and pred > 180.0:
        return "D"

    # ── Zone C (over-correction of benign glucose) ────────────────────────
    # Reference hyperglycaemic; meter says normal → might over-treat benign hyper
    if ref > 180.0 and 70.0 <= pred <= 180.0:
        return "C"
    # Reference hypoglycaemic; meter says normal → might fail to treat hypo adequately
    if ref < 70.0 and 70.0 < pred <= 180.0:
        return "C"

    # ── Zone B (inaccurate but not dangerous) ─────────────────────────────
    return "B"

_assign_clarke_zone = assign_clarke_zone


def clarke_ega(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> dict[str, float]:
    """
    Compute Clarke Error Grid Analysis zone distribution.

    Args:
        y_true: reference glucose values (mg/dL)
        y_pred: predicted glucose values (mg/dL)

    Returns:
        Dict with keys: zone_A_pct, zone_B_pct, zone_C_pct, zone_D_pct, zone_E_pct,
        zone_A_n, zone_B_n, zone_C_n, zone_D_n, zone_E_n.
        Zone A+B% represents "clinically acceptable" predictions.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    zones = [_assign_clarke_zone(r, p) for r, p in zip(y_true, y_pred)]
    n = len(zones)
    counts = {z: zones.count(z) for z in "ABCDE"}

    return {
        **{f"zone_{z}_n":   counts[z]           for z in "ABCDE"},
        **{f"zone_{z}_pct": counts[z] / n * 100 for z in "ABCDE"},
        "clinically_acceptable_pct": (counts["A"] + counts["B"]) / n * 100,
    }


# ---------------------------------------------------------------------------
# Time-in-Range metrics (AGENTS.md Rule 4)
# ---------------------------------------------------------------------------


def time_in_range(glucose_mgdL: Sequence[float]) -> float:
    """Percentage of readings in [70, 180] mg/dL."""
    g = np.asarray(glucose_mgdL, dtype=float)
    return float(np.mean((g >= TIR_LOW_MGDL) & (g <= TIR_HIGH_MGDL)) * 100.0)


def time_below_range(glucose_mgdL: Sequence[float], threshold: float = HYPO_THRESHOLD_MGDL) -> float:
    """
    Percentage of readings below threshold mg/dL.
    PRIMARY safety metric (AGENTS.md Rule 4).
    """
    g = np.asarray(glucose_mgdL, dtype=float)
    return float(np.mean(g < threshold) * 100.0)


def time_above_range(glucose_mgdL: Sequence[float], threshold: float = TIR_HIGH_MGDL) -> float:
    """Percentage of readings above threshold mg/dL."""
    g = np.asarray(glucose_mgdL, dtype=float)
    return float(np.mean(g > threshold) * 100.0)


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root mean squared error (mg/dL)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute error (mg/dL)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def mard(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute relative difference (%)."""
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ref = np.maximum(yt, 1.0)
    return float(np.mean(np.abs(yt - yp) / ref) * 100.0)


# ---------------------------------------------------------------------------
# Unified metric bundle
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    label: str = "model",
) -> dict:
    """
    Compute the full required metric set for a glucose prediction/trajectory.

    AGENTS.md Rule 3: RMSE is always reported together with Clarke EGA and TIR.
    AGENTS.md Rule 4: TBR is the PRIMARY safety metric and is listed first.

    Args:
        y_true: reference CGM values (mg/dL)
        y_pred: predicted / controlled glucose values (mg/dL)
        label:  model name for display purposes

    Returns:
        Ordered dict with all metrics. Keys include:
          TBR_pct (primary), TIR_pct, TAR_pct,
          RMSE_mgdL, MAE_mgdL, MARD_pct,
          zone_A_pct … zone_E_pct, clinically_acceptable_pct.
    """
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)

    egz   = clarke_ega(yt, yp)
    tbr   = time_below_range(yp)   # safety metric over PREDICTED/controlled glucose
    tir   = time_in_range(yp)
    tar   = time_above_range(yp)

    assert abs(tbr + tir + tar - 100.0) < 0.5, (
        f"TBR + TIR + TAR = {tbr+tir+tar:.1f} ≠ 100 — check overlap in thresholds."
    )

    r_rmse = round(rmse(yt, yp), 3)
    r_mae  = round(mae(yt, yp), 3)
    r_mard = round(mard(yt, yp), 3)
    clarke_ab = egz.get("clinically_acceptable_pct", egz.get("zone_A_pct", 0) + egz.get("zone_B_pct", 0))

    return {
        "label":                      label,
        "TBR_pct":                    round(tbr, 2),
        "tbr_percent":                round(tbr, 2),
        "TIR_pct":                    round(tir, 2),
        "tir_percent":                round(tir, 2),
        "TAR_pct":                    round(tar, 2),
        "tar_percent":                round(tar, 2),
        "RMSE_mgdL":                  r_rmse,
        "rmse":                       r_rmse,
        "MAE_mgdL":                   r_mae,
        "mae":                        r_mae,
        "MARD_pct":                   r_mard,
        "mard_percent":               r_mard,
        "clarke_ab_percent":          round(clarke_ab, 2),
        **{k: round(v, 2) for k, v in egz.items()},
    }
