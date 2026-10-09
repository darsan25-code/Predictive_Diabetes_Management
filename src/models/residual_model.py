"""
Phase 4: Hybrid Residual Model and Multi-Horizon Glucose Forecasting.

Architectures:
  1. ResidualGRU (Learned Residual Correction):
     Predicts r(t+H) = G_obs(t+H) - G_mech(t+H) from past history, IOB, COB,
     time-of-day sin/cos, and activity.
  2. PureMLModel (Pure Machine Learning Baseline):
     Identical GRU architecture predicting G_obs(t+H) directly with NO mechanistic input.
  3. HybridDigitalTwin (Composite Model):
     G_hybrid(t+H) = clip(G_mech(t+H) + r_pred(t+H), 20.0, 600.0).

Strict Split Policy (AGENTS.md Rule 1):
  Splits are ALWAYS patient-level + chronological (from split.json).
  No row from validation or test splits is ever seen during training.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mechanistic import DEFAULT_BERGMAN_PARAMS, simulate_bergman
from src.utils.units import GLUCOSE_MAX_MGDL, GLUCOSE_MIN_MGDL, HYPO_THRESHOLD_MGDL

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "glucose_mgdL",
    "iob",
    "cob",
    "sin_time",
    "cos_time",
    "activity",
    "insulin_mU_per_min",
    "meal_cho_g",
]

DEFAULT_HORIZONS_MIN = (30.0, 60.0, 120.0, 240.0)


# ---------------------------------------------------------------------------
# Neural Architectures
# ---------------------------------------------------------------------------


class ResidualGRU(nn.Module):
    """
    Recurrent Neural Network (GRU) predicting multi-horizon glucose residuals.

    Input shape:  (batch_size, seq_len, num_features)
    Output shape: (batch_size, num_horizons) -> residual corrections in mg/dL
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_horizons: int = len(DEFAULT_HORIZONS_MIN),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.gru(x)
        # Pool last hidden state
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


class PureMLModel(nn.Module):
    """
    Pure ML baseline with identical GRU architecture predicting future glucose directly.

    Input shape:  (batch_size, seq_len, num_features)
    Output shape: (batch_size, num_horizons) -> direct glucose forecast in mg/dL
    """

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_horizons: int = len(DEFAULT_HORIZONS_MIN),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_horizons = num_horizons

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_horizons),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Dataset Builder & Feature Pipeline
# ---------------------------------------------------------------------------


def extract_features_and_targets(
    df: pd.DataFrame,
    calibrated_params_map: Optional[Dict[str, Dict[str, float]]] = None,
    history_steps: int = 12,
    horizons_min: Sequence[float] = DEFAULT_HORIZONS_MIN,
    dt_min: float = 5.0,
) -> Dict[str, np.ndarray]:
    """
    Construct multi-horizon sliding windows from a preprocessed DataFrame.

    For each window at index i:
      - History: features over [i - history_steps + 1, ..., i]
      - Future horizons: steps h in [6, 12, 24, 48] (for 30, 60, 120, 240 min)
      - Mechanistic simulation: forward Bergman ODE from i to i + h
      - Residual target: G_obs(i+h) - G_mech(i+h)
      - Pure-ML target:  G_obs(i+h)
    """
    calibrated_params_map = calibrated_params_map or {}
    horizon_steps = [int(round(h / dt_min)) for h in horizons_min]
    max_h_step = max(horizon_steps)

    X_list = []
    y_res_list = []
    y_pure_list = []
    g_mech_list = []
    g_pers_list = []
    g_true_list = []

    for pid, group in df.groupby("patient_id", sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(group)
        if n_rows < history_steps + max_h_step:
            continue

        # Get calibrated or default Bergman parameters for this patient
        p_patient = dict(DEFAULT_BERGMAN_PARAMS)
        if str(pid) in calibrated_params_map:
            p_patient.update(calibrated_params_map[str(pid)])
        elif pid in calibrated_params_map:
            p_patient.update(calibrated_params_map[pid])

        # Prepare feature matrix
        for col in FEATURE_COLS:
            if col not in group.columns:
                group[col] = 0.0

        feat_vals = group[FEATURE_COLS].values.astype(np.float32)
        glucose_vals = group["glucose_mgdL"].values.astype(np.float32)
        insulin_vals = group["insulin_mU_per_min"].values.astype(np.float32)
        meal_vals = group["meal_cho_g"].values.astype(np.float32)
        # Extract patient parameter constants
        p1 = float(p_patient["p1"])
        p2 = float(p_patient["p2"])
        p3 = float(p_patient["p3"])
        n  = float(p_patient["n"])
        Gb = float(p_patient["Gb"])
        Ib = float(p_patient["Ib"])
        Vg = float(p_patient["Vg"])
        Vi = float(p_patient["Vi"])
        u_basal = n * Ib * Vi

        # Precompute 1-minute Ra timeline for entire patient trace
        n_mins = n_rows * 5
        t_mins = np.arange(n_mins, dtype=np.float32)
        Ra_trace_1min = np.zeros(n_mins, dtype=np.float32)
        k_meal, k_abs = 0.05, 0.02

        meal_step_indices = np.where(meal_vals > 0)[0]
        for m_step in meal_step_indices:
            cho = meal_vals[m_step]
            t_meal_min = m_step * 5.0
            mask = t_mins >= t_meal_min
            dt_m = t_mins[mask] - t_meal_min
            Ra_trace_1min[mask] += (k_meal * cho * 1000.0 * k_abs * np.exp(-k_abs * dt_m)).astype(np.float32)

        # Sliding window
        for i in range(history_steps - 1, n_rows - max_h_step):
            # 1. Past feature window
            x_seq = feat_vals[i - history_steps + 1 : i + 1]  # shape: (history_steps, n_feats)

            # 2. Current observed glucose for persistence baseline
            g_curr = float(glucose_vals[i])

            # 3. Fast mechanistic forward simulation across max_h_step
            # Sub-step dt = 1.0 min for high numerical stability and speed
            G_sim_h = {}
            G_curr, X_curr, I_curr = g_curr, 0.0, Ib

            # Sub-step simulation forward
            for step in range(max_h_step):
                u_val = float(insulin_vals[i + step])

                # 5 sub-steps of 1 minute each per 5-min cadence
                for sub in range(5):
                    min_idx = (i + step) * 5 + sub
                    Ra_val = float(Ra_trace_1min[min_idx]) if min_idx < n_mins else 0.0

                    dG = -(p1 + X_curr) * G_curr + p1 * Gb + Ra_val / Vg
                    dX = -p2 * X_curr + p3 * (I_curr - Ib)
                    dI = -n * (I_curr - Ib) + (u_val - u_basal) / Vi

                    G_curr += 1.0 * dG
                    X_curr += 1.0 * dX
                    I_curr += 1.0 * dI

                # Record at step + 1 (cadence step)
                cur_h_step = step + 1
                if cur_h_step in horizon_steps:
                    G_sim_h[cur_h_step] = np.clip(G_curr, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)

            y_res_h = []
            y_pure_h = []
            g_mech_h = []
            g_pers_h = []
            g_true_h = []

            for h_step in horizon_steps:
                target_idx = i + h_step
                g_true = float(glucose_vals[target_idx])
                g_mech = float(G_sim_h.get(h_step, g_curr))

                r_true = float(g_true - g_mech)

                y_res_h.append(r_true)
                y_pure_h.append(g_true)
                g_mech_h.append(g_mech)
                g_pers_h.append(g_curr)
                g_true_h.append(g_true)

            X_list.append(x_seq)
            y_res_list.append(y_res_h)
            y_pure_list.append(y_pure_h)
            g_mech_list.append(g_mech_h)
            g_pers_list.append(g_pers_h)
            g_true_list.append(g_true_h)

    if not X_list:
        empty = np.empty((0, history_steps, len(FEATURE_COLS)), dtype=np.float32)
        empty_h = np.empty((0, len(horizon_steps)), dtype=np.float32)
        return {
            "X": empty,
            "y_res": empty_h,
            "y_pure": empty_h,
            "g_mech": empty_h,
            "g_pers": empty_h,
            "g_true": empty_h,
        }

    return {
        "X": np.array(X_list, dtype=np.float32),
        "y_res": np.array(y_res_list, dtype=np.float32),
        "y_pure": np.array(y_pure_list, dtype=np.float32),
        "g_mech": np.array(g_mech_list, dtype=np.float32),
        "g_pers": np.array(g_pers_list, dtype=np.float32),
        "g_true": np.array(g_true_list, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------


def weighted_hypo_mse_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    g_true: torch.Tensor,
    hypo_weight: float = 3.0,
) -> torch.Tensor:
    """
    Weighted MSE loss placing higher penalty on predictions when true glucose is in hypoglycaemia.
    """
    weights = torch.ones_like(y_target)
    hypo_mask = g_true < HYPO_THRESHOLD_MGDL
    weights[hypo_mask] = hypo_weight
    loss = (weights * (y_pred - y_target) ** 2).mean()
    return loss


def train_residual_and_pure_models(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    hypo_weight: float = 3.0,
    patience: int = 10,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[ResidualGRU, PureMLModel, Dict[str, Any]]:
    """
    Train both ResidualGRU and PureMLModel models using strictly isolated train and val sets.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    # Convert training tensors
    X_tr = torch.tensor(train_data["X"], dtype=torch.float32)
    y_res_tr = torch.tensor(train_data["y_res"], dtype=torch.float32)
    y_pure_tr = torch.tensor(train_data["y_pure"], dtype=torch.float32)
    g_true_tr = torch.tensor(train_data["g_true"], dtype=torch.float32)

    # Convert validation tensors
    X_va = torch.tensor(val_data["X"], dtype=torch.float32, device=dev)
    y_res_va = torch.tensor(val_data["y_res"], dtype=torch.float32, device=dev)
    y_pure_va = torch.tensor(val_data["y_pure"], dtype=torch.float32, device=dev)
    g_true_va = torch.tensor(val_data["g_true"], dtype=torch.float32, device=dev)

    # Dataloaders
    dataset_res = TensorDataset(X_tr, y_res_tr, g_true_tr)
    dataset_pure = TensorDataset(X_tr, y_pure_tr, g_true_tr)

    loader_res = DataLoader(dataset_res, batch_size=batch_size, shuffle=True)
    loader_pure = DataLoader(dataset_pure, batch_size=batch_size, shuffle=True)

    residual_model = ResidualGRU(input_dim=len(FEATURE_COLS)).to(dev)
    pure_model = PureMLModel(input_dim=len(FEATURE_COLS)).to(dev)

    opt_res = torch.optim.Adam(residual_model.parameters(), lr=lr, weight_decay=1e-4)
    opt_pure = torch.optim.Adam(pure_model.parameters(), lr=lr, weight_decay=1e-4)

    # 1. Train Residual Model
    best_loss_res = float("inf")
    best_state_res = None
    no_imp_res = 0

    for epoch in range(n_epochs):
        residual_model.train()
        for b_x, b_y, b_gt in loader_res:
            b_x, b_y, b_gt = b_x.to(dev), b_y.to(dev), b_gt.to(dev)
            opt_res.zero_grad()
            preds = residual_model(b_x)
            loss = weighted_hypo_mse_loss(preds, b_y, b_gt, hypo_weight=hypo_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(residual_model.parameters(), 1.0)
            opt_res.step()

        residual_model.eval()
        with torch.no_grad():
            va_preds = residual_model(X_va)
            va_loss = weighted_hypo_mse_loss(va_preds, y_res_va, g_true_va, hypo_weight=hypo_weight).item()

        if va_loss < best_loss_res:
            best_loss_res = va_loss
            best_state_res = {k: v.clone() for k, v in residual_model.state_dict().items()}
            no_imp_res = 0
        else:
            no_imp_res += 1

        if no_imp_res >= patience:
            break

    if best_state_res is not None:
        residual_model.load_state_dict(best_state_res)
    residual_model.eval()

    # 2. Train Pure ML Model
    best_loss_pure = float("inf")
    best_state_pure = None
    no_imp_pure = 0

    for epoch in range(n_epochs):
        pure_model.train()
        for b_x, b_y, b_gt in loader_pure:
            b_x, b_y, b_gt = b_x.to(dev), b_y.to(dev), b_gt.to(dev)
            opt_pure.zero_grad()
            preds = pure_model(b_x)
            loss = weighted_hypo_mse_loss(preds, b_y, b_gt, hypo_weight=hypo_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(pure_model.parameters(), 1.0)
            opt_pure.step()

        pure_model.eval()
        with torch.no_grad():
            va_preds = pure_model(X_va)
            va_loss = weighted_hypo_mse_loss(va_preds, y_pure_va, g_true_va, hypo_weight=hypo_weight).item()

        if va_loss < best_loss_pure:
            best_loss_pure = va_loss
            best_state_pure = {k: v.clone() for k, v in pure_model.state_dict().items()}
            no_imp_pure = 0
        else:
            no_imp_pure += 1

        if no_imp_pure >= patience:
            break

    if best_state_pure is not None:
        pure_model.load_state_dict(best_state_pure)
    pure_model.eval()

    history = {
        "residual_val_loss": best_loss_res,
        "pure_ml_val_loss": best_loss_pure,
    }
    return residual_model, pure_model, history


# ---------------------------------------------------------------------------
# Hybrid Digital Twin Wrapper
# ---------------------------------------------------------------------------


class HybridDigitalTwin:
    """
    Composite Hybrid Model:
      G_hybrid(t+H) = clip(G_mech(t+H) + r_pred(t+H), 20.0, 600.0)

    Guarantees strict physiological bounding, zero negative glucose values,
    and fallback to mechanistic predictions when residual is unavailable.
    """

    def __init__(
        self,
        residual_model: Optional[ResidualGRU] = None,
        calibrated_params_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        self.residual_model = residual_model
        self.calibrated_params_map = calibrated_params_map or {}

    def predict(
        self,
        X_seq: Union[np.ndarray, torch.Tensor],
        g_mech: np.ndarray,
        device: str = "cpu",
    ) -> np.ndarray:
        """
        Produce hybrid glucose forecast.

        Args:
            X_seq:  History tensor of shape (batch, seq_len, num_features).
            g_mech: Mechanistic forecast of shape (batch, num_horizons).

        Returns:
            g_hybrid: Array of shape (batch, num_horizons) clipped to [20, 600] mg/dL.
        """
        g_mech = np.asarray(g_mech, dtype=np.float32)

        if self.residual_model is None:
            return np.clip(g_mech, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)

        self.residual_model.eval()
        with torch.no_grad():
            if isinstance(X_seq, np.ndarray):
                X_tensor = torch.tensor(X_seq, dtype=torch.float32, device=device)
            else:
                X_tensor = X_seq.to(device)

            r_pred = self.residual_model(X_tensor).cpu().numpy()

        g_hybrid = g_mech + r_pred
        return np.clip(g_hybrid, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)


# ---------------------------------------------------------------------------
# Multi-Horizon Evaluation Across All 4 Models
# ---------------------------------------------------------------------------


def evaluate_phase4_horizons(
    test_data: Dict[str, np.ndarray],
    residual_model: ResidualGRU,
    pure_model: PureMLModel,
    horizons_min: Sequence[float] = DEFAULT_HORIZONS_MIN,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate Persistence, Mechanistic-Only, Pure-ML, and Hybrid models across all horizons.

    Returns:
        Nested dict with full metric sets for each model and each horizon.
    """
    from src.eval.metrics import compute_metrics

    X_te = test_data["X"]
    g_true_all = test_data["g_true"]
    g_pers_all = test_data["g_pers"]
    g_mech_all = test_data["g_mech"]

    # Generate model predictions
    residual_model.eval()
    pure_model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_te, dtype=torch.float32, device=device)
        r_pred = residual_model(X_t).cpu().numpy()
        g_pure_raw = pure_model(X_t).cpu().numpy()

    g_pure_all = np.clip(g_pure_raw, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)
    g_hyb_all = np.clip(g_mech_all + r_pred, GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL)

    results: Dict[str, Dict[str, Any]] = {
        "persistence": {},
        "mechanistic_only": {},
        "pure_ml": {},
        "hybrid": {},
    }

    for h_idx, h_min in enumerate(horizons_min):
        h_str = f"{int(h_min)}min"
        y_true = g_true_all[:, h_idx]

        # 1. Persistence
        y_pers = g_pers_all[:, h_idx]
        m_pers = compute_metrics(y_true, y_pers, label=f"Persistence ({h_str})")
        results["persistence"][h_str] = m_pers

        # 2. Mechanistic Only
        y_mech = g_mech_all[:, h_idx]
        m_mech = compute_metrics(y_true, y_mech, label=f"Mechanistic-Only ({h_str})")
        results["mechanistic_only"][h_str] = m_mech

        # 3. Pure-ML
        y_pure = g_pure_all[:, h_idx]
        m_pure = compute_metrics(y_true, y_pure, label=f"Pure-ML ({h_str})")
        results["pure_ml"][h_str] = m_pure

        # 4. Hybrid
        y_hyb = g_hyb_all[:, h_idx]
        m_hyb = compute_metrics(y_true, y_hyb, label=f"Hybrid ({h_str})")
        results["hybrid"][h_str] = m_hyb

    return results


# Compatibility alias
DigitalTwinModel = HybridDigitalTwin
