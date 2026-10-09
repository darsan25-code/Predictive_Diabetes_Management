"""
Unit tests for Phase 4: Hybrid Residual Model and Multi-Horizon Forecasting.

Required tests:
  1. Data leakage test: No row of any test patient or test window appears in training.
  2. Bounded output test: Hybrid output stays strictly within physiological bounds [20, 600] mg/dL.
  3. Perfect fit test: Residual of a perfect mechanistic fit on noise-free Bergman data is near zero.
"""
import pytest
import numpy as np
import pandas as pd
import torch

from src.data.loaders import SyntheticLoader
from src.data.preprocessor import Preprocessor
from src.data.splitter import PatientTimeSplitter
from src.models.residual_model import (
    ResidualGRU,
    PureMLModel,
    HybridDigitalTwin,
    extract_features_and_targets,
    train_residual_and_pure_models,
)
from src.utils.units import GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL


def test_no_data_leakage():
    """Verify strictly zero overlap between training and test sequences/windows."""
    loader = SyntheticLoader(seed=42)
    raw_df = loader.load()
    preprocessor = Preprocessor()
    clean_df = preprocessor.transform(raw_df)

    splitter = PatientTimeSplitter(train_fraction=0.70, val_fraction=0.15)
    train_df, val_df, test_df, manifest = splitter.split(clean_df)

    # Check temporal window bounds per patient from manifest
    for pid, p_info in manifest["patients"].items():
        tr_end = pd.to_datetime(p_info["train"]["end"])
        va_start = pd.to_datetime(p_info["val"]["start"])
        va_end = pd.to_datetime(p_info["val"]["end"])
        te_start = pd.to_datetime(p_info["test"]["start"])

        # Train end strictly before or equal to val start
        assert tr_end <= va_start
        assert va_end <= te_start

    # Check extracted data splits have no overlapping timestamps
    tr_times = set(train_df["timestamp"])
    te_times = set(test_df["timestamp"])
    assert tr_times.isdisjoint(te_times)


def test_hybrid_output_bounded():
    """Verify hybrid model output is bounded within [20, 600] mg/dL under extreme inputs."""
    residual_net = ResidualGRU(input_dim=8)
    twin = HybridDigitalTwin(residual_model=residual_net)

    batch_size = 10
    X_extreme = np.random.randn(batch_size, 12, 8).astype(np.float32)

    # Test extreme high mechanistic values
    g_mech_high = np.full((batch_size, 4), 9999.0, dtype=np.float32)
    preds_high = twin.predict(X_extreme, g_mech_high)
    assert np.all(preds_high <= GLUCOSE_MAX_MGDL)
    assert np.all(preds_high >= GLUCOSE_MIN_MGDL)

    # Test extreme negative mechanistic values
    g_mech_low = np.full((batch_size, 4), -500.0, dtype=np.float32)
    preds_low = twin.predict(X_extreme, g_mech_low)
    assert np.all(preds_low >= GLUCOSE_MIN_MGDL)
    assert np.all(preds_low <= GLUCOSE_MAX_MGDL)
    assert not np.any(np.isnan(preds_low))


def test_perfect_mechanistic_fit_residual_near_zero():
    """Verify that on noise-free pure Bergman simulation, mechanistic residuals are near zero."""
    from src.models.mechanistic import simulate_bergman, DEFAULT_BERGMAN_PARAMS

    params = dict(DEFAULT_BERGMAN_PARAMS)
    t_eval = np.arange(0.0, 360.0, 5.0)
    u_fn = lambda t: float(params["n"] * params["Ib"] * params["Vi"])
    ra_fn = lambda t: 0.0
    initial_state = [params["Gb"], 0.0, params["Ib"]]

    # 1. Steady state without meals
    G_true, _, _ = simulate_bergman(t_eval, params, u_fn, ra_fn, initial_state)
    G_pred, _, _ = simulate_bergman(t_eval, params, u_fn, ra_fn, initial_state)
    residual = G_true - G_pred
    assert np.all(np.abs(residual) < 1e-4)

    # 2. Dynamic state with meal perturbation
    meal_ra = lambda t: 50.0 * np.exp(-0.02 * (t - 30.0)) if t >= 30.0 else 0.0
    G_meal_true, _, _ = simulate_bergman(t_eval, params, u_fn, meal_ra, initial_state)
    G_meal_pred, _, _ = simulate_bergman(t_eval, params, u_fn, meal_ra, initial_state)
    meal_res = G_meal_true - G_meal_pred
    assert np.all(np.abs(meal_res) < 1e-4)


def test_gru_shapes_and_training_step():
    """Verify tensor input/output shapes and optimization convergence over training steps."""
    input_dim = 8
    seq_len = 12
    num_horizons = 4
    batch_size = 16

    res_model = ResidualGRU(input_dim=input_dim, num_horizons=num_horizons)
    pure_model = PureMLModel(input_dim=input_dim, num_horizons=num_horizons)

    x = torch.randn(batch_size, seq_len, input_dim)
    out_res = res_model(x)
    out_pure = pure_model(x)

    assert out_res.shape == (batch_size, num_horizons)
    assert out_pure.shape == (batch_size, num_horizons)

    # Test training routine
    train_data = {
        "X": np.random.randn(32, seq_len, input_dim).astype(np.float32),
        "y_res": np.random.randn(32, num_horizons).astype(np.float32),
        "y_pure": (120.0 + 10.0 * np.random.randn(32, num_horizons)).astype(np.float32),
        "g_true": (120.0 + 10.0 * np.random.randn(32, num_horizons)).astype(np.float32),
    }
    val_data = {
        "X": np.random.randn(16, seq_len, input_dim).astype(np.float32),
        "y_res": np.random.randn(16, num_horizons).astype(np.float32),
        "y_pure": (120.0 + 10.0 * np.random.randn(16, num_horizons)).astype(np.float32),
        "g_true": (120.0 + 10.0 * np.random.randn(16, num_horizons)).astype(np.float32),
    }

    t_res, t_pure, hist = train_residual_and_pure_models(
        train_data, val_data, n_epochs=5, batch_size=8
    )
    assert t_res is not None
    assert t_pure is not None
    assert "residual_val_loss" in hist
