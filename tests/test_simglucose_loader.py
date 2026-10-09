"""
Unit tests for Phase 6: SimGlucose Loader and UVA/Padova Benchmark.
"""
import pytest
import numpy as np
import pandas as pd

from src.data.loaders import SimGlucoseLoader
from src.eval.metrics import time_in_range, time_below_range


def test_simglucose_loader_schema_and_units():
    """Verify simglucose loader returns the standard schema with explicit units."""
    loader = SimGlucoseLoader(seed=42)
    # Test loading a short 6-hour trace
    df = loader.load_patient("adult#001", duration_hours=6.0, seed=42)

    assert not df.empty
    assert "patient_id" in df.columns
    assert "timestamp" in df.columns
    assert "glucose_mgdL" in df.columns
    assert "insulin_mU_per_min" in df.columns
    assert "meal_cho_g" in df.columns

    # Explicit unit validation
    assert df["glucose_mgdL"].min() >= 20.0
    assert df["glucose_mgdL"].max() <= 600.0
    assert df["insulin_mU_per_min"].min() >= 0.0
    assert df["meal_cho_g"].min() >= 0.0


def test_simglucose_loader_reproducibility():
    """Verify that identical random seeds produce identical simulation traces."""
    loader1 = SimGlucoseLoader(seed=123)
    df1 = loader1.load_patient("adolescent#001", duration_hours=4.0, seed=123)

    loader2 = SimGlucoseLoader(seed=123)
    df2 = loader2.load_patient("adolescent#001", duration_hours=4.0, seed=123)

    assert np.allclose(df1["glucose_mgdL"].values, df2["glucose_mgdL"].values, atol=1e-5)
    assert np.allclose(df1["insulin_mU_per_min"].values, df2["insulin_mU_per_min"].values, atol=1e-5)


def test_simglucose_cohorts_and_metrics():
    """Verify loading across adult, adolescent, and child cohorts and metric computations."""
    loader = SimGlucoseLoader(seed=42)
    for p_name in ["adult#001", "adolescent#001", "child#001"]:
        df = loader.load_patient(p_name, duration_hours=4.0, seed=42)
        assert len(df) > 0
        g = df["glucose_mgdL"].values
        tir = time_in_range(g)
        tbr = time_below_range(g)
        assert 0.0 <= tir <= 100.0
        assert 0.0 <= tbr <= 100.0
