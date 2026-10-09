"""
Phase 1 pipeline tests: loader round-trip, 5-min alignment, split isolation, and unit conversion.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.loaders import SyntheticLoader, OHIOLoader
from src.data.preprocessor import Preprocessor
from src.data.splitter import PatientTimeSplitter
from src.utils.units import mmolL_to_mgdL, mgdL_to_mmolL

def test_unit_conversion():
    """Verify exact 90 mg/dL == 5.0 mmol/L unit conversion contract."""
    assert mmolL_to_mgdL(5.0) == 90.0
    assert mgdL_to_mmolL(90.0) == 5.0

def test_synthetic_loader_roundtrip():
    """Verify synthetic data round-trips through the loader and preprocessor."""
    loader = SyntheticLoader(seed=123)
    df_raw = loader.load_patient("patient_roundtrip", duration_hours=24)
    assert not df_raw.empty
    assert "glucose_mgdL" in df_raw.columns
    assert "insulin_mU_per_min" in df_raw.columns

    preprocessor = Preprocessor(dt_minutes=5.0)
    df_clean = preprocessor.transform(df_raw)
    assert not df_clean.empty
    assert "iob" in df_clean.columns
    assert "cob" in df_clean.columns
    assert "sin_time" in df_clean.columns

def test_grid_5min_alignment():
    """Verify resampled time grid is exactly 5-min aligned with zero gaps."""
    loader = SyntheticLoader(seed=42)
    df_raw = loader.load_patient("patient_grid", duration_hours=12)
    preprocessor = Preprocessor(dt_minutes=5.0)
    df_clean = preprocessor.transform(df_raw)

    timestamps = pd.to_datetime(df_clean["timestamp"])
    diffs = timestamps.diff().dropna()
    expected_step = pd.Timedelta(minutes=5)
    assert (diffs == expected_step).all()

def test_split_patient_isolation():
    """Verify split.json manifest never puts the same patient in train_patients and test_held_out_patients."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loader = SyntheticLoader(seed=42)
        p1 = loader.load_patient("p1", duration_hours=24)
        p1["patient_id"] = "p1"
        p2 = loader.load_patient("p2", duration_hours=24)
        p2["patient_id"] = "p2"
        p3 = loader.load_patient("p3", duration_hours=24)
        p3["patient_id"] = "p3"

        combined = pd.concat([p1, p2, p3], ignore_index=True)
        preprocessor = Preprocessor(dt_minutes=5.0)
        clean_df = preprocessor.transform(combined)

        splitter = PatientTimeSplitter(train_ratio=0.7, val_ratio=0.15)
        train_df, val_df, test_df, manifest = splitter.split(clean_df, output_dir=tmp_path)

        train_patients = set(manifest["patient_held_out_split"]["train_patients"])
        test_patients = set(manifest["patient_held_out_split"]["test_held_out_patients"])

        # Patient isolation assertion: no overlap
        assert train_patients.isdisjoint(test_patients)
        assert (tmp_path / "split.json").exists()
