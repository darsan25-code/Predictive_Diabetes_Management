"""
Tests for data loading, preprocessing, and patient+time splitting.
"""
import pytest
import tempfile
from pathlib import Path

from src.data.loaders import SyntheticLoader, OHIOLoader
from src.data.preprocessor import Preprocessor
from src.data.splitter import PatientTimeSplitter

def test_synthetic_loader():
    loader = SyntheticLoader(seed=123)
    df = loader.load_patient("patient_test", duration_hours=12)
    assert not df.empty
    assert "glucose_mgdL" in df.columns
    assert "insulin_mU_per_min" in df.columns
    assert "meal_cho_g" in df.columns

def test_preprocessor_unit_guard():
    preprocessor = Preprocessor()
    loader = SyntheticLoader(seed=42)
    df = loader.load_patient("patient_01", duration_hours=6)
    
    # Introduce out-of-bound values
    df.iloc[0, df.columns.get_loc("glucose_mgdL")] = 10.0
    df.iloc[1, df.columns.get_loc("glucose_mgdL")] = 700.0

    df_clean = preprocessor.transform(df)
    assert df_clean["glucose_mgdL"].min() >= 20.0
    assert df_clean["glucose_mgdL"].max() <= 600.0

def test_splitter_write_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        loader = SyntheticLoader(seed=42)
        df1 = loader.load_patient("p1", duration_hours=24)
        df1["patient_id"] = "p1"
        df2 = loader.load_patient("p2", duration_hours=24)
        df2["patient_id"] = "p2"
        
        import pandas as pd
        combined = pd.concat([df1, df2], ignore_index=True)
        
        splitter = PatientTimeSplitter(train_ratio=0.7, val_ratio=0.15)
        train_df, val_df, test_df, split_info = splitter.split(combined, output_dir=tmp_path)
        
        assert (tmp_path / "split.json").exists()
        assert not train_df.empty
        assert not test_df.empty
        assert "p1" in split_info["patients"] or "p2" in split_info["patients"]
