"""
Pytest configuration and shared fixtures.
"""
import pytest
import pandas as pd
import numpy as np

from src.data.loaders import SyntheticLoader
from src.data.preprocessor import Preprocessor

@pytest.fixture
def synthetic_cgm_data():
    loader = SyntheticLoader(seed=42)
    df = loader.load_patient("patient_01", duration_hours=24)
    preprocessor = Preprocessor()
    df_clean = preprocessor.transform(df)
    return df_clean

@pytest.fixture
def sample_glucose_series():
    t = np.arange(0, 100 * 5, 5)
    g_true = 120 + 40 * np.sin(t / 50)
    g_pred = g_true + np.random.normal(0, 5, size=len(g_true))
    return g_true, g_pred
