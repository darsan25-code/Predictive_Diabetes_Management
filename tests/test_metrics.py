"""
Tests for evaluation metrics: Clarke EGA, TIR/TBR/TAR, RMSE, MAE, MARD.
"""
import pytest
import numpy as np

from src.eval.metrics import compute_metrics, assign_clarke_zone

def test_clarke_zone_assignment():
    assert assign_clarke_zone(100, 100) == "A"
    assert assign_clarke_zone(100, 110) == "A"
    assert assign_clarke_zone(50, 150) in ["C", "D", "E"]
    assert assign_clarke_zone(200, 50) == "E"

def test_compute_metrics_bundle(sample_glucose_series):
    g_true, g_pred = sample_glucose_series
    m = compute_metrics(g_true, g_pred)

    assert "rmse" in m
    assert "mae" in m
    assert "mard_percent" in m
    assert "tir_percent" in m
    assert "tbr_percent" in m
    assert "tar_percent" in m
    assert "clarke_ab_percent" in m

    # Validate sum of TIR, TBR, TAR is approximately 100%
    total_range_pct = m["tir_percent"] + m["tbr_percent"] + m["tar_percent"]
    assert pytest.approx(total_range_pct, 0.1) == 100.0
