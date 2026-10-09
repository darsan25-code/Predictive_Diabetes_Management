"""
Smoke tests for local results viewer (app.py).
"""
import pytest
from pathlib import Path
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_app_config_loading():
    """Verify that configuration can be parsed properly from YAML."""
    cfg_path = REPO_ROOT / "configs" / "data_default.yaml"
    assert cfg_path.exists(), "configs/data_default.yaml must exist"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)
    assert "loader" in cfg


def test_app_artifacts_presence():
    """Verify that critical status and experiment artifacts exist for viewer display."""
    status_file = REPO_ROOT / "STATUS.md"
    assert status_file.exists(), "STATUS.md must exist for Overview page"

    exp_dir = REPO_ROOT / "experiments"
    assert exp_dir.exists(), "experiments/ directory must exist"

    # Verify at least phase plots exist
    required_plots = [
        "phase2_fit.png",
        "phase3_identifiability.png",
        "phase4_horizons.png",
        "phase5_kalman.png",
        "phase6_cohort_match.png",
        "phase7_mpc.png",
    ]
    for p in required_plots:
        plot_file = exp_dir / p
        assert plot_file.exists(), f"Plot {p} missing from experiments/"


def test_app_processed_data_schema():
    """Verify processed patient parquet files conform to expected schema."""
    processed_dir = REPO_ROOT / "data" / "processed"
    assert processed_dir.exists()
    p_file = processed_dir / "patient_synthetic_000.parquet"
    if p_file.exists():
        df = pd.read_parquet(p_file)
        assert "timestamp" in df.columns
        assert "glucose_mgdL" in df.columns
        assert df["glucose_mgdL"].min() >= 20.0
