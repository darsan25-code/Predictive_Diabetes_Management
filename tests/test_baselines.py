"""
Tests for Persistence and Mechanistic baseline models.
"""
import pytest
import numpy as np

from src.models.baselines import PersistenceBaseline, MechanisticOnlyBaseline

def test_persistence_baseline():
    baseline = PersistenceBaseline()
    history = [120.0, 125.0, 130.0]
    preds = baseline.predict(history, horizon_steps=6)
    assert len(preds) == 6
    assert np.all(np.array(preds) == 130.0)

def test_mechanistic_baseline():
    baseline = MechanisticOnlyBaseline()
    history_G = [140.0, 142.0, 145.0]
    basal = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    bolus = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    preds = baseline.predict(history_G, basal, bolus, horizon_steps=6)
    assert len(preds) == 6
    assert np.all(np.isfinite(preds))
