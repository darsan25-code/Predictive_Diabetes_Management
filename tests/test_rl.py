"""
Automated unit tests for Phase 8: Gymnasium GlucoseEnv and RLController.
"""
import pytest
import numpy as np

from src.control.rl_agent import GlucoseEnv, RLController
from src.control.safety_shield import SafetyShield
from src.models.bergman import BergmanModel


def test_glucose_gym_env_lifecycle():
    """Verify Gymnasium environment reset, step, shapes, and reward calculation."""
    bm = BergmanModel()
    shield = SafetyShield(bergman_model=bm)
    env = GlucoseEnv(bergman_model=bm, safety_shield=shield, episode_steps=20)

    obs, info = env.reset(seed=42)
    assert len(obs) == 8
    assert np.all(np.isfinite(obs))

    action = np.array([15.0], dtype=np.float32)
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert len(next_obs) == 8
    assert isinstance(reward, float)
    assert "glucose_true" in info
    assert "reward_components" in info
    assert "shield_intervened" in info


def test_rl_controller_train_and_evaluate():
    """Verify RLController can train and evaluate closed-loop trajectories."""
    controller = RLController()
    trained = controller.train(total_timesteps=100)

    env = GlucoseEnv(episode_steps=12, randomize_twin=False)
    eval_res = controller.evaluate_episode(env, deterministic=True)

    assert "tir" in eval_res
    assert "tbr" in eval_res
    assert len(eval_res["glucose_traj"]) == 12
