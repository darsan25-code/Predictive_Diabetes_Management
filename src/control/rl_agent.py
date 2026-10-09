"""
Phase 8: Reinforcement Learning Glucose Regulation with a Hard Safety Shield.

Uses Stable-Baselines3 (PPO / SAC) and Gymnasium.
Integrates:
- Validated mechanistic digital twin dynamics.
- Continuous glucose monitoring with sensor noise.
- Extended Kalman Filter (Phase 5) for state estimation.
- Mandatory unconditional Safety Shield (Phase 8).
- Multi-component reward function balancing TIR (70-180 mg/dL), hypoglycemia prevention, and insulin variability.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Sequence
import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False
    logger.warning("gymnasium not installed — falling back to base environment class.")

try:
    from stable_baselines3 import PPO, SAC
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False
    logger.warning("stable-baselines3 not installed — RLController will use heuristic fallback.")

from src.models.bergman import BergmanModel, DEFAULT_BERGMAN_PARAMS
from src.control.safety_shield import SafetyShield
from src.estimation.kalman import ExtendedKalmanFilter

_ParentEnv = gymnasium.Env if _GYM_AVAILABLE else object


class GlucoseEnv(_ParentEnv):
    """
    Gymnasium-compatible environment for closed-loop glucose regulation.
    """
    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        bergman_model=None,
        safety_shield=None,
        ekf=None,
        cfg: dict | None = None,
        episode_steps: int = 288,
        dt_minutes: float = 5.0,
        randomize_twin: bool = True,
        seed: int = 42,
    ) -> None:
        if _GYM_AVAILABLE:
            super().__init__()
        cfg = cfg or {}
        self.cfg = cfg
        self.dt = float(cfg.get("dt_minutes", dt_minutes))
        self.episode_steps = int(cfg.get("episode_length_steps", episode_steps))
        self.randomize_twin = randomize_twin
        self.rng = np.random.default_rng(seed)

        self.bergman = bergman_model if bergman_model is not None else BergmanModel()
        self.shield = safety_shield if safety_shield is not None else SafetyShield(bergman_model=self.bergman)
        self.ekf = ekf if ekf is not None else ExtendedKalmanFilter(self.bergman)

        # Action: continuous insulin infusion rate (mU/min) in [0, 50.0]
        self.u_max = float(cfg.get("constraints", {}).get("u_max_mU_per_min", 50.0))
        if _GYM_AVAILABLE:
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([self.u_max], dtype=np.float32),
                dtype=np.float32,
            )
            # Observation: [G_cgm, G_est, X_est, I_est, IOB, u_prev, sin_time, cos_time] (dim 8)
            self.observation_space = spaces.Box(
                low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([600.0, 600.0, 1.0, 200.0, 20.0, self.u_max, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )

        self.step_count = 0
        self.current_state = np.array([120.0, 0.0, 10.0], dtype=float)
        self.u_prev = 15.0
        self.iob = 0.0
        self.meal_schedule = []

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.shield.reset_episode()

        # Randomize twin parameters within ±15% if configured
        if self.randomize_twin:
            p_twin = dict(DEFAULT_BERGMAN_PARAMS)
            for k in ("p1", "p2", "p3", "n"):
                p_twin[k] *= float(self.rng.uniform(0.85, 1.15))
            p_twin["Gb"] = float(self.rng.uniform(85.0, 115.0))
            p_twin["Ib"] = float(self.rng.uniform(8.0, 12.0))
            self.bergman = BergmanModel({"parameters": p_twin})
            self.shield.bergman = self.bergman
            self.ekf = ExtendedKalmanFilter(self.bergman)

        g0 = float(self.rng.uniform(100.0, 150.0))
        self.current_state = np.array([g0, 0.0, float(self.bergman.Ib)], dtype=float)
        self.u_prev = float(self.bergman.steady_state_basal_insulin())
        self.iob = 0.0

        # Generate 2-3 random meals across 24h (aligned to 5-minute ticks)
        self.meal_schedule = [
            (float(round(self.rng.uniform(300, 420) / self.dt) * self.dt), float(self.rng.uniform(40, 70))),
            (float(round(self.rng.uniform(660, 780) / self.dt) * self.dt), float(self.rng.uniform(50, 80))),
            (float(round(self.rng.uniform(1020, 1140) / self.dt) * self.dt), float(self.rng.uniform(30, 60))),
        ]

        self.ekf.reset(G0=g0)
        cgm_obs = g0 + float(self.rng.normal(0.0, 3.0))
        obs = self._get_observation(cgm_obs)
        return obs, {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        t_curr = self.step_count * self.dt

        u_raw = float(action[0]) if hasattr(action, "__len__") else float(action)
        u_raw = float(np.clip(u_raw, 0.0, self.u_max))

        # 1. HARD SAFETY SHIELD INTERCEPTION (Mandatory Invariant)
        u_safe, intervened = self.shield(u_raw, self.current_state, iob_units=self.iob)

        # 2. Compute meal glucose appearance Ra(t) with 85% systemic bioavailability
        ra = 0.0
        k_meal, k_abs = 0.85, 0.02
        for tm, ch in self.meal_schedule:
            if t_curr >= tm:
                ra += k_meal * ch * 1000.0 * k_abs * np.exp(-k_abs * (t_curr - tm))

        # 3. Simulate true physiological state transition
        u_fn = lambda t: u_safe
        ra_fn = lambda t: ra
        _, G_sim, X_sim, I_sim = self.bergman.simulate(
            (0.0, self.dt),
            self.current_state,
            u_fn=u_fn,
            ra_fn=ra_fn,
            t_eval=np.array([self.dt]),
        )
        self.current_state = np.array([float(G_sim[-1]), float(X_sim[-1]), float(I_sim[-1])])

        # 4. Generate noisy CGM measurement
        g_true = self.current_state[0]
        cgm_noise = float(self.rng.normal(0.0, 3.0))
        cgm_meas = float(np.clip(g_true + cgm_noise, 20.0, 600.0))

        # 5. Update Phase 5 EKF state estimation
        x_est, _, _, _ = self.ekf.step(cgm_mgdL=cgm_meas, u_mU_per_min=u_safe, dt_min=self.dt)

        # Update IOB (simple linear decay over 180 min)
        self.iob = max(0.0, self.iob * 0.97 + (u_safe / 60000.0))

        # 6. Reward computation
        reward, r_components = self._compute_reward(g_true, u_safe, self.u_prev, intervened)
        self.u_prev = u_safe

        terminated = self.step_count >= self.episode_steps
        truncated = False

        info = {
            "glucose_true": g_true,
            "glucose_cgm": cgm_meas,
            "insulin_raw": u_raw,
            "insulin_executed": u_safe,
            "shield_intervened": intervened,
            "is_hypo": g_true < 70.0,
            "is_in_range": 70.0 <= g_true <= 180.0,
            "reward_components": r_components,
        }

        obs = self._get_observation(cgm_meas)
        return obs, float(reward), terminated, truncated, info

    def _get_observation(self, cgm_meas: float) -> np.ndarray:
        t_min = self.step_count * self.dt
        sin_t = float(np.sin(2 * np.pi * t_min / 1440.0))
        cos_t = float(np.cos(2 * np.pi * t_min / 1440.0))
        x_est = self.ekf.state
        obs = np.array([
            cgm_meas,
            float(x_est[0]),
            float(x_est[1]),
            float(x_est[2]),
            float(self.iob),
            float(self.u_prev),
            sin_t,
            cos_t,
        ], dtype=np.float32)
        return obs

    def _compute_reward(self, g: float, u: float, u_prev: float, intervened: bool) -> tuple[float, dict[str, float]]:
        # 1. TIR reward (70 - 180 mg/dL)
        if 70.0 <= g <= 180.0:
            r_tir = 1.0 - (abs(g - 110.0) / 70.0) * 0.4
        else:
            r_tir = 0.0

        # 2. Severe hypoglycemia penalty (exponential below 70)
        if g < 70.0:
            r_hypo = -10.0 - 0.5 * (70.0 - g)
        else:
            r_hypo = 0.0

        # 3. Hyperglycemia penalty
        if g > 180.0:
            r_hyper = -0.5 * min((g - 180.0) / 50.0, 5.0)
        else:
            r_hyper = 0.0

        # 4. Insulin variability penalty
        r_du = -0.01 * min((u - u_prev) ** 2, 100.0)

        # 5. Shield feedback
        r_shield = -5.0 if intervened else 0.0

        total_r = r_tir + r_hypo + r_hyper + r_du + r_shield
        components = {
            "r_tir": r_tir,
            "r_hypo": r_hypo,
            "r_hyper": r_hyper,
            "r_du": r_du,
            "r_shield": r_shield,
        }
        return float(total_r), components


class RLController:
    """
    PPO / SAC Reinforcement Learning Controller.
    """

    def __init__(
        self,
        bergman_model=None,
        safety_shield=None,
        cfg: dict | None = None,
        algorithm: str = "PPO",
    ) -> None:
        cfg = cfg or {}
        self.cfg = cfg
        self.env = GlucoseEnv(bergman_model, safety_shield, cfg=cfg)
        self.algorithm = algorithm
        self.model = None

        if _GYM_AVAILABLE and _SB3_AVAILABLE:
            ppo_cfg = cfg.get("ppo", {})
            if algorithm == "PPO":
                self.model = PPO(
                    policy="MlpPolicy",
                    env=self.env,
                    learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
                    n_steps=int(ppo_cfg.get("n_steps", 512)),
                    batch_size=int(ppo_cfg.get("batch_size", 64)),
                    n_epochs=int(ppo_cfg.get("n_epochs", 10)),
                    gamma=float(ppo_cfg.get("gamma", 0.99)),
                    clip_range=float(ppo_cfg.get("clip_range", 0.2)),
                    ent_coef=float(ppo_cfg.get("ent_coef", 0.01)),
                    verbose=0,
                )
            elif algorithm == "SAC":
                self.model = SAC(
                    policy="MlpPolicy",
                    env=self.env,
                    learning_rate=3e-4,
                    buffer_size=50000,
                    batch_size=64,
                    gamma=0.99,
                    verbose=0,
                )

    def train(self, total_timesteps: int = 10000) -> bool:
        if self.model is not None:
            logger.info("Training %s agent for %d timesteps...", self.algorithm, total_timesteps)
            self.model.learn(total_timesteps=total_timesteps)
            logger.info("Training complete.")
            return True
        return False

    def predict_action(self, obs: np.ndarray, deterministic: bool = True) -> float:
        if self.model is not None:
            act, _ = self.model.predict(obs, deterministic=deterministic)
            return float(act[0])
        # Fallback basal heuristic
        return float(self.env.bergman.steady_state_basal_insulin())

    def evaluate_episode(self, env: GlucoseEnv, deterministic: bool = True) -> dict[str, Any]:
        obs, _ = env.reset()
        g_traj, u_raw_traj, u_exec_traj, intervened_traj = [], [], [], []

        terminated = False
        while not terminated:
            action = self.predict_action(obs, deterministic=deterministic)
            obs, reward, terminated, _, info = env.step(action)
            g_traj.append(info["glucose_true"])
            u_raw_traj.append(info["insulin_raw"])
            u_exec_traj.append(info["insulin_executed"])
            intervened_traj.append(info["shield_intervened"])

        g_arr = np.array(g_traj)
        tir = float(np.mean((g_arr >= 70.0) & (g_arr <= 180.0))) * 100.0
        tbr = float(np.mean(g_arr < 70.0)) * 100.0
        tar = float(np.mean(g_arr > 180.0)) * 100.0

        return {
            "glucose_traj": g_arr,
            "insulin_raw": np.array(u_raw_traj),
            "insulin_executed": np.array(u_exec_traj),
            "intervened_traj": np.array(intervened_traj),
            "tir": tir,
            "tbr": tbr,
            "tar": tar,
            "hypo_count": int(np.sum(g_arr < 70.0)),
        }
