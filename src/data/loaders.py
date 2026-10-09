"""
Pluggable data loaders (AGENTS.md Rule 8).

SyntheticLoader is the default and requires no external files.
OHIOLoader documents the exact schema expected for real CGM datasets.
Use get_loader(cfg) to select the appropriate loader from config.

UNIT POLICY (Rule 9):
  glucose_mgdL     — CGM reading in mg/dL   (explicit in column name)
  insulin_mU_per_min — infusion rate in mU/min (explicit in column name)
  meal_cho_g       — carbohydrate intake in grams
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from src.utils.units import GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default Bergman parameters used for synthetic generation only.
# Inlined to avoid circular imports with src.models.
# ---------------------------------------------------------------------------
_SYNTH_PARAMS = {
    "p1": 0.028,
    "p2": 0.028,
    "p3": 5.0e-5,
    "n": 0.15,
    "Gb": 100.0,
    "Ib": 10.0,
    "Vg": 117.0,
    "Vi": 12.0,
}


def _synthetic_dynamics_rhs(
    t: float,
    state: list[float],
    params: dict,
    u_fn: Callable,
    meal_specs: list[dict],
    exercise_bouts: list[dict],
    dawn_cfg: dict,
) -> list[float]:
    """
    Physiologically enhanced ODE right-hand side with unmodeled effects:
    - Circadian insulin-sensitivity variation (dawn phenomenon)
    - Per-meal absorption variability
    - Exercise-driven acute sensitivity shifts and direct glucose uptake
    """
    G, X, I = state
    p1, p2, p3 = params["p1"], params["p2"], params["p3"]
    n, Gb, Ib  = params["n"],  params["Gb"],  params["Ib"]
    Vg, Vi     = params["Vg"], params["Vi"]

    # 1. Circadian dawn phenomenon: time-varying insulin sensitivity p3(t)
    t_hour = (t / 60.0) % 24.0
    effective_p3 = p3
    if dawn_cfg.get("enabled", True):
        amp = float(dawn_cfg.get("amplitude", 0.35))
        peak_h = float(dawn_cfg.get("peak_hour", 6.0))
        w_h = float(dawn_cfg.get("width_hours", 2.5))
        # Gaussian modulation reducing morning insulin sensitivity
        dawn_factor = 1.0 - amp * np.exp(-0.5 * ((t_hour - peak_h) / max(w_h, 0.5)) ** 2)
        effective_p3 = p3 * max(dawn_factor, 0.1)

    # 2. Exercise-driven sensitivity shifts & direct glucose uptake
    ex_uptake = 0.0
    for bout in exercise_bouts:
        t_start = bout["start"]
        t_end = bout["end"]
        # Active exercise bout or 60 min post-exercise recovery window
        if t_start <= t <= t_end + 60.0:
            effective_p3 *= bout.get("boost", 1.6)
            if t_start <= t <= t_end:
                ex_uptake += bout.get("uptake", 0.08) * G

    # 3. Meal absorption with per-meal variability
    Ra = 0.0
    for m in meal_specs:
        t_m = m["time"]
        if t >= t_m:
            dt_m = t - t_m
            k_abs = m.get("k_abs", 0.02)
            k_meal = m.get("k_meal", 0.85)  # 85% carbohydrate systemic bioavailability
            true_cho = m["true_cho"]
            Ra += k_meal * true_cho * 1000.0 * k_abs * np.exp(-k_abs * dt_m)

    u = float(u_fn(t))  # insulin infusion rate (mU/min)
    u_basal = n * Ib * Vi

    dG = -(p1 + X) * G + p1 * Gb + (Ra / Vg) - (ex_uptake / Vg)
    dX = -p2 * X + effective_p3 * (I - Ib)
    dI = -n * (I - Ib) + (u - u_basal) / Vi
    return [dG, dX, dI]


# Backward-compatible Bergman RHS (standard 1-compartment, constant parameters)
def _bergman_rhs(
    t: float,
    state: list[float],
    params: dict,
    u_fn: Callable,
    meal_times: list[float],
    meal_cho_g: list[float],
) -> list[float]:
    meal_specs = [
        {"time": tm, "true_cho": cho, "k_abs": 0.02, "k_meal": 0.85}
        for tm, cho in zip(meal_times, meal_cho_g)
    ]
    return _synthetic_dynamics_rhs(t, state, params, u_fn, meal_specs, [], {"enabled": False})


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseLoader(ABC):
    """Abstract base for all data loaders."""

    REQUIRED_COLUMNS = (
        "patient_id",
        "timestamp",
        "glucose_mgdL",
        "insulin_mU_per_min",
        "meal_cho_g",
    )

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Load data and return a DataFrame with columns:
          patient_id (str), timestamp (datetime64), glucose_mgdL (float),
          insulin_mU_per_min (float), meal_cho_g (float).
        """
        ...

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Loader output is missing columns: {missing}")


# ---------------------------------------------------------------------------
# Synthetic loader
# ---------------------------------------------------------------------------


class SyntheticLoader(BaseLoader):
    """
    Generate synthetic CGM + insulin traces via enhanced Bergman model simulation.

    Includes unmodeled physiological effects:
    - Circadian insulin sensitivity variation (dawn phenomenon)
    - Per-meal absorption variability
    - Exercise sensitivity shifts and glucose uptake
    - Realistic AR(1) sensor noise and calibration drift
    - 30-50% carbohydrate estimation error in recorded logs
    """

    def __init__(self, cfg: Optional[dict] = None, seed: int = 42) -> None:
        cfg = cfg or {}
        self.n_patients     = int(cfg.get("n_patients", 5))
        self.duration_hours = float(cfg.get("duration_hours", 72))
        self.dt_minutes     = float(cfg.get("dt_minutes", 5))
        self.seed           = int(cfg.get("seed", seed))
        self.unmodeled_cfg  = cfg.get("unmodeled_effects", {
            "enabled": True,
            "dawn_phenomenon": {"amplitude": 0.35, "peak_hour": 6.0, "width_hours": 2.5},
            "meal_variability": {"k_abs_min": 0.012, "k_abs_max": 0.035, "k_meal_min": 0.75, "k_meal_max": 0.95},
            "exercise": {"probability_per_day": 0.8, "duration_min": 45.0, "hour_start_min": 16.0, "hour_start_max": 19.0, "sensitivity_boost": 1.6, "glucose_uptake_rate": 0.08},
            "sensor_noise": {"ar1_phi": 0.70, "noise_std": 4.0, "drift_std": 2.5},
            "carb_estimation_error": {"min_error_pct": 0.30, "max_error_pct": 0.50},
        })

    def load_patient(self, patient_id: str = "patient_01", duration_hours: float = 24) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        df = self._generate_patient(patient_id, rng, duration_hours=duration_hours)
        return df

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        frames = [
            self._generate_patient(f"synthetic_{i:03d}", rng, duration_hours=self.duration_hours)
            for i in range(self.n_patients)
        ]
        df = pd.concat(frames, ignore_index=True)
        logger.info(
            "SyntheticLoader: %d patients, %d readings, %.0f min interval.",
            self.n_patients, len(df), self.dt_minutes,
        )
        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_params(self, rng: np.random.Generator) -> dict:
        """Sample Bergman parameters with ±10 % patient-level variation."""
        p = dict(_SYNTH_PARAMS)
        for key in ("p1", "p2", "p3", "n"):
            p[key] *= rng.uniform(0.90, 1.10)
        p["Gb"] = float(rng.uniform(85.0, 115.0))
        p["Ib"] = float(rng.uniform(8.0, 12.0))
        return p

    def _generate_meals(
        self, duration_min: float, rng: np.random.Generator
    ) -> list[dict]:
        """Return list of meal specs with true CHO and absorption variability."""
        meals = []
        t = rng.uniform(300.0, 480.0)   # first meal at 5–8 h
        unmodeled = self.unmodeled_cfg.get("enabled", True)
        m_var = self.unmodeled_cfg.get("meal_variability", {})
        c_err = self.unmodeled_cfg.get("carb_estimation_error", {})

        while t < duration_min - 120.0:
            true_cho = float(rng.uniform(30.0, 80.0))  # 30–80 g true CHO

            # Meal absorption rate variability
            if unmodeled:
                k_abs = float(rng.uniform(
                    m_var.get("k_abs_min", 0.012),
                    m_var.get("k_abs_max", 0.035),
                ))
                k_meal = float(rng.uniform(
                    m_var.get("k_meal_min", 0.040),
                    m_var.get("k_meal_max", 0.060),
                ))
                # Carb-estimation error: 30% to 50% discrepancy between true and logged CHO
                min_err = float(c_err.get("min_error_pct", 0.30))
                max_err = float(c_err.get("max_error_pct", 0.50))
                err_sign = rng.choice([-1.0, 1.0])
                err_magnitude = rng.uniform(min_err, max_err)
                reported_cho = max(5.0, true_cho * (1.0 + err_sign * err_magnitude))
            else:
                k_abs = 0.02
                k_meal = 0.05
                reported_cho = true_cho

            meals.append({
                "time": float(t),
                "true_cho": true_cho,
                "reported_cho": reported_cho,
                "k_abs": k_abs,
                "k_meal": k_meal,
            })
            t += rng.uniform(180.0, 360.0)  # 3–6 h gap

        return meals

    def _generate_exercise_bouts(
        self, duration_min: float, rng: np.random.Generator
    ) -> list[dict]:
        """Return scheduled daily exercise bouts."""
        if not self.unmodeled_cfg.get("enabled", True):
            return []
        ex_cfg = self.unmodeled_cfg.get("exercise", {})
        prob_day = float(ex_cfg.get("probability_per_day", 0.8))
        dur_ex = float(ex_cfg.get("duration_min", 45.0))
        h_min = float(ex_cfg.get("hour_start_min", 16.0))
        h_max = float(ex_cfg.get("hour_start_max", 19.0))
        boost = float(ex_cfg.get("sensitivity_boost", 1.6))
        uptake = float(ex_cfg.get("glucose_uptake_rate", 0.08))

        n_days = int(np.ceil(duration_min / (24.0 * 60.0)))
        bouts = []
        for day in range(n_days):
            if rng.uniform(0.0, 1.0) < prob_day:
                start_h = rng.uniform(h_min, h_max)
                t_start = day * 24.0 * 60.0 + start_h * 60.0
                t_end = t_start + dur_ex
                if t_start < duration_min:
                    bouts.append({
                        "start": t_start,
                        "end": min(t_end, duration_min),
                        "boost": boost,
                        "uptake": uptake,
                    })
        return bouts

    def _build_u_fn(
        self,
        params: dict,
        meal_specs: list[dict],
        rng: np.random.Generator,
    ) -> Callable:
        """Return u(t) → mU/min: basal + post-meal boluses based on reported CHO."""
        basal = params["n"] * params["Ib"] * params["Vi"]
        bolus_times = [m["time"] + rng.uniform(0.0, 10.0) for m in meal_specs]
        # Bolus dosed based on REPORTED CHO (patient calculates dose from logged carbs)
        bolus_amts  = [m["reported_cho"] * rng.uniform(0.08, 0.12) * 1000.0 for m in meal_specs]
        bolus_dur   = 30.0  # spread over 30 min

        def u(t: float) -> float:
            total = basal
            for t_b, amt in zip(bolus_times, bolus_amts):
                if t_b <= t <= t_b + bolus_dur:
                    total += amt / bolus_dur
            return total

        return u

    def _generate_patient(self, patient_id: str, rng: np.random.Generator, duration_hours: float | None = None) -> pd.DataFrame:
        dur_hrs = duration_hours if duration_hours is not None else self.duration_hours
        duration_min = dur_hrs * 60.0
        params = self._sample_params(rng)
        meal_specs = self._generate_meals(duration_min, rng)
        exercise_bouts = self._generate_exercise_bouts(duration_min, rng)
        u_fn = self._build_u_fn(params, meal_specs, rng)

        dawn_cfg = self.unmodeled_cfg.get("dawn_phenomenon", {}) if self.unmodeled_cfg.get("enabled", True) else {"enabled": False}

        G0 = params["Gb"] + rng.uniform(-20.0, 20.0)
        sol = solve_ivp(
            _synthetic_dynamics_rhs,
            t_span=(0.0, duration_min),
            y0=[G0, 0.0, params["Ib"]],
            t_eval=np.arange(0.0, duration_min, self.dt_minutes),
            args=(params, u_fn, meal_specs, exercise_bouts, dawn_cfg),
            method="RK45",
            max_step=1.0,
            rtol=1e-4,
            atol=1e-6,
        )

        unmodeled = self.unmodeled_cfg.get("enabled", True)
        if unmodeled:
            s_cfg = self.unmodeled_cfg.get("sensor_noise", {})
            phi = float(s_cfg.get("ar1_phi", 0.70))
            std = float(s_cfg.get("noise_std", 4.0))
            drift_std = float(s_cfg.get("drift_std", 2.5))
            drift = float(rng.normal(0.0, drift_std))

            noise = np.zeros(len(sol.t))
            white = rng.normal(0.0, std * np.sqrt(max(1.0 - phi ** 2, 0.1)), size=len(sol.t))
            noise[0] = rng.normal(0.0, std)
            for k in range(1, len(sol.t)):
                noise[k] = phi * noise[k - 1] + white[k]

            G_cgm = np.clip(
                sol.y[0] + noise + drift,
                GLUCOSE_MIN_MGDL,
                GLUCOSE_MAX_MGDL,
            )
        else:
            G_cgm = np.clip(
                sol.y[0] + rng.normal(0.0, 5.0, size=sol.y[0].shape),
                GLUCOSE_MIN_MGDL,
                GLUCOSE_MAX_MGDL,
            )

        u_samples = np.array([u_fn(t) for t in sol.t])

        # Reported meal signal with carb-estimation error recorded in dataframe
        meal_signal = np.zeros(len(sol.t))
        for m in meal_specs:
            idx = int(np.searchsorted(sol.t, m["time"]))
            if idx < len(meal_signal):
                meal_signal[idx] = m["reported_cho"]

        timestamps = pd.date_range(
            start="2024-01-01", periods=len(sol.t),
            freq=f"{int(self.dt_minutes)}min",
        )
        return pd.DataFrame({
            "patient_id":         patient_id,
            "timestamp":          timestamps,
            "glucose_mgdL":       G_cgm,
            "insulin_mU_per_min": u_samples,
            "meal_cho_g":         meal_signal,
        })


# ---------------------------------------------------------------------------
# OHIO CSV loader (skeleton for real data)
# ---------------------------------------------------------------------------


class OHIOLoader(BaseLoader):
    """
    Loader for OHIO 2018 / 2020 CGM dataset (or any similarly structured CSVs).

    Expected CSV columns (configure overrides in data_default.yaml):
      patient_id, timestamp (ISO 8601), glucose_mgdL, insulin_mU_per_min,
      meal_cho_g (optional — filled with 0 if absent).

    Usage:
      1. Place per-patient CSV files in data/raw/ohio/
      2. Set  loader: ohio_csv  in configs/data_default.yaml
    """

    def __init__(self, cfg: dict) -> None:
        ohio = cfg.get("ohio_csv", {})
        self.data_dir      = ohio.get("data_dir",      "data/raw/ohio/")
        self.glucose_col   = ohio.get("glucose_col",   "glucose_mgdL")
        self.insulin_col   = ohio.get("insulin_col",   "insulin_mU_per_min")
        self.timestamp_col = ohio.get("timestamp_col", "timestamp")
        self.patient_col   = ohio.get("patient_col",   "patient_id")

    def load(self) -> pd.DataFrame:
        import glob
        from pathlib import Path

        files = sorted(glob.glob(str(Path(self.data_dir) / "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSV files in '{self.data_dir}'. "
                "Add OHIO CSVs or switch to  loader: synthetic  in data_default.yaml."
            )
        frames = []
        for fpath in files:
            df = pd.read_csv(fpath, parse_dates=[self.timestamp_col])
            df = df.rename(columns={
                self.glucose_col:   "glucose_mgdL",
                self.insulin_col:   "insulin_mU_per_min",
                self.timestamp_col: "timestamp",
                self.patient_col:   "patient_id",
            })
            if "meal_cho_g" not in df.columns:
                df["meal_cho_g"] = 0.0
            frames.append(df)

        df_all = pd.concat(frames, ignore_index=True)
        self._validate(df_all)
        logger.info("OHIOLoader: loaded %d readings from %d file(s).", len(df_all), len(files))
        return df_all


# ---------------------------------------------------------------------------
# SimGlucose UVA/Padova Simulator Loader (Phase 6)
# ---------------------------------------------------------------------------


class SimGlucoseLoader(BaseLoader):
    """
    Loader for the open-source UVA/Padova T1D simulation benchmark (simglucose).

    Provides virtual patient cohorts:
      - Adults: adult#001 to adult#010
      - Adolescents: adolescent#001 to adolescent#010
      - Children: child#001 to child#010

    Converts all measurements to explicit project units:
      - glucose_mgdL: CGM glucose reading (mg/dL)
      - insulin_mU_per_min: Pump delivery rate in mU/min (converted from U/min)
      - meal_cho_g: Carbohydrate intake in grams
    """

    DEFAULT_PATIENTS = (
        "adult#001", "adult#002", "adult#003",
        "adolescent#001", "adolescent#002", "adolescent#003",
        "child#001", "child#002", "child#003",
    )

    def __init__(self, cfg: Optional[dict] = None, seed: int = 42) -> None:
        cfg = cfg or {}
        self.patient_names = cfg.get("patients", list(self.DEFAULT_PATIENTS))
        self.duration_hours = float(cfg.get("duration_hours", 24.0))
        self.seed = int(cfg.get("seed", seed))

    def load_patient(
        self,
        patient_name: str = "adult#001",
        duration_hours: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Simulate and load a single virtual patient trace from simglucose."""
        from datetime import datetime, timedelta
        from simglucose.patient.t1dpatient import T1DPatient
        from simglucose.sensor.cgm import CGMSensor
        from simglucose.actuator.pump import InsulinPump
        from simglucose.simulation.env import T1DSimEnv
        from simglucose.simulation.scenario_gen import RandomScenario
        from simglucose.controller.basal_bolus_ctrller import BBController
        from simglucose.simulation.sim_engine import SimObj

        dur_hrs = duration_hours if duration_hours is not None else self.duration_hours
        sim_seed = seed if seed is not None else self.seed

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        scenario = RandomScenario(start_time=start_time, seed=sim_seed)
        patient = T1DPatient.withName(patient_name)
        sensor = CGMSensor.withName("Dexcom", seed=sim_seed)
        pump = InsulinPump.withName("Insulet")
        env = T1DSimEnv(patient, sensor, pump, scenario)
        controller = BBController()

        sim_obj = SimObj(env, controller, timedelta(hours=dur_hrs), animate=False)
        sim_obj.simulate()
        raw_res = sim_obj.results()

        # Build output DataFrame with explicit units
        timestamps = pd.to_datetime(raw_res.index)
        glucose = raw_res["CGM"].values.astype(float)
        # Convert insulin from U/min to mU/min (1 U = 1000 mU)
        insulin_mU = (raw_res["insulin"].values.astype(float)) * 1000.0
        meal_cho = raw_res["CHO"].values.astype(float)

        df = pd.DataFrame({
            "patient_id": patient_name,
            "timestamp": timestamps,
            "glucose_mgdL": glucose,
            "insulin_mU_per_min": insulin_mU,
            "meal_cho_g": meal_cho,
        })
        df["insulin_mU_per_min"] = df["insulin_mU_per_min"].bfill().ffill().fillna(0.0)
        df["meal_cho_g"] = df["meal_cho_g"].fillna(0.0)
        df["glucose_mgdL"] = df["glucose_mgdL"].bfill().ffill()
        self._validate(df)
        return df

    def load(self) -> pd.DataFrame:
        frames = []
        for p_name in self.patient_names:
            df_p = self.load_patient(p_name, duration_hours=self.duration_hours, seed=self.seed)
            frames.append(df_p)

        df_all = pd.concat(frames, ignore_index=True)
        self._validate(df_all)
        logger.info(
            "SimGlucoseLoader: loaded %d readings across %d virtual patients.",
            len(df_all), len(self.patient_names)
        )
        return df_all


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_loader(cfg: dict) -> BaseLoader:
    """
    Return the appropriate loader based on the 'loader' key in cfg.

    Args:
        cfg: contents of data_default.yaml (already parsed as a dict).
    """
    loader_type = cfg.get("loader", "synthetic")
    if loader_type == "synthetic":
        return SyntheticLoader(cfg.get("synthetic", {}))
    elif loader_type == "ohio_csv":
        return OHIOLoader(cfg)
    elif loader_type == "simglucose":
        return SimGlucoseLoader(cfg.get("simglucose", {}))
    else:
        raise ValueError(
            f"Unknown loader '{loader_type}'. Valid options: 'synthetic', 'ohio_csv', 'simglucose'."
        )
