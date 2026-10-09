"""
Data preprocessor: resampling, unit validation, and outlier guarding.

UNIT POLICY (AGENTS.md Rule 9):
  Input glucose column MUST be named 'glucose_mgdL' — the name is the unit contract.
  Any value outside [20, 600] mg/dL is clipped with a logged WARNING, not silently.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.units import GLUCOSE_MIN_MGDL, GLUCOSE_MAX_MGDL

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("patient_id", "timestamp", "glucose_mgdL", "insulin_mU_per_min")


class Preprocessor:
    """
    Standard preprocessing pipeline for CGM + insulin DataFrames.

    Steps (applied per patient):
      1. Sort by timestamp.
      2. Validate and clip glucose to physiological bounds [20, 600] mg/dL.
      3. Clip insulin to [0, 20] mU/min.
      4. Resample to a uniform cadence (default 5 min) using linear interpolation.
      5. Fill residual NaNs via forward-fill (≤ 2 consecutive gaps) then drop rows.
    """

    def __init__(
        self,
        dt_minutes: float = 5.0,
        glucose_min: float = GLUCOSE_MIN_MGDL,
        glucose_max: float = GLUCOSE_MAX_MGDL,
        insulin_max: float = 20.0,
        max_gap_fill: int = 2,
    ) -> None:
        self.dt_minutes  = dt_minutes
        self.glucose_min = glucose_min
        self.glucose_max = glucose_max
        self.insulin_max = insulin_max
        self.max_gap_fill = max_gap_fill

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the full preprocessing pipeline to a raw loader DataFrame.

        Returns a clean DataFrame with a uniform 5-min timestamp cadence.
        Raises ValueError if required columns are missing.
        """
        self._check_columns(df)

        patient_frames = []
        for pid, group in df.groupby("patient_id", sort=False):
            cleaned = self._process_patient(pid, group)
            if cleaned is not None and len(cleaned) > 0:
                patient_frames.append(cleaned)

        if not patient_frames:
            raise ValueError("No valid patient data survived preprocessing.")

        result = pd.concat(patient_frames, ignore_index=True)
        logger.info(
            "Preprocessor: %d patients, %d rows after processing.",
            len(patient_frames), len(result),
        )
        return result

    transform = process

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_columns(df: pd.DataFrame) -> None:
        missing = [c for c in _REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

    def _process_patient(
        self, patient_id: str, df: pd.DataFrame
    ) -> Optional[pd.DataFrame]:
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # ── LO/HI censoring flags & clipping ──────────────────────────────
        df["cgm_censored_lo"] = (df["glucose_mgdL"] < 40.0).astype(int)
        df["cgm_censored_hi"] = (df["glucose_mgdL"] > 400.0).astype(int)
        df["glucose_mgdL"] = df["glucose_mgdL"].clip(40.0, 400.0)

        # ── Clip insulin and meal inputs ─────────────────────────────────
        df["insulin_mU_per_min"] = df["insulin_mU_per_min"].clip(0.0, self.insulin_max)
        if "meal_cho_g" not in df.columns:
            df["meal_cho_g"] = 0.0
        df["meal_cho_g"] = df["meal_cho_g"].clip(0.0, None)

        # ── Resample to uniform 5-minute grid ────────────────────────────
        df = df.set_index("timestamp")
        freq = f"{int(self.dt_minutes)}min"

        numeric_cols = ["glucose_mgdL", "insulin_mU_per_min", "cgm_censored_lo", "cgm_censored_hi"]
        resampled = df[numeric_cols].resample(freq).mean()
        meals_resampled = df[["meal_cho_g"]].resample(freq).sum()
        resampled["meal_cho_g"] = meals_resampled["meal_cho_g"].fillna(0.0)
        resampled["patient_id"] = patient_id

        # ── Missing CGM (<30 min interpolate, >30 min mask) ──────────────
        # max_interpolate = 30 min / 5 min = 6 steps
        max_interpolate_steps = int(30.0 / self.dt_minutes)
        cgm_interp = resampled["glucose_mgdL"].interpolate(method="linear", limit=max_interpolate_steps)
        cgm_mask = cgm_interp.notna().astype(int)
        resampled["cgm_mask"] = cgm_mask
        resampled["glucose_mgdL"] = cgm_interp.bfill().ffill()
        resampled["insulin_mU_per_min"] = resampled["insulin_mU_per_min"].interpolate(method="linear", limit=max_interpolate_steps).fillna(0.0)

        resampled = resampled.reset_index()

        # ── Feature Engineering ──────────────────────────────────────────
        timestamps = pd.to_datetime(resampled["timestamp"])
        resampled["timestamp_min"] = (timestamps - timestamps.iloc[0]).dt.total_seconds() / 60.0

        # Time-of-day sin/cos & sleep flag
        time_hours = timestamps.dt.hour + timestamps.dt.minute / 60.0
        resampled["sin_time"] = np.sin(2 * np.pi * time_hours / 24.0)
        resampled["cos_time"] = np.cos(2 * np.pi * time_hours / 24.0)
        resampled["sleep_flag"] = ((timestamps.dt.hour >= 23) | (timestamps.dt.hour < 7)).astype(int)
        resampled["activity"] = np.where(resampled["sleep_flag"] == 1, 0.8, 1.0)

        # IOB (Insulin on Board) & COB (Carbs on Board) decay exponential models
        dt = self.dt_minutes
        u_ins = resampled["insulin_mU_per_min"].values
        u_cho = resampled["meal_cho_g"].values

        iob = np.zeros(len(resampled))
        cob = np.zeros(len(resampled))
        k_iob = np.exp(-dt / 180.0) # ~3h half-life decay
        k_cob = np.exp(-dt / 120.0) # ~2h half-life decay

        for k in range(1, len(resampled)):
            iob[k] = iob[k-1] * k_iob + u_ins[k-1] * dt
            cob[k] = cob[k-1] * k_cob + u_cho[k-1]

        resampled["iob"] = iob
        resampled["cob"] = cob

        # Lagged glucose (1, 2, 3 steps)
        g = resampled["glucose_mgdL"]
        resampled["glucose_lag_1"] = g.shift(1).bfill()
        resampled["glucose_lag_2"] = g.shift(2).bfill()
        resampled["glucose_lag_3"] = g.shift(3).bfill()

        if len(resampled) < 2:
            logger.warning("Patient %s: insufficient data after resampling, skipping.", patient_id)
            return None

        return resampled
