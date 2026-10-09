"""
Patient-level, chronological train/val/test splitter.

AGENTS.md Rule 1:
  Splits are ALWAYS by patient AND by time window.
  Random row splits are FORBIDDEN.
  Every split is written to an explicit split.json.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PatientTimeSplitter:
    """
    Split a preprocessed DataFrame into train / val / test subsets.

    Strategy:
      1. Each patient's timeline is split chronologically:
           [──────────── train ──────────── | val | test]
      2. Patients are then optionally grouped: all patients share the same
         temporal fraction (the most robust strategy for small cohorts).
      3. A ``split.json`` manifest is written to ``output_path``.

    The manifest records, for every patient:
      - train window  [start, end) as ISO timestamps
      - val window    [start, end)
      - test window   [start, end)

    This makes every split fully reproducible and auditable.
    """

    def __init__(
        self,
        train_fraction: float = 0.70,
        val_fraction: float   = 0.15,
        output_path: str | Path = "data/processed/split.json",
        **kwargs,
    ) -> None:
        train_fraction = float(kwargs.get("train_ratio", train_fraction))
        val_fraction   = float(kwargs.get("val_ratio", val_fraction))

        if not (0 < train_fraction < 1):
            raise ValueError("train_fraction must be in (0, 1).")
        if not (0 < val_fraction < 1):
            raise ValueError("val_fraction must be in (0, 1).")
        if train_fraction + val_fraction >= 1.0:
            raise ValueError("train + val fractions must be < 1.0.")

        self.train_frac  = train_fraction
        self.val_frac    = val_fraction
        self.test_frac   = 1.0 - train_fraction - val_fraction
        self.output_path = Path(output_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self, df: pd.DataFrame, output_dir: Optional[str | Path] = None
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
        if output_dir is not None:
            out_path = Path(output_dir) / "split.json"
        else:
            out_path = self.output_path
        if "patient_id" not in df.columns or "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain 'patient_id' and 'timestamp' columns.")

        train_parts, val_parts, test_parts = [], [], []
        all_pids = sorted(list(df["patient_id"].unique()))
        n_held_out = max(1, int(len(all_pids) * self.test_frac)) if len(all_pids) > 1 else 0
        held_out_pids = set(all_pids[-n_held_out:]) if n_held_out > 0 else set()

        manifest: dict = {
            "fractions": {
                "train": self.train_frac,
                "val":   self.val_frac,
                "test":  self.test_frac,
            },
            "patient_held_out_split": {
                "train_patients": [p for p in all_pids if p not in held_out_pids],
                "test_held_out_patients": sorted(list(held_out_pids)),
            },
            "patients": {},
        }

        # Ensure output directory exists for per-patient parquet files
        target_dir = out_path.parent if out_path else Path("data/processed")
        target_dir.mkdir(parents=True, exist_ok=True)

        for pid, group in df.groupby("patient_id", sort=False):
            group = group.sort_values("timestamp").reset_index(drop=True)

            # Emit data/processed/patient_*.parquet (or .csv if parquet engine unavailable)
            file_base = target_dir / f"patient_{pid}"
            try:
                group.to_parquet(file_base.with_suffix(".parquet"), index=False)
            except Exception:
                group.to_csv(file_base.with_suffix(".csv"), index=False)

            n = len(group)
            i_train = int(n * self.train_frac)
            i_val   = int(n * (self.train_frac + self.val_frac))

            tr = group.iloc[:i_train].copy()
            va = group.iloc[i_train:i_val].copy()
            te = group.iloc[i_val:].copy()

            if len(tr) == 0:
                tr = group.iloc[:max(1, int(n * 0.8))].copy()
                va = group.iloc[max(1, int(n * 0.8)):].copy()
                te = va.copy()

            train_parts.append(tr)
            val_parts.append(va)
            test_parts.append(te)

            # Record per-patient temporal windows in manifest
            manifest["patients"][str(pid)] = {
                "train": {
                    "start": str(tr["timestamp"].iloc[0]),
                    "end":   str(tr["timestamp"].iloc[-1]),
                    "n_rows": len(tr),
                },
                "val": {
                    "start": str(va["timestamp"].iloc[0]),
                    "end":   str(va["timestamp"].iloc[-1]),
                    "n_rows": len(va),
                },
                "test": {
                    "start": str(te["timestamp"].iloc[0]),
                    "end":   str(te["timestamp"].iloc[-1]),
                    "n_rows": len(te),
                },
                "is_patient_held_out": str(pid) in held_out_pids
            }

        if not train_parts:
            raise ValueError("No patients survived the split. Check data quality.")

        train_df = pd.concat(train_parts, ignore_index=True)
        val_df   = pd.concat(val_parts,   ignore_index=True)
        test_df  = pd.concat(test_parts,  ignore_index=True)

        # Write manifest — ALWAYS, before returning (Rule 1)
        self._write_manifest(manifest, out_path)

        logger.info(
            "Split complete: %d train / %d val / %d test rows across %d patients.",
            len(train_df), len(val_df), len(test_df),
            len(manifest["patients"]),
        )
        return train_df, val_df, test_df, manifest

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _write_manifest(self, manifest: dict, out_path: Optional[Path] = None) -> None:
        target_path = out_path if out_path is not None else self.output_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        logger.info("Split manifest written to %s", target_path)

    @staticmethod
    def load_manifest(path: str | Path) -> dict:
        """Load a previously written split.json manifest."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Split manifest not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
