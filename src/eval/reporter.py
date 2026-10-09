"""
Reporter module for evaluation reports and experiment plot saving.

AGENTS.md Rule 7: Always include clinician-in-the-loop framing in all outputs.
AGENTS.md Rule 10: Every phase saves at least 1 plot to experiments/.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

CLINICIAN_DISCLAIMER = """> **CLINICIAN DECISION-SUPPORT NOTICE**
> This system is a research prototype digital twin for Type 1 Diabetes simulation and state estimation.
> It is **NOT** a certified medical device and must **NEVER** be used for direct therapy adjustment,
> automated insulin dosing without human oversight, or independent clinical diagnosis.
"""

class Reporter:
    """Generates evaluation reports and handles artifact saving for experiment phases."""

    def __init__(self, output_dir: str | Path = "experiments"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics_report(
        self,
        metrics_by_model: Dict[str, Dict[str, Any]],
        filename: str = "metrics_report.md"
    ) -> Path:
        """Writes markdown metrics comparison table with mandatory clinician disclaimer."""
        filepath = self.output_dir / filename
        
        # Build Markdown table
        headers = ["Model", "TBR (<70) %", "TIR (70-180) %", "TAR (>180) %", "RMSE (mg/dL)", "MAE (mg/dL)", "MARD %", "Clarke A+B %"]
        rows = []
        for model_name, m in metrics_by_model.items():
            row = [
                model_name,
                f"{m.get('tbr_percent', 0.0):.2f}",
                f"{m.get('tir_percent', 0.0):.2f}",
                f"{m.get('tar_percent', 0.0):.2f}",
                f"{m.get('rmse', 0.0):.2f}",
                f"{m.get('mae', 0.0):.2f}",
                f"{m.get('mard_percent', 0.0):.2f}",
                f"{m.get('clarke_ab_percent', 0.0):.2f}",
            ]
            rows.append(row)
            
        table_str = "| " + " | ".join(headers) + " |\n"
        table_str += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for r in rows:
            table_str += "| " + " | ".join(r) + " |\n"

        content = f"# Digital Twin Model Evaluation Report\n\n{CLINICIAN_DISCLAIMER}\n\n## Performance Comparison\n\n{table_str}\n"

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        # Also write CSV summary
        df_rows = []
        for model_name, m in metrics_by_model.items():
            r = {"model": model_name}
            r.update(m)
            df_rows.append(r)
        csv_path = self.output_dir / filename.replace(".md", ".csv")
        pd.DataFrame(df_rows).to_csv(csv_path, index=False)

        return filepath

    def save_plot(self, fig: Any, filename: str) -> Path:
        """Saves a matplotlib figure to output_dir (AGENTS.md Rule 10)."""
        filepath = self.output_dir / filename
        if fig is not None and _MATPLOTLIB_AVAILABLE:
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            with open(filepath.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(f"Plot placeholder for {filename}\n")
        return filepath
