"""
Phase 3: Per-Patient Calibration and Identifiability Analysis Experiment.

Generates:
- experiments/phase3_identifiability.png
- experiments/phase3_identifiability_report.md
- experiments/phase3_identifiability_report.csv
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from src.data.loaders import SyntheticLoader
from src.data.preprocessor import Preprocessor
from src.data.splitter import PatientTimeSplitter
from src.models.mechanistic import DEFAULT_BERGMAN_PARAMS
from src.models.calibration import (
    calibrate_patient,
    run_sensitivity_analysis,
    run_profile_likelihood,
    classify_identifiability,
    plot_sensitivity_heatmap,
)


def main():
    print("=== Starting Phase 3: Per-Patient Calibration & Identifiability Analysis ===")
    
    # Generate multi-patient synthetic dataset (72h duration)
    loader = SyntheticLoader({"n_patients": 5, "duration_hours": 72, "seed": 42})
    df_raw = loader.load()
    preprocessor = Preprocessor(dt_minutes=5.0)
    df_clean = preprocessor.transform(df_raw)

    # Perform patient-level & temporal splitting (70% train, 15% val, 15% test)
    splitter = PatientTimeSplitter(train_ratio=0.7, val_ratio=0.15)
    train_df, val_df, test_df, split_manifest = splitter.split(df_clean, output_dir=REPO_ROOT / "data" / "processed")

    train_patients = train_df["patient_id"].unique()
    print(f"Calibration Training Patients ({len(train_patients)}): {list(train_patients)}")

    pop_params = dict(DEFAULT_BERGMAN_PARAMS)
    calibration_records = []
    sensitivity_records = []
    identifiability_records = []

    # Run calibration for each patient in training set
    for pid in train_patients:
        pdf = train_df[train_df["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
        t0 = pdf["timestamp"].iloc[0]
        t_arr = (pdf["timestamp"] - t0).dt.total_seconds().values / 60.0
        g_obs = pdf["glucose_mgdL"].values
        u_arr = pdf["insulin_mU_per_min"].values
        cho_arr = pdf["meal_cho_g"].values

        u_fn = lambda t, t_a=t_arr, u_a=u_arr: float(np.interp(t, t_a, u_a))

        meal_idx = np.where(cho_arr > 0)[0]
        m_times = t_arr[meal_idx]
        m_chos = cho_arr[meal_idx]
        k_meal, k_abs = 0.05, 0.02

        def ra_fn(t, m_times=m_times, m_chos=m_chos):
            ra = 0.0
            for tm, ch in zip(m_times, m_chos):
                if t >= tm:
                    ra += k_meal * ch * 1000.0 * k_abs * np.exp(-k_abs * (t - tm))
            return float(ra)

        # 1. Calibrate patient parameters
        calibrated_p, diag = calibrate_patient(
            t_arr, g_obs, u_fn, ra_fn,
            pop_params=pop_params,
            l2_reg=0.05,
            n_starts=3,
        )

        cal_rec = {
            "patient_id": pid,
            "rmse_initial": diag["rmse_initial"],
            "rmse_calibrated": diag["rmse_calibrated"],
            "mae_calibrated": diag["mae_calibrated"],
            "rmse_improvement": diag["rmse_improvement"],
            "objective_cost": diag["objective_cost"],
            "success": diag["success"],
        }
        for k in ("p1", "p2", "p3", "n"):
            cal_rec[f"param_{k}"] = calibrated_p[k]
        calibration_records.append(cal_rec)

        # 2. Local sensitivity analysis
        sens_df = run_sensitivity_analysis(t_arr, g_obs, u_fn, ra_fn, calibrated_p, perturbation_pct=0.10)
        sens_df["patient_id"] = pid
        sensitivity_records.append(sens_df)

        # 3. Profile likelihood analysis
        prof_res = run_profile_likelihood(
            t_arr, g_obs, u_fn, ra_fn, calibrated_p,
            pop_params=pop_params, l2_reg=0.05, n_grid=12,
        )

        # 4. Identifiability classification
        ident_df = classify_identifiability(calibrated_p, sens_df, prof_res)
        ident_df["patient_id"] = pid
        identifiability_records.append(ident_df)

    cal_df = pd.DataFrame(calibration_records)
    all_sens_df = pd.concat(sensitivity_records, ignore_index=True)
    all_ident_df = pd.concat(identifiability_records, ignore_index=True)

    # Generate population average sensitivity heatmap
    mean_sens_df = all_sens_df.groupby("parameter").agg({
        "baseline_rmse": "mean",
        "delta_rmse_plus": "mean",
        "delta_rmse_minus": "mean",
        "pct_change_plus": "mean",
        "pct_change_minus": "mean",
        "sensitivity_score": "mean",
    }).reset_index()

    plot_sensitivity_heatmap(mean_sens_df, output_path=str(REPO_ROOT / "experiments" / "phase3_identifiability.png"))

    # Aggregate identifiability summary across patients
    agg_ident = all_ident_df.groupby("parameter").agg({
        "point_estimate": "mean",
        "ci_lo": "mean",
        "ci_hi": "mean",
        "sensitivity_score": "mean",
    }).reset_index()

    agg_ident["ci_95"] = agg_ident.apply(lambda r: f"[{r['ci_lo']:.2e}, {r['ci_hi']:.2e}]", axis=1)
    # Re-apply identifiability rule to aggregate
    agg_ident["identifiable"] = agg_ident.apply(
        lambda r: "yes" if ((r["ci_hi"] / max(r["ci_lo"], 1e-12)) < 15.0 and r["sensitivity_score"] > 0.05) else "no",
        axis=1
    )
    agg_ident["evidence"] = agg_ident.apply(
        lambda r: "Well-constrained with strong curvature" if r["identifiable"] == "yes" else "Practically non-identifiable / unconstrained from CGM observations alone",
        axis=1
    )

    print("\n" + "=" * 80)
    print("PHASE 3 PARAMETER IDENTIFIABILITY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Parameter':<12} {'Point Estimate':>16} {'95% CI':>24} {'Identifiable?':>15} {'Sensitivity (dRMSE)':>22}")
    print("-" * 92)
    for _, row in agg_ident.iterrows():
        print(f"{row['parameter']:<12} {row['point_estimate']:>16.4e} {row['ci_95']:>24} {row['identifiable']:>15} {row['sensitivity_score']:>22.3f}")

    print("\nExplicit List of Non-Identifiable Parameters:")
    non_ident = agg_ident[agg_ident["identifiable"] == "no"]
    if len(non_ident) == 0:
        print("  None (all primary parameters met identifiability criteria under the tested protocol).")
    else:
        for _, r in non_ident.iterrows():
            print(f"  - {r['parameter']}: {r['evidence']} (Sensitivity: {r['sensitivity_score']:.3f} mg/dL, CI width: {r['ci_95']})")

    # Save reports
    all_ident_df.to_csv(REPO_ROOT / "experiments" / "phase3_identifiability_report.csv", index=False)
    
    with open(REPO_ROOT / "experiments" / "phase3_identifiability_report.md", "w") as f:
        f.write("# Phase 3: Parameter Calibration and Identifiability Report\n\n")
        f.write("## 1. Per-Patient Calibration Performance\n\n")
        f.write(cal_df.to_markdown(index=False))
        f.write("\n\n## 2. Parameter Identifiability Classification Table\n\n")
        f.write(agg_ident[["parameter", "point_estimate", "ci_95", "identifiable", "sensitivity_score", "evidence"]].to_markdown(index=False))
        f.write("\n\n## 3. Sensitivity Heatmap\n\n")
        f.write("![Phase 3 Heatmap](phase3_identifiability.png)\n")

    print("\nSaved report to experiments/phase3_identifiability_report.md")


if __name__ == "__main__":
    main()
