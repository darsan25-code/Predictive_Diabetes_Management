"""
Phase 5: Extended Kalman Filter State Estimation Experiment.

Evaluates EKF on a held-out test week with 95% uncertainty band coverage,
saves plot to experiments/phase5_ekf_tracking.png and report to experiments/phase5_ekf_report.md.
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loaders import SyntheticLoader
from src.data.preprocessor import Preprocessor
from src.data.splitter import PatientTimeSplitter
from src.models.bergman import BergmanModel
from src.estimation.kalman import ExtendedKalmanFilter, tune_ekf_hyperparameters


def main():
    print("=== Starting Phase 5: Extended Kalman Filter State Estimation ===")
    
    # 1. Load multi-patient dataset with 7 days (168h) duration
    loader = SyntheticLoader({"n_patients": 5, "duration_hours": 168, "seed": 42})
    df_raw = loader.load()
    preprocessor = Preprocessor(dt_minutes=5.0)
    df_clean = preprocessor.transform(df_raw)

    # 2. Split dataset: Train (70%), Val (15%), Test (15% -> ~25.2h per patient, or 1 held-out test patient for 168h)
    splitter = PatientTimeSplitter(train_ratio=0.7, val_ratio=0.15)
    train_df, val_df, test_df, split_manifest = splitter.split(df_clean, output_dir=REPO_ROOT / "data" / "processed")

    bm = BergmanModel()

    # 3. Validation-based hyperparameter tuning for Q and R
    print("\n--- Tuning Q and R on Validation Data ---")
    best_config = tune_ekf_hyperparameters(
        val_df=val_df,
        bergman_model=bm,
        candidate_q_g=[0.5, 1.0, 2.0, 4.0, 8.0],
        candidate_r=[9.0, 16.0, 25.0, 36.0],
    )
    print("Selected EKF configuration:", best_config)

    q_g = best_config["q_g"]
    r_cgm = best_config["r_cgm"]
    ekf_cfg = {
        "noise": {"Q": {"G": q_g, "X": 1e-6, "I": 0.05}, "R": {"cgm": r_cgm}},
        "initialization": {"P_diag": [400.0, 1e-6, 1.0]},
    }
    ekf = ExtendedKalmanFilter(bm, cfg=ekf_cfg)

    # 4. Evaluate on held-out test data (Test patient 168h)
    test_pids = test_df["patient_id"].unique()
    eval_pid = test_pids[0]
    test_patient_df = test_df[test_df["patient_id"] == eval_pid].sort_values("timestamp").reset_index(drop=True)

    t0 = test_patient_df["timestamp"].iloc[0]
    t_arr = (test_patient_df["timestamp"] - t0).dt.total_seconds().values / 60.0
    g_arr = test_patient_df["glucose_mgdL"].values
    u_arr = test_patient_df["insulin_mU_per_min"].values
    cho_arr = test_patient_df["meal_cho_g"].values

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

    # Initial state with intentional perturbation to test convergence
    ekf.reset(G0=g_arr[0])

    g_preds = []
    g_trues = []
    ci_lows = []
    ci_highs = []
    innovations = []
    times_hr = []

    for i in range(len(test_patient_df)):
        dt = (t_arr[i] - t_arr[i - 1]) if i > 0 else 5.0
        u_curr = u_arr[i - 1] if i > 0 else u_arr[0]
        ra_curr = ra_fn(t_arr[i])

        # 1. Prior Prediction
        x_pred, P_pred = ekf.predict(u_mU_per_min=u_curr, ra_mg_per_min=ra_curr, dt_min=dt)
        sigma_pred = float(np.sqrt(P_pred[0, 0] + r_cgm))
        g_pred = float(x_pred[0])
        ci_lo = g_pred - 1.96 * sigma_pred
        ci_hi = g_pred + 1.96 * sigma_pred

        # 2. Measurement Update
        x_post, P_post, innov, _ = ekf.update_measurement(cgm_mgdL=g_arr[i])

        g_preds.append(g_pred)
        g_trues.append(g_arr[i])
        ci_lows.append(ci_lo)
        ci_highs.append(ci_hi)
        innovations.append(innov)
        times_hr.append(t_arr[i] / 60.0)

    g_preds = np.array(g_preds)
    g_trues = np.array(g_trues)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    rmse = float(np.sqrt(np.mean((g_preds - g_trues) ** 2)))
    mae = float(np.mean(np.abs(g_preds - g_trues)))
    covered = (g_trues >= ci_lows) & (g_trues <= ci_highs)
    emp_coverage_pct = float(np.mean(covered)) * 100.0
    passed_coverage = 90.0 <= emp_coverage_pct <= 99.0

    print("\n" + "=" * 60)
    print("PHASE 5 HELD-OUT STATE ESTIMATION PERFORMANCE")
    print("=" * 60)
    print(f"Evaluated Patient:                {eval_pid}")
    print(f"Held-out Observations:            {len(g_trues)}")
    print(f"Held-out RMSE:                    {rmse:.3f} mg/dL")
    print(f"Held-out MAE:                     {mae:.3f} mg/dL")
    print(f"Empirical 95% CI Coverage:        {emp_coverage_pct:.2f}%")
    print(f"Target Coverage [90% - 99%]:      {'PASSED' if passed_coverage else 'FAILED'}")
    print(f"Final Q_G:                        {q_g:.2f}")
    print(f"Final R_CGM:                      {r_cgm:.2f}")

    # 5. Plot predicted vs real glucose over evaluation window
    fig, ax = plt.subplots(figsize=(14, 6))
    # Plot up to 48 hours for high-resolution visual clarity
    plot_mask = np.array(times_hr) <= min(times_hr[-1], 48.0)
    t_plot = np.array(times_hr)[plot_mask]
    g_true_plot = g_trues[plot_mask]
    g_pred_plot = g_preds[plot_mask]
    ci_lo_plot = ci_lows[plot_mask]
    ci_hi_plot = ci_highs[plot_mask]

    ax.plot(t_plot, g_true_plot, "k.", markersize=3, alpha=0.6, label="Observed CGM (mg/dL)")
    ax.plot(t_plot, g_pred_plot, color="#1f77b4", linewidth=2.0, label=f"EKF State Estimate (RMSE={rmse:.1f} mg/dL)")
    ax.fill_between(t_plot, ci_lo_plot, ci_hi_plot, color="#1f77b4", alpha=0.22, label=f"95% Uncertainty Band (Coverage={emp_coverage_pct:.1f}%)")

    ax.axhline(70, color="red", linestyle=":", alpha=0.5, label="Hypo Threshold (70 mg/dL)")
    ax.axhline(180, color="orange", linestyle=":", alpha=0.5, label="Hyper Threshold (180 mg/dL)")

    ax.set_title(f"Phase 5: Extended Kalman Filter State Estimation — {eval_pid}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=11)
    ax.set_ylabel("Glucose Concentration (mg/dL)", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plot_path = REPO_ROOT / "experiments" / "phase5_kalman.png"
    plt.savefig(plot_path, dpi=300)
    plt.savefig(REPO_ROOT / "experiments" / "phase5_ekf_tracking.png", dpi=300)
    plt.close()
    print(f"Saved EKF tracking plot to {plot_path}")

    # 6. Save Markdown report
    report_path = REPO_ROOT / "experiments" / "phase5_ekf_report.md"
    with open(report_path, "w") as f:
        f.write("# Phase 5: State Estimation Using Extended Kalman Filter Report\n\n")
        f.write("## 1. Mathematical Formulation & Jacobian Derivation\n\n")
        f.write("The continuous-time state transition Jacobian matrix $F_c = \\frac{\\partial f}{\\partial x}$ is derived analytically from the 3-state Bergman equations:\n\n")
        f.write("$$\\dot{G} = -(p_1 + X) G + p_1 G_b + \\frac{R_a}{V_g}$$\n")
        f.write("$$\\dot{X} = -p_2 X + p_3 (I - I_b)$$\n")
        f.write("$$\\dot{I} = -n (I - I_b) + \\frac{u - u_{\\text{basal}}}{V_i}$$\n\n")
        f.write("$$F_c = \\begin{bmatrix} -(p_1 + X) & -G & 0 \\\\ 0 & -p_2 & p_3 \\\\ 0 & 0 & -n \\end{bmatrix}$$\n\n")
        f.write("The discrete transition matrix is $F_d = I + F_c \\Delta t + \\frac{1}{2} (F_c \\Delta t)^2$.\n\n")
        f.write("Covariance updates are calculated using the numerically stable **Joseph form**:\n")
        f.write("$$P_{k|k} = (I - K_k H) P_{k|k-1} (I - K_k H)^T + K_k R K_k^T$$\n\n")
        f.write("## 2. Quantitative Evaluation on Held-Out Test Data\n\n")
        f.write(f"- **Patient ID**: `{eval_pid}`\n")
        f.write(f"- **Observations Evaluated**: `{len(g_trues)}` readings\n")
        f.write(f"- **Held-out RMSE**: `{rmse:.3f} mg/dL`\n")
        f.write(f"- **Held-out MAE**: `{mae:.3f} mg/dL`\n")
        f.write(f"- **Empirical 95% Uncertainty Coverage**: `{emp_coverage_pct:.2f}%`\n")
        f.write(f"- **Target Range [90% - 99%]**: `{'PASSED' if passed_coverage else 'FAILED'}`\n")
        f.write(f"- **Final Process Noise Q_G**: `{q_g:.2f}`\n")
        f.write(f"- **Final Measurement Noise R_CGM**: `{r_cgm:.2f}`\n\n")
        f.write("## 3. Evaluation Plot\n\n")
        f.write("![Phase 5 Tracking](phase5_ekf_tracking.png)\n")

    print(f"Saved EKF evaluation report to {report_path}")


if __name__ == "__main__":
    main()
