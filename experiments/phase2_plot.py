"""
Phase 2 Plot: Bergman model-predicted vs true glucose for 3 synthetic patients.
Generates experiments/phase2_fit.png and prints an RMSE table.
"""
import sys
sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from src.data.loaders import SyntheticLoader
from src.models.mechanistic import (
    simulate_bergman,
    fit_parameters,
    DEFAULT_BERGMAN_PARAMS,
)

# Generate 3 synthetic patients with 24h traces
loader = SyntheticLoader({"n_patients": 3, "duration_hours": 24, "seed": 42})
df_raw = loader.load()

patients = sorted(df_raw["patient_id"].unique())
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle("Phase 2 — Bergman Minimal Model Fit vs Synthetic CGM", fontsize=15, fontweight="bold")

rmse_rows = []

for idx, pid in enumerate(patients):
    group = df_raw[df_raw["patient_id"] == pid].sort_values("timestamp").reset_index(drop=True)
    t0 = group["timestamp"].iloc[0]
    t_arr = (group["timestamp"] - t0).dt.total_seconds().values / 60.0
    g_obs = group["glucose_mgdL"].values
    u_arr = group["insulin_mU_per_min"].values
    cho_arr = group["meal_cho_g"].values

    # Build inputs
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

    # Base parameter reference from configuration
    pop_params = {
        "p1": 0.028, "p2": 0.028, "p3": 5.0e-5, "n": 0.15,
        "Gb": float(np.mean(g_obs)), "Ib": 10.0, "Vg": 117.0, "Vi": 12.0
    }
    initial_guess = {
        "p1": 0.010, "p2": 0.050, "p3": 1.0e-5, "n": 0.05,
        "Gb": float(g_obs[0]), "Ib": 10.0, "Vg": 117.0, "Vi": 12.0
    }

    # Initial state
    initial_state = [g_obs[0], 0.0, 10.0]

    # Untuned prediction (initial guess parameters)
    G_default, _, _ = simulate_bergman(t_arr, initial_guess, u_fn, ra_fn, initial_state=initial_state)
    rmse_default = float(np.sqrt(np.mean((G_default - g_obs) ** 2)))

    # Fit per-patient parameters
    fitted_params, info = fit_parameters(
        t_arr, g_obs, u_fn, ra_fn,
        initial_params=initial_guess,
        pop_params=pop_params,
        n_starts=3,
        l2_reg=0.01,
    )
    G_fit, _, _ = simulate_bergman(t_arr, fitted_params, u_fn, ra_fn, initial_state=initial_state)
    rmse_fit = info["rmse_fitted"]

    rmse_rows.append({
        "patient": pid,
        "rmse_default": rmse_default,
        "rmse_fitted": rmse_fit,
        "improvement": rmse_default - rmse_fit,
    })

    # Plot
    ax = axes[idx]
    hours = t_arr / 60.0
    ax.plot(hours, g_obs, "k.", markersize=2, alpha=0.5, label="CGM (observed)")
    ax.plot(hours, G_default, "--", color="#bbb", linewidth=1.2, label=f"Default (RMSE={rmse_default:.1f})")
    ax.plot(hours, G_fit, "-", color="#2b5c8f", linewidth=1.8, label=f"Fitted (RMSE={rmse_fit:.1f})")
    ax.axhline(70, color="r", linestyle=":", alpha=0.3)
    ax.axhline(180, color="orange", linestyle=":", alpha=0.3)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.set_title(f"{pid}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2)

axes[-1].set_xlabel("Time (hours)")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("experiments/phase2_fit.png", dpi=300)
print("Plot saved to experiments/phase2_fit.png\n")

# Print RMSE table
print(f"{'Patient':<20} {'RMSE Default':>14} {'RMSE Fitted':>14} {'Improvement':>14}")
print("-" * 66)
for row in rmse_rows:
    print(f"{row['patient']:<20} {row['rmse_default']:>14.2f} {row['rmse_fitted']:>14.2f} {row['improvement']:>14.2f}")
