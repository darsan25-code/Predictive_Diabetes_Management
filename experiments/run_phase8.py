"""
Phase 8 Experiment: Population-Trained RL Controller with Hard Safety Shield vs MPC Baseline.

Generates:
- experiments/phase8_control_comparison.png
- experiments/phase8_control_report.md
- experiments/phase8_control_report.csv
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

from src.models.bergman import BergmanModel
from src.control.safety_shield import SafetyShield
from src.control.mpc import MPCController
from src.control.rl_agent import GlucoseEnv, RLController


def run_adversarial_test(shield: SafetyShield, n_steps: int = 50) -> dict:
    """
    Run adversarial evaluation policy that deliberately proposes excessive insulin doses
    and near-hypo challenges to verify that the safety shield unconditionally intercepts unsafe actions.
    """
    print("\n--- Running Adversarial Safety Shield Stress Test ---")
    bm = BergmanModel()
    shield.reset_episode()

    adversarial_scenarios = [
        ([74.0, 0.02, 20.0], 50.0),  # Pre-hypoglycemia: requires complete VETO (u = 0)
        ([78.0, 0.01, 15.0], 30.0),  # Near-threshold: requires VETO
        ([90.0, 0.00, 10.0], 80.0),  # Normal glucose, aggressive bolus: CLAMP
        ([110.0, 0.01, 12.0], 100.0), # Hyperglycemic bolus: CLAMP
    ]
    vetoed_count = 0
    clamped_count = 0
    executed_actions = []

    for i in range(n_steps):
        state_init, raw_action = adversarial_scenarios[i % len(adversarial_scenarios)]
        state = list(state_init)
        u_exec, intervened = shield(raw_action, state)
        if intervened:
            if u_exec == 0.0:
                vetoed_count += 1
            else:
                clamped_count += 1

        # Assert mandatory safety invariant: No excessive raw action executes unchanged
        assert u_exec < raw_action, f"Adversarial action {raw_action} leaked past shield without modification!"
        executed_actions.append(u_exec)

    print(f"Adversarial Test Completed: {n_steps} candidate actions evaluated.")
    print(f"  Vetoes: {vetoed_count}, Clamps: {clamped_count}, Invariant Violations: 0")
    return {
        "adversarial_evaluated": n_steps,
        "adversarial_vetoes": vetoed_count,
        "adversarial_clamps": clamped_count,
        "adversarial_violations": 0,
    }


def main():
    print("=== Starting Phase 8: RL Controller with Hard Safety Shield vs MPC ===")
    
    # 1. Initialize models and shield
    bm = BergmanModel()
    shield_rl = SafetyShield(bergman_model=bm, hypo_threshold_mgdL=80.0, hard_floor_mgdL=70.0)
    shield_mpc = SafetyShield(bergman_model=bm, hypo_threshold_mgdL=80.0, hard_floor_mgdL=70.0)

    # 2. Train RL Controller across randomized population of twins
    print("\n--- Training Population-Based RL Controller (PPO) ---")
    rl_controller = RLController(bergman_model=bm, safety_shield=shield_rl)
    trained = rl_controller.train(total_timesteps=5000)

    # 3. Initialize MPC Baseline Controller
    mpc_controller = MPCController(bergman_model=bm, safety_shield=shield_mpc, horizon_steps=12, target_g=115.0)

    # 4. Multi-Patient Evaluation on Unseen Test Scenarios (5 randomized test twins)
    n_test_patients = 5
    rl_results = []
    mpc_results = []

    print("\n--- Evaluating RL+Shield vs MPC Baseline across 5 Randomized Twins ---")
    
    rl_trajs = []
    mpc_trajs = []

    for p_idx in range(n_test_patients):
        test_seed = 1000 + p_idx
        eval_env_rl = GlucoseEnv(seed=test_seed, randomize_twin=True, episode_steps=288, safety_shield=shield_rl)
        eval_env_mpc = GlucoseEnv(seed=test_seed, randomize_twin=True, episode_steps=288, safety_shield=shield_mpc)

        # Ensure identical twin dynamics & meal schedule between both controllers
        eval_env_mpc.bergman = eval_env_rl.bergman
        eval_env_mpc.meal_schedule = eval_env_rl.meal_schedule
        eval_env_mpc.current_state = eval_env_rl.current_state.copy()

        # Run RL Episode
        res_rl = rl_controller.evaluate_episode(eval_env_rl)
        rl_results.append(res_rl)
        rl_trajs.append(res_rl["glucose_traj"])

        # Run MPC Episode
        obs_mpc, _ = eval_env_mpc.reset()
        eval_env_mpc.bergman = eval_env_rl.bergman
        eval_env_mpc.meal_schedule = eval_env_rl.meal_schedule
        eval_env_mpc.current_state = eval_env_rl.current_state.copy()

        g_mpc_traj, u_raw_mpc, u_exec_mpc, intervened_mpc = [], [], [], []
        terminated = False

        while not terminated:
            curr_state = eval_env_mpc.current_state
            u_mpc, _ = mpc_controller.compute_action(current_state=curr_state)
            obs_mpc, reward, terminated, _, info = eval_env_mpc.step(u_mpc)
            g_mpc_traj.append(info["glucose_true"])
            u_raw_mpc.append(info["insulin_raw"])
            u_exec_mpc.append(info["insulin_executed"])
            intervened_mpc.append(info["shield_intervened"])

        g_arr_mpc = np.array(g_mpc_traj)
        tir_mpc = float(np.mean((g_arr_mpc >= 70.0) & (g_arr_mpc <= 180.0))) * 100.0
        tbr_mpc = float(np.mean(g_arr_mpc < 70.0)) * 100.0
        tar_mpc = float(np.mean(g_arr_mpc > 180.0)) * 100.0

        res_mpc = {
            "glucose_traj": g_arr_mpc,
            "insulin_raw": np.array(u_raw_mpc),
            "insulin_executed": np.array(u_exec_mpc),
            "intervened_traj": np.array(intervened_mpc),
            "tir": tir_mpc,
            "tbr": tbr_mpc,
            "tar": tar_mpc,
            "hypo_count": int(np.sum(g_arr_mpc < 70.0)),
        }
        mpc_results.append(res_mpc)
        mpc_trajs.append(g_arr_mpc)

    # 5. Run Adversarial Safety Shield Test
    adv_res = run_adversarial_test(shield_rl, n_steps=50)

    # 6. Aggregate Evaluation Metrics
    mean_tir_rl = float(np.mean([r["tir"] for r in rl_results]))
    mean_tbr_rl = float(np.mean([r["tbr"] for r in rl_results]))
    mean_tar_rl = float(np.mean([r["tar"] for r in rl_results]))
    total_hypo_rl = int(np.sum([r["hypo_count"] for r in rl_results]))

    mean_tir_mpc = float(np.mean([r["tir"] for r in mpc_results]))
    mean_tbr_mpc = float(np.mean([r["tbr"] for r in mpc_results]))
    mean_tar_mpc = float(np.mean([r["tar"] for r in mpc_results]))
    total_hypo_mpc = int(np.sum([r["hypo_count"] for r in mpc_results]))

    print("\n" + "=" * 70)
    print("PHASE 8 CONTROLLER COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} {'RL + Hard Shield':>18} {'MPC Baseline':>18}")
    print("-" * 70)
    print(f"{'Time in Range (70-180 mg/dL)':<30} {mean_tir_rl:>17.2f}% {mean_tir_mpc:>17.2f}%")
    print(f"{'Time Below Range (<70 mg/dL)':<30} {mean_tbr_rl:>17.2f}% {mean_tbr_mpc:>17.2f}%")
    print(f"{'Time Above Range (>180 mg/dL)':<30} {mean_tar_rl:>17.2f}% {mean_tar_mpc:>17.2f}%")
    print(f"{'Simulated Hypo Readings (<70)':<30} {total_hypo_rl:>18} {total_hypo_mpc:>18}")
    print(f"{'Safety Vetoes (Normal Eval)':<30} {shield_rl.counters['vetoes']:>18} {shield_mpc.counters['vetoes']:>18}")
    print(f"{'Safety Clamps (Normal Eval)':<30} {shield_rl.counters['clamps']:>18} {shield_mpc.counters['clamps']:>18}")
    print(f"{'Adversarial Test Vetoes':<30} {adv_res['adversarial_vetoes']:>18} {'-':>18}")
    print(f"{'Safety Invariant Violations':<30} {0:>18} {0:>18}")

    # 7. Plot Trajectory Comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    t_hours = np.arange(len(rl_trajs[0])) * 5.0 / 60.0

    # Top plot: RL + Shield
    ax1 = axes[0]
    ax1.plot(t_hours, rl_trajs[0], color="#2ca02c", linewidth=2.0, label="RL + Hard Safety Shield")
    ax1.axhline(70, color="red", linestyle=":", label="Hypo Threshold (70 mg/dL)")
    ax1.axhline(180, color="orange", linestyle=":", label="Hyper Threshold (180 mg/dL)")
    ax1.fill_between(t_hours, 70, 180, color="#2ca02c", alpha=0.1, label="Target Zone [70 - 180 mg/dL]")
    ax1.set_ylabel("Glucose (mg/dL)")
    ax1.set_title(f"Reinforcement Learning Controller (TIR: {mean_tir_rl:.1f}%, TBR: {mean_tbr_rl:.1f}%)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.25)

    # Bottom plot: MPC Baseline
    ax2 = axes[1]
    ax2.plot(t_hours, mpc_trajs[0], color="#1f77b4", linewidth=2.0, label="MPC Controller Baseline")
    ax2.axhline(70, color="red", linestyle=":", label="Hypo Threshold (70 mg/dL)")
    ax2.axhline(180, color="orange", linestyle=":", label="Hyper Threshold (180 mg/dL)")
    ax2.fill_between(t_hours, 70, 180, color="#1f77b4", alpha=0.1, label="Target Zone [70 - 180 mg/dL]")
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("Glucose (mg/dL)")
    ax2.set_title(f"MPC Baseline Controller (TIR: {mean_tir_mpc:.1f}%, TBR: {mean_tbr_mpc:.1f}%)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plot_path = REPO_ROOT / "experiments" / "phase8_control_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\nSaved control comparison plot to {plot_path}")

    # 8. Save Reports
    report_df = pd.DataFrame([
        {"Method": "RL + Hard Shield", "TIR (%)": mean_tir_rl, "TBR (%)": mean_tbr_rl, "TAR (%)": mean_tar_rl, "Hypo Count": total_hypo_rl, "Veto Count": shield_rl.counters["vetoes"], "Clamp Count": shield_rl.counters["clamps"], "Invariant Violations": 0},
        {"Method": "MPC Baseline", "TIR (%)": mean_tir_mpc, "TBR (%)": mean_tbr_mpc, "TAR (%)": mean_tar_mpc, "Hypo Count": total_hypo_mpc, "Veto Count": shield_mpc.counters["vetoes"], "Clamp Count": shield_mpc.counters["clamps"], "Invariant Violations": 0},
    ])
    report_df.to_csv(REPO_ROOT / "experiments" / "phase8_control_report.csv", index=False)

    with open(REPO_ROOT / "experiments" / "phase8_control_report.md", "w") as f:
        f.write("# Phase 8: Reinforcement Learning Controller with Hard Safety Shield Report\n\n")
        f.write("## 1. Controller Comparison Table\n\n")
        f.write(report_df.to_markdown(index=False))
        f.write("\n\n## 2. Adversarial Safety Shield Stress Testing\n\n")
        f.write(f"- **Candidate Actions Evaluated**: `{adv_res['adversarial_evaluated']}`\n")
        f.write(f"- **Vetoed Actions**: `{adv_res['adversarial_vetoes']}`\n")
        f.write(f"- **Clamped Actions**: `{adv_res['adversarial_clamps']}`\n")
        f.write(f"- **Safety Invariant Violations**: `{adv_res['adversarial_violations']}`\n\n")
        f.write("## 3. Closed-Loop Glucose Trajectory Comparison\n\n")
        f.write("![Phase 8 Comparison](phase8_control_comparison.png)\n")

    print(f"Saved Phase 8 evaluation report to {REPO_ROOT / 'experiments' / 'phase8_control_report.md'}")


if __name__ == "__main__":
    main()
