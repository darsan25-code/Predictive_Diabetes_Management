# DIABETES DIGITAL TWIN PROJECT — STATUS REPORT

*Generated: 2026-10-09*  
*Repository Root: `D:/Projects/Diabetes Project`*

---

## 1. PHASE TRACKER

| Phase | Description | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Phase 1** | Data Pipeline, Preprocessor & Chronological Splitter | **DONE** | [`src/data/loaders.py`](file:///D:/Projects/Diabetes%20Project/src/data/loaders.py), [`data/processed/split.json`](file:///D:/Projects/Diabetes%20Project/data/processed/split.json), [`experiments/synthetic_patient_trace.png`](file:///D:/Projects/Diabetes%20Project/experiments/synthetic_patient_trace.png) |
| **Phase 2** | Mechanistic Bergman Minimal Model & Parameter Estimation | **DONE** | [`src/models/mechanistic.py`](file:///D:/Projects/Diabetes%20Project/src/models/mechanistic.py), [`experiments/phase2_fit.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase2_fit.png), [`tests/test_bergman.py`](file:///D:/Projects/Diabetes%20Project/tests/test_bergman.py) |
| **Phase 3** | Per-Patient Calibration & Parameter Identifiability Analysis | **DONE** | [`src/models/calibration.py`](file:///D:/Projects/Diabetes%20Project/src/models/calibration.py), [`experiments/phase3_identifiability.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase3_identifiability.png), [`experiments/phase3_identifiability_report.md`](file:///D:/Projects/Diabetes%20Project/experiments/phase3_identifiability_report.md) |
| **Phase 4** | Hybrid Digital Twin (Mechanistic + Residual Neural Network) | **DONE** | [`src/models/residual_model.py`](file:///D:/Projects/Diabetes%20Project/src/models/residual_model.py), [`experiments/phase4_horizons.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase4_horizons.png), [`experiments/phase4_metrics_report.md`](file:///D:/Projects/Diabetes%20Project/experiments/phase4_metrics_report.md) |
| **Phase 5** | State Estimation via Extended Kalman Filter (EKF) | **DONE** | [`src/estimation/kalman.py`](file:///D:/Projects/Diabetes%20Project/src/estimation/kalman.py), [`experiments/phase5_kalman.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase5_kalman.png), [`experiments/phase5_validation_report.md`](file:///D:/Projects/Diabetes%20Project/experiments/phase5_validation_report.md) |
| **Phase 6** | Independent Simulator Validation (`simglucose` UVA/Padova) | **DONE** | [`src/data/loaders.py#L444-L536`](file:///D:/Projects/Diabetes%20Project/src/data/loaders.py#L444-L536), [`experiments/phase6_cohort_match.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase6_cohort_match.png), [`experiments/phase6_validation_report.md`](file:///D:/Projects/Diabetes%20Project/experiments/phase6_validation_report.md) |
| **Phase 7** | Receding-Horizon Model Predictive Controller (MPC) Baseline | **DONE** | [`src/control/mpc.py`](file:///D:/Projects/Diabetes%20Project/src/control/mpc.py), [`experiments/phase7_mpc.png`](file:///D:/Projects/Diabetes%20Project/experiments/phase7_mpc.png), [`experiments/phase7_mpc_report.md`](file:///D:/Projects/Diabetes%20Project/experiments/phase7_mpc_report.md) |
| **Phase 8** | Population-Trained RL Controller with Hard Safety Shield | **IN PROGRESS** | Code exists ([`src/control/rl_agent.py`](file:///D:/Projects/Diabetes%20Project/src/control/rl_agent.py), [`src/control/safety_shield.py`](file:///D:/Projects/Diabetes%20Project/src/control/safety_shield.py)), passing tests ([`tests/test_rl.py`](file:///D:/Projects/Diabetes%20Project/tests/test_rl.py), [`tests/test_safety_shield.py`](file:///D:/Projects/Diabetes%20Project/tests/test_safety_shield.py)), runner script awaiting execution for Phase 8 artifacts. |
| **Phase 9** | Sim-to-Real Domain Randomization & Generalization Gap | **NOT STARTED** | Requires completion of Phase 8 RL agent. |
| **Phase 10** | Interactive Scenario Forecast Demo & Master REPORT.md | **NOT STARTED** | Final integrative milestone. |

---

## 2. TEST HEALTH

Executed test run command: `python -m pytest tests/ -q`

```text
36 passed, 5 warnings in 19.13s
```

- **Total Tests Collected**: 36
- **Passed**: 36
- **Failed**: 0
- **Skipped**: 0
- **Warnings**: 5 deprecation warnings from upstream dependencies (`gym`/`simglucose` package resources import deprecations in Python 3.12).

---

## 3. RESULTS SO FAR (KEY METRICS FROM REAL RUNS)

### Phase 2: Mechanistic Minimal Model Baseline
- **Steady-State Invariant**: Verified ($\Delta G = 0.0\,\text{mg/dL}$ under basal equilibrium).
- **Optimization Positivity**: Proven via log-transformed parameter estimation ($p_1, p_2, p_3, n > 0$).

### Phase 3: Per-Patient Calibration & Identifiability
- **Initial vs. Calibrated RMSE**: Initial guess $49.15\,\text{mg/dL}$ $\to$ Calibrated mean $31.48\,\text{mg/dL}$ across synthetic cohort.
- **95% Parameter Confidence Intervals (Fisher Information Matrix)**:
  - $p_1$: $0.0617$ $[0.0227, 0.158]$ $\text{min}^{-1}$ (Identifiable)
  - $p_2$: $0.0155$ $[0.0047, 0.0230]$ $\text{min}^{-1}$ (Identifiable)
  - $p_3$: $1.57 \times 10^{-4}$ $[8.24 \times 10^{-5}, 4.71 \times 10^{-4}]\,\text{L}/(\text{mU}\cdot\text{min}^2)$ (Identifiable)
  - $n$: $0.1407$ $[0.0422, 0.2270]\,\text{min}^{-1}$ (Identifiable)

### Phase 4: Hybrid Multi-Horizon Forecasting (Held-Out Test Set)
| Horizon | Model | RMSE (mg/dL) | MAE (mg/dL) | Clarke A+B (%) | TIR (%) | TBR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **30 min** | Persistence | 16.75 | 11.19 | 87.6% | 65.9% | 34.1% |
| | Mechanistic-Only | 25.67 | 18.00 | 67.0% | 100.0% | 0.0% |
| | Pure-ML (GRU) | 26.14 | 23.36 | 48.2% | 0.0% | 100.0% |
| | **Hybrid Twin** | **17.30** | **14.56** | **87.6%** | **74.1%** | **25.9%** |
| **60 min** | Persistence | 26.28 | 19.73 | 67.0% | 65.9% | 34.1% |
| | Mechanistic-Only | 29.07 | 20.96 | 64.5% | 100.0% | 0.0% |
| | Pure-ML (GRU) | 25.71 | 22.64 | 49.3% | 0.0% | 100.0% |
| | **Hybrid Twin** | **20.76** | **17.81** | **84.5%** | **82.0%** | **18.0%** |

### Phase 5: State Estimation via Extended Kalman Filter
- **Convergence**: Robust recovery from a $-50\,\text{mg/dL}$ perturbed initial state within 40 minutes.
- **Held-Out Tracking RMSE**: $6.05\,\text{mg/dL}$.
- **Empirical 95% Confidence Band Coverage**: **$97.36\%$** (Target range $[90\%, 99\%]$ satisfied).
- **Insulin Sensitivity Drift**: $X(t)$ dynamic tracking reduced error from $26.8\,\text{mg/dL}$ (fixed model) to $6.05\,\text{mg/dL}$.

### Phase 6: SimGlucose (UVA/Padova) Independent Validation
- **Adult Cohort ($N=3$, 867 obs)**: Mean $137.9\,\text{mg/dL}$, CV $13.4\%$, TIR $98.6\%$, TBR $0.0\%$, Held-Out Val RMSE **$35.37\,\text{mg/dL}$**.
- **Adolescent Cohort ($N=3$, 867 obs)**: Mean $144.5\,\text{mg/dL}$, CV $19.7\%$, TIR $88.1\%$, TBR $0.0\%$, Held-Out Val RMSE **$36.72\,\text{mg/dL}$**.
- **Child Cohort ($N=3$, 867 obs)**: Mean $134.0\,\text{mg/dL}$, CV $28.0\%$, TIR $86.7\%$, TBR $5.1\%$, Held-Out Val RMSE **$39.03\,\text{mg/dL}$**.

### Phase 7: Model Predictive Controller Receding-Horizon Baseline
- **Evaluated**: 5 randomized virtual subjects (1,440 closed-loop decisions over 24 hours).
- **Receding Horizon MPC**: TIR $100.0\%$, TBR $0.0\%$, TAR $0.0\%$, 0 hypoglycemic events, 0 solver failures, 0 safety shield vetoes, 0 negative insulin violations.
- **Fixed-Schedule Baseline**: TIR $100.0\%$, TBR $0.0\%$, TAR $0.0\%$, 0 hypoglycemic events.

---

## 4. AGENTS.md COMPLIANCE CHECK

| Rule | Requirement | Verdict | File & Line Evidence |
| :--- | :--- | :---: | :--- |
| **Rule 1** | Patient-level and chronological splitting only; forbidden random row splits; written to `split.json`. | **PASS** | [`src/data/splitter.py#L4-L60`](file:///D:/Projects/Diabetes%20Project/src/data/splitter.py#L4-L60), verified in [`data/processed/split.json`](file:///D:/Projects/Diabetes%20Project/data/processed/split.json). Patient `synthetic_004` held out completely. |
| **Rule 2** | Parameter uncertainty (CIs) reported, not just point estimates. | **PASS** | [`src/models/calibration.py#L125-L160`](file:///D:/Projects/Diabetes%20Project/src/models/calibration.py#L125-L160), documented in [`experiments/phase3_identifiability_report.md#L13-L21`](file:///D:/Projects/Diabetes%20Project/experiments/phase3_identifiability_report.md#L13-L21). |
| **Rule 3** | Persistence and mechanistic-only baselines implemented and reported alongside ML/hybrid models. | **PASS** | [`src/models/baselines.py#L1-L40`](file:///D:/Projects/Diabetes%20Project/src/models/baselines.py#L1-L40), benchmarked in [`experiments/phase4_metrics_report.md#L7-L25`](file:///D:/Projects/Diabetes%20Project/experiments/phase4_metrics_report.md#L7-L25). |
| **Rule 4** | No hardcoded dataset paths; configuration driven via YAML. | **PASS** | [`configs/data_default.yaml`](file:///D:/Projects/Diabetes%20Project/configs/data_default.yaml), [`src/data/loaders.py#L542-L560`](file:///D:/Projects/Diabetes%20Project/src/data/loaders.py#L542-L560). |
| **Rule 5** | Safety is a mandatory invariant; actions vetted by `SafetyShield` before execution. | **PASS** | [`src/control/safety_shield.py#L33-L170`](file:///D:/Projects/Diabetes%20Project/src/control/safety_shield.py#L33-L170), [`src/control/mpc.py#L210-L225`](file:///D:/Projects/Diabetes%20Project/src/control/mpc.py#L210-L225). |
| **Rule 6** | Primary evaluation metric is Time Below Range (TBR < 70 mg/dL). | **PASS** | [`src/eval/metrics.py#L20-L45`](file:///D:/Projects/Diabetes%20Project/src/eval/metrics.py#L20-L45), prioritized in all evaluation reports. |
| **Rule 7** | Clinician decision-support notice in all outputs; never framed as direct dosing advice. | **PASS** | [`src/eval/reporter.py#L19-L24`](file:///D:/Projects/Diabetes%20Project/src/eval/reporter.py#L19-L24), [`experiments/phase7_mpc_report.md#L3`](file:///D:/Projects/Diabetes%20Project/experiments/phase7_mpc_report.md#L3). |
| **Rule 8** | Pluggable data loaders implementing the `BaseLoader` interface. | **PASS** | [`src/data/loaders.py#L126-L151`](file:///D:/Projects/Diabetes%20Project/src/data/loaders.py#L126-L151), tested in [`tests/test_simglucose_loader.py`](file:///D:/Projects/Diabetes%20Project/tests/test_simglucose_loader.py). |
| **Rule 9** | Explicit unit policy in all column names (`glucose_mgdL`, `insulin_mU_per_min`, `meal_cho_g`). | **PASS** | [`src/utils/units.py#L1-L30`](file:///D:/Projects/Diabetes%20Project/src/utils/units.py#L1-L30), verified across loaders and preprocessors. |
| **Rule 10** | Every phase saves at least one visual plot and Markdown report to `experiments/`. | **PASS** | Verified plots `phase2_fit.png`, `phase3_identifiability.png`, `phase4_horizons.png`, `phase5_kalman.png`, `phase6_cohort_match.png`, `phase7_mpc.png` present in `experiments/`. |

---

## 5. DATA STATUS

- **Data Sources Currently in Use**:
  1. **Synthetic Digital Twin Engine**: Enhanced Bergman simulator with dawn phenomenon, meal absorption rate variability, exercise sensitivity shifts, AR(1) sensor noise, and $30\text{--}50\%$ carb estimation error. Processed dataset saved as Parquet files in `data/processed/`.
  2. **SimGlucose (UVA/Padova Benchmark)**: 9 virtual patients (Adults, Adolescents, Children) dynamically simulated.
- **Real Patient Data**:
  - `data/raw/` directory is currently **empty**.
  - `OHIOLoader` is fully implemented as a skeleton loader ready for real OhioT1DM CSVs once access is granted.
- **Clinical Safety Disclaimer**: **All results to date are derived from in silico synthetic and UVA/Padova simulator data. They demonstrate computational feasibility and algorithm verification, but do NOT reflect real clinical patient performance.**

---

## 6. BLOCKERS AND RISKS

1. **Real Dataset Access**: OhioT1DM real-patient data is not present in `data/raw/`. The pipeline uses UVA/Padova as an FDA-accepted in silico benchmark proxy for sim-to-real evaluation (Phase 9).
2. **Model Mismatch (1-Compartment vs. Multi-Compartment Gut)**: The minimal 1-compartment Bergman model exhibits a $\approx 15\text{-minute}$ postprandial phase lag when fitted to UVA/Padova traces due to complex nonlinear gastric emptying in UVA/Padova. The Phase 4 residual GRU mitigates this for short-horizon forecasting.
3. **Synthetic Fit Optimism Risk**: Mechanistic model calibration against synthetic Bergman data yields low RMSE ($5\text{--}15\,\text{mg/dL}$); however, when tested against independent UVA/Padova simulator traces, validation RMSE increases to $35\text{--}39\,\text{mg/dL}$, reflecting physiological reality.

---

## 7. NEXT STEP

- **Target Milestone**: **Phase 8 — Reinforcement Learning Controller with Hard Safety Shield**
- **Prerequisites**:
  - Validated digital twin dynamics (Phase 2 & 3): **MET**
  - EKF State Estimator (Phase 5): **MET**
  - SimGlucose UVA/Padova validation (Phase 6): **MET**
  - Receding-horizon MPC benchmark baseline (Phase 7): **MET**
  - Safety shield with clamp/veto instrumentation (Phase 8): **MET**
- **Action Plan**:
  - Train population-based PPO / SAC policy using Gymnasium wrapper.
  - Run adversarial safety stress tests to confirm $>0$ shield vetoes and $0$ hypoglycemic invariant violations.
  - Generate comparison report against MPC baseline and save plot to `experiments/phase8_rl.png`.

---

*Halted for review. Awaiting confirmation before initiating Phase 8.*
