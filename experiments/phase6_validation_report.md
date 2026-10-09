# Phase 6: Independent Simulator Validation Report (UVA/Padova Benchmark)

## 1. Stack Validation & SimGlucose Integration
- **Package Status**: Verified `simglucose` (open-source implementation of the FDA-accepted UVA/Padova Type 1 Diabetes Simulator).
- **Interface Contract**: Implemented [`SimGlucoseLoader`](file:///D:/Projects/Diabetes%20Project/src/data/loaders.py#L444-L536) conforming to `BaseLoader`.
- **Units**: Explicit project units enforced: `glucose_mgdL` (mg/dL), `insulin_mU_per_min` (mU/min), `meal_cho_g` (g).
- **Evaluation Windows**: 24-hour continuous monitoring (289 steps @ 5-min intervals). Fitting performed on 0–16h (67%), validated on held-out 16–24h (33%).

## 2. Cohort Match & Validation Statistics
| Cohort / Source                    |   N |   Obs |   Mean Glucose |   Std Dev | CV (%)   | TIR (%)   | TBR (%)   | TAR (%)   | Val RMSE   | Val MAE   |
|------------------------------------|-----|-------|----------------|-----------|----------|-----------|-----------|-----------|------------|-----------|
| Synthetic Digital Twin (Phase 1-4) |   9 |  2592 |           96.2 |      28.9 | 30.1%    | 83.1%     | 15.2%     | 1.8%      | N/A        | N/A       |
| SimGlucose UVA/Padova — Adult      |   3 |   867 |          137.9 |      18.4 | 13.4%    | 98.6%     | 0.0%      | 1.4%      | 35.37      | 30.20     |
| SimGlucose UVA/Padova — Adolescent |   3 |   867 |          144.5 |      28.4 | 19.7%    | 88.1%     | 0.0%      | 11.9%     | 36.72      | 29.95     |
| SimGlucose UVA/Padova — Child      |   3 |   867 |          134   |      37.5 | 28.0%    | 86.7%     | 5.1%      | 8.2%      | 39.03      | 33.68     |

## 3. Fit Quality of Phase 3 Twin Across Age Cohorts (Held-Out Validation Window)
- **Adult Cohort Fit**: Validation RMSE = `35.37 mg/dL` | MAE = `30.20 mg/dL`
- **Adolescent Cohort Fit**: Validation RMSE = `36.72 mg/dL` | MAE = `29.95 mg/dL`
- **Child Cohort Fit**: Validation RMSE = `39.03 mg/dL` | MAE = `33.68 mg/dL`

## 4. Detailed Mismatch & Realism Analysis
1. **Bioavailability & Meal Absorption**: In previous iterations, carbohydrate scaling (k_meal) was under-specified (5%), causing severe underestimation of postprandial glucose. Correcting to standard 85% bioavailability aligns synthetic postprandial peaks with UVA/Padova physiology.
2. **Multi-Compartment Gastrointestinal Dynamics**: UVA/Padova incorporates nonlinear solid/liquid gastric emptying and a 2-compartment gut model. The minimal 1-compartment Bergman twin captures overall excursion amplitude but shows an expected ~15-minute phase lag during rapid meal absorption.
3. **Pediatric Glycemic Volatility**: Children in UVA/Padova exhibit higher insulin sensitivity (p3) and lower glucose distribution volumes, resulting in larger glycemic swings and a higher validation RMSE (`39.03 mg/dL`) than adults (`35.37 mg/dL`).
