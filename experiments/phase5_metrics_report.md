# Digital Twin Model Evaluation Report

> **CLINICIAN DECISION-SUPPORT NOTICE**
> This system is a research prototype digital twin for Type 1 Diabetes simulation and state estimation.
> It is **NOT** a certified medical device and must **NEVER** be used for direct therapy adjustment,
> automated insulin dosing without human oversight, or independent clinical diagnosis.


## Performance Comparison

| Model | TBR (<70) % | TIR (70-180) % | TAR (>180) % | RMSE (mg/dL) | MAE (mg/dL) | MARD % | Clarke A+B % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persistence Baseline | 100.00 | 0.00 | 0.00 | 16.55 | 12.31 | 39.58 | 96.18 |
| Mechanistic Bergman | 95.83 | 4.17 | 0.00 | 29.62 | 26.76 | 59.51 | 94.79 |
| Digital Twin (Composite) | 0.00 | 100.00 | 0.00 | 46.73 | 45.52 | 127.61 | 3.82 |

