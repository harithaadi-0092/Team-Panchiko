# Neonatal Triage: GA Feature Selection + Fuzzy Risk Stratification

An interpretable pipeline for early triage of critically ill neonates. A **genetic algorithm (GA)**
chooses a small, safe, low-cost set of measurements, and a **fuzzy rule system** turns them into
transparent risk tiers (LOW / WATCH / HIGH / CRITICAL).

> **Research prototype, not a medical device.** Results below come from synthetic data and show that
> the method works, not that it is clinically valid.

## The problem
Critical neonates are rare (~8% here), so accuracy is misleading: predicting "not critical" for
everyone scores ~92% and catches nobody. The model must:
- catch **>98%** of critical cases,
- avoid class collapse on imbalanced data,
- avoid leaning on demographic or institutional shortcuts (gestational age, birth weight, hospital ID),
- run fast at the bedside and stay stable when readings are missing.

## How it works
1. **Fuzzy engine:** LOW / NORMAL / HIGH trapezoidal memberships, at most 16 IF-THEN rules
   (1-2 conditions each), zero-order Takagi-Sugeno scoring. Rules are chosen by greedy
   Newton (boosting-style) selection. Missing inputs contribute zero evidence.
2. **Genetic algorithm:** binary chromosome (one bit per feature), tournament selection,
   uniform crossover, bit-flip mutation, elitism, and a **repair operator** enforcing:
   5-10 features, at most 2 demographic, at least 50% physiological, no institutional features.
3. **Fitness:** `0.40*AUROC + 0.25*Specificity - 1.0*FNR - 0.10*SensorCost - 0.05*Redundancy
   - 0.15*RobustnessDrop`, minus a heavy penalty if validation recall < 98%.
4. **Evaluation:** held-out test set the GA never sees, with Wilson confidence intervals, missing-data
   and sensor-dropout sweeps, tier breakdown, and latency.

## Results (synthetic cohort, seed 42)
| Metric | Value |
|---|---|
| Final fitness (validation) | 0.461 |
| Test AUROC | 0.950 |
| Critical recall | 98.98% (95% CI 97.4-99.6%) |
| Specificity | 55.9% |
| Latency | ~0.06 ms / patient |

**Known weaknesses:** the recall lower confidence bound is just under 98%; losing the lactate sensor
drops recall to ~93.7%. Robustness to single-sensor loss is the focus of the next iteration.

## Files
| File | Purpose |
|---|---|
| `neonatal_ga_fuzzy.py` | Trains the model (synthetic data or your own Excel/CSV) |
| `neonatal_model.json` | Trained model from the synthetic run |

## Input format
One row per patient, one column per measurement, and an outcome column (1/0, yes/no,
critical/stable, ...). Blank cells are allowed.

## Limitations
- Trained and validated on synthetic data only; do not use for real clinical decisions.
- Fuzzy cut-points are percentile-based (relative to the cohort), not clinical reference ranges.
- Small numbers of critical cases give wide recall confidence intervals.
