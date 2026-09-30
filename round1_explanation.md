# Round 1: Neonatal triage, GA + fuzzy (baseline)

**Final fitness (validation): 0.4611** | Held-out test: AUROC 0.9496, critical recall 98.98% (4 missed of 394; 95% Wilson CI 97.4-99.6%), specificity 55.9%, 0.059 ms per patient.

Data is synthetic (20,000 neonates, ~7.9% critical, 27 candidate features incl. redundant, pure-noise and institutional ones). All numbers are from `run_log_round1.txt` (seed 42).

## Representation
Binary chromosome, one bit per candidate feature (1 = used). A repair operator runs after every crossover/mutation and guarantees: 5-10 features, at most 2 static/demographic, at least 50% physiological (vitals + labs), and no institutional features (`site_id`, `nicu_level`). Verified: 0 invalid chromosomes in 2,000 random repairs.

## Fuzzy system
- Each feature has LOW / NORMAL / HIGH trapezoids from training percentiles (10/35/65/90). The three degrees always sum to 1 (Ruspini partition).
- Rules have 1-2 antecedents (AND = min). At most 16 rules, chosen by greedy Newton (boosting-style) selection so that each new rule adds information the earlier ones lack.
- Zero-order TSK output: score = sum of (rule firing x log-odds weight). A missing input makes its rules fire at 0, so it contributes neutral evidence rather than a made-up imputed value.
- Risk tiers LOW / WATCH / HIGH / CRITICAL come from score cut-offs. The LOW/WATCH boundary is calibrated on train at 99.5% critical recall (safety margin over the 98% requirement).

## Fitness
`0.40*AUROC + 0.25*specificity - 1.0*FNR - 0.10*sensor_cost - 0.05*redundancy - 0.15*robustness_drop`, minus `2.0 + 10*shortfall` if validation recall < 98%.
The threshold is set on train and recall is checked on validation, so subsets that only look safe on the data they were fit on get penalised.

## GA operators and parameters
Population 40, 40 generations, tournament size 3, uniform crossover (p=0.85), bit-flip mutation 0.05, elitism 2, repair. Memoised fitness (1,091 unique evaluations). Best fitness went 0.3736 (gen 0) to 0.4611 (last improvement gen 26); population mean went 0.251 to 0.406. See `convergence_round1.png`.

## Why CI here
GA: the subset search is combinatorial and non-differentiable, and hard constraints are easy to enforce with repair. Fuzzy: clinicians can read the rules, and missing inputs degrade gracefully instead of breaking the model.

## Known weaknesses (for Round 2 / jury)
- Recall lower confidence bound (97.4%) is below 98%, so the 98% requirement is met by the point estimate, not proven. More positives or a higher calibration target would tighten this.
- Losing `lactate` drops test recall to 93.7%, and losing `crp` drops it to 96.7%: single points of failure. Round 2 should reward backup sensors and mask features during training.
- The GA picked both `gestational_age_wk` and `birth_weight_g` (both static slots, and highly correlated).
- The naive top-10 univariate baseline scored 0.4191 fitness vs 0.4611, so the GA's gain is real but modest on this synthetic data.
- Specificity is ~56% at ~99% recall: this is the false-alarm cost of a safety-first threshold.
