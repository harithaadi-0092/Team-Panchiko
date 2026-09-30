"""
Neonatal critical-illness triage: GA feature selection + fuzzy risk stratification
ROUND 1 (baseline)

Pipeline
  1. Synthetic NICU cohort (imbalanced, noisy, redundant + irrelevant + institutional features)
  2. Stratified train / val / test split (test is never seen by the GA)
  3. Fuzzy engine: Ruspini LOW/NORMAL/HIGH memberships, <=2-antecedent rules,
     zero-order TSK with additive aggregation (a missing input contributes zero evidence);
     rules chosen by greedy Newton/boosting selection so they are complementary
  4. GA over a binary feature mask, with a repair operator enforcing the bounds
  5. Fitness = weighted objectives, with a hard gate on critical-class recall
  6. Held-out evaluation: AUROC, recall + Wilson CI, robustness sweep, latency

Run (synthetic data):   python neonatal_ga_fuzzy.py
Run (YOUR real data):   python neonatal_ga_fuzzy.py --data patients.xlsx --label critical
    optional: --schema schema.csv  --sheet Sheet1  --id-col patient_id  --exclude colA,colB
              --generations 40  --pop 40  --seed 42
    First run without --schema: the script guesses each column's group/cost from its name,
    prints the guesses and writes schema_template.csv for you to correct and pass back in.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# =============================================================================
# 0. PARAMETER CONFIGURATION (everything tunable lives here)
# =============================================================================


@dataclass(frozen=True)
class Config:
    seed: int = 42
    # --- data ---
    n_patients: int = 20000
    prevalence: float = 0.08          # ~8% critical neonates -> heavy imbalance
    # --- GA ---
    pop_size: int = 40
    generations: int = 40
    tournament_k: int = 3
    crossover_rate: float = 0.85      # uniform crossover
    mutation_rate: float = 0.05       # per-bit flip probability
    elite: int = 2
    # --- feature-selection bounds (hard constraints) ---
    k_min: int = 5
    k_max: int = 10
    max_static: int = 2               # at most 2 demographic/static features
    min_physio_frac: float = 0.5      # >= half of selected must be physiological
    # institutional features (site_id, nicu_level) are never allowed
    # --- fuzzy engine ---
    n_rules: int = 16                 # rule-base cap (readability + latency)
    rule_smoothing: float = 20.0      # Laplace-style shrinkage used for candidate shortlisting
    shortlist: int = 120              # candidate rules kept before greedy selection
    rule_ridge: float = 5.0           # ridge on Newton step (stops tiny-support rules getting huge weights)
    # --- clinical safety gate ---
    recall_floor: float = 0.98        # hard requirement on critical recall
    recall_target: float = 0.995      # threshold calibrated on TRAIN at this recall (safety margin)
    # --- fitness weights ---
    w_auc: float = 0.40
    w_spec: float = 0.25              # specificity at the deployed threshold (false-alarm burden)
    w_fnr: float = 1.00               # false-negative rate on critical cases
    w_cost: float = 0.10              # sensor acquisition cost
    w_red: float = 0.05               # near-duplicate redundancy
    w_rob: float = 0.15               # degradation under missing telemetry
    gate_penalty: float = 2.0         # subtracted if recall < floor
    # --- robustness probe used inside fitness ---
    miss_rate: float = 0.20
    n_mask_draws: int = 2
    redundancy_corr: float = 0.70     # only |corr| above this counts as redundant


TERMS = ("LOW", "NORMAL", "HIGH")

# =============================================================================
# 1. SYNTHETIC NEONATAL COHORT
# =============================================================================
# name, group, mu, sd, shift(sd units), subtype, cost, lognormal?
# groups: physio (bedside monitor), lab (blood draw), static (demographic), inst (institutional)
# subtypes: 0 sepsis, 1 respiratory, 2 cardiovascular, 3 prematurity, -1 none
SPEC = [
    ("heart_rate",        "physio", 145, 15,   1.4, 2, 1.0, False),
    ("resp_rate",         "physio",  46,  8,   1.3, 1, 1.0, False),
    ("spo2",              "physio",  96, 2.5, -2.0, 1, 1.0, False),
    ("temperature",       "physio", 36.8, 0.4, -1.3, 0, 1.0, False),
    ("mean_bp",           "physio",  45,  7,  -1.4, 2, 3.0, False),
    ("hrv_rmssd",         "physio",  30,  8,  -1.3, 0, 2.0, False),
    ("apnea_per_hr",      "physio", 0.8, 0.7,  1.6, 1, 1.0, True),
    ("cap_refill_s",      "physio", 2.0, 0.5,  2.0, 2, 1.0, False),
    ("lactate",           "lab",    1.8, 0.35, 2.2, 0, 4.0, True),
    ("blood_glucose",     "lab",     80, 20,  -0.8, 0, 2.0, False),
    ("blood_ph",          "lab",   7.35, 0.05,-1.5, 2, 4.0, False),
    ("crp",               "lab",    3.0, 0.8,  1.6, 0, 4.0, True),
    ("wbc",               "lab",     12,  4,   0.8, 0, 4.0, False),
    ("bilirubin",         "lab",      8,  3,   0.3, 3, 3.0, False),
    ("hematocrit",        "lab",     50,  6,   0.0, -1, 3.0, False),
    ("sodium",            "lab",    138,  3,   0.0, -1, 3.0, False),
    ("gestational_age_wk", "static", 38,  2,  -2.5, 3, 0.5, False),
    ("apgar_5min",        "static", 8.6,  1,  -1.5, 3, 0.5, False),
    ("sex",               "static",  0.5, 0.5, 0.0, -1, 0.5, False),
]
DERIVED = [  # redundant backups of primary sensors
    ("hr_pulseox",       "physio", 1.0),
    ("spo2_secondary",   "physio", 1.0),
    ("birth_weight_g",   "static", 0.5),
]
NOISE = [f"aux_marker_{i}" for i in (1, 2, 3)]    # pure-noise labs
INST = ["site_id", "nicu_level"]                  # institutional shortcuts (forbidden)
LEAK = 0.3      # non-primary-subtype features still shift slightly


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    names: list
    groups: np.ndarray
    cost: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def generate_cohort(cfg: Config, rng: np.random.Generator) -> Dataset:
    n = cfg.n_patients
    y = (rng.random(n) < cfg.prevalence).astype(int)

    # each critical neonate has a primary deterioration pathway (+ sometimes a second)
    primary = rng.choice(4, n, p=[0.30, 0.30, 0.25, 0.15])
    secondary = rng.choice(4, n)
    has2 = rng.random(n) < 0.25
    active = np.zeros((n, 4), bool)
    active[np.arange(n), primary] = True
    active[np.flatnonzero(has2), secondary[has2]] = True
    active &= (y[:, None] == 1)

    cols, names, groups, cost = {}, [], [], []

    def strength(sub):
        if sub < 0:
            return np.zeros(n)
        return np.where(y == 1, np.where(active[:, sub], 1.0, LEAK), 0.0)

    for name, grp, mu, sd, shift, sub, c, lg in SPEC:
        z = rng.standard_normal(n) + shift * strength(sub)
        x = np.exp(np.log(mu) + sd * z) if lg else mu + sd * z
        if name == "sex":
            x = (rng.random(n) < 0.5).astype(float)
        cols[name] = x
        names.append(name); groups.append(grp); cost.append(c)

    cols["spo2"] = np.clip(cols["spo2"], 60, 100)
    cols["gestational_age_wk"] = np.clip(cols["gestational_age_wk"], 24, 42)
    cols["apgar_5min"] = np.clip(np.round(cols["apgar_5min"]), 0, 10)

    # redundant / derived sensors
    cols["hr_pulseox"] = cols["heart_rate"] + rng.normal(0, 5, n)
    cols["spo2_secondary"] = np.clip(cols["spo2"] + rng.normal(0, 1.0, n), 60, 100)
    cols["birth_weight_g"] = np.clip(
        700 + 190 * (cols["gestational_age_wk"] - 24) + rng.normal(0, 350, n)
        - 250 * strength(3), 500, 5000)
    for name, grp, c in DERIVED:
        names.append(name); groups.append(grp); cost.append(c)

    for name in NOISE:
        cols[name] = rng.standard_normal(n)
        names.append(name); groups.append("lab"); cost.append(2.0)

    # institutional shortcut: referral centre (site 0) sees more critical babies
    p_site = np.where(y[:, None] == 1, [[.45, .20, .15, .10, .10]], [[.2] * 5])
    site = (rng.random(n)[:, None] > np.cumsum(p_site, axis=1)).sum(1).clip(0, 4)
    cols["site_id"] = site.astype(float)
    cols["nicu_level"] = np.where(site == 0, 4, np.where(site == 1, 3, 2)).astype(float)
    for name in INST:
        names.append(name); groups.append("inst"); cost.append(0.0)

    X = np.column_stack([cols[k] for k in names])
    idx = np.arange(n)
    i_tr, i_tmp = train_test_split(idx, test_size=0.5, stratify=y, random_state=cfg.seed)
    i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=cfg.seed)
    return Dataset(X, y, names, np.array(groups), np.array(cost), i_tr, i_va, i_te)



# =============================================================================
# 1b. REAL-DATA LOADER (Excel / CSV)
# =============================================================================
# Group guesses are made from column names. ALWAYS check them (or pass --schema).
_GROUP_KEYS = {
    "inst":   {"site", "hospital", "centre", "center", "institution", "facility", "unit", "ward",
               "clinic", "region", "hospitalid", "siteid", "source", "nicu"},
    "static": {"age", "sex", "gender", "weight", "ga", "bw", "apgar", "birth", "gestational",
               "gestation", "gravida", "parity", "maternal", "delivery", "race", "ethnicity",
               "height", "length", "head", "twin", "multiple"},
    "lab":    {"lactate", "crp", "glucose", "ph", "wbc", "bilirubin", "hematocrit", "hct", "sodium",
               "potassium", "creatinine", "platelet", "platelets", "hemoglobin", "procalcitonin",
               "bicarbonate", "urea", "bun", "culture", "gas", "pco2", "po2", "base", "excess",
               "neutrophil", "calcium"},
}
_DEFAULT_COST = {"physio": 1.0, "lab": 3.0, "static": 0.5, "inst": 0.0}
_POS_WORDS = {"1", "yes", "y", "true", "t", "critical", "died", "dead", "death", "deceased",
              "positive", "pos", "high", "sick", "case", "non-survivor", "nonsurvivor"}
_NEG_WORDS = {"0", "no", "n", "false", "f", "stable", "survived", "alive", "negative", "neg",
              "low", "healthy", "control", "non-critical", "noncritical", "survivor", "normal"}


def guess_group(name: str) -> str:
    low = name.lower()
    tokens = set(t for t in re.split(r"[^a-z0-9]+", low) if t)
    for grp in ("inst", "static", "lab"):
        keys = _GROUP_KEYS[grp]
        if tokens & keys or any(k in low for k in keys if len(k) >= 5):
            return grp
    return "physio"


def _read_table(path: str, sheet=None):
    import pandas as pd
    if str(path).lower().endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    return pd.read_csv(path)


def _parse_label(col) -> np.ndarray:
    """Return float array of 0/1 (NaN where unreadable/missing)."""
    out = np.full(len(col), np.nan)
    for i, v in enumerate(col):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        t = str(v).strip().lower()
        if t in _POS_WORDS:
            out[i] = 1.0
        elif t in _NEG_WORDS:
            out[i] = 0.0
        else:
            try:
                out[i] = 1.0 if float(t) > 0 else 0.0
            except ValueError:
                pass
    return out


def load_real_dataset(path: str, label: str, cfg: Config, sheet=None, schema_path: str | None = None,
                      id_col: str | None = None, exclude: list[str] | None = None,
                      max_missing: float = 0.6) -> Dataset:
    """Read a spreadsheet into the same Dataset structure the synthetic generator produces."""
    import pandas as pd
    df = _read_table(path, sheet)
    df.columns = [str(c).strip() for c in df.columns]
    if label not in df.columns:
        raise SystemExit(f"Label column '{label}' not found. Columns are: {list(df.columns)}")

    y_all = _parse_label(df[label])
    keep_rows = ~np.isnan(y_all)
    if (~keep_rows).any():
        print(f"[loader] dropped {int((~keep_rows).sum())} rows with a missing/unreadable '{label}' value")
    df, y = df.loc[keep_rows].reset_index(drop=True), y_all[keep_rows].astype(int)
    if len(np.unique(y)) < 2:
        raise SystemExit("The label column has only one class; need both critical and non-critical rows.")

    skip = {label} | set(exclude or []) | ({id_col} if id_col else set())
    schema = None
    if schema_path:
        schema = _read_table(schema_path)
        schema.columns = [str(c).strip().lower() for c in schema.columns]
        schema["feature"] = schema["feature"].astype(str).str.strip()
        schema = schema.set_index("feature")

    names, groups, costs, cols, notes = [], [], [], [], []
    for c in df.columns:
        if c in skip:
            continue
        if schema is not None and c not in schema.index:
            notes.append(f"'{c}' not in schema -> ignored")
            continue
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):                # text column (e.g. M/F): code it
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().mean() < 0.5:
                codes, uniq = pd.factorize(s.where(s.notna()))
                if len(uniq) > 6:
                    notes.append(f"'{c}' is text with {len(uniq)} categories -> ignored")
                    continue
                num = pd.Series(np.where(codes < 0, np.nan, codes), dtype=float)
                notes.append(f"'{c}' text coded as {dict(enumerate(uniq))}")
            s = num
        x = pd.to_numeric(s, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan).to_numpy()
        miss = np.isnan(x).mean()
        if miss > max_missing:
            notes.append(f"'{c}' is {miss:.0%} missing -> dropped"); continue
        if np.nanstd(x) == 0 or np.isnan(x).all():
            notes.append(f"'{c}' is constant -> dropped"); continue
        if schema is not None:
            g = str(schema.loc[c, "group"]).strip().lower()
            cost = float(schema.loc[c, "cost"]) if "cost" in schema.columns and pd.notna(schema.loc[c, "cost"]) \
                else _DEFAULT_COST.get(g, 1.0)
        else:
            g = guess_group(c)
            cost = _DEFAULT_COST[g]
        if g not in _DEFAULT_COST:
            raise SystemExit(f"Feature '{c}' has group '{g}'; use one of {list(_DEFAULT_COST)}")
        names.append(c); groups.append(g); costs.append(cost); cols.append(x)

    if not names:
        raise SystemExit("No usable numeric feature columns found.")
    X = np.column_stack(cols)
    groups, costs = np.array(groups), np.array(costs)

    print(f"[loader] {path}: {len(y)} patients, {int(y.sum())} critical ({y.mean():.2%}), {len(names)} usable features")
    for n in notes:
        print(f"[loader] note: {n}")
    print(f"[loader] {'feature':28s}{'group':8s}{'cost':>6s}{'missing':>9s}   "
          f"({'from schema' if schema is not None else 'GUESSED from names - please verify'})")
    for j, n in enumerate(names):
        print(f"[loader] {n:28s}{groups[j]:8s}{costs[j]:6.1f}{np.isnan(X[:, j]).mean():9.1%}")
    if schema is None:
        with open("schema_template.csv", "w") as f:
            f.write("feature,group,cost\n")
            for n, g, c in zip(names, groups, costs):
                f.write(f"{n},{g},{c}\n")
        print("[loader] wrote schema_template.csv (edit group/cost, then rerun with --schema schema_template.csv)")

    idx = np.arange(len(y))
    try:
        i_tr, i_tmp = train_test_split(idx, test_size=0.5, stratify=y, random_state=cfg.seed)
        i_va, i_te = train_test_split(i_tmp, test_size=0.5, stratify=y[i_tmp], random_state=cfg.seed)
    except ValueError as e:
        raise SystemExit(f"Too few critical cases to split into train/val/test: {e}")
    n_te = int(y[i_te].sum())
    if n_te < 30:
        print(f"[loader] WARNING: only {n_te} critical cases in the held-out test split -> the recall "
              f"estimate will be very noisy (see its confidence interval).")
    if int(y[i_tr].sum()) < 100:
        print(f"[loader] WARNING: only {int(y[i_tr].sum())} critical cases for training the rules; "
              f"consider more data or fewer rules (Config.n_rules).")
    return Dataset(X, y, names, groups, costs, i_tr, i_va, i_te)


def fit_config_to_data(cfg: Config, groups: np.ndarray) -> Config:
    """Shrink feature-count bounds if the real dataset has fewer eligible features than the defaults."""
    n_phys = int(np.isin(groups, ["physio", "lab"]).sum())
    n_elig = int((groups != "inst").sum())
    if n_phys < 1:
        raise SystemExit("Need at least one physiological (physio/lab) feature.")
    k_max = min(cfg.k_max, n_elig)
    k_min = min(cfg.k_min, n_phys, k_max)
    if (k_min, k_max) != (cfg.k_min, cfg.k_max):
        print(f"[loader] adjusted feature bounds to {k_min}-{k_max} (only {n_elig} eligible, {n_phys} physiological)")
    return replace(cfg, k_min=k_min, k_max=k_max)


# =============================================================================
# 2. FUZZY ENGINE
# =============================================================================
def build_knots(X_train: np.ndarray) -> np.ndarray:
    """Data-driven trapezoid knots (a,b,c,d) at the 10/35/65/90th percentiles."""
    q = np.nanquantile(X_train, [0.10, 0.35, 0.65, 0.90], axis=0).T
    q = np.maximum.accumulate(q, axis=1) + 1e-6 * np.arange(4)   # strictly increasing
    return q


def memberships(X: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """(N,F) -> (N,F,3) LOW/NORMAL/HIGH degrees. Ruspini partition: rows sum to 1.
    NaN inputs propagate as NaN memberships (= 'unknown')."""
    a, b, c, d = (knots[:, i] for i in range(4))
    low = np.clip((b - X) / (b - a), 0, 1)
    high = np.clip((X - c) / (d - c), 0, 1)
    return np.stack([low, 1 - low - high, high], axis=2)


_CAND_CACHE: dict[int, np.ndarray] = {}


def candidate_rules(k: int) -> np.ndarray:
    """All 1- and 2-antecedent rules over k local features. Rows: [f1,t1,f2,t2], f2=-1 => single."""
    if k not in _CAND_CACHE:
        rows = [[f, t, -1, 0] for f in range(k) for t in range(3)]
        rows += [[f1, t1, f2, t2] for f1 in range(k) for f2 in range(f1 + 1, k)
                 for t1 in range(3) for t2 in range(3)]
        _CAND_CACHE[k] = np.array(rows, dtype=int)
    return _CAND_CACHE[k]


def fire(M: np.ndarray, rules: np.ndarray) -> np.ndarray:
    """Firing strength (N,R). AND = min. Any unknown antecedent => rule does not fire."""
    A = M[:, rules[:, 0], rules[:, 1]]
    B = M[:, np.maximum(rules[:, 2], 0), rules[:, 3]]
    F = np.where(rules[:, 2] >= 0, np.minimum(A, B), A)
    return np.nan_to_num(F, nan=0.0)


class FuzzyRiskModel:
    """Zero-order TSK: score = sum_r firing_r * w_r, w_r = log-odds shift of rule r."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fit(self, M: np.ndarray, y: np.ndarray) -> "FuzzyRiskModel":
        """Rule learning: (1) shortlist candidates by support-weighted log-odds shift,
        (2) greedy Newton (boosting-style) selection: each step adds the rule that most
        reduces logistic loss given the rules already chosen, so rules are complementary."""
        c = self.cfg
        cand = candidate_rules(M.shape[1])
        F = fire(M, cand)
        support, pos = F.sum(0), F.T @ y
        prior, m = y.mean(), c.rule_smoothing
        p0 = np.clip((pos + prior * m) / (support + m), 1e-4, 1 - 1e-4)
        w0 = np.log(p0 / (1 - p0)) - np.log(prior / (1 - prior))
        short = np.argsort(-(np.abs(w0) * np.sqrt(support)))[: c.shortlist]
        Fs, F2 = F[:, short], F[:, short] ** 2
        self.prior_logit_ = float(np.log(prior / (1 - prior)))
        raw = np.zeros(len(y))
        chosen, weights = [], []
        for _ in range(c.n_rules):
            p = 1 / (1 + np.exp(-(self.prior_logit_ + raw)))
            g = Fs.T @ (y - p)
            h = F2.T @ (p * (1 - p)) + c.rule_ridge
            gain = g * g / h
            gain[chosen] = -1
            r = int(np.argmax(gain))
            step = float(g[r] / h[r])
            chosen.append(r); weights.append(step)
            raw += step * Fs[:, r]
        # merge duplicate picks are impossible (masked); store the model
        self.rules_ = cand[short[chosen]]
        self.weights_ = np.array(weights)
        self.scale_ = float(raw.std() + 1e-9)
        return self

    def raw(self, M: np.ndarray) -> np.ndarray:
        return fire(M, self.rules_) @ self.weights_

    def risk(self, M: np.ndarray) -> np.ndarray:
        """Calibration-free risk in [0,1] (monotone in raw score)."""
        return 1 / (1 + np.exp(-(self.raw(M) / self.scale_ + self.prior_logit_)))

    def describe(self, local_names: list) -> list[str]:
        out = []
        for (f1, t1, f2, t2), w in zip(self.rules_, self.weights_):
            ante = f"{local_names[f1]} is {TERMS[t1]}"
            if f2 >= 0:
                ante += f" AND {local_names[f2]} is {TERMS[t2]}"
            out.append(f"IF {ante} THEN risk {w:+.2f} log-odds")
        return out


# =============================================================================
# 3. METRIC HELPERS
# =============================================================================
def threshold_for_recall(scores_pos: np.ndarray, target: float) -> float:
    return float(np.quantile(scores_pos, 1 - target, method="lower"))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


# =============================================================================
# 4. CONSTRAINTS + REPAIR
# =============================================================================
class FeatureConstraints:
    def __init__(self, cfg: Config, groups: np.ndarray):
        self.cfg = cfg
        self.static = groups == "static"
        self.inst = groups == "inst"
        self.physio = np.isin(groups, ["physio", "lab"])

    def is_valid(self, m: np.ndarray) -> bool:
        k = int(m.sum())
        return (self.cfg.k_min <= k <= self.cfg.k_max
                and not (m & self.inst).any()
                and (m & self.static).sum() <= self.cfg.max_static
                and (m & self.physio).sum() >= self.cfg.min_physio_frac * k)

    def repair(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c, m = self.cfg, mask.astype(bool).copy()
        m[self.inst] = False
        s = np.flatnonzero(m & self.static)
        if len(s) > c.max_static:
            m[rng.choice(s, len(s) - c.max_static, replace=False)] = False
        while m.sum() > c.k_max:
            m[rng.choice(np.flatnonzero(m))] = False
        while (m & self.physio).sum() < c.min_physio_frac * m.sum() and (m & self.static).any():
            m[rng.choice(np.flatnonzero(m & self.static))] = False
        while m.sum() < c.k_min:
            m[rng.choice(np.flatnonzero(~m & self.physio))] = True
        return m


# =============================================================================
# 5. FITNESS EVALUATOR
# =============================================================================
class Evaluator:
    def __init__(self, cfg: Config, ds: Dataset, rng: np.random.Generator):
        self.cfg, self.ds = cfg, ds
        Xtr = ds.X[ds.idx_train]
        self.knots = build_knots(Xtr)
        self.M = {n: memberships(ds.X[i], self.knots)
                  for n, i in (("train", ds.idx_train), ("val", ds.idx_val), ("test", ds.idx_test))}
        self.y = {"train": ds.y[ds.idx_train], "val": ds.y[ds.idx_val], "test": ds.y[ds.idx_test]}
        corr = np.abs(np.corrcoef(np.nan_to_num(Xtr), rowvar=False))
        self.corr = np.nan_to_num(corr)
        top = np.sort(ds.cost)[::-1][: cfg.k_max].sum()
        self.cost_norm_den = float(top)
        nv = len(ds.idx_val)
        self.val_masks = [rng.random((nv, ds.X.shape[1])) < cfg.miss_rate
                          for _ in range(cfg.n_mask_draws)]      # fixed => deterministic fitness
        self.cache: dict[bytes, dict] = {}
        self.n_evals = 0

    def redundancy(self, S: np.ndarray) -> float:
        if len(S) < 2:
            return 0.0
        R = self.corr[np.ix_(S, S)][np.triu_indices(len(S), 1)]
        return float(np.maximum(0, R - self.cfg.redundancy_corr).mean() / (1 - self.cfg.redundancy_corr))

    def fit_model(self, S: np.ndarray) -> tuple[FuzzyRiskModel, float]:
        model = FuzzyRiskModel(self.cfg).fit(self.M["train"][:, S], self.y["train"])
        s_tr = model.raw(self.M["train"][:, S])
        thr = threshold_for_recall(s_tr[self.y["train"] == 1], self.cfg.recall_target)
        return model, thr

    def evaluate(self, mask: np.ndarray) -> dict:
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]
        self.n_evals += 1
        c, S = self.cfg, np.flatnonzero(mask)
        model, thr = self.fit_model(S)
        yv, Mv = self.y["val"], self.M["val"][:, S]
        s = model.raw(Mv)
        auc = roc_auc_score(yv, s)
        recall = float((s[yv == 1] >= thr).mean())
        spec = float((s[yv == 0] < thr).mean())
        rob = 0.0
        for mk in self.val_masks:
            Mm = Mv.copy()
            Mm[mk[:, S]] = np.nan
            sm = model.raw(Mm)
            rob += max(0, auc - roc_auc_score(yv, sm)) + max(0, recall - float((sm[yv == 1] >= thr).mean()))
        rob /= len(self.val_masks)
        cost_n = float(self.ds.cost[S].sum() / self.cost_norm_den)
        red = self.redundancy(S)
        fit = (c.w_auc * auc + c.w_spec * spec - c.w_fnr * (1 - recall)
               - c.w_cost * cost_n - c.w_red * red - c.w_rob * rob)
        if recall < c.recall_floor:
            fit -= c.gate_penalty + 10 * (c.recall_floor - recall)
        res = dict(fitness=fit, auroc=auc, recall=recall, spec=spec, cost=cost_n,
                   red=red, rob=rob, k=int(len(S)))
        self.cache[key] = res
        return res


# =============================================================================
# 6. GENETIC ALGORITHM
# =============================================================================
class GeneticSelector:
    def __init__(self, cfg: Config, cons: FeatureConstraints, ev: Evaluator, n_feat: int, rng):
        self.cfg, self.cons, self.ev, self.n, self.rng = cfg, cons, ev, n_feat, rng

    def _init_pop(self) -> list[np.ndarray]:
        pop = []
        for _ in range(self.cfg.pop_size):
            m = np.zeros(self.n, bool)
            k = self.rng.integers(self.cfg.k_min, self.cfg.k_max + 1)
            m[self.rng.choice(self.n, k, replace=False)] = True
            pop.append(self.cons.repair(m, self.rng))
        return pop

    def _tournament(self, pop, fit):
        i = self.rng.choice(len(pop), self.cfg.tournament_k, replace=False)
        return pop[i[np.argmax(fit[i])]]

    def run(self) -> tuple[np.ndarray, dict, list[dict]]:
        c, log = self.cfg, []
        pop = self._init_pop()
        t0 = time.time()
        for g in range(c.generations + 1):
            res = [self.ev.evaluate(m) for m in pop]
            fit = np.array([r["fitness"] for r in res])
            order = np.argsort(-fit)
            b = res[order[0]]
            row = dict(gen=g, best=float(fit[order[0]]), mean=float(fit.mean()),
                       best_auroc=b["auroc"], best_recall=b["recall"], best_spec=b["spec"],
                       best_k=b["k"], n_gate_pass=int(sum(r["recall"] >= c.recall_floor for r in res)),
                       unique=len({m.tobytes() for m in pop}), evals=self.ev.n_evals,
                       t=round(time.time() - t0, 1))
            log.append(row)
            print(f"gen {g:2d} | best {row['best']:.4f} mean {row['mean']:.4f} | "
                  f"AUROC {b['auroc']:.4f} rec {b['recall']:.4f} spec {b['spec']:.3f} k={b['k']} | "
                  f"gate-pass {row['n_gate_pass']:2d}/{c.pop_size} uniq {row['unique']:2d} | {row['t']}s")
            if g == c.generations:
                break
            nxt = [pop[i].copy() for i in order[: c.elite]]
            while len(nxt) < c.pop_size:
                a, bb = self._tournament(pop, fit), self._tournament(pop, fit)
                if self.rng.random() < c.crossover_rate:
                    child = np.where(self.rng.random(self.n) < 0.5, a, bb)
                else:
                    child = a.copy()
                flip = self.rng.random(self.n) < c.mutation_rate
                child = np.logical_xor(child, flip)
                nxt.append(self.cons.repair(child, self.rng))
            pop = nxt
        best = pop[int(np.argmax([self.ev.evaluate(m)["fitness"] for m in pop]))]
        return best, self.ev.evaluate(best), log


# =============================================================================
# 7. FINAL REPORT ON HELD-OUT TEST DATA
# =============================================================================
def test_metrics(ev: Evaluator, model, thr, S, M_test=None):
    M = ev.M["test"][:, S] if M_test is None else M_test
    y = ev.y["test"]
    s = model.raw(M)
    pos, neg = s[y == 1], s[y == 0]
    tp = int((pos >= thr).sum())
    lo, hi = wilson(tp, len(pos))
    return dict(auroc=roc_auc_score(y, s), recall=tp / len(pos), recall_ci=(lo, hi),
                spec=float((neg < thr).mean()), fn=int(len(pos) - tp), n_pos=int(len(pos)))


def robustness_sweep(ev: Evaluator, model, thr, S, rng):
    rows = []
    M = ev.M["test"][:, S]
    for rate in (0.0, 0.1, 0.2, 0.3, 0.4):
        aucs, recs = [], []
        for _ in range(5):
            Mm = M.copy()
            Mm[rng.random(M.shape[:2]) < rate] = np.nan
            r = test_metrics(ev, model, thr, S, Mm)
            aucs.append(r["auroc"]); recs.append(r["recall"])
        rows.append((f"random {int(rate*100)}% missing", np.mean(aucs), np.mean(recs)))
    for j, f in enumerate(S):                       # single-sensor dropout
        Mm = M.copy(); Mm[:, j, :] = np.nan
        r = test_metrics(ev, model, thr, S, Mm)
        rows.append((f"sensor lost: {ev.ds.names[f]}", r["auroc"], r["recall"]))
    return rows


def latency_ms(ev: Evaluator, model, S, n=1000) -> float:
    x = ev.ds.X[ev.ds.idx_test[:n]][:, S]
    kn = ev.knots[S]
    t0 = time.perf_counter()
    for i in range(n):
        M = memberships(x[i:i + 1], kn)
        model.raw(M)
    return (time.perf_counter() - t0) / n * 1000


TIER_NAMES = ("LOW", "WATCH", "HIGH", "CRITICAL")


def tier_cuts(ev, model, S, thr_watch):
    """Tier boundaries from positive-score quantiles of the TRAIN set (recall 99.5/90/60%)."""
    s_tr = model.raw(ev.M["train"][:, S]); y_tr = ev.y["train"]
    return [thr_watch] + [threshold_for_recall(s_tr[y_tr == 1], r) for r in (0.90, 0.60)]


def tier_table(ev, model, S, thr_watch):
    cuts = tier_cuts(ev, model, S, thr_watch)
    s, y = model.raw(ev.M["test"][:, S]), ev.y["test"]
    tier = np.digitize(s, cuts)
    rows = []
    for t, nm in enumerate(TIER_NAMES):
        sel = tier == t
        rows.append((nm, int(sel.sum()), int(y[sel].sum()),
                     float(y[sel].mean()) if sel.any() else 0.0))
    return rows


def export_model(path, ev, model, thr, S, cuts):
    """Save everything needed to score new patients (see predict_patients.py)."""
    names = [ev.ds.names[i] for i in S]
    payload = dict(
        features=names,
        knots=ev.knots[S].tolist(),
        rules=model.rules_.tolist(),
        weights=model.weights_.tolist(),
        scale=model.scale_, prior_logit=model.prior_logit_,
        threshold=thr, tier_cuts=cuts, tier_names=list(TIER_NAMES),
        recall_target=ev.cfg.recall_target,
    )
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)


def main(cfg: Config = Config(), data_args: argparse.Namespace | None = None):
    t_start = time.time()
    rng = np.random.default_rng(cfg.seed)
    real = data_args is not None and data_args.data
    if real:
        ds = load_real_dataset(data_args.data, data_args.label, cfg, sheet=data_args.sheet,
                               schema_path=data_args.schema, id_col=data_args.id_col,
                               exclude=[c for c in (data_args.exclude or "").split(",") if c])
        cfg = fit_config_to_data(cfg, ds.groups)
    else:
        ds = generate_cohort(cfg, rng)
    suffix = "_real" if real else ""
    ev = Evaluator(cfg, ds, rng)
    cons = FeatureConstraints(cfg, ds.groups)
    F = ds.X.shape[1]

    print("=" * 78)
    print(f"Cohort: {len(ds.y)} neonates, {F} candidate features, critical prevalence {ds.y.mean():.3%}")
    for nm, i in (("train", ds.idx_train), ("val", ds.idx_val), ("test", ds.idx_test)):
        print(f"  {nm:5s}: n={len(i):5d}  critical={int(ds.y[i].sum()):4d} ({ds.y[i].mean():.2%})")
    print(f"Class-collapse reference: predicting 'not critical' for all -> accuracy "
          f"{1 - ds.y[ds.idx_test].mean():.1%}, critical recall 0.0%")
    print("=" * 78)

    ga = GeneticSelector(cfg, cons, ev, F, rng)
    best_mask, best_res, log = ga.run()
    S = np.flatnonzero(best_mask)
    model, thr = ev.fit_model(S)
    names = [ds.names[i] for i in S]

    print("\n" + "=" * 78 + "\nFINAL SOLUTION")
    print(f"Selected ({len(S)}): " + ", ".join(f"{n}[{ds.groups[i]}]" for n, i in zip(names, S)))
    print(f"Constraint check -> valid={cons.is_valid(best_mask)}, static={int((best_mask & cons.static).sum())}"
          f"/{cfg.max_static}, institutional={int((best_mask & cons.inst).sum())}, "
          f"physiological={int((best_mask & cons.physio).sum())}/{len(S)}")
    print(f"FINAL FITNESS (validation): {best_res['fitness']:.4f}   "
          f"[AUROC {best_res['auroc']:.4f}, recall {best_res['recall']:.4f}, spec {best_res['spec']:.4f}, "
          f"cost {best_res['cost']:.3f}, red {best_res['red']:.3f}, rob {best_res['rob']:.4f}]")

    print("\nRule base:")
    for r in model.describe(names):
        print("  " + r)

    tm = test_metrics(ev, model, thr, S)
    print("\nHELD-OUT TEST")
    print(f"  AUROC              : {tm['auroc']:.4f}")
    print(f"  Critical recall    : {tm['recall']:.4f}  (95% Wilson CI {tm['recall_ci'][0]:.4f}-{tm['recall_ci'][1]:.4f}; "
          f"{tm['fn']} missed of {tm['n_pos']})")
    print(f"  Specificity        : {tm['spec']:.4f}")
    print(f"  >=98% recall gate  : {'PASS' if tm['recall'] >= cfg.recall_floor else 'FAIL'}")

    print("\nRisk tiers (test):")
    print(f"  {'tier':9s}{'n':>6s}{'critical':>10s}{'rate':>8s}")
    for nm, n, k, rate in tier_table(ev, model, S, thr):
        print(f"  {nm:9s}{n:6d}{k:10d}{rate:8.1%}")

    print("\nRobustness (test, fixed deployed threshold):")
    print(f"  {'scenario':38s}{'AUROC':>8s}{'recall':>8s}")
    for nm, a, r in robustness_sweep(ev, model, thr, S, rng):
        print(f"  {nm:38s}{a:8.4f}{r:8.4f}")

    print(f"\nInference latency: {latency_ms(ev, model, S):.3f} ms / patient "
          f"({len(model.rules_)} rules, {len(S)} features)")

    # naive baseline for comparison: top-10 univariate features, bounds ignored
    uni = np.array([abs(roc_auc_score(ds.y[ds.idx_train], np.nan_to_num(ds.X[ds.idx_train, j])) - 0.5)
                    for j in range(F)])
    nb = np.zeros(F, bool); nb[np.argsort(-uni)[:cfg.k_max]] = True
    rb = ev.evaluate(nb)
    print(f"\nBaseline (top-10 univariate, no bounds): fitness {rb['fitness']:.4f}, AUROC {rb['auroc']:.4f}, "
          f"valid={cons.is_valid(nb)}, picks: {[ds.names[i] for i in np.flatnonzero(nb)]}")
    g0 = log[0]
    print(f"GA improvement: gen0 best {g0['best']:.4f} -> final {log[-1]['best']:.4f} "
          f"(gen0 mean {g0['mean']:.4f} -> final mean {log[-1]['mean']:.4f})")
    print(f"Unique fitness evaluations: {ev.n_evals}   Total runtime: {time.time() - t_start:.1f}s")

    export_model(f"neonatal_model{suffix}.json", ev, model, thr, S, tier_cuts(ev, model, S, thr))
    print(f"Saved trained model -> neonatal_model{suffix}.json"
          + (f"   (score new patients: python predict_patients.py file.csv --model neonatal_model{suffix}.json)"
             if real else ""))
    with open(f"convergence_log{suffix}.json", "w") as f:
        json.dump(log, f, indent=1)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="GA + fuzzy neonatal triage (synthetic or real data)")
    ap.add_argument("--data", help="Excel (.xlsx/.xls) or CSV file with real patients; omit for synthetic data")
    ap.add_argument("--label", default="critical", help="column holding the outcome (1/0, yes/no, critical/stable...)")
    ap.add_argument("--schema", help="optional CSV/Excel with columns: feature, group, cost")
    ap.add_argument("--sheet", help="Excel sheet name (default: first sheet)")
    ap.add_argument("--id-col", dest="id_col", help="patient-id column to ignore")
    ap.add_argument("--exclude", help="comma-separated columns to ignore")
    ap.add_argument("--generations", type=int)
    ap.add_argument("--pop", type=int)
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    over = {k: v for k, v in (("generations", a.generations), ("pop_size", a.pop), ("seed", a.seed)) if v}
    main(replace(Config(), **over), a)
