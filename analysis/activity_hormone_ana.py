# -*- coding: utf-8 -*-
"""
Robust analysis of activity–hormone association (small N participants)

Input:
- activity_hormone_aligned_YYYYMMDD_HHMMSS.csv

What this script does (single, non-duplicated pipeline):
1) Load + basic cleaning
2) Map activity -> activity_group (rest vs active)
3) Analyse BOTH windows:
      - during: prefix ""
      - post:   prefix "post_20_40__"
4) For each hormone (with enough samples):
      A) Activity effect (within-subject):
         - per participant: median(active) - median(rest)
         - sign-flip permutation test on participant diffs (exact if small)
         - report median_rest/median_active and effect_median_diff
      B) Intensity association (within-subject focus):
         - intensity_z = within-participant z-score (fallback NaN if std=0)
         - Spearman rho between y and intensity_z
         - cluster bootstrap CI over participants (skips constant inputs)
5) FDR (BH) correction within each window for:
      - activity p-values
      - intensity p-values

Notes:
- No files saved. Console output only.
- Designed for sparse hormones and ~4 participants where MixedLM is unstable.

Author: (generated with ChatGPT)
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# =========================
# CONFIG
# =========================
DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_20260129_142557.csv"

EXCLUDE_PARTICIPANTS = {"CAT02"}

WINDOW_PREFIXES = ["", "post_20_40__"]  # during + post

REST_SET = {"sit", "lying"}
ACTIVE_SET = {"light_activity", "stand", "moderate_activity"}

# Minimum thresholds per hormone/window (after filtering n>0 + dropna)
MIN_N_TOTAL = 12
MIN_N_REST = 5
MIN_N_ACTIVE = 5
MIN_PARTICIPANTS = 2
MIN_PARTICIPANTS_WITH_BOTH = 2  # participants who have BOTH rest & active samples for that hormone

# Intensity CI bootstrap
BOOT_B = 5000
BOOT_SEED = 0
MIN_BOOT_KEEP_RATIO = 0.2  # if too many bootstrap draws are invalid, CI is unreliable -> set CI NaN

# Whether to log1p transform hormone concentrations (often heavy-tailed).
# Turn off if your values can be negative or already normalized.
USE_LOG1P = False

# =========================
# HELPERS
# =========================
def ensure_intensity_numeric(s: pd.Series) -> pd.Series:
    """
    Convert intensity to numeric.
    - If already numeric (>=80% parseable): keep.
    - Else map common labels, then fallback to extracting digits.
    """
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().mean() >= 0.8:
        return out

    mapping = {
        "low": 1, "light": 1,
        "medium": 2, "moderate": 2,
        "high": 3, "vigorous": 3,
    }
    s2 = s.astype(str).str.strip().str.lower()
    out2 = s2.map(mapping)
    out3 = pd.to_numeric(s2.str.extract(r"(\d+(\.\d+)?)")[0], errors="coerce")
    return out2.combine_first(out3)

def canonical_hormones(df: pd.DataFrame, prefix: str):
    """
    Detect hormones by finding columns:
      {prefix}{H}_mean and {prefix}{H}_n
    Return list of hormone base names H.
    """
    mean_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_mean")]
    n_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_n")]

    def base(c, suf):
        return re.sub(fr"{re.escape(suf)}$", "", c)

    means = {base(c, "_mean")[len(prefix):] for c in mean_cols}
    ns = {base(c, "_n")[len(prefix):] for c in n_cols}
    hormones = sorted(means & ns)
    return hormones

def label_window(prefix: str) -> str:
    return "during" if prefix == "" else prefix.rstrip("__")

def safe_transform_y(y: pd.Series) -> pd.Series:
    y = pd.to_numeric(y, errors="coerce")
    if not USE_LOG1P:
        return y
    # Only apply log1p if all non-NaN values are >=0
    yn = y.dropna()
    if len(yn) == 0:
        return y
    if (yn < 0).any():
        return y
    return np.log1p(y)

def within_person_zscore(x: pd.Series) -> pd.Series:
    """
    Within-participant z-score. If std==0 -> NaN (no within-person variation).
    """
    x = pd.to_numeric(x, errors="coerce")
    mu = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x - mu) / sd

def cliffs_delta(x, y):
    """
    Cliff's delta: P(x>y) - P(x<y). Range [-1,1].
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = sum((xi > y).sum() for xi in x)
    lt = sum((xi < y).sum() for xi in x)
    return (gt - lt) / (len(x) * len(y))

def sign_flip_pvalue(diffs: np.ndarray, rng: np.random.Generator, B: int = 20000) -> float:
    """
    Sign-flip permutation on participant-level diffs (median(active)-median(rest)).
    Uses exact enumeration if 2^n <= B, else Monte Carlo.
    Two-sided p-value for mean(diff) != 0 (can swap to median if you prefer).
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    n = len(diffs)
    if n == 0:
        return np.nan

    stat_obs = float(np.mean(diffs))

    # exact if small enough
    if (2 ** n) <= B:
        # enumerate all sign patterns
        stats = []
        for mask in range(2 ** n):
            signs = np.array([1 if (mask >> i) & 1 else -1 for i in range(n)], dtype=float)
            stats.append(float(np.mean(signs * diffs)))
        stats = np.asarray(stats)
        p = (np.sum(np.abs(stats) >= abs(stat_obs)) + 1.0) / (len(stats) + 1.0)
        return float(p)

    # Monte Carlo
    count = 0
    for _ in range(B):
        signs = rng.choice([-1.0, 1.0], size=n, replace=True)
        stat = float(np.mean(signs * diffs))
        if abs(stat) >= abs(stat_obs):
            count += 1
    p = (count + 1.0) / (B + 1.0)
    return float(p)

def cluster_bootstrap_spearman_ci(
    sub: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = "participant",
    B: int = 5000,
    seed: int = 0,
):
    """
    Cluster bootstrap over participants for Spearman rho.
    Skips invalid draws where x or y is constant (nunique<2) or too few rows.
    Returns: (rho_obs, ci_lo, ci_hi, keep_ratio)
    """
    rng = np.random.default_rng(seed)

    # observed rho
    x = sub[x_col]
    y = sub[y_col]
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        rho_obs, p_obs = np.nan, np.nan
    else:
        rho_obs, p_obs = spearmanr(x, y, nan_policy="omit")

    parts = sub[group_col].dropna().unique().tolist()
    if len(parts) == 0:
        return float(rho_obs) if np.isfinite(rho_obs) else np.nan, np.nan, np.nan, 0.0

    rhos = []
    attempts = 0
    for _ in range(B):
        attempts += 1
        sampled_parts = rng.choice(parts, size=len(parts), replace=True)
        boot = pd.concat([sub[sub[group_col] == p] for p in sampled_parts], axis=0, ignore_index=True)

        # crucial: skip constant input (prevents ConstantInputWarning + bogus CI)
        if boot[x_col].nunique(dropna=True) < 2 or boot[y_col].nunique(dropna=True) < 2:
            continue

        r, _ = spearmanr(boot[x_col], boot[y_col], nan_policy="omit")
        if np.isfinite(r):
            rhos.append(float(r))

    keep_ratio = len(rhos) / max(1, attempts)
    if len(rhos) < 50 or keep_ratio < MIN_BOOT_KEEP_RATIO:
        # CI too unstable
        return float(rho_obs) if np.isfinite(rho_obs) else np.nan, np.nan, np.nan, float(keep_ratio)

    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(rho_obs), float(lo), float(hi), float(keep_ratio)

# =========================
# LOAD + PREP
# =========================
df = pd.read_csv(DATA_PATH)
print("[INFO] Loaded:", DATA_PATH)
print("[INFO] Raw shape:", df.shape)

# participant
df["participant"] = df["participant"].astype(str)
df = df[~df["participant"].isin(EXCLUDE_PARTICIPANTS)].copy()

# intensity
if "intensity" not in df.columns:
    raise KeyError("Column 'intensity' not found in CSV.")
df["intensity"] = ensure_intensity_numeric(df["intensity"])

# activity_group
df["activity_label"] = df["activity_label"].astype(str).str.strip()
df["activity_group"] = np.where(
    df["activity_label"].isin(REST_SET), "rest",
    np.where(df["activity_label"].isin(ACTIVE_SET), "active", "other")
)

# Keep only rest/active for this analysis
df = df[df["activity_group"].isin(["rest", "active"])].copy()
df["active_bin"] = (df["activity_group"] == "active").astype(int)

# Within-person z-score intensity (key upgrade)
df["intensity_z"] = df.groupby("participant")["intensity"].transform(within_person_zscore)

print("[INFO] After exclude + group:", df.shape)
print("[INFO] Participants:", sorted(df["participant"].unique().tolist()))
print("[INFO] activity_group counts:\n", df["activity_group"].value_counts())
print("[INFO] Intensity non-NA:", int(df["intensity"].notna().sum()), "/", len(df))
print("[INFO] Intensity_z non-NA:", int(df["intensity_z"].notna().sum()), "/", len(df))

# =========================
# MAIN ANALYSIS
# =========================
rng = np.random.default_rng(0)
rows = []

for prefix in WINDOW_PREFIXES:
    wname = label_window(prefix)
    hormones = canonical_hormones(df, prefix)
    print(f"\n[WINDOW] {wname} | hormones detected: {len(hormones)}")

    # overall coverage (any hormone) in this window
    ncols = [f"{prefix}{h}_n" for h in hormones if f"{prefix}{h}_n" in df.columns]
    if ncols:
        any_cov = (df[ncols].sum(axis=1) > 0).mean()
        print(f"[INFO] Overall coverage (any hormone, this window): {any_cov:.1%}")

    for h in hormones:
        n_col = f"{prefix}{h}_n"
        y_col = f"{prefix}{h}_mean"
        if n_col not in df.columns or y_col not in df.columns:
            continue

        sub = df[df[n_col] > 0].copy()
        sub["y_raw"] = pd.to_numeric(sub[y_col], errors="coerce")
        sub["y"] = safe_transform_y(sub["y_raw"])

        # keep complete cases for group + intensity_z association
        sub = sub.dropna(subset=["y", "participant", "activity_group", "intensity_z"])

        if len(sub) < MIN_N_TOTAL:
            continue
        if sub["participant"].nunique() < MIN_PARTICIPANTS:
            continue

        # ===== A) Activity effect (within-subject) =====
        # participant-level diffs: median(active) - median(rest)
        per = []
        for pid, g in sub.groupby("participant"):
            ya = g.loc[g["activity_group"] == "active", "y"]
            yr = g.loc[g["activity_group"] == "rest", "y"]
            if len(ya) == 0 or len(yr) == 0:
                continue
            per.append({
                "participant": pid,
                "diff": float(np.median(ya) - np.median(yr)),
                "med_active": float(np.median(ya)),
                "med_rest": float(np.median(yr)),
                "n_active": int(len(ya)),
                "n_rest": int(len(yr)),
            })

        per_df = pd.DataFrame(per)
        n_with_both = int(len(per_df))

        p_act_ws = np.nan
        effect_med_diff = np.nan
        med_rest_all = np.nan
        med_act_all = np.nan

        if n_with_both >= MIN_PARTICIPANTS_WITH_BOTH:
            diffs = per_df["diff"].values
            effect_med_diff = float(np.median(diffs))
            # global medians (pooled, descriptive only)
            med_rest_all = float(np.median(sub.loc[sub["activity_group"] == "rest", "y"]))
            med_act_all = float(np.median(sub.loc[sub["activity_group"] == "active", "y"]))
            # sign-flip permutation on diffs
            p_act_ws = sign_flip_pvalue(diffs, rng=rng, B=20000)

        # ===== B) Intensity association (within-person z) =====
        # Spearman rho on pooled data using intensity_z (already within-person)
        if sub["intensity_z"].nunique(dropna=True) < 2 or sub["y"].nunique(dropna=True) < 2:
            rho = np.nan
            p_int = np.nan
            ci_lo = np.nan
            ci_hi = np.nan
            keep_ratio = 0.0
        else:
            rho, p_int = spearmanr(sub["intensity_z"], sub["y"], nan_policy="omit")
            rho, ci_lo, ci_hi, keep_ratio = cluster_bootstrap_spearman_ci(
                sub=sub,
                x_col="intensity_z",
                y_col="y",
                group_col="participant",
                B=BOOT_B,
                seed=BOOT_SEED,
            )

        rows.append({
            "window": wname,
            "hormone": h,
            "n_total": int(len(sub)),
            "n_participants": int(sub["participant"].nunique()),
            "n_participants_with_both": int(n_with_both),

            # descriptive
            "median_rest": med_rest_all,
            "median_active": med_act_all,
            "effect_median_diff_active_minus_rest": effect_med_diff,

            # activity test
            "p_activity_within_subject": float(p_act_ws) if np.isfinite(p_act_ws) else np.nan,

            # intensity
            "spearman_rho_intensity_z": float(rho) if np.isfinite(rho) else np.nan,
            "spearman_ci_lo": float(ci_lo) if np.isfinite(ci_lo) else np.nan,
            "spearman_ci_hi": float(ci_hi) if np.isfinite(ci_hi) else np.nan,
            "bootstrap_keep_ratio": float(keep_ratio),
            "p_intensity": float(p_int) if np.isfinite(p_int) else np.nan,
        })

res = pd.DataFrame(rows)

if res.empty:
    print("\n[RESULT] No hormones passed minimum thresholds.")
    print("Try lowering MIN_N_TOTAL / MIN_N_REST / MIN_N_ACTIVE or checking n-columns.")
    print("\n[DONE]")
    raise SystemExit(0)

# =========================
# FILTER: ensure rest/active minimum counts (based on RAW y availability)
# =========================
# We used intensity_z complete cases above; now ensure the activity group counts are still OK.
# Compute counts per hormone/window on the same sub-selection criteria.
# (If you want counts based purely on y availability regardless intensity_z, move dropna earlier.)
def compute_group_counts(prefix, hormone):
    n_col = f"{prefix}{hormone}_n"
    y_col = f"{prefix}{hormone}_mean"
    sub = df[df[n_col] > 0].copy()
    sub["y"] = pd.to_numeric(sub[y_col], errors="coerce")
    sub = sub.dropna(subset=["y", "participant", "activity_group", "intensity_z"])
    return (
        int((sub["activity_group"] == "rest").sum()),
        int((sub["activity_group"] == "active").sum()),
    )

# attach group counts and filter
rest_counts = []
active_counts = []
for _, r in res.iterrows():
    prefix = "" if r["window"] == "during" else "post_20_40__"
    nr, na = compute_group_counts(prefix, r["hormone"])
    rest_counts.append(nr)
    active_counts.append(na)

res["n_rest"] = rest_counts
res["n_active"] = active_counts
res = res[(res["n_total"] >= MIN_N_TOTAL) & (res["n_rest"] >= MIN_N_REST) & (res["n_active"] >= MIN_N_ACTIVE)].copy()

if res.empty:
    print("\n[RESULT] After enforcing MIN_N_REST/MIN_N_ACTIVE, no hormones remain.")
    print("Lower MIN_N_REST/MIN_N_ACTIVE or adjust intensity_z missingness.")
    print("\n[DONE]")
    raise SystemExit(0)

# =========================
# FDR (BH) within each window
# =========================
for w in sorted(res["window"].unique()):
    m = res["window"] == w

    # activity
    pvals = res.loc[m, "p_activity_within_subject"].values
    ok = np.isfinite(pvals)
    if ok.sum() >= 2:
        _, q, _, _ = multipletests(pvals[ok], method="fdr_bh")
        res.loc[m, "q_p_activity_within_subject"] = np.nan
        res.loc[m, "q_p_activity_within_subject"].iloc[np.where(ok)[0]] = q
    else:
        res.loc[m, "q_p_activity_within_subject"] = np.nan

    # intensity
    pvals = res.loc[m, "p_intensity"].values
    ok = np.isfinite(pvals)
    if ok.sum() >= 2:
        _, q, _, _ = multipletests(pvals[ok], method="fdr_bh")
        res.loc[m, "q_p_intensity"] = np.nan
        res.loc[m, "q_p_intensity"].iloc[np.where(ok)[0]] = q
    else:
        res.loc[m, "q_p_intensity"] = np.nan

# =========================
# PRINT RESULTS
# =========================
# Activity table
print("\n[TOP by Activity (within-subject) | FDR]")
colsA = [
    "window", "hormone", "n_total", "n_participants", "n_participants_with_both",
    "n_rest", "n_active",
    "median_rest", "median_active",
    "effect_median_diff_active_minus_rest",
    "p_activity_within_subject", "q_p_activity_within_subject",
]
colsA = [c for c in colsA if c in res.columns]

tmpA = res.sort_values(
    ["window", "q_p_activity_within_subject", "p_activity_within_subject"],
    ascending=[True, True, True],
    na_position="last"
)
print(tmpA[colsA].head(30).to_string(index=False))

# Intensity table
print("\n[TOP by Intensity Spearman (within-person intensity_z) | FDR + cluster bootstrap CI]")
colsI = [
    "window", "hormone", "n_total", "n_participants",
    "spearman_rho_intensity_z", "spearman_ci_lo", "spearman_ci_hi",
    "bootstrap_keep_ratio",
    "p_intensity", "q_p_intensity",
]
colsI = [c for c in colsI if c in res.columns]

tmpI = res.sort_values(
    ["window", "q_p_intensity", "p_intensity"],
    ascending=[True, True, True],
    na_position="last"
)
print(tmpI[colsI].head(30).to_string(index=False))

print("\n[NOTES]")
print("- Activity p-value: sign-flip permutation on participant-level median(active)-median(rest).")
print("- Intensity uses within-person z-scored intensity (intensity_z), reducing between-subject confounding.")
print("- Bootstrap CI skips invalid draws where intensity_z or y is constant; keep_ratio reports how many draws were usable.")
print("- If keep_ratio < {:.0%}, CI is suppressed (NaN) to avoid false precision.".format(MIN_BOOT_KEEP_RATIO))

print("\n[DONE]")


import matplotlib.pyplot as plt

def pick_best_hormone(res_df: pd.DataFrame, window: str, mode: str):
    """
    mode:
      - "activity": pick by q_p_activity_within_subject then p
      - "intensity": pick by q_p_intensity then p
    """
    d = res_df[res_df["window"] == window].copy()
    if d.empty:
        return None

    if mode == "activity":
        qcol, pcol = "q_p_activity_within_subject", "p_activity_within_subject"
    else:
        qcol, pcol = "q_p_intensity", "p_intensity"

    # Prefer finite q, else fallback to p
    d["_q_ok"] = np.isfinite(d[qcol])
    d["_p_ok"] = np.isfinite(d[pcol])

    d1 = d[d["_q_ok"]].sort_values([qcol, pcol], ascending=[True, True])
    if not d1.empty:
        return d1.iloc[0]["hormone"]

    d2 = d[d["_p_ok"]].sort_values([pcol], ascending=[True])
    if not d2.empty:
        return d2.iloc[0]["hormone"]

    return None

def fetch_sub_for_plot(df_base: pd.DataFrame, prefix: str, hormone: str) -> pd.DataFrame:
    n_col = f"{prefix}{hormone}_n"
    y_col = f"{prefix}{hormone}_mean"
    sub = df_base[df_base[n_col] > 0].copy()
    sub["y_raw"] = pd.to_numeric(sub[y_col], errors="coerce")
    sub["y"] = safe_transform_y(sub["y_raw"])
    sub = sub.dropna(subset=["y", "participant", "activity_group", "intensity_z"])
    return sub

def plot_activity_box_scatter(sub: pd.DataFrame, title: str):
    """
    Boxplot + jittered scatter by participant for rest vs active
    """
    groups = ["rest", "active"]
    data = [sub.loc[sub["activity_group"] == g, "y"].values for g in groups]

    fig = plt.figure()
    ax = plt.gca()

    ax.boxplot(data, labels=groups, showfliers=False)

    # jitter scatter
    rng = np.random.default_rng(0)
    participants = sorted(sub["participant"].unique().tolist())
    # assign each participant a marker to distinguish
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_map = {pid: markers[i % len(markers)] for i, pid in enumerate(participants)}

    for i, g in enumerate(groups, start=1):
        gsub = sub[sub["activity_group"] == g]
        for pid, gg in gsub.groupby("participant"):
            x = i + rng.normal(0, 0.05, size=len(gg))
            ax.scatter(x, gg["y"].values, marker=marker_map[pid], alpha=0.8, label=pid)

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), title="participant", fontsize=8)

    ax.set_title(title)
    ax.set_ylabel("hormone value (y)")
    plt.tight_layout()
    plt.show()

def plot_intensity_scatter(sub: pd.DataFrame, title: str):
    """
    Scatter intensity_z vs y with simple fit line (visual aid only)
    """
    fig = plt.figure()
    ax = plt.gca()

    # scatter by participant
    for pid, g in sub.groupby("participant"):
        ax.scatter(g["intensity_z"], g["y"], alpha=0.8, label=pid)

    # simple fit line
    x = sub["intensity_z"].values
    y = sub["y"].values
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 2 and np.unique(x[ok]).size >= 2:
        b1, b0 = np.polyfit(x[ok], y[ok], 1)
        xx = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 100)
        yy = b1 * xx + b0
        ax.plot(xx, yy, linewidth=2)

    rho, p = spearmanr(sub["intensity_z"], sub["y"], nan_policy="omit")
    ax.set_title(f"{title}\nSpearman rho={rho:.3f}, p={p:.3g}")
    ax.set_xlabel("intensity_z (within-person)")
    ax.set_ylabel("hormone value (y)")
    ax.legend(title="participant", fontsize=8)
    plt.tight_layout()
    plt.show()

# ---- run plotting for each window ----
for w in sorted(res["window"].unique()):
    prefix = "" if w == "during" else "post_20_40__"

    # pick best hormones
    hA = pick_best_hormone(res, w, mode="activity")
    hI = pick_best_hormone(res, w, mode="intensity")

    if hA is not None:
        subA = fetch_sub_for_plot(df, prefix, hA)
        if len(subA) > 0:
            plot_activity_box_scatter(
                subA,
                title=f"[{w}] Activity effect: {hA} (rest vs active)"
            )

    if hI is not None:
        subI = fetch_sub_for_plot(df, prefix, hI)
        if len(subI) > 0 and subI["intensity_z"].notna().sum() >= 3:
            plot_intensity_scatter(
                subI,
                title=f"[{w}] Intensity association: {hI} (intensity_z vs y)"
            )


#ana
"""
Analysis of activity–hormone association
=======================================

Data:
- activity_hormone_aligned_20260129_142557.csv

Analysis steps:
1) Load + basic cleaning
2) Coverage analysis (which activities have hormone samples)
3) Descriptive stats by activity
4) Non-parametric test (Kruskal–Wallis)
5) Mixed-effects model: hormone ~ activity + intensity + (1 | participant)

No files saved. Print results only.
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import statsmodels.formula.api as smf

# =========================
# PATH
# =========================
DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_20260129_142557.csv"

SKIP_PARTICIPANTS = {"CAT02"}
MIN_ROWS = 20           # minimum rows per hormone for modelling
TOP_K_ACTIVITIES = 6    # avoid rare activity categories

# =========================
# LOAD
# =========================
df = pd.read_csv(DATA_PATH)

print("[INFO] Raw shape:", df.shape)

df = df[~df["participant"].isin(SKIP_PARTICIPANTS)].copy()
df["participant"] = df["participant"].astype(str)

print("[INFO] After excluding CAT02:", df.shape)
print("[INFO] Participants:", sorted(df["participant"].unique()))

# =========================
# Detect hormone columns
# =========================
mean_cols = [c for c in df.columns if c.endswith("_mean") and not c.startswith("post_")]
n_cols    = [c for c in df.columns if c.endswith("_n")    and not c.startswith("post_")]

def base_name(c):
    return re.sub(r"_mean$", "", c)

HORMONES = sorted(set(map(base_name, mean_cols)))

print("[INFO] Hormones detected:", len(HORMONES))
print("[INFO] Example hormones:", HORMONES[:8])

# =========================
# Activity counts
# =========================
print("\n[STEP 1] Activity counts:")
print(df["activity_label"].value_counts())

top_acts = df["activity_label"].value_counts().head(TOP_K_ACTIVITIES).index
df = df[df["activity_label"].isin(top_acts)].copy()

# =========================
# STEP 2: Coverage analysis
# =========================
print("\n[STEP 2] Hormone coverage by activity")

coverage = []
for h in HORMONES:
    ncol = f"{h}_n"
    if ncol not in df:
        continue
    tmp = df.groupby("activity_label")[ncol].apply(lambda x: (x > 0).mean())
    for act, val in tmp.items():
        coverage.append({
            "hormone": h,
            "activity": act,
            "coverage": val
        })

cov_df = pd.DataFrame(coverage)
print(cov_df.sort_values("coverage", ascending=False).head(15))

# =========================
# STEP 3: Descriptive stats
# =========================
TARGET_HORMONE = "cortisol"  # change if you want

if f"{TARGET_HORMONE}_mean" in df.columns:
    sub = df[df[f"{TARGET_HORMONE}_n"] > 0].copy()
    print(f"\n[STEP 3] Descriptive stats for {TARGET_HORMONE} by activity")
    print(
        sub.groupby("activity_label")[f"{TARGET_HORMONE}_mean"]
           .describe()[["count", "mean", "std", "min", "50%", "max"]]
    )

# =========================
# STEP 4: Kruskal–Wallis test
# =========================
print(f"\n[STEP 4] Kruskal–Wallis tests (top activities)")

for h in HORMONES:
    mean_col = f"{h}_mean"
    n_col    = f"{h}_n"

    sub = df[df[n_col] > 0]
    if len(sub) < MIN_ROWS:
        continue

    groups = []
    for act in top_acts:
        vals = sub.loc[sub["activity_label"] == act, mean_col].dropna()
        if len(vals) >= 3:
            groups.append(vals)

    if len(groups) >= 2:
        H, p = kruskal(*groups)
        print(f"  {h:15s}  H={H:6.2f}, p={p:.3e}")

# =========================
# STEP 5: Mixed-effects model
# =========================
print("\n[STEP 5] Mixed-effects model: hormone ~ activity + intensity + (1|participant)")

for h in HORMONES:
    mean_col = f"{h}_mean"
    n_col    = f"{h}_n"

    sub = df[df[n_col] > 0].copy()
    sub["y"] = pd.to_numeric(sub[mean_col], errors="coerce")
    sub["intensity"] = pd.to_numeric(sub["intensity"], errors="coerce")

    sub = sub.dropna(subset=["y", "activity_label", "intensity", "participant"])

    if len(sub) < MIN_ROWS or sub["participant"].nunique() < 2:
        continue

    try:
        model = smf.mixedlm(
            "y ~ C(activity_label) + intensity",
            sub,
            groups=sub["participant"]
        ).fit(reml=False)

        print(f"\n--- {h} ---")
        print(model.summary().tables[1])

    except Exception as e:
        print(f"[WARN] {h}: model failed ({e})")

print("\n[DONE] Analysis completed.")

# -*- coding: utf-8 -*-
"""
Further analysis: activity/intensity vs all hormones (during + post-window)

Key upgrades vs your current script:
- Analyse BOTH windows: during ("") and post ("post_20_40__")
- Merge activity categories to boost power:
    rest  = sit + lying
    active = light_activity + stand + moderate_activity
- Mixed-effects model with participant random intercept:
    m0   : y ~ 1
    mA   : y ~ active_bin
    mI   : y ~ intensity
    mAI  : y ~ active_bin + intensity
  Then Likelihood Ratio Tests (LRT) + FDR across hormones.

Default: no file saving (print only).
"""

import re
import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

# =========================
# CONFIG
# =========================
DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_20260129_142557.csv"

EXCLUDE_PARTICIPANTS = {"CAT02"}

# windows to test
WINDOW_PREFIXES = ["", "post_20_40__"]   # during and post

# activity merge map
REST_SET   = {"sit", "lying"}
ACTIVE_SET = {"light_activity", "stand", "moderate_activity"}

# modelling constraints
MIN_ROWS_PER_HORMONE = 25     # minimum rows with samples (n>0)
MIN_PARTICIPANTS = 2
USE_LOG1P = True              # stabilise heavy-tailed hormones

# output control
SAVE_RESULTS = False
OUT_CSV = r"D:\UOB\Year_3_UOB\mdm_hormone\further_analysis_results.csv"

# =========================
# HELPERS
# =========================
def safe_log1p(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    if (x < 0).any():
        return x  # don't transform if negative values exist
    return np.log1p(x)

def lrt_p(ll0, ll1, df_diff):
    LR = 2.0 * (ll1 - ll0)
    if df_diff <= 0 or not np.isfinite(LR):
        return np.nan
    return chi2.sf(LR, df_diff)

def ensure_intensity_numeric(s: pd.Series) -> pd.Series:
    """
    If intensity is already numeric -> keep.
    If it's categorical strings -> map to ordinal.
    """
    out = pd.to_numeric(s, errors="coerce")
    if out.notna().mean() >= 0.8:
        return out

    # common label mapping (adjust if needed)
    mapping = {
        "low": 1, "light": 1,
        "medium": 2, "moderate": 2,
        "high": 3, "vigorous": 3,
    }
    s2 = s.astype(str).str.strip().str.lower()
    out2 = s2.map(mapping)
    # fallback: try extract digits
    out3 = pd.to_numeric(s2.str.extract(r"(\d+(\.\d+)?)")[0], errors="coerce")
    return out2.combine_first(out3)

def canonical_hormones(df: pd.DataFrame, prefix: str):
    """
    Detect hormones by finding {prefix}X_mean and {prefix}X_n
    Return list of base hormone names X (case-sensitive as in columns).
    """
    mean_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_mean")]
    n_cols    = [c for c in df.columns if c.startswith(prefix) and c.endswith("_n")]

    # drop junk columns
    mean_cols = [c for c in mean_cols if not c[len(prefix):].lower().startswith("unnamed:")]
    n_cols    = [c for c in n_cols if not c[len(prefix):].lower().startswith("unnamed:")]

    def base(c, suf):
        return re.sub(fr"{re.escape(suf)}$", "", c)

    means = {base(c, "_mean")[len(prefix):] for c in mean_cols}
    ns    = {base(c, "_n")[len(prefix):] for c in n_cols}
    hormones = sorted(means & ns)
    return hormones

def fit_mixedlm(formula, data):
    """
    Fit MixedLM with ML (reml=False) so LRT is valid.
    Use lbfgs and limited iterations for stability.
    """
    return smf.mixedlm(formula, data=data, groups=data["participant"]).fit(
        reml=False, method="lbfgs", maxiter=200, disp=False
    )

# =========================
# LOAD + PREP
# =========================
df = pd.read_csv(DATA_PATH)
print("[INFO] Loaded:", DATA_PATH)
print("[INFO] Raw shape:", df.shape)

df = df[~df["participant"].isin(EXCLUDE_PARTICIPANTS)].copy()
df["participant"] = df["participant"].astype(str)

# intensity numeric
if "intensity" not in df.columns:
    raise KeyError("Column 'intensity' not found. Re-run alignment with intensity included.")
df["intensity"] = ensure_intensity_numeric(df["intensity"])

# merged activity group
df["activity_label"] = df["activity_label"].astype(str).str.strip()
df["activity_group"] = np.where(
    df["activity_label"].isin(REST_SET), "rest",
    np.where(df["activity_label"].isin(ACTIVE_SET), "active", "other")
)

# keep only rest/active (drop other rare types to avoid noise)
df = df[df["activity_group"].isin(["rest", "active"])].copy()
df["active_bin"] = (df["activity_group"] == "active").astype(int)

print("[INFO] After excluding + grouping:", df.shape)
print("[INFO] Participants:", sorted(df["participant"].unique().tolist()))
print("[INFO] Activity_group counts:\n", df["activity_group"].value_counts())
print("[INFO] Intensity non-NA:", int(df["intensity"].notna().sum()), "/", len(df))

# =========================
# ANALYSIS PER WINDOW
# =========================
all_results = []

for prefix in WINDOW_PREFIXES:
    hormones = canonical_hormones(df, prefix)
    print(f"\n[WINDOW] prefix='{prefix}' hormones={len(hormones)}")

    # overall coverage (any hormone) for this window
    ncols = [f"{prefix}{h}_n" for h in hormones if f"{prefix}{h}_n" in df.columns]
    if ncols:
        any_cov = (df[ncols].sum(axis=1) > 0).mean()
        print(f"[INFO] Overall coverage (any hormone, this window): {any_cov:.1%}")

    for h in hormones:
        n_col = f"{prefix}{h}_n"
        y_col = f"{prefix}{h}_mean"
        if n_col not in df.columns or y_col not in df.columns:
            continue

        sub = df[df[n_col] > 0].copy()
        if len(sub) < MIN_ROWS_PER_HORMONE:
            continue
        if sub["participant"].nunique() < MIN_PARTICIPANTS:
            continue

        # response
        sub["y"] = safe_log1p(sub[y_col]) if USE_LOG1P else pd.to_numeric(sub[y_col], errors="coerce")

        # keep complete cases
        sub = sub.dropna(subset=["y", "participant", "active_bin", "intensity"])
        if len(sub) < MIN_ROWS_PER_HORMONE or sub["participant"].nunique() < MIN_PARTICIPANTS:
            continue

        # Fit models
        try:
            m0  = fit_mixedlm("y ~ 1", sub)
            mA  = fit_mixedlm("y ~ active_bin", sub)
            mI  = fit_mixedlm("y ~ intensity", sub)
            mAI = fit_mixedlm("y ~ active_bin + intensity", sub)

            pA  = lrt_p(m0.llf,  mA.llf,  mA.df_model  - m0.df_model)
            pI  = lrt_p(m0.llf,  mI.llf,  mI.df_model  - m0.df_model)
            pAI = lrt_p(m0.llf, mAI.llf, mAI.df_model - m0.df_model)

            # also test incremental contributions:
            # activity given intensity, and intensity given activity
            pA_given_I = lrt_p(mI.llf, mAI.llf, mAI.df_model - mI.df_model)
            pI_given_A = lrt_p(mA.llf, mAI.llf, mAI.df_model - mA.df_model)

            # coefficients (from joint model)
            coef_active = mAI.params.get("active_bin", np.nan)
            coef_int    = mAI.params.get("intensity", np.nan)

            all_results.append({
                "window": "during" if prefix == "" else prefix.rstrip("__"),
                "hormone": h,
                "n_rows": len(sub),
                "n_participants": sub["participant"].nunique(),
                "coef_active_bin": float(coef_active) if np.isfinite(coef_active) else np.nan,
                "coef_intensity": float(coef_int) if np.isfinite(coef_int) else np.nan,
                "p_active_vs_base": pA,
                "p_intensity_vs_base": pI,
                "p_both_vs_base": pAI,
                "p_active_given_intensity": pA_given_I,
                "p_intensity_given_active": pI_given_A,
                "AIC_base": m0.aic,
                "AIC_both": mAI.aic,
                "deltaAIC_both_minus_base": mAI.aic - m0.aic,
            })

        except Exception:
            continue

# =========================
# SUMMARISE + FDR
# =========================
res = pd.DataFrame(all_results)
if res.empty:
    print("\n[RESULT] No eligible hormone models were fitted (too few samples).")
else:
    # FDR within each window and each p-type
    for win in res["window"].unique():
        m = (res["window"] == win)
        for pcol in ["p_active_vs_base", "p_intensity_vs_base", "p_both_vs_base",
                     "p_active_given_intensity", "p_intensity_given_active"]:
            vals = res.loc[m, pcol].values
            ok = np.isfinite(vals)
            if ok.sum() >= 2:
                _, q, _, _ = multipletests(vals[ok], method="fdr_bh")
                qcol = "q_" + pcol
                res.loc[m, qcol] = np.nan
                res.loc[m, qcol].iloc[np.where(ok)[0]] = q

    # rank: prefer joint model significance and AIC improvement
    sort_cols = ["window", "q_p_both_vs_base", "deltaAIC_both_minus_base"]
    for c in sort_cols:
        if c not in res.columns:
            sort_cols.remove(c)
    res = res.sort_values(sort_cols, ascending=[True, True, True], na_position="last")

    print("\n[TOP RESULTS] (ranked by joint model q-value, then deltaAIC)")
    show_cols = [
        "window", "hormone", "n_rows", "n_participants",
        "coef_active_bin", "coef_intensity",
        "p_both_vs_base", "q_p_both_vs_base",
        "p_active_given_intensity", "q_p_active_given_intensity",
        "p_intensity_given_active", "q_p_intensity_given_active",
        "deltaAIC_both_minus_base"
    ]
    show_cols = [c for c in show_cols if c in res.columns]
    print(res[show_cols].head(30).to_string(index=False))

    print("\n[INTERPRETATION]")
    print("- Focus first on window='post_20_40' (often more physiological latency).")
    print("- q_p_both_vs_base < 0.05 suggests activity+intensity jointly improve fit (FDR-controlled).")
    print("- p_active_given_intensity tests activity after controlling intensity.")
    print("- p_intensity_given_active tests intensity after controlling activity group.")
    print("- Negative deltaAIC_both_minus_base means the joint model fits better than baseline.")

    if SAVE_RESULTS:
        res.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        print("\n[OK] Saved results to:", OUT_CSV)

print("\n[DONE]")

# # -*- coding: utf-8 -*-
# """
# Robust further analysis for small-sample aligned activity-hormone data.
#
# Why this version:
# - Your dataset is sparse per hormone and only 4 participants -> MixedLM often fails.
# - We use nonparametric tests + effect sizes:
#     1) Activity group (rest vs active): Mann–Whitney U + Cliff's delta
#     2) Intensity association: Spearman rho
# - Run for BOTH windows: during ("") and post ("post_20_40__")
# - Multiple testing correction: FDR (BH)
#
# No files are saved. Results printed to console.
# """
#
# import re
# import numpy as np
# import pandas as pd
# from scipy.stats import mannwhitneyu, spearmanr
# from statsmodels.stats.multitest import multipletests
#
# # =========================
# # CONFIG
# # =========================
# DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_20260129_142557.csv"
#
# EXCLUDE_PARTICIPANTS = {"CAT02"}
#
# WINDOW_PREFIXES = ["", "post_20_40__"]  # during + post
#
# REST_SET   = {"sit", "lying"}
# ACTIVE_SET = {"light_activity", "stand", "moderate_activity"}
#
# MIN_N_REST = 5       # at least 5 samples in rest for a hormone
# MIN_N_ACTIVE = 5     # at least 5 samples in active for a hormone
# MIN_N_TOTAL = 12     # at least 12 total samples (n>0) for that hormone
#
# # =========================
# # HELPERS
# # =========================
# def ensure_intensity_numeric(s: pd.Series) -> pd.Series:
#     out = pd.to_numeric(s, errors="coerce")
#     if out.notna().mean() >= 0.8:
#         return out
#
#     mapping = {"low": 1, "light": 1, "medium": 2, "moderate": 2, "high": 3, "vigorous": 3}
#     s2 = s.astype(str).str.strip().str.lower()
#     out2 = s2.map(mapping)
#     out3 = pd.to_numeric(s2.str.extract(r"(\d+(\.\d+)?)")[0], errors="coerce")
#     return out2.combine_first(out3)
#
# def canonical_hormones(df: pd.DataFrame, prefix: str):
#     mean_cols = [c for c in df.columns if c.startswith(prefix) and c.endswith("_mean")]
#     n_cols    = [c for c in df.columns if c.startswith(prefix) and c.endswith("_n")]
#
#     def base(c, suf):
#         return re.sub(fr"{re.escape(suf)}$", "", c)
#
#     means = {base(c, "_mean")[len(prefix):] for c in mean_cols}
#     ns    = {base(c, "_n")[len(prefix):] for c in n_cols}
#     return sorted(means & ns)
#
# def cliffs_delta(x, y):
#     """
#     Cliff's delta effect size: P(x>y) - P(x<y)
#     Range [-1,1]. 0 means no effect.
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)
#     x = x[np.isfinite(x)]
#     y = y[np.isfinite(y)]
#     if len(x) == 0 or len(y) == 0:
#         return np.nan
#     # O(n^2) but your n is tiny
#     gt = sum((xi > y).sum() for xi in x)
#     lt = sum((xi < y).sum() for xi in x)
#     return (gt - lt) / (len(x) * len(y))
#
# def label_window(prefix):
#     return "during" if prefix == "" else prefix.rstrip("__")
#
# # =========================
# # LOAD + PREP
# # =========================
# df = pd.read_csv(DATA_PATH)
# print("[INFO] Loaded:", DATA_PATH)
# print("[INFO] Raw shape:", df.shape)
#
# df = df[~df["participant"].isin(EXCLUDE_PARTICIPANTS)].copy()
# df["participant"] = df["participant"].astype(str)
#
# # intensity numeric
# if "intensity" not in df.columns:
#     raise KeyError("Column 'intensity' not found in CSV.")
# df["intensity"] = ensure_intensity_numeric(df["intensity"])
#
# # activity group (rest vs active)
# df["activity_label"] = df["activity_label"].astype(str).str.strip()
# df["activity_group"] = np.where(
#     df["activity_label"].isin(REST_SET), "rest",
#     np.where(df["activity_label"].isin(ACTIVE_SET), "active", "other")
# )
# df = df[df["activity_group"].isin(["rest", "active"])].copy()
# df["active_bin"] = (df["activity_group"] == "active").astype(int)
#
# print("[INFO] After exclude + group:", df.shape)
# print("[INFO] Participants:", sorted(df["participant"].unique().tolist()))
# print("[INFO] activity_group counts:\n", df["activity_group"].value_counts())
# print("[INFO] Intensity non-NA:", int(df["intensity"].notna().sum()), "/", len(df))
#
# # =========================
# # ANALYSIS
# # =========================
# all_rows = []
#
# for prefix in WINDOW_PREFIXES:
#     hormones = canonical_hormones(df, prefix)
#     wname = label_window(prefix)
#     print(f"\n[WINDOW] {wname} | hormones detected: {len(hormones)}")
#
#     for h in hormones:
#         n_col = f"{prefix}{h}_n"
#         y_col = f"{prefix}{h}_mean"
#         if n_col not in df.columns or y_col not in df.columns:
#             continue
#
#         sub = df[df[n_col] > 0].copy()
#         sub["y"] = pd.to_numeric(sub[y_col], errors="coerce")
#         sub = sub.dropna(subset=["y", "intensity", "activity_group"])
#
#         if len(sub) < MIN_N_TOTAL:
#             continue
#
#         y_rest = sub.loc[sub["activity_group"] == "rest", "y"].values
#         y_act  = sub.loc[sub["activity_group"] == "active", "y"].values
#
#         if len(y_rest) < MIN_N_REST or len(y_act) < MIN_N_ACTIVE:
#             continue
#
#         # Activity association: Mann–Whitney U (two-sided)
#         try:
#             u_stat, p_act = mannwhitneyu(y_rest, y_act, alternative="two-sided")
#         except Exception:
#             p_act = np.nan
#
#         delta = cliffs_delta(y_act, y_rest)  # positive means active > rest
#
#         # Intensity association: Spearman
#         rho, p_int = spearmanr(sub["intensity"], sub["y"], nan_policy="omit")
#
#         all_rows.append({
#             "window": wname,
#             "hormone": h,
#             "n_total": int(len(sub)),
#             "n_rest": int(len(y_rest)),
#             "n_active": int(len(y_act)),
#             "median_rest": float(np.median(y_rest)),
#             "median_active": float(np.median(y_act)),
#             "cliffs_delta_active_minus_rest": float(delta),
#             "p_activity_rest_vs_active": float(p_act) if np.isfinite(p_act) else np.nan,
#             "spearman_rho_intensity": float(rho) if np.isfinite(rho) else np.nan,
#             "p_intensity": float(p_int) if np.isfinite(p_int) else np.nan,
#         })
#
# res = pd.DataFrame(all_rows)
#
# if res.empty:
#     print("\n[RESULT] No hormones passed the minimum sample thresholds.")
#     print("Try lowering MIN_N_TOTAL / MIN_N_REST / MIN_N_ACTIVE, or analyse only during window.")
# else:
#     # FDR within window for both p types
#     for w in res["window"].unique():
#         m = res["window"] == w
#
#         for pcol in ["p_activity_rest_vs_active", "p_intensity"]:
#             vals = res.loc[m, pcol].values
#             ok = np.isfinite(vals)
#             if ok.sum() >= 2:
#                 _, q, _, _ = multipletests(vals[ok], method="fdr_bh")
#                 qcol = "q_" + pcol
#                 res.loc[m, qcol] = np.nan
#                 res.loc[m, qcol].iloc[np.where(ok)[0]] = q
#
#     # Rank: prefer low q, large |effect|
#     res["abs_delta"] = res["cliffs_delta_active_minus_rest"].abs()
#     res["abs_rho"]   = res["spearman_rho_intensity"].abs()
#
#     print("\n[TOP by Activity effect (FDR)]")
#     colsA = [
#         "window", "hormone", "n_total", "n_rest", "n_active",
#         "median_rest", "median_active",
#         "cliffs_delta_active_minus_rest",
#         "p_activity_rest_vs_active", "q_p_activity_rest_vs_active"
#     ]
#     colsA = [c for c in colsA if c in res.columns]
#     topA = res.sort_values(
#         ["window", "q_p_activity_rest_vs_active", "abs_delta"],
#         ascending=[True, True, False],
#         na_position="last"
#     )
#     print(topA[colsA].head(20).to_string(index=False))
#
#     print("\n[TOP by Intensity association (FDR)]")
#     colsI = [
#         "window", "hormone", "n_total",
#         "spearman_rho_intensity",
#         "p_intensity", "q_p_intensity"
#     ]
#     colsI = [c for c in colsI if c in res.columns]
#     topI = res.sort_values(
#         ["window", "q_p_intensity", "abs_rho"],
#         ascending=[True, True, False],
#         na_position="last"
#     )
#     print(topI[colsI].head(20).to_string(index=False))
#
#     print("\n[HOW TO INTERPRET]")
#     print("- Cliff's delta: |delta| ~ 0.11 small, 0.28 medium, 0.43 large (rule-of-thumb).")
#     print("- Spearman rho: |rho| closer to 1 means stronger monotonic relationship.")
#     print("- Prefer q<0.05, but with small data you may look at effect sizes + consistency across windows.")
#     print("- If post-window shows stronger effects than during, that supports a lagged physiological response.")
#
# print("\n[DONE]")
