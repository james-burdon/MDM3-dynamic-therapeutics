

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
    precision_recall_fscore_support,
    classification_report,
)

# =========================
# CONFIG / PATH
# =========================
DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormone_interval_activity_agg.csv"

ACTIVE_THRESHOLD = 0.5
TOP_K = 3
MIN_NONMISSING = 10
N_SPLITS = 5
RANDOM_STATE = 42

LOGREG_C = 0.5
PROB_THRESHOLD = 0.5

PLOT_TOPN_STABILITY = 12
BOXPLOT_TOPN = 3


# =========================
# UTILITIES
# =========================
def merge_dupe_col(df: pd.DataFrame, keep: str, drop: str) -> None:
    if keep in df.columns and drop in df.columns:
        df[keep] = df[keep].combine_first(df[drop])
        df.drop(columns=[drop], inplace=True)
    elif drop in df.columns and keep not in df.columns:
        df.rename(columns={drop: keep}, inplace=True)

def get_candidate_hormone_cols(df: pd.DataFrame) -> list[str]:
    NON_FEATURE_COLS = {
        "cat_id", "hormone_time", "next_hormone_time", "interval_minutes", "n_samples",
        "active_ratio", "active_mean", "y_active",
    }
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().any()]  # drop all-NaN cols
    return cols

def make_logreg_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LOGREG_C,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])

def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

def point_biserial_corr_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    try:
        return float(np.corrcoef(m[xcol].values, m[ycol].values)[0, 1])
    except Exception:
        return np.nan

def mannwhitney_p_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    a = m.loc[m[ycol] == 0, xcol].values
    b = m.loc[m[ycol] == 1, xcol].values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def single_feature_auc_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(m[ycol].values, m[xcol].values))
    except Exception:
        return np.nan

def select_features_by_association(train_df: pd.DataFrame, hormone_cols: list[str],
                                   top_k: int, min_nonmissing: int) -> list[str]:
    """
    Association-based selection (TRAIN ONLY), no leakage.

    Ranking:
      1) abs(point-biserial correlation)
      2) abs(single-feature AUC - 0.5)
      3) n_nonmissing
    """
    rows = []
    ycol = "y_active"

    for col in hormone_cols:
        m = train_df[[col, ycol]].dropna()
        n = int(m.shape[0])
        if n < min_nonmissing:
            continue
        if m[ycol].nunique() < 2:
            continue

        corr = point_biserial_corr_from_m(m, col, ycol)
        if pd.isna(corr):
            continue

        auc1 = single_feature_auc_from_m(m, col, ycol)
        p = mannwhitney_p_from_m(m, col, ycol)

        rows.append({
            "feature": col,
            "n_nonmissing": n,
            "abs_corr": abs(corr),
            "auc_strength": np.nan if pd.isna(auc1) else abs(auc1 - 0.5),
            "mw_p": p
        })

    if not rows:
        return []

    s = pd.DataFrame(rows).sort_values(
        ["abs_corr", "auc_strength", "n_nonmissing"],
        ascending=[False, False, False]
    )
    return s["feature"].head(top_k).tolist()

def feasible_n_splits(y: pd.Series, desired: int) -> int:
    vc = y.value_counts()
    if vc.shape[0] < 2:
        return 0
    return int(min(desired, vc.min()))

def probs_to_pred(prob, threshold=0.5):
    prob = np.asarray(prob, dtype=float)
    return (prob >= threshold).astype(int)

def compute_class_metrics(y_true, y_pred) -> dict:
    """
    Return per-class precision/f1 in a stable dict.
    Class mapping: 1=Active, 0=Inactive
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # precision, recall, f1 per class [0,1]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )

    return {
        "inactive_precision": float(prec[0]),
        "inactive_f1": float(f1[0]),
        "active_precision": float(prec[1]),
        "active_f1": float(f1[1]),
        "support_inactive": int(sup[0]),
        "support_active": int(sup[1]),
    }

def evaluate_from_probs(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    y_pred = probs_to_pred(prob, threshold=threshold)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_roc": float(safe_auc(y_true, prob)),
    }
    out.update(compute_class_metrics(y_true, y_pred))
    return out

def plot_roc_curve(y_true, y_prob, title: str):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    plt.figure()
    if np.unique(y_true).size < 2:
        plt.title(title + " (single class — ROC undefined)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_stability(counter: dict, title: str, topn: int = 12):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topn]
    plt.figure()
    if not items:
        plt.title(title + " (none)")
        plt.show()
        return

    feats = [k for k, _ in items]
    counts = [v for _, v in items]
    x = np.arange(len(feats))

    plt.bar(x, counts)
    plt.xticks(x, feats, rotation=45, ha="right")
    plt.ylabel("Selected count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_boxplots_by_class(sub_df: pd.DataFrame, features: list[str], cat_id: int):
    """
    Boxplots of feature distributions for inactive vs active.
    For visual comparability we z-score within CAT (plot only).
    """
    dfp = sub_df.copy()
    dfp = dfp.dropna(subset=["y_active"])

    for f in features:
        if f in dfp.columns and dfp[f].notna().sum() >= 3:
            s = dfp[f].astype(float)
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd > 0:
                dfp[f] = (s - mu) / sd

    data, labels = [], []
    for f in features:
        if f not in dfp.columns:
            continue
        data.append(dfp[dfp["y_active"] == 0][f].dropna().values)
        data.append(dfp[dfp["y_active"] == 1][f].dropna().values)
        labels.append(f"{f}\nInactive")
        labels.append(f"{f}\nActive")

    plt.figure()
    if not data or all(len(d) == 0 for d in data):
        plt.title(f"CAT{cat_id}: Boxplots (no data)")
        plt.show()
        return

    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title(f"CAT{cat_id}: Top selected hormones by class (z-scored)")
    plt.ylabel("z-score (within participant)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def print_top_counts(counter: dict, title: str, topn: int = 20):
    print(f"\n=== {title} ===")
    if not counter:
        print("None")
        return
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topn]
    for f, c in items:
        print(f"{f:>18s} : {c}")

def fmt_pct(x: float) -> float:
    return round(100.0 * float(x), 1)


# =========================
# 1) Load + clean + label
# =========================
df = pd.read_csv(DATA_PATH)

merge_dupe_col(df, "Dopeg", "DOPEG")
merge_dupe_col(df, "Cortisol", "cortisol")
merge_dupe_col(df, "Cortisone", "cortisone")
merge_dupe_col(df, "Dopamine", "dopamine")

df = df.dropna(subset=["cat_id", "active_ratio"]).copy()
df["cat_id"] = df["cat_id"].astype(int)
df["y_active"] = (df["active_ratio"] > ACTIVE_THRESHOLD).astype(int)

hormone_cols = get_candidate_hormone_cols(df)

print("[INFO] shape:", df.shape)
print("[INFO] cats:", df["cat_id"].value_counts().sort_index().to_dict())
print("[INFO] candidate hormone cols:", hormone_cols)


# =========================
# 2) WITHIN-SUBJECT: nested selection inside CV
# =========================
print("\n==============================")
print("WITHIN-SUBJECT (per CAT): fold-wise association selection -> logistic regression")
print("==============================")

within_results = []
selected_counter_within = {}
selected_counter_within_by_cat = {}
within_oof = {}  # cat_id -> dict(y_true, y_prob)

for cat_id, sub in df.groupby("cat_id"):
    sub = sub.copy()

    if "hormone_time" in sub.columns:
        sub["hormone_time"] = pd.to_datetime(sub["hormone_time"], errors="coerce")
        sub = sub.sort_values("hormone_time")

    sub = sub.reset_index(drop=True)
    y = sub["y_active"].astype(int)

    n_splits = feasible_n_splits(y, N_SPLITS)
    if len(sub) < 12 or n_splits < 2:
        print(f"[SKIP] CAT{cat_id}: insufficient samples or single-class (n={len(sub)})")
        continue

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_all, y_prob_all = [], []
    selected_counter_within_by_cat.setdefault(cat_id, {})

    for fold, (tr, te) in enumerate(outer.split(sub[hormone_cols], y), start=1):
        train_df = sub.iloc[tr].copy()
        test_df = sub.iloc[te].copy()

        selected = select_features_by_association(
            train_df=train_df,
            hormone_cols=hormone_cols,
            top_k=TOP_K,
            min_nonmissing=MIN_NONMISSING
        )
        if len(selected) == 0:
            continue

        for f in selected:
            selected_counter_within[f] = selected_counter_within.get(f, 0) + 1
            selected_counter_within_by_cat[cat_id][f] = selected_counter_within_by_cat[cat_id].get(f, 0) + 1

        pipe = make_logreg_pipe()
        pipe.fit(train_df[selected], train_df["y_active"].astype(int))

        prob = pipe.predict_proba(test_df[selected])[:, 1]
        y_true_all.extend(test_df["y_active"].astype(int).tolist())
        y_prob_all.extend(prob.tolist())

    if np.unique(y_true_all).size < 2:
        print(f"[WARN] CAT{cat_id}: aggregated test single-class -> skip metrics/ROC")
        continue

    metrics = evaluate_from_probs(y_true_all, y_prob_all, threshold=PROB_THRESHOLD)
    within_results.append({
        "cat_id": cat_id,
        "n": int(len(sub)),
        **metrics
    })
    within_oof[cat_id] = {"y_true": y_true_all, "y_prob": y_prob_all}

    print(
        f"CAT{cat_id}: n={len(sub)} | "
        f"Accuracy={metrics['accuracy']:.3f} | "
        f"BalancedAccuracy={metrics['balanced_accuracy']:.3f} | "
        f"AreaUnderReceiverOperatingCharacteristicCurve={metrics['auc_roc']:.3f} | "
        f"ActivePrecision={metrics['active_precision']:.3f} ActiveF1={metrics['active_f1']:.3f} | "
        f"InactivePrecision={metrics['inactive_precision']:.3f} InactiveF1={metrics['inactive_f1']:.3f}"
    )

print("\n[SUMMARY] within-subject results")
if within_results:
    print(pd.DataFrame(within_results).sort_values("cat_id").to_string(index=False))
else:
    print("None")


# =========================
# 3) CROSS-SUBJECT: LOPO
# =========================
print("\n==============================")
print("CROSS-SUBJECT Leave-One-Participant-Out cross-validation: association selection on TRAIN -> logistic regression -> TEST")
print("==============================")

lopo_results = []
selected_counter_lopo = {}
lopo_preds = {}  # test_cat -> dict(y_true, y_prob, selected)

for test_cat in sorted(df["cat_id"].unique()):
    train_df = df[df["cat_id"] != test_cat].copy()
    test_df = df[df["cat_id"] == test_cat].copy()

    if len(test_df) < 5:
        print(f"[SKIP] Leave-One-Participant-Out cross-validation CAT{test_cat}: too few test samples (n={len(test_df)})")
        continue
    if train_df["y_active"].nunique() < 2 or test_df["y_active"].nunique() < 2:
        print(f"[SKIP] Leave-One-Participant-Out cross-validation CAT{test_cat}: single-class in train/test")
        continue

    selected = select_features_by_association(
        train_df=train_df,
        hormone_cols=hormone_cols,
        top_k=TOP_K,
        min_nonmissing=MIN_NONMISSING
    )
    if len(selected) == 0:
        print(f"[SKIP] Leave-One-Participant-Out cross-validation CAT{test_cat}: no features selected")
        continue

    for f in selected:
        selected_counter_lopo[f] = selected_counter_lopo.get(f, 0) + 1

    pipe = make_logreg_pipe()
    pipe.fit(train_df[selected], train_df["y_active"].astype(int))

    prob = pipe.predict_proba(test_df[selected])[:, 1]
    metrics = evaluate_from_probs(test_df["y_active"].astype(int).values, prob, threshold=PROB_THRESHOLD)

    lopo_results.append({
        "test_cat": test_cat,
        "n_test": int(len(test_df)),
        "features": ",".join(selected),
        **metrics
    })
    lopo_preds[test_cat] = {
        "y_true": test_df["y_active"].astype(int).values.tolist(),
        "y_prob": prob.tolist(),
        "selected": selected
    }

    print(
        f"Test CAT{test_cat}: n={len(test_df)} | feats={selected} | "
        f"Accuracy={metrics['accuracy']:.3f} | "
        f"BalancedAccuracy={metrics['balanced_accuracy']:.3f} | "
        f"AreaUnderReceiverOperatingCharacteristicCurve={metrics['auc_roc']:.3f} | "
        f"ActivePrecision={metrics['active_precision']:.3f} ActiveF1={metrics['active_f1']:.3f} | "
        f"InactivePrecision={metrics['inactive_precision']:.3f} InactiveF1={metrics['inactive_f1']:.3f}"
    )

print("\n[SUMMARY] Leave-One-Participant-Out cross-validation results")
if lopo_results:
    print(pd.DataFrame(lopo_results).sort_values("test_cat").to_string(index=False))
else:
    print("None")


# =========================
# 4) Feature stability (console)
# =========================
print_top_counts(selected_counter_within, "Most frequently selected features (within-subject folds)", topn=20)
print_top_counts(selected_counter_lopo, "Most frequently selected features (Leave-One-Participant-Out cross-validation splits)", topn=20)


# =========================
# 5) PLOTS (SHOW ONLY)
# =========================
for cat_id, d in within_oof.items():
    plot_roc_curve(
        y_true=d["y_true"],
        y_prob=d["y_prob"],
        title=f"Within-subject Receiver Operating Characteristic curve (CAT{cat_id}) — nested selection + Logistic Regression"
    )

for test_cat, d in lopo_preds.items():
    plot_roc_curve(
        y_true=d["y_true"],
        y_prob=d["y_prob"],
        title=f"Leave-One-Participant-Out cross-validation Receiver Operating Characteristic curve (Test CAT{test_cat}) — features={d['selected']}"
    )

plot_feature_stability(
    counter=selected_counter_within,
    title=f"Feature stability (within-subject folds) — TOP {PLOT_TOPN_STABILITY}",
    topn=PLOT_TOPN_STABILITY
)

plot_feature_stability(
    counter=selected_counter_lopo,
    title=f"Feature stability (Leave-One-Participant-Out cross-validation splits) — TOP {PLOT_TOPN_STABILITY}",
    topn=PLOT_TOPN_STABILITY
)

for cat_id in [3, 4]:
    if cat_id not in selected_counter_within_by_cat:
        continue
    feats_counts = sorted(selected_counter_within_by_cat[cat_id].items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, _ in feats_counts[:BOXPLOT_TOPN] if f in df.columns]
    if top_feats:
        sub_df = df[df["cat_id"] == cat_id].copy()
        plot_boxplots_by_class(sub_df=sub_df, features=top_feats, cat_id=cat_id)


# =========================
# 6) REPORT-READY SUMMARY (to fill your table)
# =========================
print("\n==============================")
print("REPORT-READY SUMMARY (percent)")
print("==============================")

# Recommend using LOPO for "generalisation" table; if you need within-subject, change here.
if lopo_results:
    # pool all LOPO predictions into one overall set
    all_y_true, all_y_prob = [], []
    for d in lopo_preds.values():
        all_y_true.extend(d["y_true"])
        all_y_prob.extend(d["y_prob"])
    overall = evaluate_from_probs(all_y_true, all_y_prob, threshold=PROB_THRESHOLD)

    print(f"Active Precision (%):  {fmt_pct(overall['active_precision'])}")
    print(f"Active F1-score (%):   {fmt_pct(overall['active_f1'])}")
    print(f"Inactive Precision (%):{fmt_pct(overall['inactive_precision'])}")
    print(f"Inactive F1-score (%): {fmt_pct(overall['inactive_f1'])}")
    print(f"Accuracy (%):          {fmt_pct(overall['accuracy'])}")
    print(f"Balanced Accuracy (%): {fmt_pct(overall['balanced_accuracy'])}")
    print(f"Area Under ROC Curve:  {overall['auc_roc']:.3f}")
else:
    print("No LOPO results available to summarise.")

print("\n[DONE]")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
    auc as sk_auc
)


DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormone_interval_activity_agg.csv"

ACTIVE_THRESHOLD = 0.5
TOP_K = 3
MIN_NONMISSING = 10
N_SPLITS = 5
RANDOM_STATE = 42

LOGREG_C = 0.5
PROB_THRESHOLD = 0.5

PLOT_TOPN_STABILITY = 12
BOXPLOT_TOPN = 3

#
def merge_dupe_col(df: pd.DataFrame, keep: str, drop: str) -> None:
    if keep in df.columns and drop in df.columns:
        df[keep] = df[keep].combine_first(df[drop])
        df.drop(columns=[drop], inplace=True)
    elif drop in df.columns and keep not in df.columns:
        df.rename(columns={drop: keep}, inplace=True)

def get_candidate_hormone_cols(df: pd.DataFrame) -> list[str]:
    NON_FEATURE_COLS = {
        "cat_id", "hormone_time", "next_hormone_time", "interval_minutes", "n_samples",
        "active_ratio", "active_mean", "y_active",
    }
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cols = [c for c in cols if df[c].notna().any()]  # drop all-NaN
    return cols

def make_logreg_pipe() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LOGREG_C,
            class_weight="balanced",
            max_iter=5000,
            solver="lbfgs",
            random_state=RANDOM_STATE
        ))
    ])

def safe_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

def point_biserial_corr_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    try:
        return float(np.corrcoef(m[xcol].values, m[ycol].values)[0, 1])
    except Exception:
        return np.nan

def mannwhitney_p_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    a = m.loc[m[ycol] == 0, xcol].values
    b = m.loc[m[ycol] == 1, xcol].values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    try:
        return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
    except Exception:
        return np.nan

def single_feature_auc_from_m(m: pd.DataFrame, xcol: str, ycol: str) -> float:
    if m.shape[0] < 6 or m[ycol].nunique() < 2:
        return np.nan
    try:
        return float(roc_auc_score(m[ycol].values, m[xcol].values))
    except Exception:
        return np.nan

def select_features_by_association(train_df: pd.DataFrame, hormone_cols: list[str],
                                   top_k: int, min_nonmissing: int) -> list[str]:
    """
    Association-based selection (TRAIN ONLY), no leakage.
    Ranking:
      1) abs(point-biserial corr)
      2) abs(AUC - 0.5)
      3) n_nonmissing
    """
    rows = []
    ycol = "y_active"

    for col in hormone_cols:
        m = train_df[[col, ycol]].dropna()
        n = int(m.shape[0])
        if n < min_nonmissing:
            continue
        if m[ycol].nunique() < 2:
            continue

        corr = point_biserial_corr_from_m(m, col, ycol)
        if pd.isna(corr):
            continue

        auc1 = single_feature_auc_from_m(m, col, ycol)
        p = mannwhitney_p_from_m(m, col, ycol)

        rows.append({
            "feature": col,
            "n_nonmissing": n,
            "abs_corr": abs(corr),
            "auc_strength": np.nan if pd.isna(auc1) else abs(auc1 - 0.5),
            "mw_p": p
        })

    if not rows:
        return []

    s = pd.DataFrame(rows).sort_values(
        ["abs_corr", "auc_strength", "n_nonmissing"],
        ascending=[False, False, False]
    )
    return s["feature"].head(top_k).tolist()

def evaluate_from_probs(y_true, prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= threshold).astype(int)
    return {
        "acc": accuracy_score(y_true, pred),
        "bal_acc": balanced_accuracy_score(y_true, pred),
        "auc": safe_auc(y_true, prob)
    }

def feasible_n_splits(y: pd.Series, desired: int) -> int:
    vc = y.value_counts()
    if vc.shape[0] < 2:
        return 0
    return int(min(desired, vc.min()))

def plot_roc_curve(y_true, y_prob, title: str):
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    plt.figure()
    if np.unique(y_true).size < 2:
        plt.title(title + " (single class — ROC undefined)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = sk_auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_stability(counter: dict, title: str, topn: int = 12):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topn]
    plt.figure()
    if not items:
        plt.title(title + " (none)")
        plt.show()
        return

    feats = [k for k, _ in items]
    counts = [v for _, v in items]
    x = np.arange(len(feats))

    plt.bar(x, counts)
    plt.xticks(x, feats, rotation=45, ha="right")
    plt.ylabel("Selected count")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_boxplots_by_class(sub_df: pd.DataFrame, features: list[str], cat_id: int):
    """
    Boxplots of feature distributions for inactive vs active.
    For visual comparability we z-score within CAT (plot only).
    """
    dfp = sub_df.copy()
    dfp = dfp.dropna(subset=["y_active"])

    # z-score for plotting
    for f in features:
        if f in dfp.columns and dfp[f].notna().sum() >= 3:
            s = dfp[f].astype(float)
            mu = s.mean()
            sd = s.std(ddof=0)
            if sd > 0:
                dfp[f] = (s - mu) / sd

    data, labels = [], []
    for f in features:
        if f not in dfp.columns:
            continue
        data.append(dfp[dfp["y_active"] == 0][f].dropna().values)
        data.append(dfp[dfp["y_active"] == 1][f].dropna().values)
        labels.append(f"{f}\nInactive")
        labels.append(f"{f}\nActive")

    plt.figure()
    if not data or all(len(d) == 0 for d in data):
        plt.title(f"CAT{cat_id}: Boxplots (no data)")
        plt.show()
        return

    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.title(f"CAT{cat_id}: Top selected hormones by class (z-scored)")
    plt.ylabel("z-score (within CAT)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =========================
# 1) Load + clean + label
# =========================
df = pd.read_csv(DATA_PATH)

merge_dupe_col(df, "Dopeg", "DOPEG")
merge_dupe_col(df, "Cortisol", "cortisol")
merge_dupe_col(df, "Cortisone", "cortisone")
merge_dupe_col(df, "Dopamine", "dopamine")

df = df.dropna(subset=["cat_id", "active_ratio"]).copy()
df["cat_id"] = df["cat_id"].astype(int)
df["y_active"] = (df["active_ratio"] > ACTIVE_THRESHOLD).astype(int)

hormone_cols = get_candidate_hormone_cols(df)

print("[INFO] shape:", df.shape)
print("[INFO] cats:", df["cat_id"].value_counts().sort_index().to_dict())
print("[INFO] candidate hormone cols:", hormone_cols)

# =========================
# 2) WITHIN-SUBJECT: nested selection inside CV
# =========================
print("\n==============================")
print("WITHIN-SUBJECT (per CAT): fold-wise association selection -> logistic regression")
print("==============================")

within_results = []
selected_counter_within = {}
selected_counter_within_by_cat = {}
within_oof = {}  # cat_id -> dict(y_true, y_prob)

for cat_id, sub in df.groupby("cat_id"):
    sub = sub.copy()

    if "hormone_time" in sub.columns:
        sub["hormone_time"] = pd.to_datetime(sub["hormone_time"], errors="coerce")
        sub = sub.sort_values("hormone_time")

    sub = sub.reset_index(drop=True)
    y = sub["y_active"].astype(int)

    n_splits = feasible_n_splits(y, N_SPLITS)
    if len(sub) < 12 or n_splits < 2:
        print(f"[SKIP] CAT{cat_id}: insufficient samples or single-class (n={len(sub)})")
        continue

    outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    y_true_all, y_prob_all = [], []
    selected_counter_within_by_cat.setdefault(cat_id, {})

    for fold, (tr, te) in enumerate(outer.split(sub[hormone_cols], y), start=1):
        train_df = sub.iloc[tr].copy()
        test_df = sub.iloc[te].copy()

        selected = select_features_by_association(
            train_df=train_df,
            hormone_cols=hormone_cols,
            top_k=TOP_K,
            min_nonmissing=MIN_NONMISSING
        )
        if len(selected) == 0:
            continue

        for f in selected:
            selected_counter_within[f] = selected_counter_within.get(f, 0) + 1
            selected_counter_within_by_cat[cat_id][f] = selected_counter_within_by_cat[cat_id].get(f, 0) + 1

        pipe = make_logreg_pipe()
        pipe.fit(train_df[selected], train_df["y_active"].astype(int))

        prob = pipe.predict_proba(test_df[selected])[:, 1]

        y_true_all.extend(test_df["y_active"].astype(int).tolist())
        y_prob_all.extend(prob.tolist())

    if np.unique(y_true_all).size < 2:
        print(f"[WARN] CAT{cat_id}: aggregated test single-class -> skip metrics/ROC")
        continue

    metrics = evaluate_from_probs(y_true_all, y_prob_all, threshold=PROB_THRESHOLD)
    within_results.append({
        "cat_id": cat_id,
        "n": int(len(sub)),
        "acc": metrics["acc"],
        "bal_acc": metrics["bal_acc"],
        "auc": metrics["auc"]
    })
    within_oof[cat_id] = {"y_true": y_true_all, "y_prob": y_prob_all}

    print(f"CAT{cat_id}: n={len(sub)} | Acc={metrics['acc']:.3f} | BalAcc={metrics['bal_acc']:.3f} | AUC={metrics['auc']:.3f}")

print("\n[SUMMARY] within-subject results")
if within_results:
    print(pd.DataFrame(within_results).sort_values("cat_id").to_string(index=False))
else:
    print("None")

# =========================
# 3) CROSS-SUBJECT: LOPO
# =========================
print("\n==============================")
print("CROSS-SUBJECT LOPO: association selection on TRAIN -> logistic regression -> TEST")
print("==============================")

lopo_results = []
selected_counter_lopo = {}
lopo_preds = {}  # test_cat -> dict(y_true, y_prob, selected)

for test_cat in sorted(df["cat_id"].unique()):
    train_df = df[df["cat_id"] != test_cat].copy()
    test_df = df[df["cat_id"] == test_cat].copy()

    if len(test_df) < 5:
        print(f"[SKIP] LOPO CAT{test_cat}: too few test samples (n={len(test_df)})")
        continue
    if train_df["y_active"].nunique() < 2 or test_df["y_active"].nunique() < 2:
        print(f"[SKIP] LOPO CAT{test_cat}: single-class in train/test")
        continue

    selected = select_features_by_association(
        train_df=train_df,
        hormone_cols=hormone_cols,
        top_k=TOP_K,
        min_nonmissing=MIN_NONMISSING
    )
    if len(selected) == 0:
        print(f"[SKIP] LOPO CAT{test_cat}: no features selected")
        continue

    for f in selected:
        selected_counter_lopo[f] = selected_counter_lopo.get(f, 0) + 1

    pipe = make_logreg_pipe()
    pipe.fit(train_df[selected], train_df["y_active"].astype(int))

    prob = pipe.predict_proba(test_df[selected])[:, 1]
    metrics = evaluate_from_probs(test_df["y_active"].astype(int).values, prob, threshold=PROB_THRESHOLD)

    lopo_results.append({
        "test_cat": test_cat,
        "n_test": int(len(test_df)),
        "features": ",".join(selected),
        "acc": metrics["acc"],
        "bal_acc": metrics["bal_acc"],
        "auc": metrics["auc"]
    })
    lopo_preds[test_cat] = {
        "y_true": test_df["y_active"].astype(int).values.tolist(),
        "y_prob": prob.tolist(),
        "selected": selected
    }

    print(f"Test CAT{test_cat}: n={len(test_df)} | feats={selected} | Acc={metrics['acc']:.3f} | BalAcc={metrics['bal_acc']:.3f} | AUC={metrics['auc']:.3f}")

print("\n[SUMMARY] LOPO results")
if lopo_results:
    print(pd.DataFrame(lopo_results).sort_values("test_cat").to_string(index=False))
else:
    print("None")

# =========================
# 4) Feature stability (console)
# =========================
def print_top_counts(counter: dict, title: str, topn: int = 20):
    print(f"\n=== {title} ===")
    if not counter:
        print("None")
        return
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)[:topn]
    for f, c in items:
        print(f"{f:>18s} : {c}")

print_top_counts(selected_counter_within, "Most frequently selected features (within-subject folds)", topn=20)
print_top_counts(selected_counter_lopo, "Most frequently selected features (LOPO splits)", topn=20)

# =========================
# 5) PLOTS (SHOW ONLY)
# =========================
# Within-subject ROC curves (out-of-fold)
for cat_id, d in within_oof.items():
    plot_roc_curve(
        y_true=d["y_true"],
        y_prob=d["y_prob"],
        title=f"Within-subject ROC (CAT{cat_id}) — nested selection + logreg"
    )

# LOPO ROC curves
for test_cat, d in lopo_preds.items():
    plot_roc_curve(
        y_true=d["y_true"],
        y_prob=d["y_prob"],
        title=f"LOPO ROC (Test CAT{test_cat}) — feats={d['selected']}"
    )

# Feature stability bar charts
plot_feature_stability(
    counter=selected_counter_within,
    title=f"Feature stability (within-subject folds) — TOP {PLOT_TOPN_STABILITY}",
    topn=PLOT_TOPN_STABILITY
)

plot_feature_stability(
    counter=selected_counter_lopo,
    title=f"Feature stability (LOPO splits) — TOP {PLOT_TOPN_STABILITY}",
    topn=PLOT_TOPN_STABILITY
)

# Boxplots for CAT3 and CAT4: top features selected within folds
for cat_id in [3, 4]:
    if cat_id not in selected_counter_within_by_cat:
        continue
    feats_counts = sorted(selected_counter_within_by_cat[cat_id].items(), key=lambda x: x[1], reverse=True)
    top_feats = [f for f, _ in feats_counts[:BOXPLOT_TOPN] if f in df.columns]
    if top_feats:
        sub_df = df[df["cat_id"] == cat_id].copy()
        plot_boxplots_by_class(sub_df=sub_df, features=top_feats, cat_id=cat_id)

print("\n[DONE]")






















# import pandas as pd
# import numpy as np
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import (
#     accuracy_score,
#     roc_auc_score,
#     classification_report,
#     confusion_matrix
# )
#
# # =========================
# # PATH
# # =========================
# DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormone_interval_activity_agg.csv"
#
# # =========================
# # CONFIG
# # =========================
# TRAIN_CAT = 3
# ACTIVE_THRESHOLD = 0.5   # active_ratio > 0.5 -> active
#
# # =========================
# # 1) Load data
# # =========================
# df = pd.read_csv(DATA_PATH)
#
# print("[INFO] Raw shape:", df.shape)
# print("[INFO] CAT counts:")
# print(df["cat_id"].value_counts().sort_index())
#
# # =========================
# # 2) Build binary label
# # =========================
# df["y_active"] = (df["active_ratio"] > ACTIVE_THRESHOLD).astype(int)
#
# print("\n[INFO] Label distribution:")
# print(df["y_active"].value_counts())
#
# # =========================
# # 3) Select hormone features
# # =========================
# NON_FEATURE_COLS = {
#     "cat_id",
#     "hormone_time",
#     "next_hormone_time",
#     "interval_minutes",
#     "n_samples",
#     "active_ratio",
#     "y_active"
# }
#
# hormone_cols = [
#     c for c in df.columns
#     if c not in NON_FEATURE_COLS
# ]
#
# print("\n[INFO] Hormone features used:")
# print(hormone_cols)
#
# X = df[hormone_cols]
# y = df["y_active"]
#
# # =========================
# # 4) Train / test split by CAT
# # =========================
# train_mask = df["cat_id"] == TRAIN_CAT
# test_mask = df["cat_id"] != TRAIN_CAT
#
# X_train, y_train = X[train_mask], y[train_mask]
# X_test, y_test = X[test_mask], y[test_mask]
#
# print(f"\n[INFO] Train samples (CAT{TRAIN_CAT}):", len(X_train))
# print("[INFO] Test samples (other CATs):", len(X_test))
# print("[INFO] Test CAT distribution:")
# print(df.loc[test_mask, "cat_id"].value_counts().sort_index())
#
# # =========================
# # 5) Model pipeline
# # =========================
# pipe = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
#     ("clf", LogisticRegression(
#         max_iter=2000,
#         class_weight="balanced",
#         solver="lbfgs"
#     ))
# ])
#
# # =========================
# # 6) Train
# # =========================
# pipe.fit(X_train, y_train)
#
# # =========================
# # 7) Evaluate
# # =========================
# y_pred = pipe.predict(X_test)
# y_prob = pipe.predict_proba(X_test)[:, 1]
#
# acc = accuracy_score(y_test, y_pred)
# auc = roc_auc_score(y_test, y_prob)
#
# print("\n=== TEST PERFORMANCE (Train=CAT3, Test=Others) ===")
# print("Accuracy:", round(acc, 4))
# print("ROC AUC :", round(auc, 4))
#
# print("\nConfusion matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# print("\nClassification report:")
# print(classification_report(y_test, y_pred, digits=3))
#
# import pandas as pd
# import numpy as np
#
# from sklearn.linear_model import LogisticRegression
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
#
# # =========================
# # PATH
# # =========================
# DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormone_interval_activity_agg.csv"
#
# # =========================
# # CONFIG
# # =========================
# TRAIN_CAT = 3
# ACTIVE_THRESHOLD = 0.5  # active_ratio > 0.5 -> active
#
# # =========================
# # HELPERS
# # =========================
# def merge_dupe_col(df: pd.DataFrame, keep: str, drop: str) -> None:
#     """
#     Merge duplicate hormone columns with different naming (case etc.).
#     keep = preferred final name, drop = alternative name.
#     """
#     if keep in df.columns and drop in df.columns:
#         df[keep] = df[keep].combine_first(df[drop])
#         df.drop(columns=[drop], inplace=True)
#     elif drop in df.columns and keep not in df.columns:
#         df.rename(columns={drop: keep}, inplace=True)
#
# # =========================
# # 1) Load
# # =========================
# df = pd.read_csv(DATA_PATH)
#
# print("[INFO] Raw shape:", df.shape)
# print("[INFO] CAT counts:")
# print(df["cat_id"].value_counts().sort_index())
#
# # =========================
# # 2) Clean duplicate hormone columns (case inconsistencies)
# # =========================
# merge_dupe_col(df, "Dopeg", "DOPEG")
# merge_dupe_col(df, "Cortisol", "cortisol")
# merge_dupe_col(df, "Cortisone", "cortisone")
#
# # =========================
# # 3) Build label
# # =========================
# if "active_ratio" not in df.columns:
#     raise ValueError("Missing 'active_ratio' in dataset. This should be produced by your interval aggregation script.")
#
# df["y_active"] = (df["active_ratio"] > ACTIVE_THRESHOLD).astype(int)
#
# print("\n[INFO] Label distribution:")
# print(df["y_active"].value_counts())
#
# # =========================
# # 4) Feature selection (HORMONES ONLY)
# #    IMPORTANT: remove leakage columns like active_mean / active_ratio
# # =========================
# NON_FEATURE_COLS = {
#     "cat_id",
#     "hormone_time",
#     "next_hormone_time",
#     "interval_minutes",
#     "n_samples",
#     "active_ratio",
#     "active_mean",   # <-- leakage, must exclude
#     "y_active",
# }
#
# hormone_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
#
# # keep only numeric columns (safety)
# hormone_cols = [c for c in hormone_cols if pd.api.types.is_numeric_dtype(df[c])]
#
# if not hormone_cols:
#     raise ValueError("No numeric hormone features found after cleaning/excluding leakage columns.")
#
# print("\n[INFO] Hormone features used:")
# print(hormone_cols)
#
# X = df[hormone_cols]
# y = df["y_active"]
#
# # =========================
# # 5) Train / test split
# # =========================
# train_mask = df["cat_id"] == TRAIN_CAT
# test_mask = df["cat_id"] != TRAIN_CAT
#
# X_train, y_train = X[train_mask], y[train_mask]
# X_test, y_test = X[test_mask], y[test_mask]
#
# print(f"\n[INFO] Train samples (CAT{TRAIN_CAT}):", len(X_train))
# print("[INFO] Test samples (other CATs):", len(X_test))
# print("[INFO] Test CAT distribution:")
# print(df.loc[test_mask, "cat_id"].value_counts().sort_index())
#
# # quick check: features all-NaN in TRAIN -> drop them explicitly
# all_nan_train = [c for c in hormone_cols if X_train[c].notna().sum() == 0]
# if all_nan_train:
#     print("\n[WARN] Dropping features that are all-NaN in TRAIN:", all_nan_train)
#     hormone_cols = [c for c in hormone_cols if c not in all_nan_train]
#     X_train = df.loc[train_mask, hormone_cols]
#     X_test = df.loc[test_mask, hormone_cols]
#
# # =========================
# # 6) Logistic Regression pipeline
# # =========================
# pipe = Pipeline(steps=[
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
#     ("clf", LogisticRegression(
#         max_iter=5000,
#         class_weight="balanced",
#         solver="lbfgs"
#     ))
# ])
#
# # =========================
# # 7) Fit
# # =========================
# pipe.fit(X_train, y_train)
#
# # =========================
# # 8) Evaluate
# # =========================
# y_pred = pipe.predict(X_test)
# y_prob = pipe.predict_proba(X_test)[:, 1]
#
# acc = accuracy_score(y_test, y_pred)
#
# # AUC requires both classes in y_test
# if len(np.unique(y_test)) == 2:
#     auc = roc_auc_score(y_test, y_prob)
# else:
#     auc = float("nan")
#
# print("\n=== TEST PERFORMANCE (Logistic Regression | Train=CAT3, Test=Others) ===")
# print("Accuracy:", round(acc, 4))
# print("ROC AUC :", "NA (single class in test)" if np.isnan(auc) else round(auc, 4))
#
# print("\nConfusion matrix:")
# print(confusion_matrix(y_test, y_pred))
#
# print("\nClassification report:")
# print(classification_report(y_test, y_pred, digits=3))
#
# # =========================
# # 9) Optional: show coefficients (interpretability)
# # =========================
# clf = pipe.named_steps["clf"]
# coefs = pd.Series(clf.coef_.ravel(), index=hormone_cols).sort_values(key=np.abs, ascending=False)
#
# print("\n=== Top coefficients (absolute) ===")
# print(coefs.head(15))
