# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\activity_hormone_bp_glucose_merged_all_20260203_002553.csv"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def make_activity_binary(df):
    if "activity_group" in df.columns:
        s = df["activity_group"].astype(str).str.lower().str.strip()
        y = np.where(s.isin(["active", "act"]), 1,
                     np.where(s.isin(["rest", "inactive"]), 0, np.nan))
        y = pd.Series(y, index=df.index)
        if y.notna().sum() > 0:
            print("[label] using activity_group")
            return y

    if "activity_label" in df.columns:
        s = df["activity_label"].astype(str).str.lower()
        y = np.where(s.str.contains("active"), 1,
                     np.where(s.str.contains("rest"), 0, np.nan))
        y = pd.Series(y, index=df.index)
        if y.notna().sum() > 0:
            print("[label] using activity_label text match")
            return y

    if "description" in df.columns:
        s = df["description"].astype(str).str.lower()

        rest_kw = ["sit", "sitting", "lying", "lay", "sleep", "rest"]
        active_kw = ["walk", "walking", "run", "running", "stand", "standing", "cycle", "stairs", "exercise"]

        rest_hit = np.zeros(len(s), dtype=bool)
        active_hit = np.zeros(len(s), dtype=bool)

        for w in rest_kw:
            rest_hit |= s.str.contains(w, regex=False, na=False)
        for w in active_kw:
            active_hit |= s.str.contains(w, regex=False, na=False)

        y = np.where(active_hit & (~rest_hit), 1,
                     np.where(rest_hit & (~active_hit), 0, np.nan))
        y = pd.Series(y, index=df.index)
        if y.notna().sum() > 0:
            print("[label] using description keywords")
            return y

    if "intensity" in df.columns:
        x = to_num(df["intensity"])
        if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
            thr = float(x.median())
            y = (x > thr).astype(float)
            print(f"[label] fallback using intensity median split (thr={thr:.3f})")
            return y

    return pd.Series([np.nan] * len(df), index=df.index)


# ---- additions start here (no deletions) ----

def safe_zscore_within_group(values: pd.Series, groups: pd.Series) -> pd.Series:
    # z-score inside each participant to reduce person-to-person baseline shifts
    out = pd.Series(index=values.index, dtype=float)
    for g in groups.astype(str).unique():
        idx = (groups.astype(str) == g)
        v = values.loc[idx].astype(float)
        m = v.mean(skipna=True)
        sd = v.std(skipna=True)
        if pd.isna(sd) or sd == 0:
            out.loc[idx] = np.nan
        else:
            out.loc[idx] = (v - m) / sd
    return out


def add_composite_features(df_use: pd.DataFrame, participants: pd.Series) -> pd.DataFrame:
    # Build composite indices from hormone columns if they exist
    df2 = df_use.copy()

    def col(name):
        return name if name in df2.columns else None

    na = col("h_noradrenaline")
    ad = col("h_adrenaline")

    mt = col("h_3_mt")
    omd = col("h_3_omd")
    met = col("h_metanephrine")
    nmet = col("h_normetanephrine")

    # sympathetic index
    if na and ad:
        z_na = safe_zscore_within_group(to_num(df2[na]), participants)
        z_ad = safe_zscore_within_group(to_num(df2[ad]), participants)
        df2["symp_index"] = z_na + z_ad
    else:
        df2["symp_index"] = np.nan

    # metabolism index
    parts = []
    for c in [mt, omd, met, nmet]:
        if c:
            parts.append(safe_zscore_within_group(to_num(df2[c]), participants))

    if len(parts) > 0:
        df2["meta_index"] = pd.concat(parts, axis=1).sum(axis=1, skipna=False)
    else:
        df2["meta_index"] = np.nan

    return df2


def run_lopo(model, X: pd.DataFrame, y: pd.Series, participants: pd.Series, tag: str):
    accs, aucs = [], []
    used = 0

    for pid in participants.unique():
        train_idx = participants != pid
        test_idx = participants == pid

        X_train, X_test = X.loc[train_idx], X.loc[test_idx]
        y_train, y_test = y.loc[train_idx], y.loc[test_idx]

        if len(y_test) < 5:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)

        if y_test.nunique() == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            aucs.append(auc)
            print(f"{tag} test={pid}: acc={acc:.2f}, auc={auc:.2f}, n={len(y_test)}")
        else:
            print(f"{tag} test={pid}: acc={acc:.2f}, auc=NA (single class), n={len(y_test)}")

        used += 1

    if used == 0:
        return {"used": 0, "acc_mean": np.nan, "auc_mean": np.nan}

    return {
        "used": used,
        "acc_mean": float(np.mean(accs)) if len(accs) else np.nan,
        "auc_mean": float(np.mean(aucs)) if len(aucs) else np.nan
    }


def plot_top_coefs(model, feature_cols, title, top_k=12):
    clf = model.named_steps["clf"]
    coefs = clf.coef_[0]

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coef": coefs,
        "abs_coef": np.abs(coefs)
    }).sort_values("abs_coef", ascending=False)

    print("\n[top features]")
    print(coef_df.head(10)[["feature", "coef"]].to_string(index=False))

    top = coef_df.head(top_k).iloc[::-1]
    plt.figure(figsize=(9, 5))
    plt.barh(top["feature"], top["coef"])
    plt.axvline(0)
    plt.title(title)
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.show()


# ---- additions end here ----


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print("[load]", os.path.basename(DATA_PATH))
    df = pd.read_csv(DATA_PATH, low_memory=False)

    if "participant" not in df.columns:
        raise ValueError("participant column is missing in the csv")

    df["y"] = make_activity_binary(df)

    hormone_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("h_")]
    other_cols = [c for c in ["glucose_mmol_L", "bp_sys", "bp_dia", "bp_map", "bp_pul"] if c in df.columns]
    feature_cols = hormone_cols + other_cols

    if len(feature_cols) == 0:
        raise ValueError("No features found (expected hormone columns starting with 'h_')")

    X = df[feature_cols].apply(to_num)
    y = df["y"]

    keep = y.notna()
    X = X.loc[keep].copy()
    y = y.loc[keep].astype(int).copy()
    participants = df.loc[keep, "participant"].astype(str)

    non_empty = X.notna().any(axis=1)
    X = X.loc[non_empty].copy()
    y = y.loc[non_empty].copy()
    participants = participants.loc[non_empty].copy()

    print(f"rows={len(X)}, features={len(feature_cols)}, participants={sorted(participants.unique().tolist())}")
    if len(X) == 0:
        print("[stop] No usable rows after labeling and feature filtering.")
        print("Check columns: activity_group / activity_label / description / intensity")
        return

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l2", max_iter=3000))
    ])

    print("\n[baseline lopo]")
    base_stats = run_lopo(model, X, y, participants, tag="base")

    # Added: composite features
    X_plus = add_composite_features(X, participants)
    plus_cols = list(X_plus.columns)

    # Drop rows where all features are still missing after adding composites
    non_empty_plus = X_plus.notna().any(axis=1)
    X_plus2 = X_plus.loc[non_empty_plus].copy()
    y2 = y.loc[non_empty_plus].copy()
    participants2 = participants.loc[non_empty_plus].copy()

    print("\n[baseline + composite lopo]")
    plus_stats = run_lopo(model, X_plus2, y2, participants2, tag="plus")

    print("\n[summary]")
    print(f"baseline: used={base_stats['used']}, acc={base_stats['acc_mean']:.2f}, auc={(base_stats['auc_mean'] if not np.isnan(base_stats['auc_mean']) else np.nan):.2f}")
    print(f"plus    : used={plus_stats['used']}, acc={plus_stats['acc_mean']:.2f}, auc={(plus_stats['auc_mean'] if not np.isnan(plus_stats['auc_mean']) else np.nan):.2f}")

    # Fit on all data (plus version) and plot coefficients
    model.fit(X_plus2, y2)
    plot_top_coefs(model, plus_cols, "Logistic regression coefficients (baseline + composite)")

    # Small sanity print, just so we know composites exist
    ok_symp = X_plus2["symp_index"].notna().sum() if "symp_index" in X_plus2.columns else 0
    ok_meta = X_plus2["meta_index"].notna().sum() if "meta_index" in X_plus2.columns else 0
    print(f"\n[check] symp_index non-missing: {ok_symp}/{len(X_plus2)}")
    print(f"[check] meta_index non-missing: {ok_meta}/{len(X_plus2)}")


if __name__ == "__main__":
    main()
