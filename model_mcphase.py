# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
)

PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\mcphases_aligned.csv"
RANDOM_STATE = 42


def safe_range(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.max() - s.min())


def slope_vs_time(ts: pd.Series, y: pd.Series, min_points: int = 10) -> float:
    ts = pd.to_datetime(ts, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")

    m = ts.notna() & y.notna()
    ts = ts[m]
    y = y[m]

    if len(y) < min_points:
        return np.nan

    t_hours = (ts - ts.min()).dt.total_seconds().to_numpy() / 3600.0
    coef = np.polyfit(t_hours, y.to_numpy(), 1)
    return float(coef[0])


def participant_split(df0: pd.DataFrame, id_col="id", test_size=0.2, random_state=42):
    ids = df0[id_col].dropna().unique()
    tr, te = train_test_split(ids, test_size=test_size, random_state=random_state)
    return df0[id_col].isin(tr), df0[id_col].isin(te)


def fit_lr_and_eval(df_feat: pd.DataFrame, feature_cols, label_col="y", title="model"):
    train_mask, test_mask = participant_split(
        df_feat, id_col="id", test_size=0.2, random_state=RANDOM_STATE
    )

    X_train = df_feat.loc[train_mask, feature_cols].copy()
    X_test = df_feat.loc[test_mask, feature_cols].copy()
    y_train = df_feat.loc[train_mask, label_col].astype(int).copy()
    y_test = df_feat.loc[test_mask, label_col].astype(int).copy()

    all_nan = [c for c in feature_cols if X_train[c].notna().sum() == 0]
    if all_nan:
        print(f"[WARN] {title}: dropping all-NaN features:", all_nan)

    feats = [c for c in feature_cols if c not in all_nan]
    X_train = X_train[feats].replace([np.inf, -np.inf], np.nan)
    X_test = X_test[feats].replace([np.inf, -np.inf], np.nan)

    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    penalty="l2",
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)

    print(f"\n=== {title} ===")
    print("AUC:", auc)
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {title} (AUC={auc:.3f})")
    plt.tight_layout()
    plt.show()

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]

    feats_arr = np.array(feats)[order]
    vals = coefs[order]

    plt.figure(figsize=(7, 4))
    plt.bar(range(len(feats_arr)), vals)
    plt.xticks(range(len(feats_arr)), feats_arr, rotation=45, ha="right")
    plt.ylabel("Coefficient")
    plt.title(f"Logistic coefficients - {title}")
    plt.tight_layout()
    plt.show()

    return auc


def add_rolling_means(
    df_in: pd.DataFrame,
    col: str,
    windows=("6h", "12h", "24h"),
    min_periods=5,
):
    out = df_in.copy()
    for w in windows:
        new_col = f"{col}_roll_{w}"
        out[new_col] = np.nan

        for pid, g in out.groupby("id", sort=False):
            g_sorted = g.sort_values("timestamp")
            r = (
                g_sorted[["timestamp", col]]
                .rolling(window=w, on="timestamp", min_periods=min_periods)[col]
                .mean()
                .to_numpy()
            )
            out.loc[g_sorted.index, new_col] = r

    return out


df = pd.read_csv(DATA_PATH)

df["y"] = (df["phase"] == "Menstrual").astype(int)
print("Positive rate (Menstrual):", df["y"].mean())

df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

CANDIDATE_FEATURES = [
    "rmssd",
    "sdnn",
    "low_frequency",
    "high_frequency",
    "glucose_value",
    "hour_sin",
    "hour_cos",
]

for c in CANDIDATE_FEATURES:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df[CANDIDATE_FEATURES] = df[CANDIDATE_FEATURES].replace([np.inf, -np.inf], np.nan)

X_all = df[CANDIDATE_FEATURES]
y_all = df["y"]

unique_ids = df["id"].dropna().unique()
train_ids, test_ids = train_test_split(
    unique_ids, test_size=0.2, random_state=RANDOM_STATE
)

train_mask = df["id"].isin(train_ids)
test_mask = df["id"].isin(test_ids)

X_train_raw = X_all[train_mask].copy()
X_test_raw = X_all[test_mask].copy()
y_train = y_all[train_mask].copy()
y_test = y_all[test_mask].copy()

print("Train samples:", X_train_raw.shape[0])
print("Test samples:", X_test_raw.shape[0])

all_nan_cols = [c for c in X_train_raw.columns if X_train_raw[c].notna().sum() == 0]
if all_nan_cols:
    print("[WARN] Dropping all-NaN features (no observed values in TRAIN):", all_nan_cols)

FEATURES = [c for c in CANDIDATE_FEATURES if c not in all_nan_cols]
X_train = X_train_raw[FEATURES]
X_test = X_test_raw[FEATURES]

pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                solver="lbfgs",
                penalty="l2",
                max_iter=3000,
                class_weight="balanced",
            ),
        ),
    ]
)

pipe.fit(X_train, y_train)

y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
acc = accuracy_score(y_test, y_pred)

print("\n=== Performance ===")
print("AUC:", auc)
print("Accuracy:", acc)
print("\nClassification report:")
print(classification_report(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

if auc < 0.5:
    print("\n[SANITY] AUC < 0.5. If you flip probabilities, AUC would be:", 1 - auc)

clf = pipe.named_steps["clf"]
coef = clf.coef_[0]
coef_df = pd.DataFrame({"feature": FEATURES, "coef": coef})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).drop(columns=["abs_coef"])

print("\n=== Model coefficients (sorted by |coef|) ===")
print(coef_df)

print("\n=== Missingness in TRAIN features (before impute) ===")
print(X_train.isna().mean().sort_values(ascending=False))

plt.figure()
df["phase"].value_counts().reindex(PHASES).plot(kind="bar")
plt.ylabel("Number of samples")
plt.title("Sample count by menstrual phase")
plt.tight_layout()
plt.show()

rmssd_data = [df.loc[df["phase"] == p, "rmssd"].dropna() for p in PHASES]
plt.figure()
plt.boxplot(rmssd_data, labels=PHASES, showfliers=False)
plt.ylabel("RMSSD")
plt.title("HRV (RMSSD) distribution across phases")
plt.tight_layout()
plt.show()

glucose_data = [df.loc[df["phase"] == p, "glucose_value"].dropna() for p in PHASES]
plt.figure()
plt.boxplot(glucose_data, labels=PHASES, showfliers=False)
plt.ylabel("Glucose value")
plt.title("Glucose distribution across phases")
plt.tight_layout()
plt.show()


df2 = df[df["phase"].isin(PHASES)].copy()
df2["y"] = (df2["phase"] == "Menstrual").astype(int)

daily = (
    df2.groupby(["id", "day_in_study", "phase"], as_index=False)
    .agg(
        rmssd_mean=("rmssd", "mean"),
        rmssd_std=("rmssd", "std"),
        rmssd_range=("rmssd", safe_range),
        glucose_mean=("glucose_value", "mean"),
        glucose_std=("glucose_value", "std"),
        glucose_range=("glucose_value", safe_range),
        n_samples=("timestamp", "count"),
        ts_min=("timestamp", "min"),
        ts_max=("timestamp", "max"),
    )
)

trend_rows = []
for (pid, day), g in df2.groupby(["id", "day_in_study"], sort=False):
    trend_rows.append(
        {
            "id": pid,
            "day_in_study": day,
            "rmssd_trend": slope_vs_time(g["timestamp"], g["rmssd"]),
            "glucose_trend": slope_vs_time(g["timestamp"], g["glucose_value"]),
        }
    )
trend_df = pd.DataFrame(trend_rows)

daily = daily.merge(trend_df, on=["id", "day_in_study"], how="left")
daily["y"] = (daily["phase"] == "Menstrual").astype(int)

print("\n[INFO] Day-level dataset shape:", daily.shape)
print("[INFO] Day-level samples per phase:\n", daily["phase"].value_counts())

for feat in [
    "rmssd_mean",
    "glucose_mean",
    "rmssd_range",
    "glucose_range",
    "rmssd_trend",
    "glucose_trend",
]:
    data = [daily.loc[daily["phase"] == p, feat].dropna() for p in PHASES]
    plt.figure()
    plt.boxplot(data, labels=PHASES, showfliers=False)
    plt.ylabel(feat)
    plt.title(f"Day-level {feat} across phases")
    plt.tight_layout()
    plt.show()

DAY_FEATURES = [
    "rmssd_mean",
    "rmssd_std",
    "rmssd_range",
    "rmssd_trend",
    "glucose_mean",
    "glucose_std",
    "glucose_range",
    "glucose_trend",
]
fit_lr_and_eval(daily, DAY_FEATURES, label_col="y", title="Day-level LR (Menstrual vs rest)")


df3 = df2.sort_values(["id", "timestamp"]).copy()
df3 = add_rolling_means(df3, "rmssd", windows=("6h", "12h", "24h"))
df3 = add_rolling_means(df3, "glucose_value", windows=("6h", "12h", "24h"))
df3["y"] = (df3["phase"] == "Menstrual").astype(int)

print("\n[INFO] Rolling-feature dataset shape:", df3.shape)

for w in ["6h", "12h", "24h"]:
    for base in ["rmssd", "glucose_value"]:
        feat = f"{base}_roll_{w}"
        data = [df3.loc[df3["phase"] == p, feat].dropna() for p in PHASES]
        plt.figure()
        plt.boxplot(data, labels=PHASES, showfliers=False)
        plt.ylabel(feat)
        plt.title(f"{feat} across phases")
        plt.tight_layout()
        plt.show()

ROLL_FEATURES = [
    "rmssd_roll_6h",
    "rmssd_roll_12h",
    "rmssd_roll_24h",
    "glucose_value_roll_6h",
    "glucose_value_roll_12h",
    "glucose_value_roll_24h",
]
fit_lr_and_eval(df3, ROLL_FEATURES, label_col="y", title="Rolling-mean LR (Menstrual vs rest)")
