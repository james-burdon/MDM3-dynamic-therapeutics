# -*- coding: utf-8 -*-
"""
mcPHASES timestamp-aligned multimodal merge (no heart_rate)

Fixes:
- merge_asof sorting: per-id asof merge
- parquet engine missing: auto-fallback to CSV

Outputs:
- mcphases_aligned.parquet if pyarrow/fastparquet available
- otherwise mcphases_aligned.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# PATHS
# =========================
DATA_DIR = Path(
    r"D:\UOB\Year_3_UOB\mdm_hormone\mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0"
)

OUT_PARQUET = Path(r"D:\UOB\Year_3_UOB\mdm_hormone\mcphases_aligned.parquet")
OUT_CSV = Path(r"D:\UOB\Year_3_UOB\mdm_hormone\mcphases_aligned.csv")

ASOF_TOL = pd.Timedelta("10min")


# =========================
# Whitelists
# =========================
HRV_COLS = ["id", "day_in_study", "timestamp", "rmssd", "sdnn", "low_frequency", "high_frequency"]
GLUCOSE_COLS = ["id", "day_in_study", "timestamp", "glucose_value"]
HORMONE_COLS = ["id", "day_in_study", "phase", "study_interval", "lh", "estrogen", "pdg"]


# =========================
# Utilities
# =========================
def resolve_file(stem: str) -> Path:
    p1 = DATA_DIR / stem
    p2 = DATA_DIR / f"{stem}.csv"
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(stem)


def read(stem: str) -> pd.DataFrame:
    return pd.read_csv(resolve_file(stem))


def standardize_day(df: pd.DataFrame) -> pd.DataFrame:
    if "day_in_study" in df.columns:
        return df
    if "sleep_end_day_in_study" in df.columns:
        return df.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    if "sleep_start_day_in_study" in df.columns:
        return df.rename(columns={"sleep_start_day_in_study": "day_in_study"})
    return df


def clean_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_day(df.copy())
    df["day_in_study"] = pd.to_numeric(df["day_in_study"], errors="coerce")
    df = df.dropna(subset=["id", "day_in_study"]).copy()
    df["day_in_study"] = df["day_in_study"].astype(int)
    return df


def keep_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]


# =========================
# Hormone day-level table
# =========================
def build_hormone_day() -> pd.DataFrame:
    horm = clean_keys(read("hormones_and_selfreport"))

    for c in ["phase", "study_interval", "lh", "estrogen", "pdg"]:
        if c not in horm.columns:
            horm[c] = np.nan

    def last_nonnull(s: pd.Series):
        s = s.dropna()
        return s.iloc[-1] if len(s) else np.nan

    agg = {
        "phase": last_nonnull,
        "study_interval": last_nonnull,
        "lh": "median",
        "estrogen": "median",
        "pdg": "median",
    }

    out = horm.groupby(["id", "day_in_study"], as_index=False).agg(agg)
    return keep_cols(out, HORMONE_COLS)


# =========================
# Per-ID asof merge (stable)
# =========================
def asof_merge_per_id(base: pd.DataFrame, feat: pd.DataFrame, tol: pd.Timedelta = ASOF_TOL) -> pd.DataFrame:
    out = []
    for pid, b in base.groupby("id", sort=False):
        f = feat[feat["id"] == pid]
        if f.empty:
            out.append(b)
            continue

        b = b.sort_values("timestamp")
        f = f.sort_values("timestamp")

        merged = pd.merge_asof(
            b,
            f.drop(columns=["id", "day_in_study"], errors="ignore"),
            on="timestamp",
            tolerance=tol,
            direction="nearest",
        )
        out.append(merged)

    return pd.concat(out, ignore_index=True)


# =========================
# Save with fallback
# =========================
def save_aligned(df: pd.DataFrame) -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"[DONE] Saved parquet: {OUT_PARQUET}")
    except ImportError as e:
        print("[WARN] Parquet engine not found (pyarrow/fastparquet). Falling back to CSV.")
        df.to_csv(OUT_CSV, index=False)
        print(f"[DONE] Saved CSV: {OUT_CSV}")
    except Exception as e:
        # Any other unexpected failure -> CSV fallback too
        print(f"[WARN] Parquet save failed ({type(e).__name__}: {e}). Falling back to CSV.")
        df.to_csv(OUT_CSV, index=False)
        print(f"[DONE] Saved CSV: {OUT_CSV}")


# =========================
# Main
# =========================
def main():
    print("[INFO] building hormone day table")
    hormone_day = build_hormone_day()
    print("[OK] hormone_day:", hormone_day.shape)

    print("[INFO] loading HRV")
    hrv = clean_keys(read("heart_rate_variability_details"))
    # Warning is fine; if you want to speed up, we can set a known format later
    hrv["timestamp"] = pd.to_datetime(hrv["timestamp"], errors="coerce")
    hrv = keep_cols(hrv, HRV_COLS)
    print("[OK] hrv:", hrv.shape)

    print("[INFO] loading glucose")
    glu = clean_keys(read("glucose"))
    glu["timestamp"] = pd.to_datetime(glu["timestamp"], errors="coerce")
    if "glucose_value" not in glu.columns and "value" in glu.columns:
        glu = glu.rename(columns={"value": "glucose_value"})
    glu = keep_cols(glu, GLUCOSE_COLS)
    print("[OK] glucose:", glu.shape)

    print("[INFO] building timeline")
    timeline = pd.concat(
        [hrv[["id", "timestamp", "day_in_study"]], glu[["id", "timestamp", "day_in_study"]]],
        ignore_index=True,
    ).dropna(subset=["timestamp"])

    timeline = (
        timeline.groupby(["id", "timestamp"], as_index=False)
        .agg({"day_in_study": "first"})
        .sort_values(["id", "timestamp"])
    )
    print("[OK] timeline:", timeline.shape)

    print("[INFO] asof merge HRV")
    aligned = asof_merge_per_id(timeline, hrv, tol=ASOF_TOL)

    print("[INFO] asof merge glucose")
    aligned = asof_merge_per_id(aligned, glu, tol=ASOF_TOL)

    print("[INFO] attach hormone/day features")
    aligned = aligned.merge(hormone_day, on=["id", "day_in_study"], how="left")

    aligned = aligned.sort_values(["id", "timestamp"]).reset_index(drop=True)

    print("[INFO] saving")
    save_aligned(aligned)

    print("[DONE] final shape:", aligned.shape)
    print("[DONE] phase missing rate:", float(aligned["phase"].isna().mean()))


if __name__ == "__main__":
    main()
