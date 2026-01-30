# -*- coding: utf-8 -*-
"""
Activity–hormone alignment (DATE + time_of_day) with intensity included.

- Activity diary provides: sampling_id, date, time_of_day, minutes, activity_label, intensity
- Hormone workbook: multi-sheet CATxx, each sheet has timestamp + numeric hormones
- For each activity interval:
    * aggregate hormones during [start_time, end_time]
    * optionally aggregate post window [end_time+20, end_time+40]
- CAT02 is skipped (kept NaN) as requested

Output: one CSV
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime

# =========================
# PATHS
# =========================
ACTIVITY_WB = r"D:\UOB\Year_3_UOB\mdm_hormone\cat234511.xlsx"
HORMONE_WB  = r"D:\UOB\Year_3_UOB\mdm_hormone\OneDrive_2026-01-27\Data files\Hormones\MD data for Wellcome 2026 January.xlsx"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_CSV = rf"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_{ts}.csv"

# =========================
# SETTINGS
# =========================
SKIP_PARTICIPANTS = {"CAT02"}
POST_LAG_WINDOW_MIN = (20, 40)  # set None to disable

ACTIVITY_SHEET = "Sheet1"
COL_SAMPLING_ID = "sampling_id"
COL_DATE        = "date"
COL_TIME        = "time_of_day"
COL_MINUTES     = "minutes"
COL_LABEL       = "activity_label"
COL_INTENSITY   = "intensity"   # <-- added

HORM_TIME_COL_CANDIDATES = ["Unnamed: 0", "timestamp", "time", "datetime", "date_time"]

# =========================
# HELPERS
# =========================
def sampling_id_to_cat(x):
    if pd.isna(x):
        return np.nan
    return f"CAT{int(float(x)):02d}"

def parse_activity_start(date_series, time_series):
    """
    Parse activity datetime from date + time_of_day.
    date: day-first (UK style) -> parsed with dayfirst=True
    time_of_day: HH:MM (or HH:MM:SS)
    """
    d = pd.to_datetime(date_series, errors="coerce", dayfirst=True)
    t = time_series.astype(str).str.strip()

    dt_str = d.dt.strftime("%Y-%m-%d") + " " + t

    out = pd.to_datetime(dt_str, format="%Y-%m-%d %H:%M", errors="coerce")
    mask = out.isna() & dt_str.notna()
    if mask.any():
        out.loc[mask] = pd.to_datetime(dt_str.loc[mask], errors="coerce")
    return out

def detect_hormone_time_col(df):
    for c in HORM_TIME_COL_CANDIDATES:
        if c in df.columns:
            return c
    best_col, best_rate = None, 0.0
    for c in df.columns:
        s = pd.to_datetime(df[c], errors="coerce")
        rate = s.notna().mean()
        if rate > best_rate and rate >= 0.6:
            best_col, best_rate = c, rate
    return best_col

def parse_hormone_timestamp(series):
    # DO NOT force dayfirst for hormone timestamps
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    return pd.to_datetime(series, errors="coerce")

def detect_numeric_cols(df, exclude):
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() > 0:
            cols.append(c)
    return cols

def read_hormone_workbook(path):
    xls = pd.ExcelFile(path)
    rows = []
    hormone_cols = set()

    for sh in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sh)
        participant = sh.strip()

        time_col = detect_hormone_time_col(df)
        if time_col is None:
            print(f"[WARN] Skip {participant}: cannot detect timestamp column")
            continue

        df = df.copy()
        df["participant"] = participant
        df["timestamp"] = parse_hormone_timestamp(df[time_col])
        df = df.dropna(subset=["timestamp"]).copy()

        exclude = {"participant", "timestamp", time_col}
        numeric = detect_numeric_cols(df, exclude)

        hormone_cols.update(numeric)
        rows.append(df[["participant", "timestamp"] + numeric])

    horm = pd.concat(rows, ignore_index=True)
    hormone_cols = sorted(hormone_cols)

    for c in hormone_cols:
        if c not in horm.columns:
            horm[c] = np.nan

    horm = horm[["participant", "timestamp"] + hormone_cols].copy()
    horm = horm.sort_values(["participant", "timestamp"], kind="stable").reset_index(drop=True)
    return horm, hormone_cols

def aggregate_window(horm, participant, t0, t1, hormone_cols, prefix=""):
    mask = (
        (horm["participant"] == participant) &
        (horm["timestamp"] >= t0) &
        (horm["timestamp"] <= t1)
    )
    sub = horm.loc[mask, hormone_cols]

    out = {}
    for c in hormone_cols:
        x = pd.to_numeric(sub[c], errors="coerce")
        out[f"{prefix}{c}_n"] = int(x.notna().sum())
        out[f"{prefix}{c}_mean"] = x.mean()
    return out

def print_time_sanity(act, horm):
    print("\n[SANITY] Time ranges by participant")
    for p in sorted(act["participant"].unique()):
        a0 = act.loc[act["participant"] == p, "start_time"].min()
        a1 = act.loc[act["participant"] == p, "end_time"].max()

        h_sub = horm[horm["participant"] == p]
        h0 = h_sub["timestamp"].min() if len(h_sub) else pd.NaT
        h1 = h_sub["timestamp"].max() if len(h_sub) else pd.NaT

        print(f"  {p}: activity {a0} → {a1} | hormone {h0} → {h1}")

# =========================
# MAIN
# =========================
def main():
    # ---- Load activity
    act_raw = pd.read_excel(ACTIVITY_WB, sheet_name=ACTIVITY_SHEET)
    act = act_raw.copy()

    # required columns check
    required = [COL_SAMPLING_ID, COL_DATE, COL_TIME, COL_MINUTES, COL_LABEL, COL_INTENSITY]
    missing = [c for c in required if c not in act.columns]
    if missing:
        raise KeyError(f"Missing activity columns: {missing}. Found: {act.columns.tolist()}")

    act["participant"] = act[COL_SAMPLING_ID].apply(sampling_id_to_cat)
    act["activity_label"] = act[COL_LABEL].astype(str)
    act["intensity"] = act[COL_INTENSITY]  # keep as-is, we’ll numeric-cast later if needed

    act["start_time"] = parse_activity_start(act[COL_DATE], act[COL_TIME])
    act["end_time"] = act["start_time"] + pd.to_timedelta(
        pd.to_numeric(act[COL_MINUTES], errors="coerce"), unit="min"
    )

    act = act.dropna(subset=["participant", "start_time", "end_time"]).copy()
    act = act.sort_values(["participant", "start_time"], kind="stable").reset_index(drop=True)

    print("[INFO] Activity intervals:", len(act))
    print("[INFO] Activity participants:", sorted(act["participant"].unique().tolist()))

    # ---- Load hormones
    horm, hormone_cols = read_hormone_workbook(HORMONE_WB)
    print("[INFO] Hormone rows:", len(horm))
    print("[INFO] Hormone participants:", sorted(horm["participant"].unique().tolist()))
    print("[INFO] Hormone numeric cols:", len(hormone_cols))

    # ---- Sanity check
    print_time_sanity(act, horm)

    # ---- Aggregate
    rows = []
    for _, a in act.iterrows():
        p = a["participant"]
        base = {
            "participant": p,
            "activity_label": a["activity_label"],
            "intensity": a["intensity"],   # <-- included in output
            "start_time": a["start_time"],
            "end_time": a["end_time"],
            "minutes": float(a[COL_MINUTES]) if pd.notna(a[COL_MINUTES]) else np.nan,
        }

        if p in SKIP_PARTICIPANTS:
            for c in hormone_cols:
                base[f"{c}_n"] = 0
                base[f"{c}_mean"] = np.nan
                if POST_LAG_WINDOW_MIN is not None:
                    lag0, lag1 = POST_LAG_WINDOW_MIN
                    base[f"post_{lag0}_{lag1}__{c}_n"] = 0
                    base[f"post_{lag0}_{lag1}__{c}_mean"] = np.nan
            rows.append(base)
            continue

        # during
        base.update(aggregate_window(horm, p, a["start_time"], a["end_time"], hormone_cols, prefix=""))

        # post
        if POST_LAG_WINDOW_MIN is not None:
            lag0, lag1 = POST_LAG_WINDOW_MIN
            p0 = a["end_time"] + pd.Timedelta(minutes=lag0)
            p1 = a["end_time"] + pd.Timedelta(minutes=lag1)
            base.update(aggregate_window(horm, p, p0, p1, hormone_cols, prefix=f"post_{lag0}_{lag1}__"))

        rows.append(base)

    out = pd.DataFrame(rows).sort_values(["participant", "start_time"], kind="stable").reset_index(drop=True)

    # quick coverage (excluding skipped)
    mask_ok = ~out["participant"].isin(SKIP_PARTICIPANTS)
    n_cols = [c for c in out.columns if c.endswith("_n") and not c.startswith("post_")]
    if mask_ok.any() and n_cols:
        covered = (out.loc[mask_ok, n_cols].sum(axis=1) > 0).mean()
        print(f"\n[INFO] Coverage DURING (excluding skipped): {covered:.1%}")

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print("\n[OK] Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()



# import re
# import numpy as np
# import pandas as pd
# from datetime import datetime
#
# ACTIVITY_WB = r"D:\UOB\Year_3_UOB\mdm_hormone\cat234511.xlsx"
# HORMONE_WB  = r"D:\UOB\Year_3_UOB\mdm_hormone\OneDrive_2026-01-27\Data files\Hormones\MD data for Wellcome 2026 January.xlsx"
#
# ts = datetime.now().strftime("%Y%m%d_%H%M%S")
# OUTPUT_CSV = rf"D:\UOB\Year_3_UOB\mdm_hormone\activity_hormone_aligned_{ts}.csv"
#
# SKIP_PARTICIPANTS = {"CAT02"}
# POST_LAG_WINDOW_MIN = (20, 40)
#
# # Column names in activity sheet
# ACTIVITY_SHEET = "Sheet1"
# COL_SAMPLING_ID = "sampling_id"
# COL_DATE        = "date"
# COL_TIME        = "time_of_day"
# COL_MINUTES     = "minutes"
# COL_LABEL       = "activity_label"
#
# HORM_TIME_COL_CANDIDATES = ["Unnamed: 0", "timestamp", "time", "datetime", "date_time"]
#
# def to_dt_dayfirst(x):
#     """Parse datetime assuming UK-style day-first format."""
#     return pd.to_datetime(x, errors="coerce", dayfirst=True)
#
# def sampling_id_to_cat(x):
#     """Convert numeric sampling_id (e.g. 2) to CAT02."""
#     if pd.isna(x):
#         return np.nan
#     return f"CAT{int(float(x)):02d}"
#
# def detect_hormone_time_col(df):
#     """Detect the most likely datetime column in a hormone sheet."""
#     for c in HORM_TIME_COL_CANDIDATES:
#         if c in df.columns:
#             return c
#     best_col, best_rate = None, 0.0
#     for c in df.columns:
#         s = to_dt_dayfirst(df[c])
#         rate = s.notna().mean()
#         if rate > best_rate and rate >= 0.6:
#             best_col, best_rate = c, rate
#     return best_col
#
# def detect_numeric_cols(df, exclude):
#     """Detect numeric hormone columns."""
#     cols = []
#     for c in df.columns:
#         if c in exclude:
#             continue
#         s = pd.to_numeric(df[c], errors="coerce")
#         if s.notna().sum() > 0:
#             cols.append(c)
#     return cols
#
# def read_hormone_workbook(path):
#     """Read and standardise all hormone sheets."""
#     xls = pd.ExcelFile(path)
#     rows = []
#     hormone_cols = set()
#
#     for sh in xls.sheet_names:
#         df = pd.read_excel(path, sheet_name=sh)
#         participant = sh.strip()
#
#         time_col = detect_hormone_time_col(df)
#         if time_col is None:
#             continue
#
#         df["participant"] = participant
#         df["timestamp"] = to_dt_dayfirst(df[time_col])
#         df = df.dropna(subset=["timestamp"])
#
#         exclude = {"participant", "timestamp", time_col}
#         numeric = detect_numeric_cols(df, exclude)
#
#         hormone_cols.update(numeric)
#         rows.append(df[["participant", "timestamp"] + numeric])
#
#     horm = pd.concat(rows, ignore_index=True)
#     hormone_cols = sorted(hormone_cols)
#
#     for c in hormone_cols:
#         if c not in horm.columns:
#             horm[c] = np.nan
#
#     horm = horm[["participant", "timestamp"] + hormone_cols]
#     horm = horm.sort_values(["participant", "timestamp"]).reset_index(drop=True)
#     return horm, hormone_cols
#
# def aggregate_window(horm, participant, t0, t1, hormone_cols, prefix=""):
#     """Aggregate hormone values within a time window."""
#     mask = (
#         (horm["participant"] == participant) &
#         (horm["timestamp"] >= t0) &
#         (horm["timestamp"] <= t1)
#     )
#     sub = horm.loc[mask, hormone_cols]
#
#     out = {}
#     for c in hormone_cols:
#         x = pd.to_numeric(sub[c], errors="coerce")
#         out[f"{prefix}{c}_n"] = int(x.notna().sum())
#         out[f"{prefix}{c}_mean"] = x.mean()
#     return out
#
#
# def main():
#     # ---- Load activity diary
#     act_raw = pd.read_excel(ACTIVITY_WB, sheet_name=ACTIVITY_SHEET)
#
#     act = act_raw.copy()
#     act["participant"] = act[COL_SAMPLING_ID].apply(sampling_id_to_cat)
#     act["activity_label"] = act[COL_LABEL].astype(str)
#
#     date_str = to_dt_dayfirst(act[COL_DATE]).dt.date.astype(str)
#     time_str = act[COL_TIME].astype(str).str.strip()
#
#     act["start_time"] = to_dt_dayfirst(date_str + " " + time_str)
#     act["end_time"] = act["start_time"] + pd.to_timedelta(
#         pd.to_numeric(act[COL_MINUTES], errors="coerce"), unit="min"
#     )
#
#     act = act.dropna(subset=["participant", "start_time", "end_time"])
#     act = act.sort_values(["participant", "start_time"]).reset_index(drop=True)
#
#
#     horm, hormone_cols = read_hormone_workbook(HORMONE_WB)
#
#     rows = []
#     for _, a in act.iterrows():
#         p = a["participant"]
#         base = {
#             "participant": p,
#             "activity_label": a["activity_label"],
#             "start_time": a["start_time"],
#             "end_time": a["end_time"],
#         }
#
#         if p in SKIP_PARTICIPANTS:
#             for c in hormone_cols:
#                 base[f"{c}_n"] = 0
#                 base[f"{c}_mean"] = np.nan
#                 if POST_LAG_WINDOW_MIN is not None:
#                     lag0, lag1 = POST_LAG_WINDOW_MIN
#                     base[f"post_{lag0}_{lag1}__{c}_n"] = 0
#                     base[f"post_{lag0}_{lag1}__{c}_mean"] = np.nan
#             rows.append(base)
#             continue
#
#         base.update(
#             aggregate_window(horm, p, a["start_time"], a["end_time"], hormone_cols)
#         )
#
#         if POST_LAG_WINDOW_MIN is not None:
#             lag0, lag1 = POST_LAG_WINDOW_MIN
#             p0 = a["end_time"] + pd.Timedelta(minutes=lag0)
#             p1 = a["end_time"] + pd.Timedelta(minutes=lag1)
#             base.update(
#                 aggregate_window(horm, p, p0, p1, hormone_cols,
#                                  prefix=f"post_{lag0}_{lag1}__")
#             )
#
#         rows.append(base)
#
#     out = pd.DataFrame(rows).sort_values(["participant", "start_time"])
#
#     #Save CSV
#     out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
#     print("[OK] Saved:", OUTPUT_CSV)
#
# if __name__ == "__main__":
#     main()
