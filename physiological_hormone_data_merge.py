import pandas as pd
import re

#
PHYS_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\physiological_with_activity_labels.txt"
HORMONE_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormones.xlsx"
OUT_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\edf_hormonecsv.csv"
KEEP_CATS = {2, 3, 4, 5, 11}
SKIP_SHEETS = {"CAT14"}

ACTIVE_SET = {"light_activity", "moderate_activity", "vigorous_activity", "stand"}

print("\n=== STEP 1: Load physiological data (.txt) ===")
phys = pd.read_csv(PHYS_PATH, sep=",", low_memory=False)

phys = phys.rename(columns={"user": "cat_id", "activity_label": "activity"})
required = {"cat_id", "timestamp", "activity"}
missing = required - set(phys.columns)
if missing:
    raise ValueError(f"[ERROR] Missing required columns in phys: {missing}")

phys["timestamp"] = pd.to_datetime(phys["timestamp"], errors="coerce")
phys["cat_id"] = pd.to_numeric(phys["cat_id"], errors="coerce")

print("[RAW] phys shape:", phys.shape)
print("[RAW] activity missing:", phys["activity"].isna().sum())

# clean
phys = phys.dropna(subset=["timestamp", "cat_id", "activity"]).copy()
phys["cat_id"] = phys["cat_id"].astype(int)

# keep only selected cats
phys = phys[phys["cat_id"].isin(KEEP_CATS)].copy()

# normalize  labels
phys["_act_norm"] = (
    phys["activity"].astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)

# binary active/rest
phys["is_active"] = phys["_act_norm"].isin(ACTIVE_SET).astype(int)
phys = phys.sort_values(["cat_id", "timestamp"])

print("[CLEAN] phys shape:", phys.shape)
print("[CLEAN] phys cats:", phys["cat_id"].value_counts().sort_index().to_dict())
print("[CLEAN] activity top:", phys["_act_norm"].value_counts().head(10).to_dict())
print("[TIME] phys per-cat:")
print(phys.groupby("cat_id")["timestamp"].agg(["min", "max", "count"]).sort_index())

print("[NOTE] phys columns now:", list(phys.columns))
print("\n=== STEP 2: Load hormone sheets (excluding CAT14) ===")
xls = pd.ExcelFile(HORMONE_PATH)
hormone_tables = []

for sheet in xls.sheet_names:
    sheet_u = str(sheet).upper()
    if sheet_u in SKIP_SHEETS:
        continue

    m = re.match(r"CAT(\d+)", sheet_u)
    if not m:
        continue

    cat_id = int(m.group(1))
    if cat_id not in KEEP_CATS:
        continue

    df = pd.read_excel(xls, sheet_name=sheet)

    ts_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["cat_id"] = cat_id

    raw_n = len(df)
    df = df.dropna(subset=["timestamp"])
    print(f"[HORMONE] {sheet_u}: raw={raw_n}, parsed_ts={len(df)}")

    # remove unnamed columns (excel junk)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    # hormones -> numeric
    for c in df.columns:
        if c not in {"timestamp", "cat_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    hormone_tables.append(df)

if not hormone_tables:
    raise ValueError("[ERROR] No hormone sheets loaded. Check file + sheet names.")

hormone = pd.concat(hormone_tables, ignore_index=True)
hormone = hormone.sort_values(["cat_id", "timestamp"])

hormone_cols = [c for c in hormone.columns if c not in {"timestamp", "cat_id"}]

print("[SUMMARY] hormone shape:", hormone.shape)
print("[SUMMARY] hormone cats:", hormone["cat_id"].value_counts().sort_index().to_dict())
print("[TIME] hormone per-cat:")
print(hormone.groupby("cat_id")["timestamp"].agg(["min", "max", "count"]).sort_index())

print("[NOTE] hormone columns:", list(hormone.columns))

print("\n=== STEP 3: Interval aggregation ===")
rows = []
diag = []

for cat_id, h_cat in hormone.groupby("cat_id"):
    h_cat = h_cat.sort_values("timestamp").reset_index(drop=True)
    p_cat = phys[phys["cat_id"] == cat_id].sort_values("timestamp")

    h_n = len(h_cat)
    p_n = len(p_cat)

    if p_cat.empty or h_n < 2:
        diag.append({
            "cat_id": cat_id,
            "hormone_rows": h_n,
            "phys_rows": p_n,
            "intervals_possible": max(h_n - 1, 0),
            "empty_seg": None,
            "kept": 0,
            "reason": "phys_empty_or_hormone<2"
        })
        continue

    empty_seg = 0
    kept = 0

    for i in range(h_n - 1):
        t0 = h_cat.loc[i, "timestamp"]
        t1 = h_cat.loc[i + 1, "timestamp"]

        seg = p_cat[(p_cat["timestamp"] >= t0) & (p_cat["timestamp"] < t1)]
        if seg.empty:
            empty_seg += 1
            continue

        row = {
            "cat_id": cat_id,
            "hormone_time": t0,
            "next_hormone_time": t1,
            "interval_minutes": (t1 - t0) / pd.Timedelta("1min"),
            "n_samples": int(len(seg)),
            "active_ratio": float(seg["is_active"].mean()),
            "active_mean": float(seg["is_active"].mean()),
        }

        for c in hormone_cols:
            row[c] = h_cat.loc[i, c]

        rows.append(row)
        kept += 1

    diag.append({
        "cat_id": cat_id,
        "hormone_rows": h_n,
        "phys_rows": p_n,
        "intervals_possible": h_n - 1,
        "empty_seg": empty_seg,
        "kept": kept,
        "reason": "ok"
    })

interval_df = pd.DataFrame(rows)

print("\n=== DIAGNOSTIC SUMMARY ===")
diag_df = pd.DataFrame(diag).sort_values("cat_id")
print(diag_df.to_string(index=False))

print("\n=== FINAL CHECK BEFORE SAVE ===")
print("[FINAL] interval_df shape:", interval_df.shape)
if not interval_df.empty:
    print("[FINAL] cats:", interval_df["cat_id"].value_counts().sort_index().to_dict())
    print("[FINAL] active_ratio summary:", interval_df["active_ratio"].describe())
    print("[FINAL] interval_minutes summary:", interval_df["interval_minutes"].describe())
    print("\n[FINAL] head:")
    print(interval_df.head(5))

#Save
interval_df.to_csv(OUT_PATH, index=False)
print("\n[SAVED]", OUT_PATH)
print("[DONE]")


import pandas as pd

p = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\edf_hormonecsv.csv"
df = pd.read_csv(p)

print(df.shape)
print(df["cat_id"].value_counts())
print(((pd.to_datetime(df["next_hormone_time"]) - pd.to_datetime(df["hormone_time"]))
       .dt.total_seconds()/60).value_counts())
print("dup cat+time:", df.duplicated(["cat_id","hormone_time"]).sum())



# tuned version
import pandas as pd
import re

# PATHS
PHYS_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\physiological_with_activity_labels.txt"
HORMONE_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\hormones.xlsx"

OUT_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\edf_hormone_tuned.csv"

# keep cats
KEEP_CATS = {2, 3, 4, 5, 11}
SKIP_SHEETS = {"CAT14"}

# activity binarization
ACTIVE_SET = {"light_activity", "moderate_activity", "vigorous_activity", "stand"}

# tuning knobs
WINDOW_MINUTES = 60
MIN_SAMPLES = 60

# optional
EXTRA_PHYS_COLS = ["HRV", "ECG"]

# 1) Load phys (.txt)
print("\n=== STEP 1: Load physiological data (.txt) ===")
phys = pd.read_csv(PHYS_PATH, sep=",", low_memory=False)

phys = phys.rename(columns={"user": "cat_id", "activity_label": "activity"})
required = {"cat_id", "timestamp", "activity"}
missing = required - set(phys.columns)
if missing:
    raise ValueError(f"[ERROR] Missing required columns in phys: {missing}. Found: {phys.columns.tolist()}")

phys["timestamp"] = pd.to_datetime(phys["timestamp"], errors="coerce")
phys["cat_id"] = pd.to_numeric(phys["cat_id"], errors="coerce")

print("[RAW] phys shape:", phys.shape)
print("[RAW] activity missing:", phys["activity"].isna().sum())

phys = phys.dropna(subset=["timestamp", "cat_id", "activity"]).copy()
phys["cat_id"] = phys["cat_id"].astype(int)

phys = phys[phys["cat_id"].isin(KEEP_CATS)].copy()

phys["_act_norm"] = (
    phys["activity"].astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"\s+", "_", regex=True)
)
phys["is_active"] = phys["_act_norm"].isin(ACTIVE_SET).astype(int)

phys = phys.sort_values(["cat_id", "timestamp"])

print("[CLEAN] phys shape:", phys.shape)
print("[CLEAN] phys cats:", phys["cat_id"].value_counts().sort_index().to_dict())
print("[CLEAN] activity top:", phys["_act_norm"].value_counts().head(10).to_dict())
print("[TIME] phys per-cat:")
print(phys.groupby("cat_id")["timestamp"].agg(["min", "max", "count"]).sort_index())

print("[NOTE] phys columns now:", list(phys.columns))

print("\n=== STEP 2: Load hormone sheets (excluding CAT14) ===")
xls = pd.ExcelFile(HORMONE_PATH)
hormone_tables = []

for sheet in xls.sheet_names:
    sheet_u = str(sheet).upper()
    if sheet_u in SKIP_SHEETS:
        print(f"[SKIP] {sheet_u}")
        continue

    m = re.match(r"CAT(\d+)", sheet_u)
    if not m:
        continue

    cat_id = int(m.group(1))
    if cat_id not in KEEP_CATS:
        continue

    df = pd.read_excel(xls, sheet_name=sheet)

    ts_col = "Unnamed: 0" if "Unnamed: 0" in df.columns else df.columns[0]
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["cat_id"] = cat_id

    raw_n = len(df)
    df = df.dropna(subset=["timestamp"])
    print(f"[HORMONE] {sheet_u}: raw={raw_n}, parsed_ts={len(df)}")

    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")], errors="ignore")

    for c in df.columns:
        if c not in {"timestamp", "cat_id"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    hormone_tables.append(df)

if not hormone_tables:
    raise ValueError("[ERROR] No hormone sheets loaded. Check file/sheet names or KEEP_CATS.")

hormone = pd.concat(hormone_tables, ignore_index=True).sort_values(["cat_id", "timestamp"])
hormone_cols = [c for c in hormone.columns if c not in {"timestamp", "cat_id"}]

print("[SUMMARY] hormone shape:", hormone.shape)
print("[SUMMARY] hormone cats:", hormone["cat_id"].value_counts().sort_index().to_dict())
print("[TIME] hormone per-cat:")
print(hormone.groupby("cat_id")["timestamp"].agg(["min", "max", "count"]).sort_index())

print("[NOTE] hormone columns:", list(hormone.columns))

# 3) Fixed-window aggregation
print("\n=== STEP 3: Fixed-window aggregation ===")
rows = []
diag = []
win = pd.Timedelta(minutes=WINDOW_MINUTES)

for cat_id, h_cat in hormone.groupby("cat_id"):
    h_cat = h_cat.sort_values("timestamp").reset_index(drop=True)
    p_cat = phys[phys["cat_id"] == cat_id].sort_values("timestamp")

    if p_cat.empty:
        diag.append({
            "cat_id": cat_id,
            "hormone_rows": len(h_cat),
            "phys_rows": 0,
            "kept": 0,
            "dropped_too_few_samples": None,
            "reason": "phys_empty"
        })
        continue

    kept = 0
    too_few = 0

    for i in range(len(h_cat)):
        t0 = h_cat.loc[i, "timestamp"]
        t1 = t0 + win

        seg = p_cat[(p_cat["timestamp"] >= t0) & (p_cat["timestamp"] < t1)]
        if len(seg) < MIN_SAMPLES:
            too_few += 1
            continue

        row = {
            "cat_id": cat_id,
            "hormone_time": t0,
            "window_end": t1,
            "window_minutes": WINDOW_MINUTES,
            "n_samples": int(len(seg)),
            "active_ratio": float(seg["is_active"].mean()),
            "active_mean": float(seg["is_active"].mean()),
        }

        for extra in EXTRA_PHYS_COLS:
            if extra in seg.columns:
                row[f"{extra}_mean"] = pd.to_numeric(seg[extra], errors="coerce").mean()

        for c in hormone_cols:
            row[c] = h_cat.loc[i, c]

        rows.append(row)
        kept += 1

    diag.append({
        "cat_id": cat_id,
        "hormone_rows": len(h_cat),
        "phys_rows": int(len(p_cat)),
        "kept": kept,
        "dropped_too_few_samples": too_few,
        "reason": "ok"
    })

out = pd.DataFrame(rows)

print("\n=== DIAGNOSTIC SUMMARY ===")
print(pd.DataFrame(diag).sort_values("cat_id").to_string(index=False))

print("\n=== FINAL CHECK BEFORE SAVE ===")
print("[FINAL] out shape:", out.shape)
if not out.empty:
    print("[FINAL] cats:", out["cat_id"].value_counts().sort_index().to_dict())
    if "active_ratio" in out.columns:
        print("[FINAL] active_ratio summary:\n", out["active_ratio"].describe())
    print("\n[FINAL] head:")
    print(out.head(5))

# Save
out.to_csv(OUT_PATH, index=False)
print("\n[SAVED]", OUT_PATH)
print("[DONE]")
