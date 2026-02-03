import csv
import re
import time
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def norm_participant(val):
    if pd.isna(val):
        return None
    s = str(val).strip().upper()
    if s == "":
        return None
    m = re.search(r"CAT\s*0*(\d+)", s)
    if m:
        return f"CAT{int(m.group(1)):02d}"
    m = re.fullmatch(r"0*(\d+)", s)
    if m:
        return f"CAT{int(m.group(1)):02d}"
    m = re.search(r"(\d+)", s)
    if m and len(m.group(1)) <= 2:
        return f"CAT{int(m.group(1)):02d}"
    return s


def excel_serial_to_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_object_dtype(s):
        parsed = pd.to_datetime(s, errors="coerce")
        if parsed.notna().sum() > 0:
            return parsed
    s_num = pd.to_numeric(s, errors="coerce")
    return pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")


def parse_time_of_day(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float, np.integer, np.floating)):
        return pd.to_timedelta(float(x), unit="D")
    s = str(x).strip()
    if s == "" or s == "0":
        return pd.NaT
    m = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?", s)
    if m:
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3)) if m.group(3) else 0
        return pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m") + pd.to_timedelta(ss, unit="s")
    try:
        return pd.to_timedelta(s)
    except Exception:
        return pd.NaT


def normalize_hormone_col(name):
    if name is None:
        return None
    s = str(name).strip()
    if s == "" or s.lower().startswith("unnamed"):
        return None
    s = s.replace("\n", " ").strip()
    s = s.replace(" ", "_").replace("-", "_").replace("/", "_")
    s = "_".join([p for p in s.split("_") if p])
    return s.lower()


def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        base = c if c else "col"
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def file_fingerprint(p: Path):
    p = Path(p).resolve()
    st = p.stat()
    md5 = hashlib.md5(p.read_bytes()).hexdigest()[:12]
    print("[FILE]", str(p))
    print("  size_bytes:", st.st_size)
    print("  md5_12:", md5)
    return md5


def must_exist(p: str, hint_name: str) -> Path:
    p0 = Path(p)
    if p0.exists():
        return p0
    here = Path(__file__).resolve().parent
    hits = list(here.rglob(p0.name))
    if hits:
        print(f"[WARN] {hint_name} not found at: {p0}")
        print(f"[WARN] Using found file: {hits[0]}")
        return hits[0]
    raise FileNotFoundError(f"{hint_name} not found. Tried: {p0} ; searched under: {here}")


def safe_out_path(out_path):
    p = Path(out_path)
    if not p.exists():
        return p
    ts = time.strftime("%Y%m%d_%H%M%S")
    return p.with_name(f"{p.stem}_{ts}{p.suffix}")


def print_summary(name, df):
    if df is None or df.empty:
        print(f"[{name}] rows=0")
        return
    parts = sorted(df["participant"].dropna().unique().tolist()) if "participant" in df.columns else []
    print(f"[{name}] rows={len(df)} participants={parts}")


def read_activity(activity_xlsx: Path) -> pd.DataFrame:
    df = pd.read_excel(activity_xlsx, engine="openpyxl")
    for col in ["sampling_id", "date", "time_of_day"]:
        if col not in df.columns:
            raise ValueError(f"Activity file missing column '{col}'")

    df["participant"] = df["sampling_id"].apply(norm_participant)
    df["activity_date"] = excel_serial_to_datetime(df["date"]).dt.date
    df["activity_time_td"] = df["time_of_day"].apply(parse_time_of_day)
    df["activity_timestamp"] = pd.to_datetime(df["activity_date"]) + df["activity_time_td"]

    df = df.dropna(subset=["activity_timestamp", "participant"]).copy()
    df = df.sort_values(["participant", "activity_timestamp"]).reset_index(drop=True)
    return df


def read_hormone(hormone_xlsx: Path) -> pd.DataFrame:
    all_rows = []
    xls = pd.ExcelFile(hormone_xlsx, engine="openpyxl")
    for sheet in xls.sheet_names:
        df = pd.read_excel(hormone_xlsx, sheet_name=sheet, engine="openpyxl")
        if df.empty:
            continue
        cols = list(df.columns)
        if len(cols) < 3:
            continue

        c1, c2, c3 = cols[0], cols[1], cols[2]
        df = df.rename(columns={c1: "sample_serial", c2: "participant_raw", c3: "sample_id"})

        df["sample_datetime"] = excel_serial_to_datetime(df["sample_serial"])
        df["participant"] = df["participant_raw"].fillna(sheet).apply(norm_participant)
        df = df.dropna(subset=["sample_datetime", "participant"]).copy()

        base_cols = ["participant", "sample_datetime", "sample_id"]
        analyte_cols = [c for c in df.columns if c not in base_cols and c not in ["sample_serial", "participant_raw"]]

        rename_map = {}
        keep = []
        for c in analyte_cols:
            nc = normalize_hormone_col(c)
            if nc is None:
                continue
            rename_map[c] = f"h_{nc}"
            keep.append(c)

        df = df[base_cols + keep].rename(columns=rename_map)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df.columns = make_unique(df.columns)
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["participant", "sample_datetime"]).reset_index(drop=True)
    return out


def read_bp_single(bp_csv: Path) -> pd.DataFrame:
    lines = bp_csv.read_text(encoding="utf-8", errors="ignore").splitlines()

    patient = None
    if len(lines) >= 2 and "PATIENT NAME" in lines[0]:
        try:
            patient = next(csv.reader([lines[1]]))[0]
        except Exception:
            patient = None
    if not patient:
        patient = norm_participant(bp_csv.stem)

    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"DATE"'):
            header_idx = i
            break
    if header_idx is None:
        return pd.DataFrame()

    reader = csv.DictReader(lines[header_idx:])
    rows = []

    def to_num(x):
        try:
            return float(x)
        except Exception:
            return np.nan

    for r in reader:
        date_str = r.get("DATE")
        time_str = r.get("TIME")
        if not date_str or not time_str:
            continue
        dt = pd.to_datetime(f"{date_str} {time_str}", format="%Y-%m-%d %H:%M", errors="coerce")
        if pd.isna(dt):
            continue

        exc_bool = str(r.get("EXC(O)")).strip().lower() == "true"

        rows.append({
            "participant": norm_participant(patient),
            "bp_datetime": dt,
            "bp_sys": to_num(r.get("SYS(O)")),
            "bp_dia": to_num(r.get("DIA(O)")),
            "bp_pul": to_num(r.get("PUL(O)")),
            "bp_map": to_num(r.get("MAP(O)")),
            "bp_err": r.get("ERR(O)"),
            "bp_exc": exc_bool,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["bp_is_valid"] = df["bp_sys"].notna() & (~df["bp_exc"])
    df = df.sort_values(["participant", "bp_datetime"]).reset_index(drop=True)
    return df


def read_bp_dir(bp_dir: Path, debug=False) -> pd.DataFrame:
    bp_dir = Path(bp_dir)
    all_bp = []
    for f in bp_dir.glob("*.csv"):
        df = read_bp_single(f)
        if not df.empty:
            if debug:
                parts = df["participant"].dropna().unique().tolist()
                print("[DEBUG] BP file:", f.name, "participant:", parts[:5])
            all_bp.append(df)
    if not all_bp:
        return pd.DataFrame()
    return pd.concat(all_bp, ignore_index=True)


def read_glucose_single(glucose_csv: Path) -> pd.DataFrame:
    m = re.search(r"(CAT\s*0*\d+)", glucose_csv.name.upper())
    p = norm_participant(m.group(1)) if m else norm_participant(glucose_csv.stem)

    df = pd.read_csv(glucose_csv, skiprows=2, engine="python")
    if df.empty or "Device Timestamp" not in df.columns:
        return pd.DataFrame()

    df["participant"] = p
    df["glucose_datetime"] = pd.to_datetime(df["Device Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")

    g_hist = pd.to_numeric(df.get("Historic Glucose mmol/L", np.nan), errors="coerce")
    g_scan = pd.to_numeric(df.get("Scan Glucose mmol/L", np.nan), errors="coerce")
    df["glucose_mmol_L"] = g_hist.where(g_hist.notna(), g_scan)

    df = df.dropna(subset=["glucose_datetime", "glucose_mmol_L"]).copy()
    df = df.sort_values(["participant", "glucose_datetime"]).reset_index(drop=True)
    return df[["participant", "glucose_datetime", "glucose_mmol_L"]]


def read_glucose_dir(glucose_dir: Path) -> pd.DataFrame:
    glucose_dir = Path(glucose_dir)
    all_g = []
    for f in glucose_dir.glob("*.csv"):
        if "GLUCOSE" not in f.name.upper():
            continue
        df = read_glucose_single(f)
        if not df.empty:
            all_g.append(df)
    if not all_g:
        return pd.DataFrame()
    return pd.concat(all_g, ignore_index=True)


def window_slice(df, t0, time_col, window_min):
    w = pd.Timedelta(minutes=float(window_min))
    lo, hi = t0 - w, t0 + w
    return df[(df[time_col] >= lo) & (df[time_col] <= hi)]


def mean_cols(sub, cols):
    out = {}
    for c in cols:
        out[c] = pd.to_numeric(sub[c], errors="coerce").mean() if (c in sub.columns and len(sub) > 0) else np.nan
    return out


def nearest_row(df, t0, time_col):
    if df.empty:
        return None, np.nan
    idx = (df[time_col] - t0).abs().idxmin()
    diff_min = abs((df.loc[idx, time_col] - t0).total_seconds()) / 60.0
    return df.loc[idx], diff_min


def hormone_align(hdf, t0, window_min, max_nearest_min, hormone_cols):
    out = {}
    if hdf.empty or not hormone_cols:
        out.update({c: np.nan for c in hormone_cols})
        out["hormone_n_in_window"] = 0
        out["hormone_align_method"] = "none"
        out["hormone_nearest_diff_min"] = np.nan
        return out

    sub = window_slice(hdf, t0, "sample_datetime", window_min)
    n = len(sub)
    out["hormone_n_in_window"] = int(n)

    if n > 0:
        out.update(mean_cols(sub, hormone_cols))
        out["hormone_align_method"] = f"window_mean_pm{window_min}"
        out["hormone_nearest_diff_min"] = 0.0
        return out

    if max_nearest_min is None:
        out.update({c: np.nan for c in hormone_cols})
        out["hormone_align_method"] = "none"
        out["hormone_nearest_diff_min"] = np.nan
        return out

    row, diff = nearest_row(hdf, t0, "sample_datetime")
    out["hormone_nearest_diff_min"] = diff
    if row is not None and diff <= max_nearest_min:
        for c in hormone_cols:
            out[c] = pd.to_numeric(row.get(c, np.nan), errors="coerce")
        out["hormone_align_method"] = f"nearest_le_{max_nearest_min}min"
    else:
        out.update({c: np.nan for c in hormone_cols})
        out["hormone_align_method"] = "none"
    return out


def bp_align_and_impute(bdf_valid, t0, window_min, max_nearest_min, max_interp_gap_min, bp_cols):
    out = {c: np.nan for c in bp_cols}
    out["bp_n_in_window"] = 0
    out["bp_fill_method"] = "none"
    out["bp_fill_diff_min"] = np.nan
    out["bp_fill_gap_min"] = np.nan

    if bdf_valid.empty:
        return out

    sub = window_slice(bdf_valid, t0, "bp_datetime", window_min)
    if len(sub) > 0:
        out["bp_n_in_window"] = int(len(sub))
        out.update(mean_cols(sub, bp_cols))
        out["bp_fill_method"] = f"window_mean_pm{window_min}"
        out["bp_fill_diff_min"] = 0.0
        out["bp_fill_gap_min"] = 0.0
        return out

    if max_nearest_min is not None:
        row, diff = nearest_row(bdf_valid, t0, "bp_datetime")
        if row is not None and diff <= max_nearest_min:
            for c in bp_cols:
                out[c] = pd.to_numeric(row.get(c, np.nan), errors="coerce")
            out["bp_fill_method"] = f"nearest_le_{max_nearest_min}min"
            out["bp_fill_diff_min"] = diff
            out["bp_fill_gap_min"] = diff
            return out

    if max_interp_gap_min is not None:
        prev = bdf_valid[bdf_valid["bp_datetime"] <= t0].tail(1)
        nxt = bdf_valid[bdf_valid["bp_datetime"] >= t0].head(1)
        if len(prev) == 1 and len(nxt) == 1:
            t_prev = prev["bp_datetime"].iloc[0]
            t_nxt = nxt["bp_datetime"].iloc[0]
            gap_min = abs((t_nxt - t_prev).total_seconds()) / 60.0
            if 0 < gap_min <= max_interp_gap_min:
                w = (t0 - t_prev).total_seconds() / (t_nxt - t_prev).total_seconds()
                for c in bp_cols:
                    v0 = pd.to_numeric(prev[c].iloc[0], errors="coerce")
                    v1 = pd.to_numeric(nxt[c].iloc[0], errors="coerce")
                    out[c] = (1 - w) * v0 + w * v1 if (pd.notna(v0) and pd.notna(v1)) else np.nan
                out["bp_fill_method"] = f"linear_interp_gap_le_{max_interp_gap_min}min"
                out["bp_fill_gap_min"] = gap_min
                return out

    return out


def glucose_align(gdf, t0, window_min):
    out = {"glucose_mmol_L": np.nan, "glucose_n_in_window": 0, "glucose_align_method": "none"}
    if gdf.empty:
        return out
    sub = window_slice(gdf, t0, "glucose_datetime", window_min)
    if len(sub) == 0:
        return out
    out["glucose_mmol_L"] = pd.to_numeric(sub["glucose_mmol_L"], errors="coerce").mean()
    out["glucose_n_in_window"] = int(len(sub))
    out["glucose_align_method"] = f"window_mean_pm{window_min}"
    return out


def _fmt_sec(dt):
    return f"{dt:.2f}s"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--activity", default=r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\cat234511.xlsx")
    ap.add_argument("--hormone", default=r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\Hormones\hormone.xlsx")
    ap.add_argument("--bp-dir", default=r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\OneDrive_2026-01-27\Data files\Blood pressure")
    ap.add_argument("--glucose-dir", default=r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\OneDrive_1_28-01-2026")
    ap.add_argument("--out", default=r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\activity_hormone_bp_glucose_merged_all.csv")

    ap.add_argument("--hormone-window-min", type=float, default=30.0)
    ap.add_argument("--hormone-nearest-max-min", type=float, default=12 * 60)

    ap.add_argument("--bp-window-min", type=float, default=30.0)
    ap.add_argument("--bp-nearest-max-min", type=float, default=6 * 60)
    ap.add_argument("--bp-interp-gap-max-min", type=float, default=8 * 60)

    ap.add_argument("--glucose-window-min", type=float, default=30.0)

    ap.add_argument("--participants", default="CAT02,CAT03,CAT04,CAT05,CAT11")
    ap.add_argument("--debug-bp", action="store_true")

    args = ap.parse_args()

    t_all0 = time.time()

    activity_path = must_exist(args.activity, "activity(cat234511)")
    hormone_path = must_exist(args.hormone, "hormone.xlsx")
    bp_dir = must_exist(args.bp_dir, "bp-dir")
    glucose_dir = must_exist(args.glucose_dir, "glucose-dir")

    print("RUNNING:", Path(__file__).name)
    print("[INFO] activity:", activity_path); file_fingerprint(activity_path)
    print("[INFO] hormone :", hormone_path); file_fingerprint(hormone_path)
    print("[INFO] bp-dir  :", bp_dir)
    print("[INFO] glucose-dir:", glucose_dir)

    target = {p.strip().upper() for p in args.participants.split(",") if p.strip()}
    print("[INFO] participants:", sorted(target))
    print()

    t0 = time.time()
    activity = read_activity(activity_path)
    print(f"[LOAD] activity done ({_fmt_sec(time.time() - t0)})")

    t0 = time.time()
    hormone = read_hormone(hormone_path)
    print(f"[LOAD] hormone done ({_fmt_sec(time.time() - t0)})")

    t0 = time.time()
    bp = read_bp_dir(bp_dir, debug=args.debug_b피) if False else read_bp_dir(bp_dir, debug=args.debug_bp)
    print(f"[LOAD] bp done ({_fmt_sec(time.time() - t0)})")

    t0 = time.time()
    glucose = read_glucose_dir(glucose_dir)
    print(f"[LOAD] glucose done ({_fmt_sec(time.time() - t0)})")
    print()

    print_summary("activity(raw)", activity)
    print_summary("hormone(raw)", hormone)
    print_summary("bp(raw)", bp)
    print_summary("glucose(raw)", glucose)
    print()

    activity = activity[activity["participant"].isin(target)].copy()
    hormone = hormone[hormone["participant"].isin(target)].copy()
    bp = bp[bp["participant"].isin(target)].copy()
    glucose = glucose[glucose["participant"].isin(target)].copy()

    print_summary("activity(filtered)", activity)
    print_summary("hormone(filtered)", hormone)
    print_summary("bp(filtered)", bp)
    print_summary("glucose(filtered)", glucose)
    print()

    hormone_cols = [c for c in hormone.columns if c not in {"participant", "sample_datetime", "sample_id"}]
    bp_cols = ["bp_sys", "bp_dia", "bp_pul", "bp_map"]

    hormone_by = {p: d.sort_values("sample_datetime") for p, d in hormone.groupby("participant")} if not hormone.empty else {}
    bp_by = {p: d.sort_values("bp_datetime") for p, d in bp.groupby("participant")} if not bp.empty else {}
    glucose_by = {p: d.sort_values("glucose_datetime") for p, d in glucose.groupby("participant")} if not glucose.empty else {}

    h_nearest_max = float(args.hormone_nearest_max_min)
    if h_nearest_max <= 0:
        h_nearest_max = None

    b_nearest_max = float(args.bp_nearest_max_min)
    if b_nearest_max <= 0:
        b_nearest_max = None

    b_interp_gap = float(args.bp_interp_gap_max_min)
    if b_interp_gap <= 0:
        b_interp_gap = None

    print("[MERGE] building rows ...")
    t0 = time.time()

    rows = []
    for i, (_, r) in enumerate(activity.iterrows(), 1):
        if i in {1, 25, 50, 75, 100} or i == len(activity):
            print(f"  progress: {i}/{len(activity)}")

        p = r["participant"]
        t_anchor = pd.to_datetime(r["activity_timestamp"], errors="coerce")
        out = r.to_dict()

        hdf = hormone_by.get(p, pd.DataFrame())
        out.update(hormone_align(hdf, t_anchor, args.hormone_window_min, h_nearest_max, hormone_cols))

        bdf = bp_by.get(p, pd.DataFrame())
        if not bdf.empty and "bp_is_valid" in bdf.columns:
            bdf_valid = bdf[bdf["bp_is_valid"] == True].copy()
        else:
            bdf_valid = pd.DataFrame()
        out.update(bp_align_and_impute(bdf_valid, t_anchor, args.bp_window_min, b_nearest_max, b_interp_gap, bp_cols))

        gdf = glucose_by.get(p, pd.DataFrame())
        out.update(glucose_align(gdf, t_anchor, args.glucose_window_min))

        rows.append(out)

    merged = pd.DataFrame(rows)
    print(f"[MERGE] done ({_fmt_sec(time.time() - t0)})")
    print()

    merged = merged.loc[:, ~merged.columns.astype(str).str.startswith("Unnamed")]
    merged = merged.dropna(axis=1, how="all")

    out_path = safe_out_path(args.out)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print("[SAVE] ->", out_path)
    print("[SAVE] shape:", merged.shape)

    if "participant" in merged.columns:
        print("[CHECK] rows per participant:", merged["participant"].value_counts().to_dict())

    if "glucose_n_in_window" in merged.columns:
        print("[CHECK] glucose points/window (overall):", merged["glucose_n_in_window"].describe().to_dict())
        print("[CHECK] glucose points/window (by participant):",
              merged.groupby("participant")["glucose_n_in_window"].mean().round(2).to_dict())

    if "hormone_align_method" in merged.columns:
        print("[CHECK] hormone methods:", merged["hormone_align_method"].value_counts().to_dict())
        hn = merged[merged["hormone_align_method"].astype(str).str.startswith("nearest")]
        if not hn.empty and "hormone_nearest_diff_min" in hn.columns:
            print("[CHECK] hormone nearest diff (min):", hn["hormone_nearest_diff_min"].describe().to_dict())

    if "bp_fill_method" in merged.columns:
        print("[CHECK] bp methods:", merged["bp_fill_method"].value_counts().to_dict())
        bn = merged[merged["bp_fill_method"].astype(str).str.startswith("nearest")]
        if not bn.empty and "bp_fill_diff_min" in bn.columns:
            print("[CHECK] bp nearest diff (min):", bn["bp_fill_diff_min"].describe().to_dict())

    print(f"[DONE] total runtime: {_fmt_sec(time.time() - t_all0)}")


if __name__ == "__main__":
    main()
