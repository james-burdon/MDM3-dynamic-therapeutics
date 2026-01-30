# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np

DATA_PATH = Path(r"D:\UOB\Year_3_UOB\mdm_hormone\mcphases_raw_long.csv")
REPORT_PATH = Path(r"D:\UOB\Year_3_UOB\mdm_hormone\sanity_report.txt")

def main():
    print("[START] check_merge.py running...")

    if not DATA_PATH.exists():
        print(f"[ERROR] File not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"[OK] Loaded: {DATA_PATH}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")

    # Basic missingness
    miss = df.isna().mean().sort_values(ascending=False)
    print("\n[INFO] Missingness (top 10):")
    print(miss.head(10))

    # Required columns
    required = {"id","day_in_study","phase","timestamp","source","signal","value"}
    missing_cols = required - set(df.columns)
    print(f"\n[CHECK] Missing required columns: {missing_cols if missing_cols else 'None'}")

    # Phase uniqueness per (id, day)
    if "phase" in df.columns:
        phase_nunique = df.groupby(["id","day_in_study"])["phase"].nunique(dropna=True)
        counts = phase_nunique.value_counts().sort_index()
        print("\n[CHECK] phase nunique per (id, day_in_study):")
        print(counts)

        bad = phase_nunique[phase_nunique > 1]
        print(f"\n[CHECK] days with >1 phase: {len(bad)}")
        if len(bad) > 0:
            print("[EXAMPLE] First 10 problematic (id, day):")
            print(bad.head(10))

    # Coverage by source/signal
    print("\n[INFO] Non-null value counts by source (top):")
    print(df.groupby("source")["value"].count().sort_values(ascending=False).head(20))

    print("\n[INFO] Top 20 (source, signal) by non-null count:")
    top_sig = df.groupby(["source","signal"])["value"].count().sort_values(ascending=False).head(20)
    print(top_sig)

    # Write a small report file
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(f"File: {DATA_PATH}\n")
        f.write(f"Shape: {df.shape}\n\n")
        f.write("Columns:\n")
        f.write(", ".join(df.columns) + "\n\n")

        f.write("Missingness (top 20):\n")
        f.write(miss.head(20).to_string() + "\n\n")

        f.write("phase nunique per (id, day_in_study):\n")
        if "phase" in df.columns:
            f.write(counts.to_string() + "\n\n")
            f.write(f"days with >1 phase: {len(bad)}\n")
        else:
            f.write("phase column missing\n")

        f.write("\nNon-null value counts by source:\n")
        f.write(df.groupby("source")["value"].count().sort_values(ascending=False).to_string() + "\n")

    print(f"\n[DONE] Report written to: {REPORT_PATH}")
    print("[END] check_merge.py finished.")

if __name__ == "__main__":
    main()
