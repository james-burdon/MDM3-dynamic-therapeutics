import os
import pandas as pd

BASE = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\DataPaper"
SKIP = {11, 21}
OUT = os.path.join(BASE, "sleep_cortisol_merged.csv")

def main():
    rows = []
    users = sorted(
        [d for d in os.listdir(BASE) if d.startswith("user_") and os.path.isdir(os.path.join(BASE, d))],
        key=lambda x: int(x.split("_")[1])
    )

    print("Found users:", users)

    for u in users:
        uid = int(u.split("_")[1])
        if uid in SKIP:
            print(f"[SKIP] user_{uid}")
            continue

        p = os.path.join(BASE, u)
        sp, hp = os.path.join(p, "sleep.csv"), os.path.join(p, "saliva.csv")
        if not (os.path.exists(sp) and os.path.exists(hp)):
            print(f"[FAIL] user_{uid}: missing file")
            continue

        try:
            sleep = pd.read_csv(sp).iloc[0].to_dict()          # 保留 sleep 所有列（第一行）
            saliva = pd.read_csv(hp)
            before = saliva[saliva["SAMPLES"] == "before sleep"].iloc[0].to_dict()
            wake = saliva[saliva["SAMPLES"] == "wake up"].iloc[0].to_dict()
        except Exception as e:
            print(f"[FAIL] user_{uid}: {e}")
            continue

        row = {"user_id": uid, **sleep}
        row.update({f"{k} (before sleep)": v for k, v in before.items()})
        row.update({f"{k} (wake up)": v for k, v in wake.items()})
        row["Delta_cortisol"] = float(wake["Cortisol NORM"]) - float(before["Cortisol NORM"])

        rows.append(row)
        print(f"[OK] user_{uid}")

    pd.DataFrame(rows).sort_values("user_id").to_csv(OUT, index=False)
    print("\nSaved to:", OUT, "| N =", len(rows))

if __name__ == "__main__":
    main()
