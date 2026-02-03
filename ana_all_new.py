# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\activity_hormone_bp_glucose_merged_all_20260203_002553.csv"


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)


def inv_fisher_z(z):
    return np.tanh(z)


def pick_activity_x(df):
    if "intensity" in df.columns:
        x = to_num(df["intensity"])
        if x.notna().sum() >= 10:
            return x, "intensity"

    if "activity_label" in df.columns:
        s = df["activity_label"].astype(str).str.lower()
        x = np.where(s.str.contains("active"), 1,
                     np.where(s.str.contains("rest"), 0, np.nan))
        x = pd.Series(x, index=df.index)
        if x.notna().sum() >= 10:
            print("[note] using binary activity from activity_label")
            return x, "activity_binary"

    raise ValueError("No usable activity variable found")


def filter_hormone_rows(df, mode, nearest_limit=180.0):
    if "hormone_align_method" not in df.columns:
        return df.copy()

    if mode == "STRICT":
        return df[df["hormone_align_method"] == "window_mean_pm30.0"].copy()

    if mode == "EXPANDED":
        m1 = df["hormone_align_method"] == "window_mean_pm30.0"
        m2 = (
            df["hormone_align_method"].astype(str).str.startswith("nearest") &
            (to_num(df["hormone_nearest_diff_min"]) <= nearest_limit)
        )
        return df[m1 | m2].copy()

    return df.copy()


def meta_spearman(df, x_col, y_col, min_n=6):
    zs, ws, parts, rhos, ns = [], [], [], [], []

    for pid, g in df.groupby("participant"):
        x = to_num(g[x_col])
        y = to_num(g[y_col])
        m = x.notna() & y.notna()
        n = int(m.sum())

        if n < min_n:
            continue
        if x[m].nunique() <= 1 or y[m].nunique() <= 1:
            continue

        rho, _ = spearmanr(x[m], y[m])
        if np.isnan(rho):
            continue

        zs.append(fisher_z(rho))
        ws.append(max(n - 3, 1))
        parts.append(pid)
        rhos.append(rho)
        ns.append(n)

    if not zs:
        return None

    z_bar = np.sum(np.array(zs) * np.array(ws)) / np.sum(ws)
    return {
        "rho_meta": float(inv_fisher_z(z_bar)),
        "participants": parts,
        "rhos": rhos,
        "ns": ns,
        "n_participants": len(parts),
        "n_total": int(np.sum(ns)),
    }


def pooled_pvalue(df, x_col, y_col):
    x = to_num(df[x_col])
    y = to_num(df[y_col])
    m = x.notna() & y.notna()
    if m.sum() < 10:
        return np.nan
    if x[m].nunique() <= 1 or y[m].nunique() <= 1:
        return np.nan
    _, p = spearmanr(x[m], y[m])
    return p


def build_table(df, x_col, hormone_cols):
    rows = []

    for h in hormone_cols:
        sub = df[["participant", x_col, h]].copy()
        res = meta_spearman(sub, x_col, h)
        if res is None or res["n_participants"] < 2:
            continue

        p = pooled_pvalue(sub, x_col, h)

        rows.append({
            "hormone": h,
            "rho_meta": res["rho_meta"],
            "n_participants": res["n_participants"],
            "n_total": res["n_total"],
            "p_screen": p,
        })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    pvals = out["p_screen"].values
    mask = np.isfinite(pvals)

    q = np.full(len(pvals), np.nan)
    if mask.sum() > 0:
        q[mask] = multipletests(pvals[mask], method="fdr_bh")[1]

    out["q_fdr"] = q
    out["abs_rho"] = out["rho_meta"].abs()
    return out.sort_values("abs_rho", ascending=False).drop(columns="abs_rho")


def plot_bar(tab, title):
    if tab.empty:
        return

    top = tab.head(8)

    plt.figure(figsize=(9, 4.5))
    plt.barh(
        top["hormone"].str.replace("h_", ""),
        top["rho_meta"]
    )
    plt.axvline(0)
    plt.xlabel("Meta Spearman rho")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_scatter(df, x_col, y_col, title):
    d = df[[x_col, y_col, "participant"]].copy()
    d[x_col] = to_num(d[x_col])
    d[y_col] = to_num(d[y_col])
    d = d.dropna()

    if len(d) < 10:
        return

    rho, p = spearmanr(d[x_col], d[y_col])

    plt.figure(figsize=(6, 4.5))
    for pid, g in d.groupby("participant"):
        plt.scatter(g[x_col], g[y_col], label=pid, alpha=0.8)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{title}\nrho={rho:.3f}, p={p:.3g}, n={len(d)}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def analyse_mode(df0, mode, x_name, hormone_cols):
    df = filter_hormone_rows(df0, mode)

    print(f"\n[{mode}] rows={len(df)}, participants={sorted(df['participant'].unique())}")

    tab_act = build_table(df, x_name, hormone_cols)
    tab_glu = build_table(df, "glucose_mmol_L", hormone_cols) if "glucose_mmol_L" in df.columns else pd.DataFrame()

    if not tab_act.empty:
        best = tab_act.iloc[0]
        print(f"activity top hit: {best['hormone']} (rho={best['rho_meta']:.3f})")
        plot_bar(tab_act, f"{mode}: activity vs hormones")
        plot_scatter(df, x_name, best["hormone"], f"{mode}: activity top hit")

    if not tab_glu.empty:
        best = tab_glu.iloc[0]
        print(f"glucose top hit: {best['hormone']} (rho={best['rho_meta']:.3f})")
        plot_bar(tab_glu, f"{mode}: glucose vs hormones")
        plot_scatter(df, "glucose_mmol_L", best["hormone"], f"{mode}: glucose top hit")


def main():
    t0 = time.time()

    print("[load]", os.path.basename(DATA_PATH))
    df0 = pd.read_csv(DATA_PATH)

    hormone_cols = [c for c in df0.columns if isinstance(c, str) and c.startswith("h_")]
    x_series, x_name = pick_activity_x(df0)
    df0[x_name] = x_series

    print(f"rows={len(df0)}, hormones={len(hormone_cols)}, X={x_name}")

    analyse_mode(df0, "STRICT", x_name, hormone_cols)
    analyse_mode(df0, "EXPANDED", x_name, hormone_cols)

    print(f"\ndone in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
