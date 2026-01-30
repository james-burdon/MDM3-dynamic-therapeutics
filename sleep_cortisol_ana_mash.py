import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.multitest import multipletests


DATA_PATH = r"D:\UOB\Year_3_UOB\mdm_hormone\MDM3-dynamic-therapeutics\DataPaper\sleep_cortisol_merged.csv"

sleep_cols = [
    "Efficiency",
    "Total Sleep Time (TST)",
    "Wake After Sleep Onset (WASO)",
    "Fragmentation Index"
]

targets = ["Delta_cortisol", "LogChange_cortisol"]

perm_B = 20000
boot_B = 5000
alpha = 0.05
lowess_frac = 0.6
plot_flag = True
seed = 0


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def add_cortisol_changes(df):
    c_before = to_num(df["Cortisol NORM (before sleep)"])
    c_wake = to_num(df["Cortisol NORM (wake up)"])

    eps = 1e-12
    df = df.copy()
    df["Delta_cortisol"] = c_wake - c_before
    df["LogChange_cortisol"] = np.log(c_wake + eps) - np.log(c_before + eps)
    df["RelChange_cortisol"] = (c_wake - c_before) / (c_before + eps)
    return df


def perm_spearman(x, y, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 5:
        return np.nan, np.nan

    r_obs, _ = spearmanr(x, y)
    cnt = 0

    for _ in range(B):
        y_perm = rng.permutation(y)
        r_p, _ = spearmanr(x, y_perm)
        if abs(r_p) >= abs(r_obs):
            cnt += 1

    p_perm = (cnt + 1) / (B + 1)
    return r_obs, p_perm


def boot_ci_spearman(x, y, B=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    n = len(x)
    if n < 5:
        return np.nan, np.nan

    rs = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        rs.append(r)

    lo = np.quantile(rs, 0.025)
    hi = np.quantile(rs, 0.975)
    return lo, hi


def plot_lowess(x, y, xlab, ylab):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if len(x) < 5:
        return

    order = np.argsort(x)
    smth = lowess(y[order], x[order], frac=lowess_frac)

    plt.figure()
    plt.scatter(x, y)
    plt.plot(smth[:, 0], smth[:, 1])
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.tight_layout()
    plt.show()


def run_one_target(df, target):
    y = to_num(df[target]).to_numpy()
    out = []

    for col in sleep_cols:
        x = to_num(df[col]).to_numpy()
        n_pair = int(np.sum(np.isfinite(x) & np.isfinite(y)))

        r_s, p_s = spearmanr(x, y, nan_policy="omit")
        _, p_perm = perm_spearman(x, y, B=perm_B, seed=seed)
        ci_lo, ci_hi = boot_ci_spearman(x, y, B=boot_B, seed=seed)

        out.append({
            "feature": col,
            "N_pairwise": n_pair,
            "spearman_r": r_s,
            "p_spearman": p_s,
            "p_perm": p_perm,
            "r_ci95_lo": ci_lo,
            "r_ci95_hi": ci_hi
        })

    out = pd.DataFrame(out).sort_values("p_perm").reset_index(drop=True)

    rej, p_fdr, *_ = multipletests(out["p_perm"].values, alpha=alpha, method="fdr_bh")
    out["p_perm_fdr"] = p_fdr
    out["sig_fdr_0.05"] = rej

    return out


def main():
    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")].copy()
    df = add_cortisol_changes(df)

    print("N =", len(df))
    print("sleep cols =", sleep_cols)
    print("targets =", targets)

    for t in targets:
        res = run_one_target(df, t)

        print(f"\n---- {t} ----")
        print(res[[
            "feature",
            "N_pairwise",
            "spearman_r",
            "r_ci95_lo",
            "r_ci95_hi",
            "p_perm",
            "p_perm_fdr",
            "sig_fdr_0.05"
        ]])

        if plot_flag:
            for col in sleep_cols:
                x = to_num(df[col]).to_numpy()
                y = to_num(df[t]).to_numpy()
                plot_lowess(x, y, col, t)


if __name__ == "__main__":
    main()
