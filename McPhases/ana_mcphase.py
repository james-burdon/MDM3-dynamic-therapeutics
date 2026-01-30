# -*- coding: utf-8 -*-
"""
mcPHASES – Fully portable analysis script

How to run:
  python mcphases_analysis.py

Requirements:
  - This script must sit NEXT TO the mcPHASES dataset folder
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from statsmodels.nonparametric.smoothers_lowess import lowess

PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]


# ======================================================
# Locate dataset automatically (portable)
# ======================================================
ROOT = Path(__file__).resolve().parent

DATASET = next(
    (p for p in ROOT.iterdir()
     if p.is_dir() and p.name.startswith("mcphases-a-dataset")),
    None
)

if DATASET is None:
    sys.exit(
        "\nERROR: mcPHASES dataset folder not found.\n"
        "Place this script in the SAME directory as:\n"
        "  mcphases-a-dataset-of-physiological-hormonal-and-self-reported-events-and-symptoms-for-menstrual-health-tracking-with-wearables-1.0.0\n"
    )


def read_mcphases(name: str) -> pd.DataFrame:
    """Read CSVs with or without .csv extension."""
    for f in (DATASET / name, DATASET / f"{name}.csv"):
        if f.exists():
            return pd.read_csv(f)
    raise FileNotFoundError(f"Missing file: {name}")


# ======================================================
# Cycle assignment (robust, per-subject)
# ======================================================
def assign_cycles(df):
    df = df.sort_values(["id", "day_in_study"]).copy()
    df["cycle"] = np.nan

    for pid, g in df.groupby("id"):
        c = 1
        seen = set()
        start = None

        for i, row in g.iterrows():
            if start is None:
                start = i

            if pd.notna(row["phase"]) and row["phase"] in PHASES:
                seen.add(row["phase"])

            if len(seen) == 4:
                df.loc[start:i, "cycle"] = c
                c += 1
                seen = set()
                start = None

    return df.dropna(subset=["cycle"])


# ======================================================
# Normalised cycle index (FIXED shift bug)
# ======================================================
def add_cycle_percent(df):
    prev = df.groupby("id")["phase"].shift()
    new = (df["phase"] == "Menstrual") & (prev != "Menstrual")
    df["cycle_m"] = new.groupby(df["id"]).cumsum()

    n = df.groupby(["id", "cycle_m"])["phase"].transform("count")
    i = df.groupby(["id", "cycle_m"]).cumcount() + 1
    df["percent"] = 100 * i / n
    return df


# ======================================================
# Hormone curves
# ======================================================
def plot_hormones(df):
    signals = ["lh", "estrogen", "pdg"]
    grid = np.linspace(0, 100, 100)

    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for ax, s in zip(axs, signals):
        curves = []

        for (_, _), g in df.groupby(["id", "cycle_m"]):
            g = g.dropna(subset=[s])
            if len(g) < 2:
                continue
            f = interp1d(g["percent"], g[s], fill_value="extrapolate")
            curves.append(f(grid))

        curves = np.array(curves)
        mean = np.nanmean(curves, axis=0)
        sem = np.nanstd(curves, axis=0) / np.sqrt(curves.shape[0])

        mean = lowess(mean, grid, frac=0.3, return_sorted=False)
        sem = lowess(sem, grid, frac=0.3, return_sorted=False)

        ax.plot(grid, mean)
        ax.fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.3)
        ax.set_ylabel(s.upper())

    axs[-1].set_xlabel("Cycle progress (%)")
    plt.tight_layout()
    plt.show()


# ======================================================
# Phase-wise physiology
# ======================================================
def plot_physio(df):
    rhr = read_mcphases("resting_heart_rate")
    temp = read_mcphases("computed_temperature")

    if "sleep_end_day_in_study" in temp:
        temp = temp.rename(columns={"sleep_end_day_in_study": "day_in_study"})

    rhr = rhr.groupby(["id", "day_in_study"])["value"].median().reset_index()
    temp = temp.groupby(["id", "day_in_study"])["nightly_temperature"].median().reset_index()

    df = df.merge(rhr, on=["id", "day_in_study"], how="left")
    df = df.merge(temp, on=["id", "day_in_study"], how="left")

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sig, lab in zip(
        axs,
        ["nightly_temperature", "value"],
        ["Temperature (°C)", "RHR (BPM)"]
    ):
        data = [df[df.phase == p][sig].dropna() for p in PHASES]
        ax.boxplot(data, labels=PHASES, showfliers=False)
        ax.set_ylabel(lab)

    plt.tight_layout()
    plt.show()


# ======================================================
# MAIN
# ======================================================
print(f"Using dataset: {DATASET.name}")

horm = read_mcphases("hormones_and_selfreport")
horm = assign_cycles(horm)
horm = add_cycle_percent(horm)

plot_hormones(horm)
plot_physio(horm)
