"""Online Supplement Fig. S1 -- per-fold LOGO error, with and without GAN augmentation.

Source: `results_comparison_bayesian.csv`, written by the LOGO evaluation in commit
8bf0737 (Bayesian-tuned SVR, real data vs. GAN-augmented training set), which
holds the fold-level MSE that the conference paper summarised as two averages.
Showing all nine folds is the point of putting this in the supplement: the
augmentation penalty is not a uniform shift, it is concentrated in a few
geometries.

Paired dot plot rather than grouped bars -- the comparison is within-fold, and a
connector makes the per-fold direction of change readable at a glance.

Usage:  python figS1_logo_folds.py [--out ../figs/figS1_logo_folds.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from sivp_style import (
    COLUMN_W,
    INK,
    INK_MUTED,
    apply_style,
    save,
    strip_spines,
)
from sivp_style import AMBER, VIOLET

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "..", "results_comparison_bayesian.csv")
DEFAULT_OUT = os.path.join(HERE, "..", "figs", "figS1_logo_folds.pdf")


def main(src, out):
    apply_style()

    df = pd.read_csv(src)
    df = df[df["Fold"].astype(str).str.isdigit()].copy()
    df["Fold"] = df["Fold"].astype(int)
    df = df.sort_values("Fold").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.75))

    y = df["Fold"]
    ax.hlines(
        y=y,
        xmin=df["Baseline_MSE"],
        xmax=df["Augmented_MSE"],
        color=INK_MUTED,
        linewidth=0.7,
        zorder=2,
    )
    ax.scatter(
        df["Baseline_MSE"], y, s=40, c=VIOLET, marker="o", zorder=4,
        edgecolor="white", linewidth=0.8, label="SVR, real data only",
    )
    ax.scatter(
        df["Augmented_MSE"], y, s=44, c=AMBER, marker="s", zorder=4,
        edgecolor="white", linewidth=0.8, label="SVR + GAN-generated samples",
    )

    ax.set_yticks(range(1, 10))
    ax.set_yticklabels([f"D{i}" for i in range(1, 10)], color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 2.95)
    ax.set_xlabel("Held-out MSE, $\\log_{10}$ scaled loss", color=INK)
    ax.set_ylabel("Held-out geometry", color=INK)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    # Legend below the plot: inside the axes it collides with the D1/D2 marks.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        fontsize=7,
        handletextpad=0.3,
        columnspacing=1.2,
        borderpad=0.2,
    )
    strip_spines(ax, keep=("left",))

    save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.src, a.out)
