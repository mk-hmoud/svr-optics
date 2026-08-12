"""Figure 4 -- Monte Carlo manufacturing-tolerance volatility per design variable.

New artwork for the journal version: the conference paper reported these Monte
Carlo numbers in running text only.  Source data is
`robustness_guarded_results.csv`, the output of scripts/robustness_guarded.py.

Rendered as a Cleveland dot plot on a logarithmic axis rather than another bar
chart, so it does not read as a repeat of Fig. 2, and so the 160x spread between
the pitch and the innermost hole stays legible.  The dominant contributor is the
only mark carrying the accent colour; everything else is the single series hue.

SUPERSEDES the published numbers.  src/robustness.py evaluated at a baseline
three of whose four geometric values lie outside the sampled design space (the
Fig. 1 radii written into diameter columns, and a pitch in um written into a
column stored in units of 10 um), so the surrogate was extrapolating and its
argmax peak finder returned grid endpoints.  These values come from a re-run at a
sampled geometry with a boundary guard: 0/100 trials rejected at every level.
Wavelength is in nanometres (the lambda column is in units of 100 nm).

Usage:  python fig4_robustness.py [--out ../figs/fig4_robustness.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from sivp_style import (
    COLUMN_W,
    INK,
    INK_MUTED,
    SERIES,
    SERIES_ACCENT,
    apply_style,
    save,
    strip_spines,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "..", "robustness_guarded_results.csv")
DEFAULT_OUT = os.path.join(HERE, "..", "figs", "fig4_robustness.pdf")

SYMBOLS = {
    "Pitch (um)": r"$\Lambda$",
    "d2 (um)": r"$d_2$",
    "d3 (um)": r"$d_3$",
    "d1 (um)": r"$d_1$",
}


def main(src, out):
    apply_style()

    df = pd.read_csv(src)
    df = df[df["variable"] != "all"].copy()
    df["label"] = df["variable"].map(SYMBOLS).fillna(df["variable"])
    df = df.rename(columns={"jitter_nm": "Volatility_nm"})
    df = df.sort_values("Volatility_nm").reset_index(drop=True)

    top = df["Volatility_nm"].idxmax()
    colours = [SERIES_ACCENT if i == top else SERIES for i in df.index]

    fig, ax = plt.subplots(figsize=(COLUMN_W, 1.85))

    ax.hlines(
        y=df.index,
        xmin=df["Volatility_nm"].min() * 0.35,
        xmax=df["Volatility_nm"],
        color=INK_MUTED,
        linewidth=0.7,
        zorder=2,
    )
    ax.scatter(
        df["Volatility_nm"],
        df.index,
        s=42,
        c=colours,
        zorder=4,
        edgecolor="white",
        linewidth=0.8,
    )

    for i, (v, lab) in enumerate(zip(df["Volatility_nm"], df["label"])):
        ax.text(
            v * 1.35,
            i,
            f"{v:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color=INK_MUTED,
        )

    ax.set_xscale("log")
    ax.set_xlim(0.1, 40.0)
    ax.set_yticks(df.index)
    ax.set_yticklabels(df["label"], fontsize=9, color=INK)
    ax.set_xlabel("Resonance jitter, std (nm)", color=INK)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    strip_spines(ax, keep=("left",))

    save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.src, a.out)
