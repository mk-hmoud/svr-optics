"""Figure 5 -- measured training and inference cost of each surrogate.

Reads timing_results.csv, written by benchmark_timing.py. All models were timed
on the same machine, the same 432-sample dataset and the same CPU-only
configuration, so the ratios are meaningful even though the absolute seconds are
hardware-specific.

Horizontal bars on a logarithmic axis, because the range spans four orders of
magnitude: an SVR fit is milliseconds while WGAN-GP training is minutes. Two
panels rather than a dual axis -- training cost and per-sample inference cost are
different quantities and must not share a scale.

Usage:  python fig5_timing.py [--out ../figs/fig5_timing.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
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
DEFAULT_SRC = os.path.join(HERE, "..", "..", "timing_results.csv")
DEFAULT_OUT = os.path.join(HERE, "..", "figs", "fig5_timing.pdf")

# Short labels; the caption carries the detail.
LABELS = {
    ("SVR", "fit only"): "SVR fit",
    ("SVR", "with hyperparameter search"): "SVR + search",
    ("GPR", "fit only"): "GPR fit",
    ("ANN", "500 epochs"): "ANN, 500 ep.",
    ("WGAN-GP", "2000 epochs, generator only"): "WGAN-GP, 2000 ep.",
}


def fmt_time(s):
    if s < 1:
        return f"{s*1000:.0f} ms"
    if s < 90:
        return f"{s:.1f} s"
    return f"{s/60:.1f} min"


def main(src, out):
    apply_style()
    df = pd.read_csv(src)
    df["label"] = [LABELS.get((m, st), f"{m} {st}") for m, st in zip(df.model, df.stage)]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(COLUMN_W, 3.15), gridspec_kw={"height_ratios": [5, 3]}
    )

    # --- panel (a): training cost -------------------------------------------
    d = df.sort_values("train_s").reset_index(drop=True)
    top = d["train_s"].idxmax()
    colours = [SERIES_ACCENT if i == top else SERIES for i in d.index]
    ax1.barh(d.index, d["train_s"], height=0.6, color=colours, zorder=3)
    for i, v in enumerate(d["train_s"]):
        ax1.text(v * 1.25, i, fmt_time(v), va="center", ha="left",
                 fontsize=7.5, color=INK_MUTED)
    ax1.set_xscale("log")
    ax1.set_xlim(0.01, 4e4)
    ax1.set_yticks(d.index)
    ax1.set_yticklabels(d["label"], fontsize=8, color=INK)
    ax1.set_xlabel("Training time (s, log scale)", color=INK)
    ax1.set_title("(a) cost to produce the model", fontsize=8.5,
                  color=INK, loc="left", pad=4)

    # --- panel (b): inference cost ------------------------------------------
    p = df.dropna(subset=["predict_us_per_sample"]).drop_duplicates("model")
    p = p.sort_values("predict_us_per_sample").reset_index(drop=True)
    ax2.barh(p.index, p["predict_us_per_sample"], height=0.55,
             color=SERIES, zorder=3)
    for i, v in enumerate(p["predict_us_per_sample"]):
        ax2.text(v * 1.05, i, f"{v:.0f}", va="center", ha="left",
                 fontsize=7.5, color=INK_MUTED)
    ax2.set_xlim(0, max(p["predict_us_per_sample"]) * 1.35)
    ax2.set_yticks(p.index)
    ax2.set_yticklabels(p["model"], fontsize=8, color=INK)
    ax2.set_xlabel(r"Inference time (µs per sample)", color=INK)
    ax2.set_title("(b) cost to evaluate one design point", fontsize=8.5,
                  color=INK, loc="left", pad=4)

    for ax in (ax1, ax2):
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.invert_yaxis()
        strip_spines(ax, keep=("left",))

    fig.tight_layout(pad=0.3, h_pad=1.4)
    save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.src, a.out)
