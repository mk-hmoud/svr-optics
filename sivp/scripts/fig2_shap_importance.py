"""Figure 2 -- global SHAP attribution ranking.

Regenerates the feature-attribution figure for the SIVP submission from
`feature_importance_ranking.csv` (the numeric output of src/explainability.py).

Differences from the conference rendering, which used horizontal dodger-blue
bars on a default matplotlib frame:

  * vertical columns instead of horizontal bars
  * single validated series colour (a magnitude chart gets one hue, never a
    value ramp keyed to bar length)
  * axis rescaled to share-of-total attribution (%) rather than raw mean |SHAP|
  * axis limit extended to 38% so the value labels sit clear of the frame
  * feature names typeset as the physical symbols used in the manuscript
  * box frame removed, hairline horizontal grid only

Usage:  python fig2_shap_importance.py [--out ../figs/fig2_shap_importance.pdf]
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from sivp_style import COLUMN_W, INK, INK_MUTED, SERIES, apply_style, save, strip_spines

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "..", "feature_importance_ranking.csv")
DEFAULT_OUT = os.path.join(HERE, "..", "figs", "fig2_shap_importance.pdf")

# dataset column name -> manuscript symbol
SYMBOLS = {
    "d3 (um)": r"$d_3$",
    "d2 (um)": r"$d_2$",
    "Pitch (um)": r"$\Lambda$",
    "lambda": r"$\lambda$",
    "Analyte": r"$n_a$",
    "d1 (um)": r"$d_1$",
}


def main(src, out):
    apply_style()

    df = pd.read_csv(src)
    df["label"] = df["Feature"].map(SYMBOLS).fillna(df["Feature"])
    df["share"] = 100 * df["Mean_Absolute_SHAP"] / df["Mean_Absolute_SHAP"].sum()
    df = df.sort_values("share", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.15))

    # One series -> one colour.  Bar length already encodes magnitude; a ramp
    # keyed to that same magnitude would burn the hue channel on nothing.
    ax.bar(df["label"], df["share"], width=0.62, color=SERIES, zorder=3)

    for x, share in enumerate(df["share"]):
        ax.text(
            x,
            share + 0.7,
            f"{share:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=INK_MUTED,
        )

    ax.set_ylim(0, 38)
    ax.set_ylabel("Attribution share (%)", color=INK)
    ax.set_xlabel("Design variable", color=INK)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=9, colors=INK, length=0)
    strip_spines(ax, keep=("bottom",))

    save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default=DEFAULT_SRC)
    p.add_argument("--out", default=DEFAULT_OUT)
    a = p.parse_args()
    main(a.src, a.out)
