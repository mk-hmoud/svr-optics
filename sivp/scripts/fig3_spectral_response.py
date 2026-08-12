"""Figure 3 -- surrogate-predicted spectral loss response vs analyte index.

Retrains the optimised SVR (RBF, C=100, gamma=0.1, epsilon=0.01) on the full
432-sample FV-FEM dataset and sweeps wavelength at fixed geometry for each
analyte, exactly as in src/sensitivity.py.  Nothing about the model or the data
changes; only the rendering is new.

Differences from the conference rendering, which used the default matplotlib
tab10 cycle, a boxed frame, a dashed vertical rule per peak and a red cross:

  * validated colourblind-safe categorical trio
  * one marker shape and one dash pattern per analyte, so the three curves stay
    separable in greyscale print and under CVD
  * markers thinned to every 80th sample instead of every 50th
  * peaks flagged with an open ring and a single drift annotation rather than a
    dashed rule and a red cross per curve
  * curves direct-labelled at their right end, where they are well separated
    (the palette carries a sub-3:1 contrast slot, which obliges visible labels)
  * y-axis limits tightened to the resonance region; box frame removed

NOTE ON UNITS: the `lambda` column of data.xlsx is in units of 100 nm and is
converted to nanometres here.  Confirmed by the Sellmeier index of fused silica,
which over 500-800 nm gives n = 1.462-1.453, matching the dataset's Re(n_eff) of
1.42-1.46; at 5-8 um it gives 1.34-0.64 and at 5-8 nm about 1.0.

NOTE ON REPRODUCIBILITY: src/sensitivity.py imports `train_best_svr_bayesian`
from src/evaluate_logo.py, but that function is absent from the current working
tree -- evaluate_logo.py was later overwritten with an ANN-only evaluator, so
src/sensitivity.py and src/explainability.py no longer import.  The search space
below is the one recovered from commit 8bf0737 ("feat: implement Bayesian
Optimization for SVR tuning"), and it reproduces the published result exactly:
peak shifts of 0.2793 and 0.2733 dataset units, mean 0.2763, matching
spectral_sensitivity_results.csv.  Pass --fixed to refit instead with the static
hyperparameters kept in src/robustness.py (RBF, C=100, gamma=0.1, epsilon=0.01),
which are ~10x faster but give a mean shift of 0.297.

Usage:  python fig3_spectral_response.py [--out ../figs/fig3_spectral_response.pdf]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR

from sivp_style import (
    CATEGORICAL,
    COLUMN_W,
    DASHES,
    INK,
    INK_MUTED,
    MARKERS,
    apply_style,
    save,
    strip_spines,
)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_data, preprocess_data  # noqa: E402

DEFAULT_OUT = os.path.join(HERE, "..", "figs", "fig3_spectral_response.pdf")

# Static fallback (src/robustness.py).
FIXED_PARAMS = dict(kernel="rbf", C=100, gamma=0.1, epsilon=0.01)
CONFIG_COLS = ["Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]


def train_best_svr_bayesian(X_train, y_train):
    """Bayesian hyperparameter search, recovered verbatim from commit 8bf0737."""
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Real

    search_spaces = {
        "C": Real(1, 2000, prior="log-uniform"),
        "gamma": Real(1e-4, 1e0, prior="log-uniform"),
        "epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
        "kernel": Categorical(["rbf"]),
    }
    opt = BayesSearchCV(
        SVR(),
        search_spaces,
        n_iter=32,
        cv=3,
        n_jobs=-1,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    opt.fit(X_train, y_train)
    return opt.best_estimator_


def main(out, fixed):
    apply_style()

    df = load_data(os.path.join(PROJECT_ROOT, "data", "data.xlsx"))
    X_scaled, y, _groups, scaler = preprocess_data(df)

    if fixed:
        model = SVR(**FIXED_PARAMS).fit(X_scaled, y)
    else:
        model = train_best_svr_bayesian(X_scaled, y)
    p = model.get_params()
    print(f"SVR: C={p['C']:.4g} gamma={p['gamma']:.4g} epsilon={p['epsilon']:.4g}")

    base_config = df[CONFIG_COLS].iloc[0].to_dict()
    analytes = sorted(df["Analyte"].unique())
    NM = 100.0  # dataset lambda unit -> nanometres
    wl = np.linspace(df["lambda"].min(), df["lambda"].max(), 1000)

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.55))

    peaks = []
    for i, analyte in enumerate(analytes):
        scan = pd.DataFrame(
            [dict(Analyte=analyte, **{"lambda": w}, **base_config) for w in wl]
        )[X_scaled.columns]
        pred = model.predict(scaler.transform(scan))

        colour = CATEGORICAL[i % len(CATEGORICAL)]
        ax.plot(
            wl * NM,
            pred,
            color=colour,
            linewidth=1.4,
            dashes=DASHES[i % len(DASHES)],
            zorder=3,
        )
        ax.plot(
            wl[::80] * NM,
            pred[::80],
            linestyle="none",
            marker=MARKERS[i % len(MARKERS)],
            markersize=3.4,
            markerfacecolor="white",
            markeredgecolor=colour,
            markeredgewidth=0.9,
            zorder=4,
        )

        k = int(np.argmax(pred))
        peaks.append((wl[k] * NM, pred[k]))
        ax.plot(
            wl[k] * NM,
            pred[k],
            marker="o",
            markersize=6.5,
            markerfacecolor="none",
            markeredgecolor=colour,
            markeredgewidth=1.1,
            zorder=5,
        )
        # Direct label at the right end of each curve, where the three are well
        # separated -- identity never rests on hue alone.
        ax.annotate(
            f"$n_a = {analyte:.2f}$",
            xy=(wl[-1] * NM, pred[-1]),
            xytext=(4, -1),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color=colour,
        )
        print(f"analyte {analyte}: resonance at {wl[k]*NM:.2f} nm")

    shifts = np.diff([p[0] for p in peaks])
    print("peak shifts (nm):", np.round(shifts, 2), "mean:", round(shifts.mean(), 2))
    print(f"sensitivity: {shifts.mean()/0.01:.0f} nm/RIU")

    ax.set_xlim(485, 905)
    ax.set_ylim(6.55, 7.52)
    ax.set_xticks([500, 550, 600, 650, 700, 750, 800])
    ax.set_xlabel(r"Wavelength $\lambda$ (nm)", color=INK)
    ax.set_ylabel(r"$\log_{10}(L_c \times 10^{8})$", color=INK)

    # One annotation for the whole redshift, instead of a rule per curve.
    y_bar = 7.40
    ax.annotate(
        "",
        xy=(peaks[-1][0], y_bar),
        xytext=(peaks[0][0], y_bar),
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, linewidth=0.7),
    )
    ax.text(
        (peaks[0][0] + peaks[-1][0]) / 2,
        y_bar + 0.02,
        "resonance redshift",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color=INK_MUTED,
    )
    strip_spines(ax)

    save(fig, out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument(
        "--fixed",
        action="store_true",
        help="skip the Bayesian search and use src/robustness.py hyperparameters",
    )
    a = p.parse_args()
    main(a.out, a.fixed)
