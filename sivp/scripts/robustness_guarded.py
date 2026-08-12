"""Re-run the Monte Carlo tolerance analysis with a boundary-guarded peak finder.

src/robustness.py locates the resonance with argmax over a bounded wavelength
grid. When a perturbed geometry pushes the resonance outside that grid the argmax
returns a grid endpoint, which is not a resonance at all. In the published run
the peak range was 2.96 against a 3.00-wide grid, i.e. the endpoints were being
returned often enough to dominate the statistic.

This script repeats the experiment, discarding any trial whose argmax lands in
the outermost `--edge` fraction of the grid, and reports the raw and guarded
statistics side by side together with the rejection rate. Both the original fixed
hyperparameters (src/robustness.py) and the Bayesian-optimised ones used
everywhere else in the manuscript are evaluated, so the effect of the guard can
be separated from the effect of the model.

Wavelength is reported in nanometres: the `lambda` column of data.xlsx is in
units of 100 nm, confirmed by the Sellmeier index of fused silica over
500-800 nm (n = 1.462-1.453) matching the dataset's Re(n_eff) of 1.42-1.46.

    python robustness_guarded.py [--trials 100] [--edge 0.02]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.svm import SVR

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_data, preprocess_data  # noqa: E402

NM_PER_UNIT = 100.0  # dataset lambda unit -> nanometres

MODELS = {
    "published (C=100, g=0.1, eps=0.01)": dict(kernel="rbf", C=100, gamma=0.1, epsilon=0.01),
    "Bayesian (C=2000, g=0.0172, eps=1e-4)": dict(kernel="rbf", C=2000, gamma=0.0172, epsilon=1e-4),
}
# src/robustness.py hard-codes {"Pitch (um)": 2.0, "d1": 0.225, "d2": 0.375,
# "d3": 0.175}. Three of those four lie OUTSIDE the sampled design space
# (pitch 0.15-0.24, d1 0.25-0.45, d2 0.55-0.75): 0.225/0.375/0.175 are the
# RADII r1/r2/r3 of Fig. 1 written into DIAMETER columns, and 2.0 is the pitch
# in micrometres written into a column stored in units of 10 um. The tolerance
# analysis was therefore evaluated where the surrogate extrapolates, which is
# what degenerates the peak finder. The baseline is taken from the dataset here.
BASE_LEGACY = {"Analyte": 1.33, "Pitch (um)": 2.0, "d1 (um)": 0.225,
               "d2 (um)": 0.375, "d3 (um)": 0.175}
BASE = None  # set in main() from the dataset
GEOM = ["Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]
ORDER = ["Analyte", "lambda", "Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]


def peak(model, scaler, cfg, wl):
    scan = pd.DataFrame([dict(cfg, **{"lambda": w}) for w in wl])[ORDER]
    pred = model.predict(scaler.transform(scan))
    return int(np.argmax(pred))


def run(model, scaler, wl, levels, trials, edge, rng, features=None):
    """Return {level: (raw_std, guarded_std, n_rejected)} in nanometres."""
    lo, hi = int(len(wl) * edge), int(len(wl) * (1 - edge))
    out = {}
    for lvl in levels:
        idxs = []
        for _ in range(trials):
            cfg = dict(BASE)
            for f in (features or GEOM):
                cfg[f] += rng.normal(0, lvl * BASE[f])
            idxs.append(peak(model, scaler, cfg, wl))
        idxs = np.array(idxs)
        keep = (idxs > lo) & (idxs < hi)
        raw = wl[idxs].std() * NM_PER_UNIT
        guarded = wl[idxs[keep]].std() * NM_PER_UNIT if keep.sum() > 1 else float("nan")
        out[lvl] = (raw, guarded, int((~keep).sum()))
    return out


def main(trials, edge, seed, legacy=False):
    global BASE
    df = load_data(os.path.join(PROJECT_ROOT, "data", "data.xlsx"))
    if legacy:
        BASE = dict(BASE_LEGACY)
    else:
        cfg = df[GEOM].iloc[0].to_dict()
        BASE = {"Analyte": 1.33, **cfg}
    print("baseline geometry:", {k: round(v, 4) for k, v in BASE.items()})
    X, y, _g, scaler = preprocess_data(df)
    wl = np.linspace(df["lambda"].min(), df["lambda"].max(), 500)
    levels = [0.01, 0.03, 0.05]

    print(f"grid: {wl.min()*NM_PER_UNIT:.0f}-{wl.max()*NM_PER_UNIT:.0f} nm, "
          f"{trials} trials/level, edge guard {edge:.0%}\n")

    for name, params in MODELS.items():
        model = SVR(**params).fit(X, y)
        rng = np.random.default_rng(seed)
        res = run(model, scaler, wl, levels, trials, edge, rng)
        print(name)
        print(f"  {'level':>6} {'raw std':>10} {'guarded':>10} {'rejected':>10}")
        for lvl, (raw, gd, nrej) in res.items():
            gd_s = "n/a" if np.isnan(gd) else f"{gd:8.1f} nm"
            print(f"  {lvl:>5.0%} {raw:8.1f} nm {gd_s:>10} {nrej:>7}/{trials}")

        rng = np.random.default_rng(seed)
        print(f"  per-variable at 3% (guarded):")
        rows = []
        for f in GEOM:
            r = run(model, scaler, wl, [0.03], trials, edge, rng, features=[f])
            raw, gd, nrej = r[0.03]
            rows.append((f, raw, gd, nrej))
        tot = sum(g * g for _, _, g, _ in rows if not np.isnan(g))
        for f, raw, gd, nrej in sorted(rows, key=lambda t: -(0 if np.isnan(t[2]) else t[2])):
            share = "n/a" if np.isnan(gd) or tot == 0 else f"{100*gd*gd/tot:5.1f}%"
            gd_s = "n/a" if np.isnan(gd) else f"{gd:7.2f} nm"
            print(f"    {f:12s} raw {raw:7.2f} nm  guarded {gd_s}  var share {share}"
                  f"  rejected {nrej}/{trials}")
        if name.startswith("Bayesian"):
            out = os.path.join(PROJECT_ROOT, "robustness_guarded_results.csv")
            recs = [{"level": f"{lvl:.0%}", "variable": "all",
                     "jitter_nm": round(res[lvl][1], 3), "rejected": res[lvl][2]}
                    for lvl in levels]
            recs += [{"level": "3%", "variable": f, "jitter_nm": round(gd, 3),
                      "rejected": nrej} for f, _raw, gd, nrej in rows]
            pd.DataFrame(recs).to_csv(out, index=False)
            print(f"  wrote {out}")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--edge", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--legacy", action="store_true",
                   help="use the out-of-domain baseline from src/robustness.py")
    a = p.parse_args()
    main(a.trials, a.edge, a.seed, a.legacy)
