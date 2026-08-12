"""Reproduce the GPR+GAN leave-one-geometry-out average reported at SIU (52.86).

Same computation as src/evaluate_logo_gpr_gan.py, which was never deleted -- only
its output CSV is missing. This runner differs in being resumable and observable:

  * each fold is written to the CSV as soon as it completes, so a crash costs one
    fold rather than the whole run;
  * an existing CSV is read back on startup and completed folds are skipped;
  * output is flushed per fold, so progress is visible while it runs. A plain
    `python script > log` block-buffers stdout, which is why the first attempt
    looked silent and then vanished.

Roughly 7 minutes per fold (1000 WGAN-GP epochs + a GPR fit on ~1384 samples),
so about an hour for all nine from cold.

    python rerun_gpr_gan_logo.py [--out ../../results_logo_gpr_gan.csv]
"""

import argparse
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_data, get_logo_folds  # noqa: E402
from src.wgan_paper import train_wgan_paper, generate_samples_paper  # noqa: E402

FEATURES = ["Analyte", "Re(eff)", "lambda", "Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]
CONFIG = ["Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]


def load(out):
    if os.path.exists(out):
        df = pd.read_csv(out)
        return {int(r.Fold): float(r.GPR_GAN_MSE) for r in df.itertuples()
                if not pd.isna(r.GPR_GAN_MSE)}
    return {}


def save(done, out):
    pd.DataFrame(sorted(done.items()), columns=["Fold", "GPR_GAN_MSE"]).to_csv(out, index=False)


def main(out, epochs, n_synth):
    df = load_data(os.path.join(PROJECT_ROOT, "data", "data.xlsx"))
    X = df[FEATURES]
    y = np.log10(np.clip(df["loss"] * 10**8, a_min=1e-10, a_max=None))

    cfgs = df[CONFIG].drop_duplicates().reset_index(drop=True)
    cfgs["group_id"] = range(len(cfgs))
    groups = df.merge(cfgs, on=CONFIG, how="left")["group_id"].values

    Xs = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=FEATURES)
    folds = list(get_logo_folds(Xs, y, groups))

    kernel = (C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
              + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e-1)))

    done = load(out)
    if done:
        print(f"resuming: folds {sorted(done)} already complete", flush=True)

    for fold, (tr, te) in enumerate(folds, 1):
        if fold in done:
            continue
        t0 = time.perf_counter()
        X_tr, X_te = Xs.iloc[tr], Xs.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]

        real = np.hstack([X_tr.values, y_tr.values.reshape(-1, 1)])
        gen = train_wgan_paper(real, epochs=epochs)
        synth = generate_samples_paper(gen, num_samples=n_synth)
        synth = np.clip(synth, real.min(axis=0), real.max(axis=0))

        X_aug = np.vstack([X_tr.values, synth[:, :-1]])
        y_aug = np.concatenate([y_tr, synth[:, -1]])

        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2,
                                       alpha=0.0, random_state=42)
        try:
            gpr.fit(X_aug, y_aug)
            mse = mean_squared_error(y_te, gpr.predict(X_te.values))
        except Exception as exc:
            print(f"fold {fold}: FAILED ({type(exc).__name__}: {exc})", flush=True)
            mse = float("nan")

        done[fold] = mse
        save(done, out)
        print(f"fold {fold}/9  MSE {mse:12.4f}   [{time.perf_counter()-t0:5.1f}s]", flush=True)

        del gen, gpr, X_aug, y_aug, synth, real
        gc.collect()

    vals = [v for v in done.values() if not np.isnan(v)]
    print(f"\n{len(vals)}/9 folds complete")
    if vals:
        print(f"mean GPR+GAN LOGO MSE = {np.mean(vals):.4f}   (SIU reported 52.86)")
    print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "results_logo_gpr_gan.csv"))
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--synthetic", type=int, default=1000)
    a = p.parse_args()
    main(a.out, a.epochs, a.synthetic)
