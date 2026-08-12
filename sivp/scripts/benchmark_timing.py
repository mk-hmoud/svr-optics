"""Measure training and inference cost for every model in the comparison.

All models are timed on the same machine, the same data and the same CPU-only
configuration, so the numbers are comparable to each other. They are not
portable: absolute values depend on hardware, and the point of the table is the
ratio between models, not the seconds.

Writes timing_results.csv next to the repository root and prints a summary.

    python benchmark_timing.py [--out ../../timing_results.csv] [--quick]

--quick shortens the two expensive runs (ANN, WGAN-GP) for a smoke test; the
reported numbers are then not usable in the paper.
"""

import argparse
import os
import platform
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data import load_data, preprocess_data  # noqa: E402

# Optimised configuration reported in the manuscript.
SVR_PARAMS = dict(kernel="rbf", C=2000, gamma=0.0172, epsilon=1e-4)


def timed(fn, repeats=1):
    """Run fn repeats times, return (median seconds, result of last run)."""
    times, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def bench_svr(X, y, quick):
    from sklearn.svm import SVR

    fit_s, model = timed(lambda: SVR(**SVR_PARAMS).fit(X, y), repeats=5)
    pred_s, _ = timed(lambda: model.predict(X), repeats=20)
    return fit_s, pred_s / len(X)


def bench_svr_search(X, y, quick):
    """Full Bayesian search, i.e. what the surrogate actually costs to produce."""
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Real
    from sklearn.svm import SVR

    def run():
        opt = BayesSearchCV(
            SVR(),
            {
                "C": Real(1, 2000, prior="log-uniform"),
                "gamma": Real(1e-4, 1e0, prior="log-uniform"),
                "epsilon": Real(1e-4, 1e-1, prior="log-uniform"),
                "kernel": Categorical(["rbf"]),
            },
            n_iter=4 if quick else 32,
            cv=3,
            n_jobs=-1,
            scoring="neg_mean_squared_error",
            random_state=42,
        )
        opt.fit(X, y)
        return opt

    return timed(run, repeats=1)[0]


def bench_gpr(X, y, quick):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-3)

    fit_s, model = timed(
        lambda: GaussianProcessRegressor(kernel=kernel, normalize_y=True).fit(X, y),
        repeats=3,
    )
    pred_s, _ = timed(lambda: model.predict(X), repeats=10)
    return fit_s, pred_s / len(X)


def bench_ann(X, y, quick):
    import tensorflow as tf

    from src.models.researcher_ann import build_researcher_ann

    epochs = 20 if quick else 500

    def run():
        tf.keras.backend.clear_session()
        m = build_researcher_ann(input_dim=X.shape[1])
        m.compile(optimizer="adam", loss="mse")
        m.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
        return m

    fit_s, model = timed(run, repeats=1)
    Xa = np.asarray(X, dtype="float32")
    model.predict(Xa, verbose=0)  # warm up the graph before timing
    pred_s, _ = timed(lambda: model.predict(Xa, verbose=0), repeats=5)
    tf.keras.backend.clear_session()
    return fit_s, pred_s / len(X)


# The generator in src/wgan_paper.py is built for 8 columns: the seven features
# used in src/find_gan_contradictions.py plus the target. That is a wider feature
# set than the regressors use, so the GAN input is rebuilt here rather than
# reusing the preprocessed X.
GAN_FEATURES = ["Analyte", "Re(eff)", "lambda", "Pitch (um)", "d1 (um)", "d2 (um)", "d3 (um)"]


def bench_wgan(df, quick):
    """Cost of producing the synthetic samples, before any regressor is fitted."""
    from sklearn.preprocessing import MinMaxScaler

    from src.wgan_paper import train_wgan_paper

    Xg = MinMaxScaler().fit_transform(df[GAN_FEATURES])
    yg = np.log10(np.clip(df["loss"] * 10**8, a_min=1e-10, a_max=None)).values
    yg = MinMaxScaler().fit_transform(yg.reshape(-1, 1))
    combined = np.hstack([Xg, yg])

    epochs = 20 if quick else 2000
    fit_s, _ = timed(lambda: train_wgan_paper(combined, epochs=epochs), repeats=1)
    return fit_s


def main(out, quick, only=None, append=False):
    df = load_data(os.path.join(PROJECT_ROOT, "data", "data.xlsx"))
    X, y, _groups, _scaler = preprocess_data(df)
    print(f"dataset: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"host   : {platform.processor() or platform.machine()}, "
          f"{os.cpu_count()} threads, CPU only\n")

    rows = []
    want = (lambda name: only is None or only == name)

    if want("svr"):
        print("SVR (fixed hyperparameters) ...")
        fit, pred = bench_svr(X, y, quick)
        rows.append(("SVR", "fit only", fit, pred))

        print("SVR (Bayesian search, 32 evaluations) ...")
        search = bench_svr_search(X, y, quick)
        rows.append(("SVR", "with hyperparameter search", search, pred))

    if want("gpr"):
        print("GPR ...")
        fit, pred = bench_gpr(X, y, quick)
        rows.append(("GPR", "fit only", fit, pred))

    # TensorFlow and torch are each run in their own process (--only ann /
    # --only wgan): loading both in one interpreter reliably gets the run
    # OOM-killed on this machine.
    if want("ann"):
        print("ANN (500 epochs) ...")
        fit, pred = bench_ann(X, y, quick)
        rows.append(("ANN", "500 epochs", fit, pred))

    if want("wgan"):
        print("WGAN-GP (2000 epochs) ...")
        gan = bench_wgan(df, quick)
        rows.append(("WGAN-GP", "2000 epochs, generator only", gan, float("nan")))

    res = pd.DataFrame(rows, columns=["model", "stage", "train_s", "predict_s_per_sample"])
    if append and os.path.exists(out):
        res = pd.concat([pd.read_csv(out).drop(columns=["predict_us_per_sample"],
                                               errors="ignore"), res],
                        ignore_index=True)
    res["predict_us_per_sample"] = res["predict_s_per_sample"] * 1e6
    res.to_csv(out, index=False)

    print(f"\n{'model':10s} {'stage':30s} {'train (s)':>11s} {'predict (us/sample)':>21s}")
    for _, r in res.iterrows():
        us = "n/a" if pd.isna(r.predict_us_per_sample) else f"{r.predict_us_per_sample:.2f}"
        print(f"{r.model:10s} {r.stage:30s} {r.train_s:11.2f} {us:>21s}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(PROJECT_ROOT, "timing_results.csv"))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", choices=["svr", "gpr", "ann", "wgan"])
    p.add_argument("--append", action="store_true")
    a = p.parse_args()
    main(a.out, a.quick, a.only, a.append)
