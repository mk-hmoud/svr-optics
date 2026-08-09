# SIVP submission package

Journal version of the SIU 2026 conference paper, ported to the official Springer
Nature template for *Signal, Image and Video Processing*.

## Contents

| File | What it is |
|---|---|
| `main.tex` | Main manuscript. `[iicol]` two-column, 10 pages. |
| `supplement.tex` | Online Supplement, 7 pages, single column. |
| `refs.bib` | Shared bibliography for both documents. |
| `sn-jnl.cls`, `sn-*.bst` | Official Springer Nature template files (Dec 2024, v3.1). |
| `scripts/` | Figure regeneration: one script per figure, plus shared style. |
| `figs/` | Generated figure PDFs, committed so the manuscript builds without Python. |

## Building

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
pdflatex supplement && bibtex supplement && pdflatex supplement && pdflatex supplement
```

The class pulls in `cuted`, `wrapfig`, `threeparttable`, `vruler` and `appendix`.
A full TeX Live or Overleaf has all of them; a minimal install may not.

Figure 1 is inline TikZ, because Springer asks for a single `.tex` with no
`\input`. `scripts/fig1_cross_section.tex` holds the same drawing wrapped in
`standalone` for editing it in isolation — edit whichever you prefer, but the
manuscript reads the inline copy.

## Regenerating figures

```
cd scripts
python fig2_shap_importance.py     # reads feature_importance_ranking.csv
python fig3_spectral_response.py   # refits the SVR; ~2 min (--fixed for a fast path)
python fig4_robustness.py          # reads robustness_volatility_ranking.csv
python figS1_logo_folds.py         # reads results_comparison_bayesian.csv
```

Run them from `scripts/` with the project venv (`../../venv/bin/python`); paths
are relative to the repo root. Each script's docstring records exactly how its
rendering differs from the conference version.

`fig3` refits the surrogate rather than reading a CSV, because the published
spectra were never saved. It uses `train_best_svr_bayesian`, which
`src/sensitivity.py` still imports but which no longer exists in
`src/evaluate_logo.py` — that file was overwritten with an ANN-only evaluator.
The search space was recovered from commit `8bf0737` and reproduces the published
peak shifts (0.2793 / 0.2733, mean 0.2763) exactly.

## Open items for the authors

1. **Wavelength units.** The `lambda` column of `data/data.xlsx` runs 5.0–8.0.
   Silica/gold SPR physics puts this resonance at 500–800 nm, so the column looks
   like units of 100 nm. If that is right, the sensitivity is ~2763 nm/RIU, not
   the 27.63 nm/RIU printed in the conference paper — a factor of 100 — and the
   Monte Carlo jitter is ~127 nm, not 1.27 nm. The conference values are
   reproduced here unchanged and all wavelength-derived quantities are stated in
   dataset units; see the flagged comment block in `main.tex` above
   Sect. 5.5. Resolve before submission.

2. **Monte Carlo peak detection.** `src/robustness.py` finds the resonance by
   `argmax` over a bounded grid. The recorded peak range is 2.96 against a
   3.0-wide grid, so the argmax is hitting the grid boundary in many trials. The
   manuscript reports the numbers as published and flags them as upper bounds
   (Sect. 5.4 and Supplement S6.3). A boundary-rejecting peak finder would settle
   it.

3. **Dropped claim.** The conference paper reported a GPR+GAN LOGO average of
   52.86. No file in the repo reproduces that figure, so the journal version
   reports only the fixed-partition value of 23.844 from
   `results_comparison_final.csv`. Restore the 52.86 if you can locate its
   source.
