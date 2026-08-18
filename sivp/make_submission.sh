#!/usr/bin/env bash
# Build a submission-ready copy with SIVP's requested file names.
#
# The working repo uses descriptive figure names; the journal asks for "Fig" +
# number ("Fig1.eps") and "ESM_n" for supplementary files. The internal names
# also no longer match the printed numbers -- fig5_timing.pdf prints as Fig. 3,
# and three figures moved into the supplement -- so this maps by printed number,
# not by filename.
set -euo pipefail
cd "$(dirname "$0")"
OUT=submission
rm -rf "$OUT"; mkdir -p "$OUT"

# printed number -> source file
cp figs/fig2_shap_importance.pdf  "$OUT/Fig2.pdf"   # main   Fig. 2
cp figs/fig5_timing.pdf           "$OUT/Fig3.pdf"   # main   Fig. 3
cp figs/figS1_logo_folds.pdf      "$OUT/FigS1.pdf"  # ESM_1  Fig. S1
cp figs/fig4_robustness.pdf       "$OUT/FigS2.pdf"  # ESM_1  Fig. S2
cp figs/fig3_spectral_response.pdf "$OUT/FigS3.pdf" # ESM_1  Fig. S3
# Fig. 1 is inline TikZ and has no file.

# Rewrite the \includegraphics paths to the renamed files.
sed -e 's|figs/fig2_shap_importance\.pdf|Fig2.pdf|' \
    -e 's|figs/fig5_timing\.pdf|Fig3.pdf|' main.tex > "$OUT/main.tex"
sed -e 's|figs/figS1_logo_folds\.pdf|FigS1.pdf|' \
    -e 's|figs/fig4_robustness\.pdf|FigS2.pdf|' \
    -e 's|figs/fig3_spectral_response\.pdf|FigS3.pdf|' supplement.tex > "$OUT/ESM_1.tex"

cp refs.bib sn-jnl.cls sn-basic.bst "$OUT/"

# Build in place so the PDFs match the renamed sources.
if [ -d .texmf ]; then
  for d in .texmf/*/; do TEXINPUTS="${TEXINPUTS:-}:$PWD/${d%/}"; done
  export TEXINPUTS=".:${TEXINPUTS#:}:"
fi
cd "$OUT"
for doc in main ESM_1; do
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
  bibtex "$doc" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
  printf '%-12s %s pages\n' "$doc" "$(pdfinfo "$doc.pdf" | awk '/^Pages/{print $2}')"
done
rm -f *.aux *.log *.out *.blg
