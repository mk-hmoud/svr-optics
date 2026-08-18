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

# EPS, not PDF: Snapp accepts PNG, JPEG, SVG or EPS and does not list PDF, and
# the artwork guidelines name EPS as the preferred vector format. pdflatex reads
# EPS via epstopdf, which the class already pulls in through graphicx.
# printed number -> source file
cp figs/fig2_shap_importance.eps   "$OUT/Fig2.eps"   # main   Fig. 2
cp figs/fig5_timing.eps            "$OUT/Fig3.eps"   # main   Fig. 3
cp figs/figS1_logo_folds.eps       "$OUT/FigS1.eps"  # ESM_1  Fig. S1
cp figs/fig4_robustness.eps        "$OUT/FigS2.eps"  # ESM_1  Fig. S2
cp figs/fig3_spectral_response.eps "$OUT/FigS3.eps"  # ESM_1  Fig. S3
# Fig. 1 is inline TikZ and has no file.

# Rewrite the \includegraphics paths to the renamed files.
sed -e 's|figs/fig2_shap_importance\.pdf|Fig2.eps|' \
    -e 's|figs/fig5_timing\.pdf|Fig3.eps|' \
    -e 's|\\usepackage{graphicx}|\\usepackage{graphicx}\\usepackage{epstopdf}|' main.tex > "$OUT/main.tex"
sed -e 's|figs/figS1_logo_folds\.pdf|FigS1.eps|' \
    -e 's|figs/fig4_robustness\.pdf|FigS2.eps|' \
    -e 's|figs/fig3_spectral_response\.pdf|FigS3.eps|' \
    -e 's|\\usepackage{graphicx}|\\usepackage{graphicx}\\usepackage{epstopdf}|' supplement.tex > "$OUT/ESM_1.tex"

cp sn-jnl.cls sn-basic.bst "$OUT/"

# Strip whole-line comments from the submitted source. The repo keeps them --
# they record why things are the way they are -- but they are working notes, not
# manuscript content, and Snapp receives the source, not just the PDF. Only lines
# whose first non-space character is % are removed, so trailing %-continuations
# and escaped \% are untouched.
sed -e '/^[[:space:]]*%/d' refs.bib > "$OUT/refs.bib"
sed -i -e '/^[[:space:]]*%/d' "$OUT/main.tex"
sed -i -e '/^[[:space:]]*%/d' "$OUT/ESM_1.tex"

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
rm -f *.aux *.log *.out *.blg *-eps-converted-to.pdf

# Snapp takes the manuscript as a .zip of LaTeX source and compiles it itself,
# so the archive must build standalone. ESM_1 is deliberately excluded -- a
# second \documentclass in the same archive would confuse their compiler; it is
# uploaded as supplementary material instead.
zip -q manuscript.zip main.tex main.bbl refs.bib sn-jnl.cls sn-basic.bst Fig2.eps Fig3.eps
echo "manuscript.zip  $(unzip -l manuscript.zip | tail -1 | awk '{print $2}') files"
