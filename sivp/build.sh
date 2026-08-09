#!/usr/bin/env bash
# Build the SIVP manuscript and its Online Supplement.
#
# sn-jnl.cls needs cuted, wrapfig, threeparttable, vruler and appendix. A full
# TeX Live or Overleaf already has them; this repo carries a local copy in
# .texmf/ so the build works on a minimal install too.
#
#   ./build.sh            # build both
#   ./build.sh main       # build one
set -euo pipefail

cd "$(dirname "$0")"

LOCAL=".texmf"
if [ -d "$LOCAL" ]; then
  for d in "$LOCAL"/*/; do TEXINPUTS="${TEXINPUTS:-}:$PWD/${d%/}"; done
  export TEXINPUTS=".:${TEXINPUTS#:}:"
fi

DOCS=("$@"); [ ${#DOCS[@]} -eq 0 ] && DOCS=(main supplement)

for doc in "${DOCS[@]}"; do
  echo "=== $doc"
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
  bibtex "$doc" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null 2>&1 || true

  if [ ! -f "$doc.pdf" ]; then
    echo "  FAILED - no PDF produced; see $doc.log"
    grep -E "^! " -A3 "$doc.log" | head -20
    exit 1
  fi
  printf '  %s pages | %s errors | %s overfull\n' \
    "$(pdfinfo "$doc.pdf" | awk '/^Pages/{print $2}')" \
    "$(grep -cE '^! ' "$doc.log" || true)" \
    "$(grep -cE '^Overfull' "$doc.log" || true)"
done
