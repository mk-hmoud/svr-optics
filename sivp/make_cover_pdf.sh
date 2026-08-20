#!/usr/bin/env bash
# Build cover_letter.pdf from cover_letter.txt.
#
# Snapp takes the cover letter as a file upload, not a text field. The text file
# stays the source of truth; this only typesets it, so edit the .txt and re-run.
set -euo pipefail
cd "$(dirname "$0")"

python3 - <<'PYEOF'
import re
txt = open('cover_letter.txt', encoding='utf-8').read().strip()
paras = [re.sub(r'\s+', ' ', p).strip() for p in txt.split('\n\n') if p.strip()]

# Last two blocks are "Yours sincerely," and the signature; everything between
# the salutation and those is body. Slicing them off by a fixed count silently
# dropped four paragraphs the first time.
salutation = paras[0]
body = paras[1:-2]
valediction = paras[-2]

def esc(t):
    for a, b in [('&', r'\&'), ('%', r'\%'), ('#', r'\#'), ('_', r'\_')]:
        t = t.replace(a, b)
    # Straight quotes render as two closing quotes under fontspec; pair them up.
    parts = t.split('"')
    if len(parts) > 1:
        t = parts[0]
        for i, seg in enumerate(parts[1:]):
            t += ('``' if i % 2 == 0 else "''") + seg
    return t

sign = '\\\\\n'.join(esc(l) for l in txt.split('\n\n')[-1].splitlines())

doc = [
    r'\documentclass[11pt,a4paper]{article}',
    r'\usepackage{fontspec}',
    r'\setmainfont{Liberation Serif}',     # TrueType, present, covers s-cedilla and dotless i
    r'\usepackage[margin=1in]{geometry}',
    r'\usepackage{parskip}',
    r'\pagestyle{empty}',
    r'\begin{document}', '',
    esc(salutation), '',
]
doc += [esc(p) + '\n' for p in body]
doc += ['', esc(valediction), '', r'\vspace{0.5em}',
        r'\noindent ' + sign, r'\end{document}']
open('cover_letter.tex', 'w', encoding='utf-8').write('\n'.join(doc))
PYEOF

xelatex -interaction=nonstopmode cover_letter.tex >/dev/null 2>&1 || true
xelatex -interaction=nonstopmode cover_letter.tex >/dev/null 2>&1 || true
rm -f cover_letter.aux cover_letter.log cover_letter.tex
if [ -f cover_letter.pdf ]; then
  echo "cover_letter.pdf  $(pdfinfo cover_letter.pdf | awk '/^Pages/{print $2}') page(s)"
else
  echo "FAILED"; exit 1
fi
