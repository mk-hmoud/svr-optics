"""Shared plotting style for the SIVP journal figures.

Every figure in the journal version is regenerated from the raw data with this
module, so the SIVP artwork is an independent rendering rather than a reuse of
the SIU conference plots.  The palette, marker set and axis conventions here are
deliberately different from the conference figures.

Palette provenance
------------------
Categorical slots are validated colourblind-safe on an all-pairs basis
(OKLab CVD deltaE 9.1, normal-vision deltaE 22.9, both clear of the floors).
Because two of the three slots sit below 3:1 contrast on a white surface, every
categorical figure also carries secondary encoding -- distinct marker shapes,
distinct dash patterns and direct end-of-curve labels -- so identity never rests
on hue alone.  This also keeps the figures readable in greyscale print.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- categorical slots (identity) -------------------------------------------
VIOLET = "#4a3aa7"
AQUA = "#1baf7a"
AMBER = "#eda100"
CATEGORICAL = [VIOLET, AQUA, AMBER]

# --- single-series fill (magnitude bar charts get ONE colour, not a ramp) ----
SERIES = VIOLET
SERIES_ACCENT = "#e34948"  # used only to mark the single value under discussion

# --- ink -------------------------------------------------------------------
INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#dcdcdc"

# secondary encoding, applied in the same order as CATEGORICAL
MARKERS = ["o", "s", "^"]
DASHES = [(None, None), (5.5, 2.0), (1.5, 1.6)]


def apply_style():
    """Install the journal figure defaults."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",  # solid hairline grid, never dashed
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
        }
    )


def strip_spines(ax, keep=("left", "bottom")):
    """Drop the box frame; keep only the axes that carry a scale."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# Column widths of the sn-jnl [iicol] layout, in inches.
COLUMN_W = 3.35
FULLWIDTH_W = 6.95


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")
