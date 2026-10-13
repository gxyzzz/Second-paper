from __future__ import annotations

from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# Okabe-Ito chromatic system for semantic consistency.
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"
OI_ORANGE = "#E69F00"
OI_SKY = "#56B4E9"
OI_PURPLE = "#CC79A7"

TEXT = "#202020"
NEUTRAL = "#6E6E6E"
NEUTRAL_DARK = "#4A4A4A"
NEUTRAL_LIGHT = "#B9B9B9"
GRID = "#E4E4E4"
PAPER = "#FFFFFF"

BACKBONE = NEUTRAL
COLIFT = OI_BLUE
PDBR = OI_VERMILLION
VALIDATION = OI_BLUE
TEST = OI_VERMILLION
TRUE_USER = OI_VERMILLION
SHUFFLED = NEUTRAL

BLUE_SEQ = LinearSegmentedColormap.from_list(
    "okabe_blue_seq", ["#F7FBFD", "#DCEEF6", "#91C8E3", OI_BLUE]
)
VERMILLION_SEQ = LinearSegmentedColormap.from_list(
    "okabe_vermillion_seq", ["#FFF9F4", "#F8DFCF", "#E9A779", OI_VERMILLION]
)


def _pick_serif() -> str:
    for family in ("Times New Roman", "STIXGeneral", "Liberation Serif", "DejaVu Serif"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
            return family
        except ValueError:
            continue
    return "DejaVu Serif"


SERIF = _pick_serif()

BASE_RC = {
    "font.family": SERIF,
    "font.size": 8.0,
    "axes.labelsize": 8.4,
    "axes.titlesize": 8.4,
    "axes.labelcolor": TEXT,
    "axes.edgecolor": NEUTRAL_DARK,
    "axes.linewidth": 0.65,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 7.4,
    "legend.frameon": False,
    "lines.linewidth": 1.45,
    "lines.markersize": 4.8,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
}


def paper_style():
    """Scoped manuscript style. Figure geometry is always chosen per plot."""
    return mpl.rc_context(rc=BASE_RC)


def clean_axes(ax, grid_axis: str | None = None, *, hide_left: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
        ax.set_axisbelow(True)


def save_figure(fig, outbase: Path | str, *, dpi: int = 400) -> None:
    """Export vector PDF and 400-dpi PNG without changing physical dimensions."""
    outbase = Path(outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix(".pdf"), facecolor=PAPER, dpi=dpi)
    fig.savefig(outbase.with_suffix(".png"), facecolor=PAPER, dpi=dpi)
