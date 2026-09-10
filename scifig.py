"""SciFig Design System (multi-color mode) applied to the CPD-TDCC figure set.

Tokens are the canonical values from the "SciFig Design System" project, not
local inventions. Two colour vocabularies live there and they do different jobs:

* **Data figures** (these ones) take series colours from the Okabe-Ito
  categorical palette and the ``SEMANTIC`` aliases -- so ``excitatory`` is
  vermillion in every figure of the paper, ``inhibitory`` is blue, and so on.
* **Diagram figures** (Figure 1, the pipeline schematic) take their stage
  containers from ``THEMES``. Figures 2-6 borrow only the pale ``fill`` tier,
  for washes and state bands, which is what visually ties them to Figure 1.

Point sizes in the style sheet are specified at final print size; see SCALE.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

__all__ = [
    'COLORS', 'SEMANTIC', 'CMAPS', 'FIGSIZE', 'THEMES', 'ACTIVITY', 'DETECT',
    'SEGMENT', 'CAUSAL', 'STRUCTURE', 'EI_PAIR', 'METHOD_PAIR',
    'CONNECTION_PAIR', 'STATE_BANDS', 'SCALE', 'PANEL_LABEL_SIZE', 'tint',
    'use_scifig', 'panel_labels', 'state_strip', 'highlight_span',
    'hist_with_kde', 'ftest_style',
]

# --- Okabe-Ito categorical palette (SciFig COLORS) -------------------------
COLORS = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky': '#56B4E9',
    'green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'gray': '#7F7F7F',
    'lightgray': '#BFBFBF',
}

# --- Fixed neuroscience aliases (SciFig SEMANTIC) --------------------------
SEMANTIC = {
    'excitatory': COLORS['vermillion'],
    'inhibitory': COLORS['blue'],
    'modulatory': COLORS['purple'],
    'input': COLORS['gray'],
    'output': COLORS['black'],
    'highlight': COLORS['orange'],
}

CMAPS = {
    'sequential': 'viridis',
    'diverging': 'PRGn',
    'cyclic': 'twilight',
}

FIGSIZE = {
    'single': (3.50, 2.60),
    'single_tall': (3.50, 3.50),
    'onehalf': (5.50, 3.60),
    'double': (7.20, 3.60),
    'double_tall': (7.20, 5.40),
}

# --- Multi-hue stage themes (SciFig THEMES) --------------------------------
THEMES = {
    'blue': dict(fill='#DCEEF9', edge='#3E7CA6', accent='#1B4F72'),
    'green': dict(fill='#E1F3E1', edge='#4C9A4C', accent='#1E5B1E'),
    'purple': dict(fill='#F0E6F6', edge='#8B5FA3', accent='#5B3A70'),
    'orange': dict(fill='#FCE9D6', edge='#E0862F', accent='#9C5518'),
    'teal': dict(fill='#DFF2EF', edge='#3E9C8F', accent='#1F5C53'),
    'pink': dict(fill='#FBE6EC', edge='#C96A87', accent='#833650'),
}

# Figure 1's stage order, kept fixed across the paper.
ACTIVITY = THEMES['blue']       # 1. neural activity
DETECT = THEMES['green']        # 2. change-point detection
SEGMENT = THEMES['orange']      # 3. segmentation
CAUSAL = THEMES['purple']       # 4. causal inference
STRUCTURE = THEMES['teal']      # 5. structural connectivity

# --- Within-panel role pairs ----------------------------------------------
# Each cleared by dataviz/scripts/validate_palette.js --pairs all --mode light.
EI_PAIR = (SEMANTIC['excitatory'], SEMANTIC['inhibitory'])   # ΔE 21.9 protan
METHOD_PAIR = (COLORS['black'], COLORS['green'])             # raw / CPD, ΔE 61.9
CONNECTION_PAIR = (COLORS['black'], COLORS['purple'])        # unconn / conn, ΔE 64.0

# SciFig's mid gray (#7F7F7F) is deliberately NOT used as a series colour here:
# against Okabe-Ito green it collapses to ΔE 3.3 under deuteranopia. Black --
# SEMANTIC['output'] -- is the baseline ink instead.

# Print scale: the style sheet is specified at FIGSIZE['double'] (7.2 in) and
# these figures are drawn at 16 in, so every point size is multiplied by this.
# Set to 1.0 if the figures are ever rebuilt at FIGSIZE['double'].
SCALE = 2.0

PANEL_LABEL_SIZE = 9 * SCALE

_SCALED_PARAMS = (
    'font.size', 'axes.titlesize', 'axes.labelsize', 'axes.linewidth',
    'xtick.labelsize', 'ytick.labelsize', 'xtick.major.width',
    'ytick.major.width', 'xtick.minor.width', 'ytick.minor.width',
    'xtick.major.size', 'ytick.major.size', 'lines.linewidth',
    'lines.markersize', 'patch.linewidth', 'legend.fontsize',
)


def tint(color, amount):
    """Blend a hex colour toward white. ``amount`` 0 = unchanged, 1 = white."""
    color = color.lstrip('#')
    rgb = (int(color[i:i + 2], 16) for i in (0, 2, 4))
    return '#%02x%02x%02x' % tuple(round(c + (255 - c) * amount) for c in rgb)


# W1..Wn bands: states are ordered in time, so the order lives in the
# lightness. Derived from the teal theme; accent-on-band clears 4.5:1.
STATE_BANDS = tuple(tint(STRUCTURE['edge'], a) for a in (0.85, 0.65, 0.45, 0.25))


def use_scifig(scale=SCALE):
    """Apply the SciFig style sheet, rescaled from print size to this canvas."""
    plt.style.use(Path(__file__).with_name('scifig.mplstyle'))
    plt.rcParams.update({k: plt.rcParams[k] * scale for k in _SCALED_PARAMS})
    plt.rcParams['image.cmap'] = CMAPS['sequential']


def panel_labels(fig, labels):
    """Draw panel letters. ``labels`` is an iterable of ``(x, y, text)``."""
    for x, y, text in labels:
        fig.text(x, y, text, fontsize=PANEL_LABEL_SIZE, fontweight='bold',
                 color=COLORS['black'])


def state_strip(ax, n_states, ymin=-0.5, ymax=0.5, fontsize=24,
                hide_spines=('left',)):
    """The W1..Wn connectivity-state bar that sits above the time axes."""
    for i in range(n_states):
        ax.fill_between([i, i + 1], ymin, ymax, color=STATE_BANDS[i], lw=0,
                        zorder=0)
        ax.text(i + 0.5, 0.0, r'$\mathbf{W}_{%d}$' % (i + 1), fontsize=fontsize,
                fontweight='bold', color=COLORS['black'], ha='center',
                va='center')
    for spine in hide_spines:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, n_states)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.set_yticks([])


def highlight_span(ax, x0, x1, theme=SEGMENT):
    """Wash marking the analysed window, painted behind the data as in fig1."""
    return ax.axvspan(x0, x1, color=theme['fill'], lw=0, zorder=0)


def hist_with_kde(ax, cached_histogram, legend_title='connection'):
    """Connection histograms with their KDE overlay.

    Colour follows the ``connection`` value, not the row order, so a panel
    whose categories come back in a different order still reads the same way.
    """
    handles = []
    for index, (connection, density, edges, kde_x, kde_y) in enumerate(cached_histogram):
        try:
            color = CONNECTION_PAIR[int(connection)]
        except (TypeError, ValueError, IndexError):
            color = CONNECTION_PAIR[index % len(CONNECTION_PAIR)]
        fill = tint(color, 0.86)
        ax.stairs(density, edges, fill=True, facecolor=fill, edgecolor=color,
                  lw=1.0 * SCALE, zorder=1)
        ax.plot(kde_x, kde_y, lw=1.0 * SCALE, color=color, zorder=2)
        handles.append(Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor=color,
                                 label=str(connection)))
    ax.legend(handles=handles, title=legend_title)


def ftest_style():
    """Marker/line kwargs for the F-statistic traces."""
    return dict(color=COLORS['green'], marker='o', ms=2.0 * SCALE, mfc='white',
                mew=0.6 * SCALE, lw=0.8 * SCALE, clip_on=False)
