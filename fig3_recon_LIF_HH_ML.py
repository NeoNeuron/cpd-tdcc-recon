# %%
# New Figure 3: reconstruction examples for all three neuron models (LIF, HH,
# ML), combining the previous Fig4 (LIF only) and Fig5 (HH, ML) into a single
# 3-row (model) x 4-column (panel type) grid:
#   col 1: example dynamics (projection curve + F-statistic, stacked)
#   col 2: CPD-windowed CC histogram
#   col 3: all-data CC histogram
#   col 4: ROC curve (raw data vs CPD-windowed data)
# From old Fig4 this keeps panels B, C, D, H, L (dropping A, the raster plot,
# which has no counterpart for HH/ML); Fig5's A-H are reused in full.
from common import *
from scifig import *
from pdif.myplot import sci_formatter
use_scifig()
path = Path(__file__).parents[0]

# Enlarge all default text (axis labels, tick numbers, legend entries) beyond
# the shared scifig scale, on top of the explicit label-size bumps below.
for _key in ('axes.labelsize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize'):
    plt.rcParams[_key] = plt.rcParams[_key] * 1.3

fig4_data_file = path / 'fig4_data.npz'
fig5_data_file = path / 'fig5_data.npz'
if not fig4_data_file.exists():
    raise SystemExit(f'{fig4_data_file} not found. Run fig4.py first.')
if not fig5_data_file.exists():
    raise SystemExit(f'{fig5_data_file} not found. Run fig5.py first.')

fig4_data = np.load(fig4_data_file, allow_pickle=True)
lif_ts = fig4_data['ts']
lif_projection_curve = fig4_data['projection_curve']
lif_ftest_x = fig4_data['ftest_x']
lif_ftest_f = fig4_data['ftest_f']
lif_histogram_data = fig4_data['histogram_data']  # [0:4]=CPD-windowed (part), [4:8]=all-data
lif_roc_all = fig4_data['roc_all']
lif_roc_part = fig4_data['roc_part']
lif_auc_all = fig4_data['auc_all']
lif_auc_part = fig4_data['auc_part']

fig5_data = np.load(fig5_data_file, allow_pickle=True)
gen_ts = fig5_data['ts']
gen_projection_curves = fig5_data['projection_curves']  # [0]=HH, [1]=ML
gen_ftest_curves = fig5_data['ftest_curves']
gen_histogram_data = fig5_data['histogram_data']  # [0]=HH all, [1]=ML all, [2]=HH part, [3]=ML part
gen_roc_all = fig5_data['roc_all']       # [0]=HH, [1]=ML
gen_roc_part = fig5_data['roc_part']
gen_auc_all = fig5_data['auc_all']
gen_auc_part = fig5_data['auc_part']

# Per-model data assembled row-by-row: (name, ts, proj_curve, ftest_x, ftest_f,
# highlight_end_index, cpd_hist, cpd_hist_xlim, all_hist, all_hist_xlim,
# roc_all, roc_part, auc_all, auc_part)
rows = [
    dict(
        name='LIF', ts=lif_ts, proj=lif_projection_curve,
        ftest_x=lif_ftest_x, ftest_f=lif_ftest_f, highlight_end=499999,
        cpd_hist=lif_histogram_data[0], cpd_xlim=(-10, -4),
        all_hist=lif_histogram_data[4], all_xlim=(-11, -4),
        roc_all=lif_roc_all[0], roc_part=lif_roc_part[0],
        auc_all=float(lif_auc_all[0]), auc_part=float(lif_auc_part[0]),
    ),
    dict(
        name='HH', ts=gen_ts, proj=gen_projection_curves[0],
        ftest_x=gen_ftest_curves[0][0], ftest_f=gen_ftest_curves[0][1], highlight_end=4999,
        cpd_hist=gen_histogram_data[2], cpd_xlim=(-10, -2),
        all_hist=gen_histogram_data[0], all_xlim=(-10, -2),
        roc_all=gen_roc_all[0], roc_part=gen_roc_part[0],
        auc_all=float(gen_auc_all[0]), auc_part=float(gen_auc_part[0]),
    ),
    dict(
        name='ML', ts=gen_ts, proj=gen_projection_curves[1],
        ftest_x=gen_ftest_curves[1][0], ftest_f=gen_ftest_curves[1][1], highlight_end=4999,
        cpd_hist=gen_histogram_data[3], cpd_xlim=(-10, -2),
        all_hist=gen_histogram_data[1], all_xlim=(-10, -2),
        roc_all=gen_roc_all[1], roc_part=gen_roc_part[1],
        auc_all=float(gen_auc_all[1]), auc_part=float(gen_auc_part[1]),
    ),
]

#%%
fig = plt.figure(figsize=(25, 12.5))

# Column widths follow a 2.5:1:1:1 ratio (time-series : CPD-hist : all-hist : ROC).
# Figure widened (16in -> 20in) at the same fractional layout so every panel
# gets more absolute room for its ticks/legend without touching proportions.
# GAP (wspace-equivalent) and the row spacing below (hspace-equivalent) were
# both widened further, with figure height increased to 12.5in to absorb it
# without shrinking the rows.
LEFT, RIGHT, GAP = 0.060, 0.985, 0.045
_unit = (RIGHT - LEFT - 3 * GAP) / 5.5
col_ts = (LEFT, LEFT + 2.5 * _unit)
col_cpd = (col_ts[1] + GAP, col_ts[1] + GAP + _unit)
col_all = (col_cpd[1] + GAP, col_cpd[1] + GAP + _unit)
col_roc = (col_all[1] + GAP, col_all[1] + GAP + _unit)

row_top = [0.92, 0.6133, 0.3067]
row_bottom = [0.6733, 0.3667, 0.06]

col_headers = [
    (sum(col_ts) / 2, 'Example dynamics'),
    (sum(col_cpd) / 2, 'CPD-windowed CC'),
    (sum(col_all) / 2, 'All-data CC'),
    (sum(col_roc) / 2, 'ROC'),
]
for x, text in col_headers:
    fig.text(x, 0.995, text, fontsize=22, fontweight='bold', ha='center')

# W1-W4 connectivity-state strip, restored above panel A: the LIF example
# (unlike HH/ML) is drawn from a single recording spanning 4 subnetwork
# windows, matching the 4 CC histograms/ROC points computed from it elsewhere.
gs = fig.add_gridspec(1, 1, left=col_ts[0], right=col_ts[1], top=0.975, bottom=0.945)
ax_strip = fig.add_subplot(gs[0, 0])
state_strip(ax_strip, 4, fontsize=20)

panel_letters = 'ABCDEFGHIJKL'
letter_i = 0
for row, top, bottom in zip(rows, row_top, row_bottom):
    fig.text(0.00, (top + bottom) / 2, row['name'], fontsize=24, fontweight='bold',
              va='center', ha='left', rotation=0)

    # column 1: projection curve (top) + F-statistic (bottom), stacked
    gd = fig.add_gridspec(2, 1, left=col_ts[0], right=col_ts[1], top=top, bottom=bottom,
                           hspace=0.3, height_ratios=[1, 0.5])
    ax_proj, ax_ftest = (fig.add_subplot(gd[0, 0]), fig.add_subplot(gd[1, 0]))
    ax_proj.plot(row['ts'], row['proj'], color=COLORS['green'], lw=0.5)
    ax_proj.set_rasterized(True)
    highlight_span(ax_proj, row['ts'][0], row['ts'][row['highlight_end']])
    ax_proj.set_xlim(0, 4e5)
    ax_proj.set_ylim(0, row['proj'].max())
    ax_proj.set_ylabel(r'$\left|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle\right|$', fontsize=17)
    ax_proj.set_xticks([0, 1e5, 2e5, 3e5, 4e5], ['', '', '', '', ''])
    ax_ftest.plot(row['ftest_x'], row['ftest_f'], **ftest_style())
    ax_ftest.set_xlim(0, 4e5)
    ax_ftest.ticklabel_format(style='sci', scilimits=(0, 0), axis='x', useMathText=True)
    ax_ftest.set_xlabel('Time (ms)')
    ax_ftest.set_ylabel('F statistics')

    # column 2: CPD-windowed CC histogram
    gs = fig.add_gridspec(1, 1, left=col_cpd[0], right=col_cpd[1], top=top, bottom=bottom)
    ax = fig.add_subplot(gs[0, 0])
    hist_with_kde(ax, row['cpd_hist'])
    ax.set_xlim(*row['cpd_xlim'])
    ax.set_xlabel('CC')
    ax.xaxis.set_major_formatter(sci_formatter)

    # column 3: all-data CC histogram
    gs = fig.add_gridspec(1, 1, left=col_all[0], right=col_all[1], top=top, bottom=bottom)
    ax = fig.add_subplot(gs[0, 0])
    hist_with_kde(ax, row['all_hist'])
    ax.set_xlim(*row['all_xlim'])
    ax.set_xlabel('CC')
    ax.xaxis.set_major_formatter(sci_formatter)

    # column 4: ROC curve
    gs = fig.add_gridspec(1, 1, left=col_roc[0], right=col_roc[1], top=top, bottom=bottom)
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(row['roc_all'][0], row['roc_all'][1], color=METHOD_PAIR[0], lw=2.5,
            label=f"raw data: {row['auc_all']:.3f}", clip_on=False)
    ax.plot(row['roc_part'][0], row['roc_part'][1], color=METHOD_PAIR[1], lw=2.5,
            label=f"CPD data: {row['auc_part']:.3f}", clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    ax.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
    ax.legend(loc='lower right', fontsize=14)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')

    panel_labels(fig, [(col_ts[0] - 0.035, top + 0.005, panel_letters[letter_i])])
    letter_i += 1
    for col in (col_cpd, col_all, col_roc):
        panel_labels(fig, [(col[0] - 0.035, top + 0.005, panel_letters[letter_i])])
        letter_i += 1

fig.savefig(path / 'figures' / 'fig3_recon_LIF_HH_ML.pdf', dpi=600)

# %%
