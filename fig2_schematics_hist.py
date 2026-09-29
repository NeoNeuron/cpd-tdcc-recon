# %%
# New Figure 2: CPD method schematics (previously Fig3) with the
# second-derivative-of-voltage histograms (previously Fig2 panels C-D) added
# as a new top row. Reuses the cached data from fig2.py and fig3.py.
# Panel C (covariance schematic) is drawn by covariance_schematic() below.
from common import *
from scifig import *
from matplotlib.image import imread
from matplotlib.patches import Ellipse, FancyArrowPatch
use_scifig()
compact_ticks()
path = Path(__file__).parents[0]

fig2_data_file = path / 'fig2_data.npz'
fig3_data_file = path / 'fig3_data.npz'
if not fig2_data_file.exists():
    raise SystemExit(f'{fig2_data_file} not found. Run fig2.py first.')
if not fig3_data_file.exists():
    raise SystemExit(f'{fig3_data_file} not found. Run fig3.py first.')

fig2_data = np.load(fig2_data_file, allow_pickle=True)
histograms = fig2_data['histograms']
theory_x = fig2_data['theory_x']
theory_curves = fig2_data['theory_curves']

fig3_data = np.load(fig3_data_file, allow_pickle=True)
ts = fig3_data['ts']
projected_curves = fig3_data['projected_curves']
singular_curves = fig3_data['singular_curves']
tps = fig3_data['tps']
tcd_x = fig3_data['tcd_x']
tcd_f = fig3_data['tcd_f']
tcd_delta = float(fig3_data['tcd_delta'])

def covariance_schematic(ax, sigma1=THEMES['orange'], sigma2=THEMES['purple'], seed=3):
    """Panel C: covariance ellipses of Delta^2 v before (Sigma_1) and after (Sigma_2) the change
    point, and the distributions of the projection onto alpha, the minimum-variance direction of
    Sigma_1. Illustrative 2-D covariances, not simulation data."""
    rng = np.random.default_rng(seed)
    ink, grey = COLORS['black'], '#6b6b6b'

    def rot(deg):
        t = np.deg2rad(deg)
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    th1, (a1, b1) = 8, (1.9, 0.26)      # Sigma_1: long / short axis s.d.
    th2, (a2, b2) = 58, (1.45, 0.75)    # Sigma_2: independent orientation
    R1, R2 = rot(th1), rot(th2)
    S1 = R1 @ np.diag([a1**2, b1**2]) @ R1.T
    S2 = R2 @ np.diag([a2**2, b2**2]) @ R2.T
    alpha, u = R1[:, 1], R1[:, 0]       # short and long axes of Sigma_1
    s1, s2 = np.sqrt(alpha @ S1 @ alpha), np.sqrt(alpha @ S2 @ alpha)

    ax.set_aspect('equal')
    ax.axis('off')
    # samples and 2-s.d. ellipses
    for S, (a, b), th, theme in ((S2, (a2, b2), th2, sigma2), (S1, (a1, b1), th1, sigma1)):
        x = rng.multivariate_normal([0, 0], S, 160)
        ax.scatter(*x.T, s=4, color=theme['edge'], alpha=0.45, lw=0, zorder=2)
        ax.add_patch(Ellipse((0, 0), 4 * a, 4 * b, angle=th, fc=theme['fill'], ec='none', alpha=0.8, zorder=1))
        ax.add_patch(Ellipse((0, 0), 4 * a, 4 * b, angle=th, fc='none', ec=theme['edge'], lw=2.2, zorder=3))
    ax.add_patch(FancyArrowPatch((0, 0), 2.0 * alpha, arrowstyle='-|>', mutation_scale=18, lw=3,
                                 color=ink, zorder=5))
    ax.text(*(2.0 * alpha + np.array([0.15, 0.0])), r'$\boldsymbol{\alpha}$', fontsize=22, color=ink,
            ha='left', va='center')

    # projection axis parallel to alpha, offset along the long axis of Sigma_1
    c, L = 4.6 * u, 3.1
    ax.add_patch(FancyArrowPatch(c - L * alpha, c + L * alpha, arrowstyle='-|>', mutation_scale=12,
                                 lw=1.3, color=grey, zorder=6))
    ax.text(*(c + (L + 0.15) * alpha), r'$\langle\boldsymbol{\alpha},\Delta^2\mathbf{v}\rangle$',
            color=grey, fontsize=15, ha='center', va='bottom')
    for sd, theme in ((s1, sigma1), (s2, sigma2)):   # projection of the 2-s.d. extent
        for p in (-2 * sd, 2 * sd):
            ax.plot(*np.c_[p * alpha + 0.3 * u, c + p * alpha], color=theme['edge'], lw=1.0,
                    ls=(0, (2, 2)), zorder=0)
    s = np.linspace(-L, L, 400)
    for sd, theme, h in ((s2, sigma2, 0.75), (s1, sigma1, 1.5)):
        ss = s[np.abs(s) < 3 * sd]
        base = c[:, None] + np.outer(alpha, ss)
        pts = base + np.outer(u, h * np.exp(-ss**2 / (2 * sd**2)))
        ax.fill(*np.c_[base, pts[:, ::-1]], color=theme['edge'], alpha=0.3, lw=0, zorder=3)
        ax.plot(*pts, color=theme['edge'], lw=2.2, zorder=4)
    ax.text(*(c + 1.6 * u + 0.05 * alpha), r'$\lambda_{\min}(\boldsymbol{\Sigma}_1)$',
            color=sigma1['accent'], fontsize=15, ha='left', va='center')
    ax.text(*(c + 0.85 * u - 1.5 * alpha), r'$\boldsymbol{\alpha}^{\top}\boldsymbol{\Sigma}_2\boldsymbol{\alpha}$',
            color=sigma2['accent'], fontsize=15, ha='left', va='center')
    ax.text(*(R1 @ [-2 * a1 - 0.1, 0]), r'$\boldsymbol{\Sigma}_1\ (\mathbf{W}_1)$', color=sigma1['accent'],
            fontsize=18, ha='right', va='center')
    ax.text(*(R2 @ [-2 * a2 + 0.2, -2 * b2 - 0.05]), r'$\boldsymbol{\Sigma}_2\ (\mathbf{W}_2)$',
            color=sigma2['accent'], fontsize=18, ha='center', va='top')
    ax.relim()
    ax.autoscale_view()


#%%
# Old fig3 spanned figure-fraction [0.05, 1.00] top-to-bottom; it is reused
# here linearly remapped into [0.05, SCHEM_TOP], preserving its bottom margin
# and all internal proportions (hspace, height_ratios), so it now sits below
# the new histogram row with a clear gap for that row's x-labels.
H_NEW = 9.5
SCHEM_TOP = 0.78
_OLD_MIN, _OLD_MAX = 0.05, 1.00
_SCALE_Y = (SCHEM_TOP - _OLD_MIN) / (_OLD_MAX - _OLD_MIN)

def rescale(*vals):
    return tuple(_OLD_MIN + (v - _OLD_MIN) * _SCALE_Y for v in vals)

PROJ_LABEL_SIZE = 14

fig = plt.figure(figsize=(12, H_NEW))

# --- new top row: E/I second-derivative-of-voltage histograms (old Fig2 C-D)
# A+B together span [0.0, 0.98], matching the full width used by the
# schematics content below (images at left=0.0, projected/singular curves
# and F-statistic at right=0.98).
gs = fig.add_gridspec(1, 3, wspace=0.2, hspace=0.2, top=0.96, bottom=0.84, left=0.05, right=0.48)
ax_top = [fig.add_subplot(gsi) for gsi in gs]
for axis, histogram in zip(ax_top, histograms[:3]):
    axis.stairs(histogram[0], histogram[1], fill=True, facecolor=tint(EI_PAIR[0], 0.86),
                edgecolor=EI_PAIR[0], lw=1.5)
theory_kw = dict(ls='--', lw=2.0, color=COLORS['black'], label='theory')
legend_kw = dict(loc='lower left', bbox_to_anchor=(0.05, 0.92), borderaxespad=0.0)
ax_top[0].plot(theory_x[0], theory_curves[0], **theory_kw)
ax_top[1].plot(theory_x[0], theory_curves[0], **theory_kw)
ax_top[2].plot(theory_x[1], theory_curves[1], **theory_kw)
ax_top[0].legend(**legend_kw)
ax_top[2].ticklabel_format(style='sci', scilimits=(0, 0), axis='y', useMathText=True)
ax_top[0].set_xlabel(r'$\Delta^2 v_i$')
ax_top[1].set_xlabel(r'$\Delta^2 v_i^\mathrm{rec}$')
ax_top[2].set_xlabel(r'$\Delta^2 v_i^\mathrm{ext}$')
for axi in ax_top:
    axi.set_ylabel('Density')

gs = fig.add_gridspec(1, 3, wspace=0.2, hspace=0.2, top=0.96, bottom=0.84, left=0.55, right=0.98)
ax_top = [fig.add_subplot(gsi) for gsi in gs]
for axis, histogram in zip(ax_top, histograms[3:]):
    axis.stairs(histogram[0], histogram[1], fill=True, facecolor=tint(EI_PAIR[1], 0.86),
                edgecolor=EI_PAIR[1], lw=1.5)
ax_top[0].plot(theory_x[0], theory_curves[2], **theory_kw)
ax_top[1].plot(theory_x[0], theory_curves[2], **theory_kw)
ax_top[2].plot(theory_x[1], theory_curves[3], **theory_kw)
ax_top[0].legend(**legend_kw)
ax_top[2].ticklabel_format(style='sci', scilimits=(0, 0), axis='y', useMathText=True)
ax_top[2].set_yticks([0, 400, 800, 1200])
ax_top[0].set_xlabel(r'$\Delta^2 v_i$')
ax_top[1].set_xlabel(r'$\Delta^2 v_i^\mathrm{rec}$')
ax_top[2].set_xlabel(r'$\Delta^2 v_i^\mathrm{ext}$')
for axi in ax_top:
    axi.set_ylabel('Density')

# --- old fig3 content (schematics), rescaled into the bottom band ----------
top, bottom = rescale(0.95, 0.15)
gd = fig.add_gridspec(2, 1, left=0.0, right=0.37, top=top, bottom=bottom, hspace=0.25, height_ratios=[1, 1.4])
ax = [fig.add_subplot(gdi) for gdi in gd]
covariance_schematic(ax[0])
pdf_image = imread('DDV_SVD.png')
ax[1].imshow(pdf_image)
ax[1].axis('off')

top, bottom = rescale(1.00, 0.95)
gs = fig.add_gridspec(1, 1, left=0.5, right=0.98, top=top, bottom=bottom)
ax = fig.add_subplot(gs[0, 0])
state_strip(ax, 2, ymin=-0.5, ymax=0.8, fontsize=18,
            hide_spines=('top', 'right', 'left'))

top, bottom = rescale(0.94, 0.63)
gd = fig.add_gridspec(4, 1, left=0.5, right=0.98, top=top, bottom=bottom, hspace=0.10)
ax = [fig.add_subplot(gdi) for gdi in gd]
for axis, curve, curve_ts in zip(ax, projected_curves, (ts[1:-1], ts[1:-1], ts[1:], ts[1:])):
    axis.plot(curve_ts, curve, label='', color=COLORS['sky'], lw=0.9)
for axi in ax:
    axi.set_rasterized(True)
ylabels = [r'$\left|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}\rangle\right|$',
           r'$\left|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle\right|$',
           r'$\left|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle\right|$',
           r'$\left|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle\right|$']
for i, ylabel in enumerate(ylabels):
    ax[i].set_xlim(0, 2000)
    ax[i].set_ylim(0, 0.6)
    ax[i].set_ylabel(ylabel, fontsize=PROJ_LABEL_SIZE, rotation=0, va='center', ha='right')
    if i == 3:
        ax[i].set_xlabel('Time (ms)')
    else:
        ax[i].set_xticklabels([])

top, bottom = rescale(0.53, 0.24)
gd = fig.add_gridspec(4, 1, left=0.5, right=0.98, top=top, bottom=bottom)
ax = [fig.add_subplot(gdi) for gdi in gd]
for axis, curve, curve_ts in zip(ax, singular_curves, (ts[1:-1], ts[1:-1], ts[1:], ts[1:])):
    axis.plot(curve_ts, curve, label='', color=COLORS['green'], lw=0.9)
for axi in ax:
    axi.set_rasterized(True)

ylabels = [r'$\left|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}\rangle\right|$',
           r'$\left|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle\right|$',
           r'$\left|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle\right|$',
           r'$\left|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle\right|$']
for i, ylabel in enumerate(ylabels):
    highlight_span(ax[i], ts[0], ts[24999])
    ax[i].set_xlim(0, 2000)
    ax[i].set_ylim(0, 0.6)
    ax[i].set_ylabel(ylabel, fontsize=PROJ_LABEL_SIZE, rotation=0, va='center', ha='right')
    if i == 3:
        ax[i].set_xlabel('Time (ms)')
    else:
        ax[i].set_xticklabels([])

top, bottom = rescale(0.14, 0.08)
gs = fig.add_gridspec(1, 1, left=0.5, right=0.98, top=top, bottom=bottom)
ax = fig.add_subplot(gs[0, 0])
print(tps)
ax.plot(tcd_x, tcd_f, **ftest_style())
ax.set_xlim(0, 2000)
for tp in tps:
    changepoint_band(ax, tp, tcd_delta)
ax.set_ylim(0, 5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('F statistics', rotation=0, fontsize=14, va='center', ha='right')

label_CD, = rescale(0.95)
label_EF, = rescale(0.55)
label_G, = rescale(0.15)
panel_labels(fig, [
    (0.01, 0.995, 'A'), (0.510, 0.995, 'B'),
    (0.01, label_CD, 'C'), (0.38, label_CD, 'D'),
    (0.01, label_EF, 'E'), (0.38, label_EF, 'F'),
    (0.38, label_G, 'G'),
])
fig.savefig(path / 'figures' / 'fig2_schematics_hist.pdf', bbox_inches='tight')

# %%
