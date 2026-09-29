# %%
# Supplementary Figure S1: excitatory/inhibitory second-derivative-of-voltage
# time series (previously Fig2 panels A-B). The companion histogram panels
# (previously Fig2 C-D) now open fig2_schematics_hist.py instead.
from common import *
from scifig import *
use_scifig()
compact_ticks()

path = Path(__file__).parents[0]
fig2_data_file = path / 'fig2_data.npz'
if not fig2_data_file.exists():
    raise SystemExit(
        f'{fig2_data_file} not found. Run fig2.py first to (re)generate the '
        'cached simulation data this figure reuses.')

data = np.load(fig2_data_file, allow_pickle=True)
curve_ts = data['curve_ts']
curves = data['curves']
E_fr = float(data['E_fr'])
I_fr = float(data['I_fr'])
print(E_fr, I_fr)

#%%
fig, ax = plt.subplots(4, 2, figsize=(16, 8),
                       gridspec_kw={'hspace': 0.2, 'top': 0.90, 'bottom': 0.12, 'left': 0.05},
                       sharex=True, sharey='col')

for axis, index, label in zip(ax[:, 0], range(4), ('V', 'W', 'P', 'ion')):
    axis.semilogy(curve_ts[index], curves[index], label=label, color=EI_PAIR[0], lw=0.9)
for axis, index, label in zip(ax[:, 1], range(4, 8), ('V', 'W', 'P', 'ion')):
    axis.semilogy(curve_ts[index - 4], curves[index], label=label, color=EI_PAIR[1], lw=0.9)
ax[0,0].set_title('Excitatory populations', fontsize=18)
ax[0,1].set_title('Inhibitory populations', fontsize=18)
ax[0,0].set_ylabel(r'$\left|\langle\Delta^2 v_i\rangle_i\right|$')
ax[0,1].set_ylabel(r'$\left|\langle\Delta^2 v_i\rangle_i\right|$')
ax[1,0].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{rec}\rangle_i\right|$')
ax[1,1].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{rec}\rangle_i\right|$')
ax[2,0].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{ext}\rangle_i\right|$')
ax[2,1].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{ext}\rangle_i\right|$')
ax[3,0].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{ion}\rangle_i\right|$')
ax[3,1].set_ylabel(r'$\left|\langle\Delta^2 v_i^\mathrm{ion}\rangle_i\right|$')
ax[3,0].set_xlabel('Time (ms)')
ax[3,1].set_xlabel('Time (ms)')
ax[0,0].set_xlim(0,200)
ax[0,0].set_ylim(1e-7, 1e-1)
ax[0,1].set_ylim(1e-7, 1e-1)
ax[0,0].set_yticks([1e-7, 1e-4, 1e-1])
ax[0,1].set_yticks([1e-7, 1e-4, 1e-1])
for axi in ax.flatten():
    axi.set_rasterized(True)

panel_labels(fig, [
    (0.01, 0.94, 'A'), (0.47, 0.94, 'B'),
])
fig.savefig(path / 'figures' / 'figS1_EI32k_timeseries.pdf', dpi=300, bbox_inches='tight')

# %%
