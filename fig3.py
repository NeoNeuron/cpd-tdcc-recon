#%%
from common import *
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})
from matplotlib.image import imread
path = Path(__file__).parents[0]
data_path = path / 'N4000'
conn_paths = [data_path / f"connect_matrix-p=0.020-s{i:d}.npy" for i in range(2)]

v_rec = np.load(data_path / 'conn_s0_SM_sv.npy')
fig3_data_file = path / 'fig3_data.npz'
using_fig3_cache = fig3_data_file.exists()
if using_fig3_cache:
    fig3_data = np.load(fig3_data_file, allow_pickle=True)
    ts = fig3_data['ts']
    v = fig3_data['v']
    projected_curves = fig3_data['projected_curves']
    singular_curves = fig3_data['singular_curves']
    tps = fig3_data['tps']
    tcd_x = fig3_data['tcd_x']
    tcd_f = fig3_data['tcd_f']
    tcd_delta = float(fig3_data['tcd_delta'])
else:
    data_buff = np.load(data_path / 'N4000_chunk0&1_data.npz')
    ts = data_buff['ts']
    curP = data_buff['curP']
    curW = data_buff['curW']
    V = data_buff['V']
    DDV = np.diff(V, n=2, axis=0)
    DDV_ion = np.diff(np.diff(V, n=1, axis=0) - curP[1:] - curW[1:], n=1, axis=0)
    DDV_P = np.diff(curP, n=1, axis=0)
    DDV_W = np.diff(curW, n=1, axis=0)

    import scipy.sparse as sp
    _, s, v = sp.linalg.svds(DDV[0:25000], k=1, return_singular_vectors='vh', which='SM', maxiter=1000)
    projected_curves = np.array([
        np.abs(DDV @ v_rec),
        np.abs(DDV_ion @ v_rec),
        np.abs(DDV_P @ v_rec),
        np.abs(DDV_W @ v_rec),
    ], dtype=object)
    singular_curves = np.array([
        np.abs(DDV @ v.T).reshape(-1),
        np.abs(DDV_ion @ v.T).reshape(-1),
        np.abs(DDV_P @ v.T).reshape(-1),
        np.abs(DDV_W @ v.T).reshape(-1),
    ], dtype=object)
    tps, tcd_x, tcd_f, _ = TCD_Ftest(
        ts[1:-1], (DDV @ v.T).flatten(), window_size=int(20/0.02),
        p_thresh=1e-18, return_delta_mean=True,
    )
    tcd_delta = tcd_x[1] - tcd_x[0]
    np.savez(
        fig3_data_file,
        ts=ts, v=v, projected_curves=projected_curves, singular_curves=singular_curves,
        tps=tps, tcd_x=tcd_x, tcd_f=tcd_f, tcd_delta=tcd_delta,
    )
#%%
fig = plt.figure(figsize=(12, 7))
gd = fig.add_gridspec(2,1, left=0.0, right=0.37, top=0.95, bottom=0.15, hspace=0.25, height_ratios=[1,1.4])
ax = [fig.add_subplot(gdi) for gdi in gd]
pdf_image = imread('schematics.png', )
ax[0].imshow(pdf_image)
ax[0].axis('off')
pdf_image = imread('DDV_SVD.png', )
ax[1].imshow(pdf_image)
ax[1].axis('off')
# ax[0].add_artist(ab)
# ax[0].axis('off')
# ax[1].plot(range(10))

gs = fig.add_gridspec(1, 1, left=0.5, right=0.98, top=1.00, bottom=0.95)
ax = fig.add_subplot(gs[0, 0])
ax.fill_between([0,1], -0.5, 1, color='C2', alpha=0.4, lw=0)
ax.fill_between([1,2], -0.5, 1, color='C3', alpha=0.4, lw=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(0,2)
ax.set_ylim(-0.5, 0.8)
# ax.set_xlabel('Time', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.text(0.5, 0.0, r'$\mathbf{W}_1$', fontsize=18, fontweight='bold', ha='center', va='center')
ax.text(1.5, 0.0, r'$\mathbf{W}_2$', fontsize=18, fontweight='bold', ha='center', va='center')


gd = fig.add_gridspec(4,1, left=0.5, right=0.98, top=0.94, bottom=0.63, hspace=0.10)
ax = [fig.add_subplot(gdi) for gdi in gd]
for axis, curve, curve_ts in zip(ax, projected_curves, (ts[1:-1], ts[1:-1], ts[1:], ts[1:])):
    axis.plot(curve_ts, curve, label='')
for axi in ax:
    axi.set_rasterized(True)
ylabels = [r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle|$']
for i, ylabel in enumerate(ylabels):
    ax[i].set_xlim(0, 2000)
    ax[i].set_ylim(0, 0.6)
    ax[i].set_ylabel(ylabel, fontsize=10, rotation=0, va='center', ha='right')
    if i == 3:
        ax[i].set_xlabel('Time (ms)')
    else:
        ax[i].set_xticklabels([])

gd = fig.add_gridspec(4,1, left=0.5, right=0.98, top=0.53, bottom=0.24)
ax = [fig.add_subplot(gdi) for gdi in gd]
for axis, curve, curve_ts in zip(ax, singular_curves, (ts[1:-1], ts[1:-1], ts[1:], ts[1:])):
    axis.plot(curve_ts, curve, label='')
for axi in ax:
    axi.set_rasterized(True)

ylabels = [r'$|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}\rangle|$',
           r'$|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle|$',
           r'$|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle|$',
           r'$|\langle \hat\mathbf{v}_n, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle|$']
for i, ylabel in enumerate(ylabels):
    ax[i].fill_between(ts[:25000], 0, 0.6, color='C1', alpha=0.2, lw=0, zorder=10)
    ax[i].set_xlim(0, 2000)
    ax[i].set_ylim(0, 0.6)
    ax[i].set_ylabel(ylabel, fontsize=10, rotation=0, va='center', ha='right')
    if i == 3:
        ax[i].set_xlabel('Time (ms)')
    else:
        ax[i].set_xticklabels([])

gs = fig.add_gridspec(1, 1, left=0.5, right=0.98, top=0.14, bottom=0.08)
ax = fig.add_subplot(gs[0, 0])
print(tps)
ax.plot(tcd_x, tcd_f, '-o', ms=4, clip_on=False)
ax.set_xlim(0, 2000)
for tp in tps:
    ax.fill_between([tp-tcd_delta/2, tp+tcd_delta/2], 0, 5, color='C3', alpha=0.6, lw=0, zorder=10)
ax.set_ylim(0,5)
ax.set_xlabel('Time (ms)')
ax.set_ylabel('F statistics', rotation=0, fontsize=14, va='center', ha='right')

fig.text(0.01, 0.95, 'A', fontsize=24)
fig.text(0.38, 0.95, 'B', fontsize=24)
fig.text(0.01, 0.55, 'C', fontsize=24)
fig.text(0.38, 0.55, 'D', fontsize=24)
fig.text(0.38, 0.15, 'E', fontsize=24)
fig.savefig(path / 'figures' / 'fig3_schematics.pdf', dpi=600 )
#%%
if using_fig3_cache:
    raise SystemExit('Loaded compact fig3_data.npz; first figure is complete.')

#%%
#%% run model
import brainpy as bp
import brainpy.math as bm
bm.set_platform('gpu')
print(bp.__version__)
from EINet import LIFNet_monitor
bm.set_dt(0.02)
warmup_time = 20        # ms
simulation_time = 1000   # ms
curP_list, curW_list, V_list, spikes_list = [], [], [], []
state = None
for conn_path in conn_paths:
    model = LIFNet_monitor(num_neurons=4000, K=40, mu=10, conn_path=conn_path, method='euler')
    if state is None:
        indices = np.arange(int((warmup_time+simulation_time)/bm.get_dt()))
        curW2E, curP2E, curW2I, curP2I, E_V, I_V, E_spike, I_spike = bm.for_loop(
            model.step_run, indices, progress_bar=True)
        curP2E = curP2E[int(warmup_time/bm.get_dt()):, :]
        curP2I = curP2I[int(warmup_time/bm.get_dt()):, :]
        curW2E = curW2E[int(warmup_time/bm.get_dt()):, :]
        curW2I = curW2I[int(warmup_time/bm.get_dt()):, :]
        E_V = E_V[int(warmup_time/bm.get_dt()):, :]
        I_V = I_V[int(warmup_time/bm.get_dt()):, :]
        E_spike = E_spike[int(warmup_time/bm.get_dt()):, :]
        I_spike = I_spike[int(warmup_time/bm.get_dt()):, :]
        state = model.save_neu_state()
    else:
        model.load_neu_state(state)
        indices = np.arange(int(simulation_time/bm.get_dt()))
        curW2E, curP2E, curW2I, curP2I, E_V, I_V, E_spike, I_spike = bm.for_loop(
            model.step_run, indices, progress_bar=True)

    curP = np.concatenate([curP2E, curP2I], axis=1)
    curW = np.concatenate([curW2E, curW2I], axis=1)
    V = np.concatenate([E_V, I_V], axis=1)
    spikes = np.concatenate([E_spike, I_spike], axis=1)
    curP_list.append(curP)
    curW_list.append(curW)
    V_list.append(V)
    spikes_list.append(spikes)
#%
ts = np.arange(int(len(conn_paths)*simulation_time/bm.get_dt())) * bm.get_dt()
curP = np.concatenate(curP_list, axis=0)
curW = np.concatenate(curW_list, axis=0)
V = np.concatenate(V_list, axis=0)
spikes = np.concatenate(spikes_list, axis=0)
#%%
np.savez(data_path / 'N4000_chunk0&1_data.npz', ts = ts, curP=curP, curW=curW, V=V, spikes=spikes)
# %%
plt.plot(ts, V[:, 100:].mean(1), label='E neuron')
# plt.axvline(200, color='k', lw=1)
plt.xlim(490,610)
#%%
print(f'{spikes.sum()/spikes.shape[1]/simulation_time/len(conn_paths)*1e3:.3f} Hz')
bp.visualize.raster_plot(ts, spikes[:, :100], show=True)
#%%
DDV = np.diff(V, n=2, axis=0)
DDV_ion = np.diff(np.diff(V, n=1, axis=0) - curP[1:] - curW[1:], n=1, axis=0)
DDV_P = np.diff(curP, n=1, axis=0)
DDV_W = np.diff(curW, n=1, axis=0)
fig, ax = plt.subplots(4,1, figsize=(12, 4), sharex=True, sharey=True)
ax[0].plot(ts[1:-1], np.abs(DDV.mean(1)), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion.mean(1)), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P.mean(1)), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W.mean(1)), label='rec')
#%%
fig, ax = plt.subplots(4,1, figsize=(12, 4), sharex=True, sharey=True)
ax[0].plot(ts[1:-1], np.abs(DDV@v.T).reshape(-1), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v.T).reshape(-1), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v.T).reshape(-1), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v.T).reshape(-1), label='rec')
#%%
import scipy.sparse as sp
sparse_ids = np.load(conn_paths[1], allow_pickle=True)
num_e = 4000
dense_conn = np.zeros((num_e, num_e))
dense_conn[sparse_ids[0], sparse_ids[1]] = 1.0
u, s, vh = np.linalg.svd(dense_conn)
print(s[-1])
#%%
v = vh[-1].flatten()
np.save(data_path / 'conn_s1_SM_sv.npy', v)
#%%
ug, s, vg = sp.linalg.svds(DDV, k=20, which='LM', maxiter=1000)
s_denoise = DDV @ vg.T @ vg
print(s_denoise.shape)
#%%
DDV_denoise = ug @ np.diag(s) @ vg 
#%%
_, s, v = sp.linalg.svds(DDV[0:25000], k=1, return_singular_vectors='vh', which='SM', maxiter=1000)
s.shape, v.shape
#%%
v_DDV = v.flatten().copy()
#%%
plt.plot(ts[1:-1], np.abs(DDV[:, :]@v.flatten()), label='E neuron')
plt.xlim(0,2000)
plt.ylim(0)