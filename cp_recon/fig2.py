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
path = Path(__file__).parents[1]
data_path = path / 'N4000'
conn_paths = [data_path / f"connect_matrix-p=0.020-s{i:d}.npy" for i in range(2)]

v_rec = np.load(data_path / 'conn_s0_SM_sv.npy')
data_buff = np.load(data_path / 'N4000_chunk0&1_data.npz')
ts = data_buff['ts']
curP=data_buff['curP']
curW=data_buff['curW']
V=data_buff['V']
spikes=data_buff['spikes']
DDV = np.diff(V, n=2, axis=0)
DDV_ion = np.diff(np.diff(V, n=1, axis=0) - curP[:-1] - curW[:-1], n=1, axis=0)
DDV_P = np.diff(curP, n=1, axis=0)
DDV_W = np.diff(curW, n=1, axis=0)
#%%
import scipy.sparse as sp
_, s, v = sp.linalg.svds(DDV[0:25000], k=1, return_singular_vectors='vh', which='SM', maxiter=1000)
#%%
fig = plt.figure(figsize=(12, 6))
gd = fig.add_gridspec(2,1, left=0.0, right=0.37, top=0.95, bottom=0.05, hspace=0.3, height_ratios=[1,1.5])
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
gd = fig.add_gridspec(4,1, left=0.5, right=0.98, top=1.0, bottom=0.62, hspace=0.10)
ax = [fig.add_subplot(gdi) for gdi in gd]
ax[0].plot(ts[1:-1], np.abs(DDV@v_rec), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v_rec), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v_rec), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v_rec), label='rec')
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

gd = fig.add_gridspec(4,1, left=0.5, right=0.98, top=0.52, bottom=0.17)
ax = [fig.add_subplot(gdi) for gdi in gd]
ax[0].plot(ts[1:-1], np.abs(DDV@v.T).reshape(-1), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v.T).reshape(-1), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v.T).reshape(-1), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v.T).reshape(-1), label='rec')

ylabels = [r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle|$',
           r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle|$']
for i, ylabel in enumerate(ylabels):
    ax[i].fill_between(ts[:25000], 0, 0.6, color='C0', alpha=0.2, lw=0)
    ax[i].set_xlim(0, 2000)
    ax[i].set_ylim(0, 0.6)
    ax[i].set_ylabel(ylabel, fontsize=10, rotation=0, va='center', ha='right')
    if i == 3:
        ax[i].set_xlabel('Time (ms)')
    else:
        ax[i].set_xticklabels([])

gs = fig.add_gridspec(1, 1, left=0.5, right=0.98, top=0.07, bottom=0.01)
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


fig.text(0.01, 0.95, 'a', fontsize=24, fontweight='bold')
fig.text(0.01, 0.5, 'b', fontsize=24, fontweight='bold')
fig.text(0.38, 0.95, 'c', fontsize=24, fontweight='bold')
fig.text(0.38, 0.5, 'd', fontsize=24, fontweight='bold')
fig.savefig('fig2_schematics.pdf', dpi=600 )
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