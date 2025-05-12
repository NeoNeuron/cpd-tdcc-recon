#%%
from common import *
import causal4.utils as c4u
path = Path(__file__).parents[1]
data_path = path / 'N4000'
from causal4.myplot import sci_formatter
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

fname = 'LIFNet-K=40mu=50_T=4.00e+05'
spk_data = np.fromfile(data_path / (fname+'_spike_train.dat'), dtype=float).reshape(-1,2)
# T_single = 1e5
# for i in range(4):
#     buff = spk_data[(spk_data[:,0]>=i*T_single)*(spk_data[:,0]<(i+1)*T_single)].copy()
#     buff[:, 0] -= i*T_single
#     c4u.save2bin(data_path / (fname+f'_part{i:d}_spike_train.dat'), buff)

def apply_mask(df, mask_file):
    mask = np.load(mask_file)
    # mask = mask[:, (mask[0] < 160)+(mask[0] >= 3960)]
    # mask = mask[:, (mask[1] < 160)+(mask[1] >= 3960)]
    row_list = []
    for id_pairs in mask.T:
        row_list.append(df[(df['pre_id'] == id_pairs[0]) & (df['post_id'] == id_pairs[1])].index.values)
    row_list = np.concatenate(row_list)
    df = df.loc[row_list]
    return df
#%%
fname = 'LIFNet-K=40mu=50_T=4.00e+05'
vol_data = c4u.fetch_voltage(data_path / (fname+'_voltage.dat'), 200, (0, 4e5))
ts = vol_data[1:-1, 0].copy()
DDV = np.diff(vol_data[:, 1:], n=2, axis=0)
DDV_mean = np.abs(DDV.mean(1))
# %%
plt.figure(figsize=(10, 5))
plt.plot(ts, np.abs(DDV.mean(1)))
# %%
from scipy.sparse.linalg import svds
_,s,v = svds(DDV[:500000], k=1, return_singular_vectors='vh', which='SM')
s.shape, v.shape
s
#%%
# np.save('projections_topology_change.npy', (DDV@v.T).flatten()[int(5e6-2e5):int(5e6+2e5)])
#%% # Load the voltage data
fig = plt.figure(figsize=(16, 14))

gs = fig.add_gridspec(1, 1, left=0.07, right=0.97, top=1., bottom=0.97)
ax = fig.add_subplot(gs[0, 0])
for i in range(4):
    ax.fill_between([i,i+1], -0.5, 0.5, color=f'C{i:d}', alpha=0.4, lw=0)
    ax.text(i+0.5, 0.0, r'$\mathbf{W}_{%d}$'%(i+1), fontsize=24, fontweight='bold', ha='center', va='center')
ax.spines['left'].set_visible(False)
ax.set_xlim(0,4)
ax.set_ylim(-0.5, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

gs = fig.add_gridspec(3, 1, left=0.07, right=0.97, top=0.96, bottom=0.67, hspace=0.15, height_ratios=[1, 1, 0.8])
axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
spk_data_plot = spk_data[::1000]
axs[0].plot(spk_data_plot[spk_data_plot[:,1]<160, 0], spk_data_plot[spk_data_plot[:,1]<160,1], '.', clip_on=False)
axs[0].plot(spk_data_plot[spk_data_plot[:,1]>=160, 0], spk_data_plot[spk_data_plot[:,1]>=160,1], '.', clip_on=False)
axs[0].set_xlim(0, 4e5)
axs[0].set_ylim(0, 200)
axs[0].set_yticks([1,160,200])
axs[0].set_ylabel('Neuronal ID', fontsize=12)
axs[0].set_xlim(0, 4e5)
axs[0].set_xticks([0, 1e5, 2e5, 3e5, 4e5], ['', '', '', '', ''])
axs[1].plot(ts, np.abs(DDV@v.flatten()))
axs[1].set_rasterized(True)
axs[1].fill_between(ts[:500000], 0, 1, color='C1', alpha=0.4, lw=0, zorder=100)
axs[1].set_xticks([0, 1e5, 2e5, 3e5, 4e5])
axs[1].set_xlim(0,4e5)
axs[1].set_ylim(0,1)
axs[1].set_ylabel(r'$|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$', fontsize=13)
axs[1].set_xticks([0, 1e5, 2e5, 3e5, 4e5], ['', '', '', '', ''])
tps, x, f, p = TCD_Ftest(ts, (DDV@v.T).flatten(), window_size=int(400/0.02), p_thresh=1e-18, return_delta_mean=True)
print(tps)
dT = x[1]-x[0]
axs[2].plot(x, f, '-o', ms=4, clip_on=False)
# axs[2].semilogy(x, p, '-o', ms=4, clip_on=False)
axs[2].set_xlim(0, 4e5)
axs[2].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
# for tp in tps:
#     axs[2].fill_between([tp-dT/2, tp+dT/2], 0.85, 1.6, color='C3', alpha=0.6, lw=0, zorder=10)
# axs[2].set_ylim(0.85,1.6)
axs[2].set_xlabel('Time (ms)', fontsize=16)
axs[2].set_ylabel('F statistics', fontsize=12)# rotation=0, va='center', ha='right')

roc_all = []
roc_part = []
auc_all = []
auc_part = []

gs = fig.add_gridspec(3, 4, left=0.07, right=0.97, top=0.61, bottom=0.05, wspace=0.4, hspace=0.4)
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

for i, axi in enumerate(axs):
    estimator = CausalityEstimator(
        path=data_path,
        spk_fname=f'LIFNet-K=40mu=50_T=4.00e+05_part{i:d}',
        N=200, T=1e5, n_thread=120, delay=0.0, dt=0.1, order=(1, 1), DT=1e4)
    # Fetch the causality data as a pandas dataframe
    data = estimator.fetch_data(new_run=True)
    data_matched = c4u.match_features(data, N=200, conn_file=data_path/f'connect_matrix-p=0.020-s{i:d}_subnet.npy')
    data_matched = apply_mask(data_matched, data_path/f'connect_matrix-p=0.020-s{i:d}_mask.npy')
    data_recon, fig_data = c4u._reconstruction_analysis(
        data_matched, x='log-CC', hist_range = (-10,-4), nbins = 100, # not implemented yet
        algorithm='curve_fit')
    sns.histplot(data=data_recon, x='log-CC', hue='connection', kde=True, bins=30, binrange=(-10, -4), ax=axi)
    axi.set_xlim(-10, -4)
    axi.set_xlabel('CC')
    axi.xaxis.set_major_formatter(sci_formatter)
    roc_part.append(fig_data['roc_gt'])
    auc_part.append(fig_data['auc_svm'])

axs = [fig.add_subplot(gs[1, i]) for i in range(4)]
estimator = CausalityEstimator(
    path=data_path,
    spk_fname='LIFNet-K=40mu=50_T=4.00e+05',
    N=200, T=4e5, n_thread=120, delay=0.0, dt=0.1, order=(1, 1), DT=1e4)
# Fetch the causality data as a pandas dataframe
data = estimator.fetch_data(new_run=True)
for i, axi in enumerate(axs):
    data_matched = c4u.match_features(data, N=200, conn_file=data_path/f'connect_matrix-p=0.020-s{i:d}_subnet.npy')
    data_matched = apply_mask(data_matched, data_path/f'connect_matrix-p=0.020-s{i:d}_mask.npy')
    data_recon, fig_data = c4u._reconstruction_analysis(
        data_matched, x='log-CC', hist_range = None, nbins = 100, # not implemented yet
        algorithm='curve_fit')
    sns.histplot(data=data_recon, x='log-CC', hue='connection', kde=True, bins=30, binrange=(-11, -4), ax=axi)
    axi.set_xlabel('CC')
    axi.set_xlim(-11, -4)
    axi.xaxis.set_major_formatter(sci_formatter)
    roc_all.append(fig_data['roc_gt'])
    auc_all.append(fig_data['auc_svm'])

axs = [fig.add_subplot(gs[2, i]) for i in range(4)]
for i, axi in enumerate(axs):
    axi.plot(roc_all[i][0], roc_all[i][1], color='C2', lw=4, label=f'raw data: {auc_all[i]:.3f}', clip_on=False)
    axi.plot(roc_part[i][0], roc_part[i][1], color='C3', lw=4, label=f'TCP data: {auc_part[i]:.3f}', clip_on=False)
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    axi.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
    axi.legend(loc='lower right', fontsize=10)
    axi.set_xlabel('FPR', fontsize=14)
    axi.set_ylabel('TPR', fontsize=14)

fig.text(0.02, 0.97, 'a', fontsize=24, fontweight='bold')
fig.text(0.02, 0.844, 'b', fontsize=24, fontweight='bold')
fig.text(0.02, 0.746, 'c', fontsize=24, fontweight='bold')
for i, lab in enumerate('defg'):
    fig.text(0.02+i*0.245, 0.605, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('hijk'):
    fig.text(0.02+i*0.245, 0.40, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('lmno'):
    fig.text(0.02+i*0.245, 0.19, lab, fontsize=24, fontweight='bold')

fig.savefig('fig3_reconLIF.pdf', dpi=600)
# %%
