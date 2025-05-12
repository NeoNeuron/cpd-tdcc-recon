#%%
from common import *
import causal4.utils as c4u
path = Path(__file__).parents[1]
data_path = path / 'N4000'
from causal4.myplot import sci_formatter
from scipy.sparse.linalg import svds
import pickle as pkl
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

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
network_types = ['HH', 'ML']
# # proj_vecs = []
# DDV_projs = []
# for nettype in network_types:
#     fname = nettype+'Net-K=40mu=50_T=4.00e+05'
#     vol_data = c4u.fetch_voltage(data_path / (fname+'_voltage.dat'), 200, (0, 4e5))
#     DDV = np.diff(vol_data[:, 1:], n=2, axis=0)
#     # DDV_mean = np.abs(DDV.mean(1))
#     _,s,v = svds(DDV[:3000000], k=1, return_singular_vectors='vh', which='SM')
#     # proj_vecs.append(v)
#     DDV_projs.append((DDV@v.T).flatten())
# with open(data_path / 'DDV_projs_to_chunk0_6e4.pkl', 'wb') as f:
#     pkl.dump(DDV_projs, f)
#%%
with open(data_path / 'DDV_projs_to_chunk0_1e4.pkl', 'rb') as f:
    DDV_projs = pkl.load(f)
DDV_projs = DDV_projs[1:3]
ts = np.arange(DDV_projs[0].shape[0])*0.02
#%% # Load the voltage data
fig = plt.figure(figsize=(16, 9))

gs = fig.add_gridspec(1, 1, left=0.07, right=0.97, top=1., bottom=0.96)
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

gs = fig.add_gridspec(4, 1, left=0.07, right=0.97, top=0.95, bottom=0.4, hspace=0.3, height_ratios=[1, 0.8, 1, 0.8])
axs = np.array([fig.add_subplot(gs[i, 0]) for i in range(4)])
axs = axs.reshape(2, 2)
for axi, DDV_proj in zip(axs, DDV_projs):
    axi[0].plot(ts, np.abs(DDV_proj).flatten())
    ymax = np.abs(DDV_proj).max()
    axi[0].fill_between(ts[:500000], 0, ymax, color='C1', alpha=0.4, lw=0, zorder=100)
    axi[0].set_xlim(0,4e5)
    axi[0].set_ylim(0,ymax)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    axi[0].set_ylabel(
        r'$|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$', fontsize=16)
    axi[0].set_rasterized(True)
    tps, x, f, p = TCD_Ftest(
        ts, DDV_proj.flatten(), window_size=int(200/0.02),
        p_thresh=1e-18, return_delta_mean=True)
    print(tps)
    dT = x[1]-x[0]
    axi[1].plot(x, f, '-o', ms=4, clip_on=False)
    # axi[1].semilogy(x, p, '-o', ms=4, clip_on=False)
    axi[1].set_xlim(0, 4e5)
    axi[1].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    # for tp in tps:
    #     axs[2].fill_between([tp-dT/2, tp+dT/2], 0.85, 1.6, color='C3', alpha=0.6, lw=0, zorder=10)
    # axs[2].set_ylim(0.85,1.6)
    axi[1].set_xlabel('Time (ms)', fontsize=16)
    axi[1].set_ylabel('F statistics', fontsize=12)# rotation=0, va='center', ha='right')
    axi[0].set_xticks([0, 1e5, 2e5, 3e5, 4e5], ['', '', '', '', ''])
    axi[1].set_xticks([0, 1e5, 2e5, 3e5, 4e5])
    axi[1].set_xlabel('Time (ms)', fontsize=16)
#%

roc_all = []
roc_part = []
auc_all = []
auc_part = []

gs = fig.add_gridspec(1, 6, left=0.07, right=0.97, top=0.30, bottom=0.09, wspace=0.3)
axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
for axi, nettype in zip(axs[1::3], network_types):
    estimator = CausalityEstimator(
        path=data_path,
        spk_fname=nettype+'Net-K=40mu=50_T=4.00e+05',
        N=200, T=4e5, n_thread=120, delay=0.0, dt=1.0, order=(1, 1), DT=1e4)
    # Fetch the causality data as a pandas dataframe
    data = estimator.fetch_data(new_run=True)
    data_matched = c4u.match_features(data, N=200, conn_file=data_path/f'connect_matrix-p=0.020-s0_subnet.npy')
    data_matched = apply_mask(data_matched, data_path/f'connect_matrix-p=0.020-s0_mask.npy')
    data_recon, fig_data = c4u._reconstruction_analysis(
        data_matched, x='log-CC', hist_range = (-10,-2), nbins = 100, # not implemented yet
        algorithm='curve_fit')
    sns.histplot(data=data_recon, x='log-CC', hue='connection', kde=True, bins=30, binrange=(-10, -2), ax=axi)
    axi.set_xlabel('CC')
    axi.set_xlim(-10, -2)
    axi.xaxis.set_major_formatter(sci_formatter)
    roc_all.append(fig_data['roc_gt'])
    auc_all.append(fig_data['auc_svm'])

#%
for axi, nettype in zip(axs[::3], network_types):
    estimator = CausalityEstimator(
        path=data_path,
        spk_fname=nettype+'Net-K=40mu=50_T=4.00e+05',
        N=200, T=1e5, n_thread=120, delay=0.0, dt=1.0, order=(1, 1), DT=1e4)
    # Fetch the causality data as a pandas dataframe
    data = estimator.fetch_data(new_run=True)
    data_matched = c4u.match_features(data, N=200, conn_file=data_path/f'connect_matrix-p=0.020-s0_subnet.npy')
    data_matched = apply_mask(data_matched, data_path/f'connect_matrix-p=0.020-s0_mask.npy')
    data_recon, fig_data = c4u._reconstruction_analysis(
        data_matched, x='log-CC', hist_range = (-10,-2), nbins = 100, # not implemented yet
        algorithm='curve_fit')
    sns.histplot(data=data_recon, x='log-CC', hue='connection', kde=True, bins=30, binrange=(-10, -2), ax=axi)
    axi.set_xlabel('CC')
    axi.set_xlim(-10, -2)
    axi.xaxis.set_major_formatter(sci_formatter)
    roc_part.append(fig_data['roc_gt'])
    auc_part.append(fig_data['auc_svm'])

for i, axi in enumerate(axs[2::3]):
    axi.plot(roc_all[i][0], roc_all[i][1], color='C2', lw=4, label=f'raw data: {auc_all[i]:.3f}', clip_on=False)
    axi.plot(roc_part[i][0], roc_part[i][1], color='C3', lw=4, label=f'CPD data: {auc_part[i]:.3f}', clip_on=False)
    axi.set_xlim(0, 1)
    axi.set_ylim(0, 1)
    axi.set_xticks([0, 0.5, 1], ['0', '0.5', '1'])
    axi.set_yticks([0, 0.5, 1], ['0', '0.5', '1'])
    axi.legend(loc='lower right', fontsize=10)
    axi.set_xlabel('FPR', fontsize=14)
    axi.set_ylabel('TPR', fontsize=14)

fig.text(0.02, 0.95, 'a', fontsize=24, fontweight='bold')
fig.text(0.02, 0.63, 'b', fontsize=24, fontweight='bold')
for i, lab in enumerate('cde'):
    fig.text(0.02+i*0.165, 0.30, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('fgh'):
    fig.text(0.51+i*0.155, 0.30, lab, fontsize=24, fontweight='bold')
fig.savefig('fig4_reconGeneral.pdf', dpi=600)
# %%
