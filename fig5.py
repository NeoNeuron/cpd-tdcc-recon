#%%
from common import *
import pdif.utils as c4u
path = Path(__file__).parents[0]
data_path = path / 'N4000'
from pdif.myplot import sci_formatter
from scipy.sparse.linalg import svds
from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle
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
fig5_data_file = path / 'fig5_data.npz'
if fig5_data_file.exists():
    fig5_data = np.load(fig5_data_file, allow_pickle=True)
    ts = fig5_data['ts']
    projection_curves = fig5_data['projection_curves']
    ftest_curves = fig5_data['ftest_curves']
    ftest_tps = fig5_data['ftest_tps']
    histogram_data = fig5_data['histogram_data']
    roc_all = fig5_data['roc_all']
    roc_part = fig5_data['roc_part']
    auc_all = fig5_data['auc_all']
    auc_part = fig5_data['auc_part']
else:
    with open(data_path / 'DDV_projs_to_chunk0_1e4.pkl', 'rb') as f:
        DDV_projs = pkl.load(f)[1:3]
    ts = np.arange(DDV_projs[0].shape[0])*0.02
    projection_curves = []
    ftest_curves = []
    ftest_tps = []
    for DDV_proj in DDV_projs:
        projection_curves.append(np.abs(DDV_proj).flatten())
        tps, x, f, p = TCD_Ftest(
            ts, DDV_proj.flatten(), window_size=int(200/0.02),
            p_thresh=1e-18, return_delta_mean=True)
        ftest_curves.append((x, f))
        ftest_tps.append(tps)
    ts = ts[::100]
    projection_curves = np.array(projection_curves, dtype=object)[:,::100]
    ftest_curves = np.array(ftest_curves, dtype=object)
    ftest_tps = np.array(ftest_tps, dtype=object)

def get_histogram_data(data_recon):
    result = []
    for connection in data_recon['connection'].unique():
        values = data_recon.loc[data_recon['connection'] == connection, 'log-CC'].to_numpy()
        density, edges = np.histogram(values, bins=30, range=(-10, -2), density=True)
        kde_x = np.linspace(-10, -2, 200)
        kde_y = gaussian_kde(values)(kde_x) if values.size > 1 else np.zeros_like(kde_x)
        result.append((connection, density, edges, kde_x, kde_y))
    return np.array(result, dtype=object)

def as_object_array(values):
    result = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        result[i] = value
    return result
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
for axi, projection_curve, ftest_curve, tps in zip(
        axs, projection_curves, ftest_curves, ftest_tps):
    axi[0].plot(ts, projection_curve)
    ymax = projection_curve.max()
    axi[0].fill_between(ts[:5000], 0, ymax, color='C1', alpha=0.4, lw=0, zorder=100)
    axi[0].set_xlim(0,4e5)
    axi[0].set_ylim(0,ymax)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='both', useMathText=True)
    axi[0].set_ylabel(
        r'$|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$', fontsize=16)
    axi[0].set_rasterized(True)
    print(tps)
    x, f = ftest_curve
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

if not fig5_data_file.exists():
    roc_all = []
    roc_part = []
    auc_all = []
    auc_part = []
    histogram_data = []

gs = fig.add_gridspec(1, 6, left=0.07, right=0.97, top=0.30, bottom=0.09, wspace=0.3)
axs = [fig.add_subplot(gs[0, i]) for i in range(6)]
def plot_cached_histogram(axis, cached_histogram):
    legend_handles = []
    for c_i, (connection, density, edges, kde_x, kde_y) in enumerate(cached_histogram):
        color = f'C{1-c_i:d}'
        axis.stairs(density, edges, fill=True,
                    facecolor=color, alpha=0.5, edgecolor='black')
        axis.plot(kde_x, kde_y, lw=1, color=color)
        legend_handles.append(Rectangle((0, 0), 1, 1, facecolor=color,
                                        edgecolor='black', alpha=0.5,
                                        label=str(connection)))
    axis.legend(handles=legend_handles, title='connection',
                fontsize=10, title_fontsize=12)

for i, (axi, nettype) in enumerate(zip(axs[1::3], network_types)):
    if fig5_data_file.exists():
        plot_cached_histogram(axi, histogram_data[i])
        axi.set_xlabel('CC')
        axi.set_xlim(-10, -2)
        axi.xaxis.set_major_formatter(sci_formatter)
        continue
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
    histogram_data.append(get_histogram_data(data_recon))
    plot_cached_histogram(axi, histogram_data[-1])
    axi.set_xlabel('CC')
    axi.set_xlim(-10, -2)
    axi.xaxis.set_major_formatter(sci_formatter)
    roc_all.append(fig_data['roc_gt'])
    auc_all.append(fig_data['auc_svm'])

#%
for i, (axi, nettype) in enumerate(zip(axs[::3], network_types)):
    if fig5_data_file.exists():
        plot_cached_histogram(axi, histogram_data[len(network_types) + i])
        axi.set_xlabel('CC')
        axi.set_xlim(-10, -2)
        axi.xaxis.set_major_formatter(sci_formatter)
        continue
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
    histogram_data.append(get_histogram_data(data_recon))
    plot_cached_histogram(axi, histogram_data[-1])
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

if not fig5_data_file.exists():
    np.savez(
        fig5_data_file,
        ts=ts,
        projection_curves=projection_curves,
        ftest_curves=as_object_array(ftest_curves),
        ftest_tps=as_object_array(ftest_tps),
        histogram_data=as_object_array(histogram_data),
        roc_all=as_object_array(roc_all),
        roc_part=as_object_array(roc_part),
        auc_all=np.array(auc_all),
        auc_part=np.array(auc_part),
    )

fig.text(0.02, 0.95, 'A', fontsize=24)
fig.text(0.02, 0.63, 'B', fontsize=24)
for i, lab in enumerate('CDE'):
    fig.text(0.02+i*0.165, 0.30, lab, fontsize=24)
for i, lab in enumerate('FGH'):
    fig.text(0.51+i*0.155, 0.30, lab, fontsize=24)
fig.savefig(path / 'figures' / 'fig5_reconGeneral.pdf', dpi=600)
# %%
