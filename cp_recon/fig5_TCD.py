#%%
from common import *
import causal4.utils as c4u
path = Path(__file__).parents[1]
data_path = path / 'N4000-2-normal'
from causal4.myplot import sci_formatter
from scipy.sparse.linalg import svds
plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
})

def merge_weights(data, conn_file, weight_file, K:int=40):
    conn_pairs = np.load(conn_file)
    weight_pairs = np.load(weight_file)
    weight_pairs *= 1.0/np.sqrt(K)
    weight_pairs[(conn_pairs[0] >= 160) * (conn_pairs[1] < 160)] *= 2
    weight_pairs[(conn_pairs[0] >= 160) * (conn_pairs[1] >= 160)] *= 1.8
    weight_df = pd.DataFrame(
        {'pre_id':conn_pairs[0], 'post_id':conn_pairs[1],
         'weight':weight_pairs})
    data = data.merge(weight_df, how='left', on=['pre_id', 'post_id'])
    return data

# fname = 'LIFNet-K=40mu=50s0w0.5_T=4.00e+05'
# spk_data = np.fromfile(data_path / (fname+'_spike_train.dat'), dtype=float).reshape(-1,2)
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
sample_range = 20000 # in unit ms
window_sizes = np.arange(10000, 100001, 10000)
tps_list = []
for i in np.arange(0.1, 1.3, 0.1):
    fname = f'LIFNet-K=40mu=50s0w{i:.1f}_T=4.00e+05'
    if (data_path / (fname+'TCD_vs_window_size.npz')).exists():
        buff = np.load(data_path / (fname+'TCD_vs_window_size.npz'))
        tps_list.append(buff['tps'])
    else:
        vol_data = c4u.fetch_voltage(data_path / (fname+'_voltage.dat'), 200, (0, 4e5))
        ts = vol_data[1:-1, 0].copy()
        DDV = np.diff(vol_data[:, 1:], n=2, axis=0)
        DDV_mean = np.abs(DDV.mean(1))

        if not (data_path / (fname+f'_ddv-vn{sample_range/1e3:.0f}.npy')).exists():
            _,s,v = svds(DDV[:int(sample_range/0.02)], k=1, return_singular_vectors='vh', which='SM')
            np.save(data_path / (fname+f'_ddv-vn{sample_range/1e3:.0f}.npy'), v.flatten())
        else:
            v = np.load(data_path / (fname+f'_ddv-vn{sample_range/1e3:.0f}.npy'))
        v.shape

        tmp = np.abs(DDV@v.T).flatten()
        tps = [TCD(ts, tmp, window_size=i, return_delta_mean=False) for i in window_sizes]
        plt.plot(window_sizes, tps, '-o')
        plt.ylim(0, 4e5)
        np.savez(data_path / (fname+'TCD_vs_window_size.npz'), window_sizes=window_sizes, tps=tps)
        tps_list.append(tps)
#%%
tps_array = np.asarray(tps_list)
#%%
sns.heatmap(tps_array, cmap='Blues', cbar_kws={'label': 'TCD'}, linecolor='gray', linewidths=0)
plt.xticks(np.arange(len(window_sizes))+.5, window_sizes, rotation=45)
plt.yticks(np.arange(len(tps_list))+.5, [f'{i:.1f}' for i in np.arange(0.1, 1.3, 0.1)], rotation=0)
#%%
tmp_subnet = np.load(data_path/'connect_matrix-p=0.020-s0.npy')
tmp_weight = np.load(data_path/'connect_matrix-p=0.020-s0-d0.5-w0.npy')
mask = tmp_weight > 0 
tmp_subnet = tmp_subnet[:,mask]

print(tmp_subnet.shape, tmp_weight.shape)
plt.hist(tmp_weight.flatten(), bins=100)
print(np.sum(tmp>0)/tmp.size)
# %%
# deltas = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
buff = np.load(data_path / 'projection_ddv_delta_2e6.npz')
ts=buff['ts']
deltas = buff['delta']
proj=buff['proj']
tmp = np.abs(proj.reshape(len(deltas), 2, -1))
projs_mean = tmp.mean(-1)
projs_mean_relative_change = np.diff(projs_mean, axis=1).flatten()/projs_mean[:,0]
#%%
projs_99 = np.quantile(tmp, 0.995, axis=-1)
projs_99_relative_change = np.diff(projs_99, axis=1).flatten()/projs_99[:,0]
#%%
plt.plot(np.diff(projs_99, axis=1), '-o')
#%%
data_lens = [2e6, 1e6]
rhoE, rhoI = [], []
rocs, aucs = [], []
for delta in deltas:
    for _T in data_lens:
        estimator = CausalityEstimator(
            path=data_path,
            spk_fname=f'LIFNet-K=40mu=50delay=2.0conn=randoms0w{delta:.2f}_T=2.00e+06',
            N=200, T=_T, n_thread=120, delay=0.0, dt=0.5, order=(1, 1), DT=1e4)
        # Fetch the causality data as a pandas dataframe
        data = estimator.fetch_data(new_run=True)

        data_matched = c4u.match_features(data, N=160, Ni=40, conn_file=data_path / f'connect_matrix-random-p=0.020-s0_subnet.npy')

        data_matched = merge_weights(
            data_matched,
            conn_file=data_path / f'connect_matrix-random-p=0.020-s0_subnet.npy',
            weight_file=data_path / f'connect_matrix-random-p=0.020-s0-d{delta:.2f}-w0_subnet.npy', K=40)
        data_matched['sum(CC2)'] = np.sqrt(data_matched['sum(CC2)'])

        tmpE = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('E')]
        tmpI = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('I')]
        rhoE.append(tmpE['weight'].corr(tmpE['sum(CC2)']))
        rhoI.append(tmpI['weight'].corr(tmpI['sum(CC2)']))
        hist_range = (-10, -1)
        data_matched = apply_mask(data_matched, data_path / f'connect_matrix-random-p=0.020-s0_mask.npy')
        data_recon, fig_data = c4u._reconstruction_analysis(
            data_matched, x='log-CC', hist_range = hist_range, nbins = 100, # not implemented yet
            algorithm='curve_fit')
        rocs.append(fig_data['roc_gt'])
        aucs.append(fig_data['auc_svm'])

rhoE = np.asarray(rhoE).reshape(len(deltas), len(data_lens))
rhoI = np.asarray(rhoI).reshape(len(deltas), len(data_lens))
aucs = np.asarray(aucs).reshape(len(deltas), len(data_lens))

#%% # Load the voltage data
fig = plt.figure(figsize=(16, 14))

gs = fig.add_gridspec(1, 1, left=0.07, right=0.5, top=1., bottom=0.97)
ax = fig.add_subplot(gs[0, 0])
for i in range(2):
    ax.fill_between([i,i+1], -0.5, 0.5, color=f'C{i+2:d}', alpha=0.4, lw=0)
    ax.text(i+0.5, 0.0, r'$\mathbf{W}_{%d}$'%(i+1), fontsize=24, fontweight='bold', ha='center', va='center')
ax.spines['left'].set_visible(False)
ax.set_xlim(0,2)
ax.set_ylim(-0.5, 0.5)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])

axs = []
for i in range(3):
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.5,
        top=0.96-i*0.24, bottom=0.795-i*0.24, hspace=0.2,
        height_ratios=[1, 0.6])
    for j in range(2):
        axs.append(fig.add_subplot(gs[j, 0]))
axs = np.array(axs)
axs = axs.reshape(-1, 2)
deltas_subset = [0.1, 0.6, 1.2]
sample_range = 20000
for delta, axi in zip(deltas_subset,axs):
    axi[0].plot(ts[::1], np.abs(proj[int(np.nonzero(deltas==delta)[0])])[::1], color='C0', lw=2)
    axi[0].fill_between(ts[:int(sample_range/0.02)], 0, axi[0].get_ylim()[1], color='C1', alpha=0.4, lw=0, zorder=100)
    axi[0].set_xlim(0,2e6)
    axi[0].set_ylim(0,)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    # axi[0].set_xlabel('Time (ms)', fontsize=16)
    axi[0].set_ylabel(r'$|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$', fontsize=16)
    axi[0].set_rasterized(True)
    tps, x, f, p = TCD_Ftest(
        ts, proj[int(np.nonzero(deltas==delta)[0])], window_size=int(500/0.02),
        p_thresh=1e-18, return_delta_mean=True)
    print(tps)
    dT = x[1]-x[0]
    axi[1].plot(x, f, '-o', ms=3, clip_on=False)
    # axi[1].semilogy(x, p, '-o', ms=4, clip_on=False)
    axi[1].set_xlim(0, 2e6)
    axi[1].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    # for tp in tps:
    #     axs[2].fill_between([tp-dT/2, tp+dT/2], 0.85, 1.6, color='C3', alpha=0.6, lw=0, zorder=10)
    # axs[2].set_ylim(0.85,1.6)
    axi[1].set_ylabel('F statistics', fontsize=14)# rotation=0, va='center', ha='right')
    axi[0].set_xticks([0, 5e5, 1e6, 15e5, 2e6], ['', '', '', '', ''])
    axi[1].set_xticks([0, 5e5, 1e6, 15e5, 2e6])
    if delta == 1.2:
        axi[1].set_ylim(0.5, 3)
    else:
        axi[1].set_ylim(0.9, 1.6)
    axi[1].set_xlabel('Time (ms)', fontsize=16)
#%
# axs[-1].axis('off')
# axs[-1].plot(ts[::1000], np.abs(proj[-2])[::1000], color='C0', lw=2)
# axs[-1].axhline(projs_99[-2,0], xmin=0, xmax=0.5, color='C2', lw=4, zorder=10)
# axs[-1].axhline(projs_99[-2,1], xmin=0.5, xmax=1, color='C3', lw=4, zorder=10)
# axs[-1].set_xticks([0, 5e5, 1e6, 1.5e6, 2e6])
# axs[-1].set_xlim(0,2e6)
# axs[-1].set_ylim(0,)
# axs[-1].set_xlabel('Time (ms)', fontsize=16)
# axs[-1].set_ylabel(r'$|\langle \boldsymbol{\alpha}, \Delta^2 \mathbf{v}\rangle|$', fontsize=16)
# axs[-1].set_rasterized(True)
# axs[-1].axhline(projs_99[-2,0], xmin=0.5, xmax=1.1, color='grey', ls='--', lw=4, zorder=10, clip_on=False)
# axs[-1].axhline(projs_99[-2,1], xmin=1.0, xmax=1.1, color='grey', ls='--', lw=4, zorder=10, clip_on=False)
# axs[-1].text(2.2e6, 0.6, 'relative diff.', fontsize=24, fontweight='bold', ha='left', va='center')
# axs[-1].annotate(
#     '', xy=(0.5, 0.5), xytext=(0.5, 0.5),
#     arrowprops=dict(arrowstyle='<->', color='grey', lw=2), zorder=10,
#     # transform=axs[-1].transData,
#     clip_on=False)

# arrow = FancyArrowPatch((0, 0), (1, 0),
#                         arrowstyle='<->', mutation_scale=20, color='red')
# axs[-1].add_patch(arrow)



gs = fig.add_gridspec(3, 2, left=0.58, right=0.97, top=0.96, bottom=0.31, wspace=0.3, hspace=0.6)
axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
axs = np.asarray(axs)
hist_range = (-10, -2)

counter = 0
Ts = [2e6, 1e6]
for delta, ax_row in zip(deltas_subset, axs):
    for axi, T_ in zip(ax_row, Ts):
        estimator = CausalityEstimator(
            path=data_path,
            spk_fname=f'LIFNet-K=40mu=50delay=2.0conn=randoms0w{delta:.2f}_T=2.00e+06',
            N=200, T=T_, n_thread=120, delay=0.0, dt=0.5, order=(1, 1), DT=1e5)
        # Fetch the causality data as a pandas dataframe
        data = estimator.fetch_data(new_run=True)
        data_matched = c4u.match_features(data, N=160, Ni=40,
            conn_file = data_path / f'connect_matrix-random-p=0.020-s0_subnet.npy')
        data_matched = merge_weights(
            data_matched, conn_file=data_path/f'connect_matrix-random-p=0.020-s0_subnet.npy',
            weight_file=data_path/f'connect_matrix-random-p=0.020-s0-d{delta:.2f}-w0_subnet.npy')
        data_matched['sum(CC2)'] = np.sqrt(data_matched['sum(CC2)'])
        # data_matched = data_matched.fillna(0)
        # data_matched = apply_mask(data_matched, data_path/f'connect_matrix-p=0.020-s0_mask.npy')
        # data_recon, fig_data = c4u._reconstruction_analysis(
        #     data_matched, x='log-CC', hist_range = hist_range, nbins = 100, # not implemented yet
        #     algorithm='curve_fit')
        # sns.histplot(data=data_recon, x='log-CC', hue='connection', kde=True, bins=30, binrange=hist_range, ax=axi)
        tmpE = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('E')]#*data_matched['post_cell_type'].eq('E')]
        tmpI = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('I')]#*data_matched['post_cell_type'].eq('I')]
        rhoE_ = tmpE['weight'].corr(tmpE['sum(CC2)'])
        rhoI_ = tmpI['weight'].corr(tmpI['sum(CC2)'])
        sns.regplot(data=tmpE, x='weight', y='sum(CC2)', ax=axi, scatter_kws=dict(alpha=0.2, zorder=10), line_kws=dict(label=f'Exc.: {rhoE_:.2f}'))
        sns.regplot(data=tmpI, x='weight', y='sum(CC2)', ax=axi, scatter_kws=dict(alpha=0.2), line_kws=dict(label=f'Inh.: {rhoI_:.2f}'))
        # axi.set_title(f"correlation: ({rhoE_:.2}, {rhoI_:.2})")
        axi.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
        axi.legend(title='correlation', fontsize=10)
        # sns.scatterplot(data=data_matched[data_matched['connection'].eq(1)], x='log-CC', y='weight', alpha=0.3, ax=axi)
        axi.set_ylabel('TDCC')
        axi.set_xlabel(r'coupling strength ($S$)')
        axi.set_xlim(0)
        axi.set_ylim(0)
        # axi.set_xlim(*hist_range)
        # axi.xaxis.set_major_formatter(sci_formatter)
        # roc_all.append(fig_data['roc_gt'])
        # auc_all.append(fig_data['auc_svm'])
        # fig.text(0.76, 0.97-counter*0.2, r'$\sigma_S$='+f'{delta:.1f}', fontsize=20, ha='center', va='center')
        # counter += 1
# axs[-1,0].axis('off')
# axs[-1,0].barplot()
# axs[-1,0]
# axs[-1,0]
# axs[-1,0]

# deltas_subset = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
# mask = [False, True, True, True, False, True, True, True, True]
gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.25, bottom=0.10, wspace=0.15)
axs = [fig.add_subplot(gs[0, j]) for j in range(2)]
axs = np.asarray(axs)

axs[0].bar(np.arange(len(deltas))-0.1, projs_mean[:,0], width=0.2, color='C2', alpha=0.8, label=r'$\mathbf{W}_1$')
axs[0].bar(np.arange(len(deltas))+0.1, projs_mean[:,1], width=0.2, color='C3', alpha=0.8, label=r'$\mathbf{W}_2$')
axs[0].legend(loc=(0.02, 0.70), fontsize=16)
axs[0].set_xticks(np.arange(len(deltas)), deltas)
axs[0].set_xlabel(r'$\sigma_S$', fontsize=16)
axs[0].set_ylabel(r'$\langle|\langle \hat{\mathbf{v}}_n, \Delta^2\mathbf{v}\rangle|\rangle_t$', fontsize=16)

axs[1].plot(projs_mean_relative_change*100, np.diff(rhoE, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Exc.', clip_on=False)
axs[1].plot(projs_mean_relative_change*100, np.diff(rhoI, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Inh.', clip_on=False)
# axs[1].plot(np.diff(projs_mean, axis=1)*100, np.diff(rhoE, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Exc.', clip_on=False)
# axs[1].plot(np.diff(projs_mean, axis=1)*100, np.diff(rhoI, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Inh.', clip_on=False)
axs[1].legend(loc='lower right', fontsize=16)
axs[1].set_ylim(0)
axs[1].set_xlim(0)
axs[1].set_xlabel(r'relative diff. of $\langle|\langle \hat{\mathbf{v}}_n, \Delta^2\mathbf{v}\rangle|\rangle_t$ (%)', fontsize=14)
axs[1].set_ylabel(r'$\rho$ improvement', fontsize=14)

# axs[1].bar(deltas_subset-0.02, rhoE[mask,0], width=0.04, color='C2', alpha=0.8)
# axs[1].bar(deltas_subset+0.02, rhoE[mask,1], width=0.04, color='C3', alpha=0.8)
# axs[1].set_xlabel(r'$\sigma_S$', fontsize=16)
# axs[1].set_ylabel(r'$\rho_E$', fontsize=16)

for i, lab in enumerate('adg'):
    fig.text(0.02, 0.95-i*0.23, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('beh'):
    fig.text(0.55, 0.95-i*0.23, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('cfi'):
    fig.text(0.765, 0.95-i*0.23, lab, fontsize=24, fontweight='bold')
for i, lab in enumerate('jk'):
    fig.text(0.02+i*0.5, 0.26, lab, fontsize=24, fontweight='bold')

fig.savefig('fig5_reconLIF_vary_weights.pdf', dpi=600)
#%%
