#%%
from common import *
from scifig import *
import pdif.utils as c4u
path = Path(__file__).parents[0]
data_path = path / 'N4000-2-normal'
from scipy.sparse.linalg import svds
use_scifig()

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

def as_object_array(values):
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result
# %%
# deltas = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
fig6_data_file = path / 'fig6_data.npz'
fig6_data_keys = {
    'ts', 'deltas', 'proj_curves', 'projs_std_relative_change', 'ftest_curves',
    'ftest_tps', 'rhoE', 'rhoI', 'rocs', 'aucs', 'regression_data',
}
if fig6_data_file.exists():
    with np.load(fig6_data_file, allow_pickle=True) as fig6_data:
        if fig6_data_keys.issubset(fig6_data.files):
            ts = fig6_data['ts']
            deltas = fig6_data['deltas']
            proj_curves = fig6_data['proj_curves']
            projs_std_relative_change = fig6_data['projs_std_relative_change']
            ftest_curves = fig6_data['ftest_curves']
            ftest_tps = fig6_data['ftest_tps']
            rhoE = fig6_data['rhoE']
            rhoI = fig6_data['rhoI']
            rocs = fig6_data['rocs']
            aucs = fig6_data['aucs']
            regression_data = fig6_data['regression_data']
            fig6_data_loaded = True
        else:
            fig6_data_loaded = False
else:
    fig6_data_loaded = False

if not fig6_data_loaded:
    buff = np.load(data_path / 'projection_ddv_delta_2e6.npz')
    ts = buff['ts']
    deltas = buff['delta']
    proj = buff['proj']
    tmp = np.abs(proj.reshape(len(deltas), 2, -1))
    projs_mean = tmp.mean(-1)
    projs_mean_relative_change = np.diff(projs_mean, axis=1).flatten()/projs_mean[:,0]

    tmp = proj.reshape(len(deltas), 2, -1)
    projs_std = tmp.std(-1)
    projs_std_relative_change = np.diff(projs_std, axis=1).flatten()/projs_std[:,0]
# #%%
# projs_99 = np.quantile(tmp, 0.995, axis=-1)
# projs_99_relative_change = np.diff(projs_99, axis=1).flatten()/projs_99[:,0]
# #%%
# plt.plot(np.diff(projs_99, axis=1), '-o')
#%%
deltas_subset = [0.4, 0.8, 1.2]
data_lens = [2e6, 1e6]
if not fig6_data_loaded:
    proj_curves = []
    ftest_curves = []
    ftest_tps = []
    for delta in deltas_subset:
        delta_index = int(np.nonzero(deltas == delta)[0])
        proj_curves.append(np.abs(proj[delta_index])[::100])
        tps, x, f, p = TCD_Ftest(
            ts, proj[delta_index], window_size=int(500/0.02),
            p_thresh=1e-18, return_delta_mean=True)
        ftest_curves.append((x, f))
        ftest_tps.append(tps)

    rhoE, rhoI = [], []
    rocs, aucs = [], []
    regression_data = []
    for delta in deltas:
        for _T in data_lens:
            estimator = CausalityEstimator(
                path=data_path,
                spk_fname=f'LIFNet-K=40mu=50delay=2.0conn=randoms0w{delta:.2f}_T=2.00e+06',
                N=200, T=_T, n_thread=120, delay=0.0, dt=0.5, order=(1, 1), DT=1e4)
            data = estimator.fetch_data(new_run=True)
            data_matched = c4u.match_features(
                data, N=160, Ni=40,
                conn_file=data_path / f'connect_matrix-random-p=0.020-s0_subnet.npy')
            data_matched = merge_weights(
                data_matched,
                conn_file=data_path / f'connect_matrix-random-p=0.020-s0_subnet.npy',
                weight_file=data_path / f'connect_matrix-random-p=0.020-s0-d{delta:.2f}-w0_subnet.npy', K=40)
            data_matched['sum(CC2)'] = np.sqrt(data_matched['sum(CC2)'])

            tmpE = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('E')]
            tmpI = data_matched[data_matched['connection'].eq(1)*data_matched['pre_cell_type'].eq('I')]
            rhoE.append(tmpE['weight'].corr(tmpE['sum(CC2)']))
            rhoI.append(tmpI['weight'].corr(tmpI['sum(CC2)']))
            if delta in deltas_subset:
                regression_data.append((
                    tmpE['weight'].to_numpy(), tmpE['sum(CC2)'].to_numpy(),
                    tmpI['weight'].to_numpy(), tmpI['sum(CC2)'].to_numpy(),
                ))
            data_matched = apply_mask(
                data_matched, data_path / f'connect_matrix-random-p=0.020-s0_mask.npy')
            data_recon, fig_data = c4u._reconstruction_analysis(
                data_matched, x='log-CC', hist_range=(-10, -1), nbins=100,
                algorithm='curve_fit')
            rocs.append(fig_data['roc_gt'])
            aucs.append(fig_data['auc_svm'])

    rhoE = np.asarray(rhoE).reshape(len(deltas), len(data_lens))
    rhoI = np.asarray(rhoI).reshape(len(deltas), len(data_lens))
    aucs = np.asarray(aucs).reshape(len(deltas), len(data_lens))
    ts = ts[::100]
    np.savez(
        fig6_data_file,
        ts=ts,
        deltas=deltas,
        proj_curves=proj_curves,
        projs_std_relative_change=projs_std_relative_change,
        ftest_curves=as_object_array(ftest_curves),
        ftest_tps=as_object_array(ftest_tps),
        rhoE=rhoE,
        rhoI=rhoI,
        rocs=as_object_array(rocs),
        aucs=aucs,
        regression_data=as_object_array(regression_data),
    )

#%% # Load the voltage data
fig = plt.figure(figsize=(16, 14))

gs = fig.add_gridspec(1, 1, left=0.07, right=0.5, top=1., bottom=0.97)
ax = fig.add_subplot(gs[0, 0])
state_strip(ax, 2)

axs = []
for i in range(3):
    gs = fig.add_gridspec(2, 1, left=0.07, right=0.5,
        top=0.96-i*0.24, bottom=0.795-i*0.24, hspace=0.2,
        height_ratios=[1, 0.6])
    for j in range(2):
        axs.append(fig.add_subplot(gs[j, 0]))
axs = np.array(axs)
axs = axs.reshape(-1, 2)
deltas_subset = [0.4, 0.8, 1.2]
sample_range = 20000
for (delta, axi), proj_curve, ftest_curve, tps in zip(zip(deltas_subset, axs), proj_curves, ftest_curves, ftest_tps):
    axi[0].plot(ts, proj_curve, color=COLORS['green'], lw=0.5)
    highlight_span(axi[0], ts[0], ts[int(sample_range/2) - 1])
    axi[0].set_xlim(0,2e6)
    axi[0].set_ylim(0,)
    axi[0].ticklabel_format(style='sci', scilimits=(0,0), axis='x', useMathText=True)
    # axi[0].set_xlabel('Time (ms)', fontsize=16)
    axi[0].set_ylabel(r'$|\langle \hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$', fontsize=16)
    axi[0].set_rasterized(True)
    x, f = ftest_curve
    dT = x[1]-x[0]
    axi[1].plot(x, f, **ftest_style())
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
    elif delta == 0.8:
        axi[1].set_ylim(0.8, 2.4)
    else:
        axi[1].set_ylim(0.9, 1.6)
    axi[1].set_xlabel('Time (ms)', fontsize=16)


gs = fig.add_gridspec(3, 2, left=0.58, right=0.97, top=0.96, bottom=0.31, wspace=0.3, hspace=0.6)
axs = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(3)]
axs = np.asarray(axs)
Ts = [2e6, 1e6]
counter = 0
for ax_row in axs:
    for axi, _T in zip(ax_row, Ts):
        weightE, tdccE, weightI, tdccI = regression_data[counter]
        rhoE_ = rhoE[np.where(deltas == deltas_subset[counter // len(Ts)])[0][0], counter % len(Ts)]
        rhoI_ = rhoI[np.where(deltas == deltas_subset[counter // len(Ts)])[0][0], counter % len(Ts)]
        tmpE = pd.DataFrame({'weight': weightE, 'sum(CC2)': tdccE})
        tmpI = pd.DataFrame({'weight': weightI, 'sum(CC2)': tdccI})
        sns.regplot(data=tmpE, x='weight', y='sum(CC2)', ax=axi, color=EI_PAIR[0],
                    scatter_kws=dict(alpha=0.2, s=12, zorder=10),
                    line_kws=dict(label=f'Exc.: {rhoE_:.2f}', lw=2.5))
        sns.regplot(data=tmpI, x='weight', y='sum(CC2)', ax=axi, color=EI_PAIR[1],
                    scatter_kws=dict(alpha=0.2, s=12),
                    line_kws=dict(label=f'Inh.: {rhoI_:.2f}', lw=2.5))
        axi.ticklabel_format(style='sci', scilimits=(0,0), axis='y', useMathText=True)
        axi.legend(title='correlation', fontsize=10)
        axi.set_ylabel('TDCC')
        axi.set_xlabel(r'coupling strength ($S$)')
        axi.set_xlim(0)
        axi.set_ylim(0)
        counter += 1

# deltas_subset = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
# mask = [False, True, True, True, False, True, True, True, True]
gs = fig.add_gridspec(1, 2, left=0.07, right=0.97, top=0.24, bottom=0.06, wspace=0.15)
axs = [fig.add_subplot(gs[0, j]) for j in range(2)]
axs = np.asarray(axs)

axs[0].bar(np.arange(len(deltas)), projs_std_relative_change*100, width=0.5, color=COLORS['green'])
# axs[0].bar(np.arange(len(deltas))-0.1, projs_mean[:,0], width=0.2, color='C2', alpha=0.8, label=r'$\mathbf{W}_1$')
# axs[0].bar(np.arange(len(deltas))+0.1, projs_mean[:,1], width=0.2, color='C3', alpha=0.8, label=r'$\mathbf{W}_2$')
# axs[0].legend(loc=(0.02, 0.70), fontsize=16)
axs[0].set_xticks(np.arange(len(deltas)), deltas)
# axs[0].set_xlabel(r'$\sigma_S$', fontsize=16)
axs[0].set_xlabel(r'Heterogeneity of coupling strength $\sigma_S$', fontsize=16)
# axs[0].set_ylabel(r'$\langle|\langle \hat{\mathbf{v}}_n, \Delta^2\mathbf{v}\rangle|\rangle_t$', fontsize=16)
axs[0].set_ylabel('relative change\nof '+ r'$\mathrm{std}\left(\langle \hat{\mathbf{v}}_n, \Delta^2\mathbf{v}\rangle\right)$ (%)', fontsize=14)

axs[1].bar(np.arange(len(deltas))-0.15, np.diff(rhoE, axis=1).flatten(), width=0.3, color=EI_PAIR[0], label='Exc.')
axs[1].bar(np.arange(len(deltas))+0.15, np.diff(rhoI, axis=1).flatten(), width=0.3, color=EI_PAIR[1], label='Inh.')
# axs[1].plot(projs_mean_relative_change*100, np.diff(rhoE, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Exc.', clip_on=False)
# axs[1].plot(projs_mean_relative_change*100, np.diff(rhoI, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Inh.', clip_on=False)
# axs[1].plot(np.diff(projs_mean, axis=1)*100, np.diff(rhoE, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Exc.', clip_on=False)
# axs[1].plot(np.diff(projs_mean, axis=1)*100, np.diff(rhoI, axis=1), lw=4, marker='o', ms=6, mfc='w', label='Inh.', clip_on=False)
axs[1].legend(loc='upper left', fontsize=16)
axs[1].set_xticks(np.arange(len(deltas)), deltas)
axs[1].set_xlabel(r'Heterogeneity of coupling strength $\sigma_S$', fontsize=16)
# axs[1].set_xlabel(r'relative diff. of $\langle|\langle \hat{\mathbf{v}}_n, \Delta^2\mathbf{v}\rangle|\rangle_t$ (%)', fontsize=14)
axs[1].set_ylabel(r'$\rho$ improvement', fontsize=14)

# axs[1].bar(deltas_subset-0.02, rhoE[mask,0], width=0.04, color='C2', alpha=0.8)
# axs[1].bar(deltas_subset+0.02, rhoE[mask,1], width=0.04, color='C3', alpha=0.8)
# axs[1].set_xlabel(r'$\sigma_S$', fontsize=16)
# axs[1].set_ylabel(r'$\rho_E$', fontsize=16)

panel_labels(fig, [
    *[(0.02, 0.96-i*0.24, lab) for i, lab in enumerate('ADG')],
    *[(0.55, 0.96-i*0.24, lab) for i, lab in enumerate('BEH')],
    *[(0.765, 0.96-i*0.24, lab) for i, lab in enumerate('CFI')],
    *[(0.028+i*0.5, 0.25, lab) for i, lab in enumerate('JK')],
])

fig.savefig(path / 'figures' / 'fig6_reconLIF_vary_weights.pdf', dpi=600)
#%%
