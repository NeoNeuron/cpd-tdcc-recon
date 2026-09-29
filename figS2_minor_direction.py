# %%
"""Supplementary Figure S2: variance of Delta^2 v along the minimum-variance direction of Sigma_1.

Covariance of the second-order voltage difference in state k (Bernoulli spikes, Poisson external
input, LIF reset treated as a self-coupling -(v_th - v_reset)):
    Sigma_k = 2 dt [ (W_k - R) D (W_k - R)^T + diag(F^2 nu) ]
alpha = eigenvector of Sigma_1 with the smallest eigenvalue. Detection signal:
    r = alpha^T Sigma_2 alpha / alpha^T Sigma_1 alpha      (r = 1 when W_2 = W_1).
Uses random E-I coupling matrices only (no simulation data); results are cached in figS2_data.npz.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scifig import COLORS, use_scifig

path = Path(__file__).parents[0]
figS2_data_file = path / 'figS2_data.npz'

NE, NI = 3200, 800
N = NE + NI
K = 40
dt = 0.5                  # ms, sampling step of Delta^2 v
mE = mI = 0.05            # 1/ms, firing rates
nu0 = 0.05                # 1/ms
J = dict(EE=1.0, IE=1.0, EI=-2.0, II=-1.8, E=1.0, I=0.8)
dV = dict(E=1.0, I=0.7)   # v_th - v_reset
SEEDS = range(3)
SIGMAS = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
NSUB = [25, 50, 100, 200, 400, 800, 1600, 4000]
NSAMPLE = 200

rates = np.r_[np.full(NE, mE), np.full(NI, mI)]
R = np.diag(np.r_[np.full(NE, dV['E']), np.full(NI, dV['I'])])
ext = 2 * dt * np.r_[np.full(NE, J['E'] ** 2), np.full(NI, J['I'] ** 2)] * nu0
Jm = np.empty((N, N))
Jm[:NE, :NE], Jm[NE:, :NE] = J['EE'], J['IE']
Jm[:NE, NE:], Jm[NE:, NE:] = J['EI'], J['II']

# external floor c_ext and per-neuron recurrent variance c_rec (population-mean mode excluded)
v_col = {q: 2 * dt * np.r_[J[q + 'E'] ** 2 * mE * (1 - K / NE) / NE * np.ones(NE),
                           J[q + 'I'] ** 2 * mI * (1 - K / NI) / NI * np.ones(NI)] for q in 'EI'}
w = NE / N, NI / N
c_ext = w[0] * ext[0] + w[1] * ext[-1]
c_rec = (w[0] * (v_col['E'].sum() + 2 * dt * dV['E'] ** 2 * mE)
         + w[1] * (v_col['I'].sum() + 2 * dt * dV['I'] ** 2 * mI))


def make_A(rng):
    A = np.zeros((N, N), bool)
    A[:, :NE] = rng.random((N, NE)) < K / NE
    A[:, NE:] = rng.random((N, NI)) < K / NI
    np.fill_diagonal(A, False)
    return A


def make_S(rng, sig):
    """S ~ N(J, (J sig)^2), clipped at zero so that Dale's law is preserved."""
    S = Jm * (1 + sig * rng.standard_normal((N, N)))
    return np.where(np.sign(S) == np.sign(Jm), S, 0.0) / np.sqrt(K)


def sigma(W, rows):
    Weff = (W - R)[rows]
    return 2 * dt * (Weff * rates) @ Weff.T + np.diag(ext[rows])


def minor(S1):
    lam, U = np.linalg.eigh(S1)
    return lam, U[:, 0]


def ratio(S1, S2, a):
    return (a @ S2 @ a) / (a @ S1 @ a)


def compute():
    out = {k: [] for k in ('topo_full', 'topo_sub', 'wts_full', 'wts_sub', 'nsweep', 'ctrl')}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        A1, A2 = make_A(rng), make_A(rng)
        S0 = make_S(rng, 0.0)
        W1, W2 = S0 * A1, S0 * A2
        perm = rng.permutation(N)
        sub = perm[:NSAMPLE]
        # (A) spectra + (D) direction control, topology change
        for mode, rows in (('full', np.arange(N)), ('sub', sub)):
            S1, S2 = sigma(W1, rows), sigma(W2, rows)
            lam, a = minor(S1)
            if seed == 0:
                out[f'lam_{mode}'], out[f'q2_{mode}'] = lam, a @ S2 @ a
            out[f'topo_{mode}'].append(ratio(S1, S2, a))
            if mode == 'sub':
                u = rng.standard_normal(len(rows))
                u /= np.linalg.norm(u)
                one = np.ones(len(rows)) / np.sqrt(len(rows))
                out['ctrl'].append([ratio(S1, S2, a), ratio(S1, S2, u), ratio(S1, S2, one)])
        # (C) observed-population size, topology change
        out['nsweep'].append([ratio(sigma(W1, perm[:n]), sigma(W2, perm[:n]), minor(sigma(W1, perm[:n]))[1])
                              for n in NSUB])
        # (B) weight change on a fixed adjacency matrix
        for mode, rows in (('full', np.arange(N)), ('sub', sub)):
            row = []
            for sig in SIGMAS:
                S1, S2 = sigma(make_S(rng, sig) * A1, rows), sigma(make_S(rng, sig) * A1, rows)
                row.append(ratio(S1, S2, minor(S1)[1]))
            out[f'wts_{mode}'].append(row)
        print(f'seed {seed} done')
    return {k: np.asarray(v) for k, v in out.items()}


if figS2_data_file.exists():
    data = dict(np.load(figS2_data_file))
else:
    data = compute()
    np.savez(figS2_data_file, **data)

print(f'c_ext={c_ext:.3f}  c_rec={c_rec:.3f}  1+c_rec/c_ext={1 + c_rec / c_ext:.2f}')
print('topology ratio full/sub:', data['topo_full'].mean(), data['topo_sub'].mean())
print('n sweep:', dict(zip(NSUB, data['nsweep'].mean(0).round(2))))
print('weights full:', data['wts_full'].mean(0).round(2), 'sub:', data['wts_sub'].mean(0).round(2))
print('direction control (minor, random, mean):', data['ctrl'].mean(0).round(2))

# %%
use_scifig(scale=1.0)       # drawn at print size, FIGSIZE['double'] width
C = {'full': COLORS['green'], 'sub': COLORS['orange']}
LAB = {'full': f'all $N={N}$ neurons', 'sub': f'$n={NSAMPLE}$ sampled neurons'}
MUTED = COLORS['gray']
fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1), layout='constrained')

# A: eigenvalue spectra of Sigma_1, with alpha^T Sigma_2 alpha
for mode, x0 in (('full', -0.04), ('sub', 0.04)):
    lam = data[f'lam_{mode}']
    ax[0].plot(np.linspace(0, 1, len(lam)), lam, color=C[mode])
    ax[0].plot(x0, lam[0], 'o', ms=4, color=C[mode])
    ax[0].plot(x0, data[f'q2_{mode}'], 'D', ms=4, mfc='white', mec=C[mode], mew=1.5)
ax[0].axhline(c_ext, color=MUTED, lw=0.8, ls=':')
ax[0].text(1, c_ext, r'$c_\mathrm{ext}$', ha='right', va='bottom', color=MUTED, fontsize=7)
ax[0].axhline(c_ext + c_rec, color=MUTED, lw=0.8, ls=':')
ax[0].text(0.15, c_ext + c_rec, r'$c_\mathrm{ext}+c_\mathrm{rec}$', ha='left', va='bottom', color=MUTED,
           fontsize=7)
ax[0].set(xlabel='eigenvalue rank / $n$', ylabel=r'eigenvalue of $\Sigma_1$', ylim=(0, 0.75))
ax[0].plot([], [], 'ko', ms=4, label=r'$\lambda_{\min}(\Sigma_1)$')
ax[0].plot([], [], 'kD', ms=4, mfc='white', label=r'$\alpha^\top\Sigma_2\alpha$')
ax[0].legend(fontsize=6, loc='upper left')

RATIO = r'$\alpha^\top\Sigma_2\alpha\,/\,\alpha^\top\Sigma_1\alpha$'
# B: weight change on a fixed adjacency matrix
for mode in ('full', 'sub'):
    y = data[f'wts_{mode}']
    ax[1].errorbar(SIGMAS, y.mean(0), y.std(0), color=C[mode], marker='o', ms=3.5, capsize=2)
ax[1].axhline(1, color=MUTED, lw=0.8, ls='--')
ax[1].set(yscale='log', yticks=[1, 2, 5, 10], yticklabels=['1', '2', '5', '10'],
          xlabel=r'coupling heterogeneity $\sigma_S$', ylabel=RATIO)

# C: number of observed neurons, topology change
y = data['nsweep']
ax[2].errorbar(NSUB, y.mean(0), y.std(0), color=C['sub'], marker='o', ms=3.5, capsize=2, label='simulation')
ax[2].axhline(1 + c_rec / c_ext, color=COLORS['black'], lw=1, ls=':', label=r'$1+c_\mathrm{rec}/c_\mathrm{ext}$')
ax[2].axhline(1, color=MUTED, lw=0.8, ls='--')
ax[2].set(xscale='log', yscale='log', yticks=[1, 2, 4, 8], yticklabels=['1', '2', '4', '8'],
          xlabel='number of observed neurons $n$', ylabel=RATIO)
ax[2].legend(fontsize=6, loc='center left')

# D: choice of projection direction (n = NSAMPLE, topology change)
y = data['ctrl']
xs = np.arange(3)
ax[3].bar(xs, y.mean(0), yerr=y.std(0), width=0.55, color=C['sub'], capsize=2)
ax[3].axhline(1, color=MUTED, lw=0.8, ls='--')
ax[3].set_xticks(xs, [r'$\boldsymbol{\alpha}$', 'random', r'$\mathbf{1}/\sqrt{n}$'])
ax[3].set(xlabel='projection direction $u$', ylabel=r'$u^\top\Sigma_2 u\,/\,u^\top\Sigma_1 u$')

for a_ in ax[1:3]:
    a_.yaxis.set_minor_formatter(plt.NullFormatter())
for a_, l in zip(ax, 'ABCD'):
    a_.set_title(l, loc='left', fontweight='bold', fontsize=9)
fig.legend([plt.Line2D([], [], color=C[m]) for m in C], [LAB[m] for m in C],
           loc='outside upper center', ncol=2, fontsize=7)
fig.savefig(path / 'figures' / 'figS2_minor_direction.pdf')

# %%
