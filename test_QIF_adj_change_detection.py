# %%
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import brainpy as bp
import brainpy.math as bm
bm.set_platform('gpu')
print(bp.__version__)
from EINet import LIFNet_monitor, get_IJcomm, IzhNet_monitor, QIFNet_monitor

from scipy.stats import norm
plt.rcParams.update({'axes.spines.top': False,
                 'axes.spines.right': False,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12,
                 })

def estimate_std(rate:float, type:str='total',
                 dt:float=0.02, rate_I:float=None,
                 KN_ratio:float=0.01, EI_ratio:float=4.0,
                 JEE:float = 1.0,  JIE:float = 1.0,
                 JEI:float = -2.0, JII:float = -1.8):
    """ estiamte std of second order derivative of voltage
    Args:
        rate (float): firing rate of excitatory neurons, in unit of kHz
        dt (float): numerical time step, in unit of ms
        rate_I (float): firing rate of inhibitory neurons
        KN_ratio (float): ratio of noise to signal
        EI_ratio (float): ratio of excitatory to inhibitory neurons
        JEE, JIE, JEI, JII: synaptic weights
    Returns:
        std (float): standard deviation of second order derivative of voltage

    """
    if rate_I is None:
        rate_I = rate
    if type == 'total':
        W_E = JEE + JIE / EI_ratio
        W_I = JEI * EI_ratio + JII
        return np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    elif type == 'E':
        W_E = JEE
        W_I = JEI * EI_ratio
        return (1+1/EI_ratio)*np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    elif type == 'I':
        W_E = JIE / EI_ratio
        W_I = JII
        return (1+EI_ratio)*np.sqrt(2*KN_ratio*dt*(
            EI_ratio * rate * W_E**2  + rate_I * W_I**2
            )/ (EI_ratio + 1.0))
    else:
        raise ValueError('type should be total, E or I')

from scipy.ndimage import gaussian_filter1d
from functools import partial
gf_window = 1
gf = partial(gaussian_filter1d, sigma=gf_window)

path = Path(__file__).parents[0]
data_path = path / 'N4000'
conn_paths = [data_path / f"connect_matrix-p=0.020-s{i:d}.npy" for i in range(2)]

#%% run model
bm.set_dt(0.02)
warmup_time = 20         # ms
simulation_time = 1000   # ms
curP_list, curW_list, V_list, spikes_list = [], [], [], []
state = None
# conn_path = data_path / f"connect_matrix-p=0.020-s0.npy"
# weight_paths = [data_path / f"connect_matrix-p=0.020-s0-d1.0-w{i:d}.npy" for i in range(2)]
for i, conn_path in enumerate(conn_paths):
    model = IzhNet_monitor(num_neurons=4000, K=40, mu=50, conn_path=conn_path, poisson_seed=i)
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
#%
DDV = np.diff(V, n=2, axis=0)
DDV_ion = np.diff(np.diff(V, axis=0) - curP[1:] - curW[1:], n=1, axis=0)
DDV_P = np.diff(curP, n=1, axis=0)
DDV_W = np.diff(curW, n=1, axis=0)
#%%
plt.plot(ts, V[:, 0], '-o', label='V')
plt.plot(ts[:-1], np.diff(V[:, 0]), '-o', label='dV')
plt.plot(ts[:-1], np.diff(V[:, 0]) - curP[1:,0], - curW[1:,0], '-o', label='Ion')
plt.plot(ts, curP[:, 0], '-o', label='P')
plt.plot(ts, curW[:, 0], '-o', label='W')
plt.legend()
plt.xlim(0,0.2)
plt.grid()
#%%
import scipy.sparse as sp
scan_range = 800 # in unit ms
_, s, v = sp.linalg.svds(DDV[0:int(scan_range/0.02)], k=1, return_singular_vectors='vh', which='SM', maxiter=1000)
#%%
_, s, vh = np.linalg.svd(DDV[0:int(scan_range/0.02)], full_matrices=False)
print(s[-1])
v = vh[-1]
#%%
fig, ax = plt.subplots(4, 1, figsize=(16, 8), 
                       gridspec_kw={'hspace': 0.2, 'top': 0.95, 'bottom': 0.4, 'left': 0.05},
                       sharex=True, sharey=True)
E_fr = E_spike.sum()/simulation_time/E_spike.shape[1]
I_fr = I_spike.sum()/simulation_time/I_spike.shape[1]
print(E_fr.mean(), I_fr.mean())

ax[0].plot(ts[1:-1], np.abs(DDV.mean(1)), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion.mean(1)), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P.mean(1)), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W.mean(1)), label='rec')
ax[0].set_ylabel(r'$|\langle\Delta^2 v_i\rangle_i|$')
ax[1].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{ion}\rangle_i|$')
ax[2].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{ext}\rangle_i|$')
ax[3].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{rec}\rangle_i|$')
ax[3].set_xlabel('Time (ms)')
ax[0].set_xlim(0,2000)
# ax[0].set_xlim(990,1010)
ax[0].set_ylim(0,)
#%%
fig, ax = plt.subplots(4, 1, figsize=(16, 8), 
                       gridspec_kw={'hspace': 0.2, 'top': 0.95, 'bottom': 0.4, 'left': 0.05},
                       sharex=True, sharey=True)
E_fr = E_spike.sum()/simulation_time/E_spike.shape[1]
I_fr = I_spike.sum()/simulation_time/I_spike.shape[1]
print(E_fr.mean(), I_fr.mean())
ax[0].plot(ts[1:-1], np.abs(DDV@v.T), label='V')
# ax[0].axhline(np.abs(DDV@v.T)[:50000].std(), color='C1', lw=2, ls='--')
# ax[0].axhline(np.abs(DDV@v.T)[50000:].std(), color='C2', lw=2, ls='--')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v.T), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v.T), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v.T), label='rec')
for axi in ax:
    axi.fill_between(
        ts[:int(scan_range/0.02)], 0, axi.get_ylim()[1], color='C0', alpha=0.2, lw=0)
ax[0].set_ylabel(r'$|\langle\hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}\rangle|$')
ax[1].set_ylabel(r'$|\langle\hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle|$')
ax[2].set_ylabel(r'$|\langle\hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle|$')
ax[3].set_ylabel(r'$|\langle\hat{\mathbf{v}}_n, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle|$')
ax[3].set_xlabel('Time (ms)')
ax[0].set_xlim(0,2000)
# ax[0].set_xlim(990,1010)
ax[0].set_ylim(0,)

#%%
v_rec = np.load(data_path / 'conn_s0_SM_sv.npy')
fig, ax = plt.subplots(4, 1, figsize=(16, 8), 
                       gridspec_kw={'hspace': 0.2, 'top': 0.95, 'bottom': 0.4, 'left': 0.05},
                       sharex=True, sharey=True)
E_fr = E_spike.sum()/simulation_time/E_spike.shape[1]
I_fr = I_spike.sum()/simulation_time/I_spike.shape[1]
print(E_fr.mean(), I_fr.mean())
ax[0].plot(ts[1:-1], np.abs(DDV@v_rec.T), label='V')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v_rec.T), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v_rec.T), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v_rec.T), label='rec')
ax[0].set_ylabel(r'$|\langle\boldsymbol{\alpha}, \Delta^2 \mathbf{v}\rangle|$')
ax[1].set_ylabel(r'$|\langle\boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ion}\rangle|$')
ax[2].set_ylabel(r'$|\langle\boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{ext}\rangle|$')
ax[3].set_ylabel(r'$|\langle\boldsymbol{\alpha}, \Delta^2 \mathbf{v}^\mathrm{rec}\rangle|$')
ax[3].set_xlabel('Time (ms)')
ax[0].set_xlim(0,2000)
# ax[0].set_xlim(990,1010)
ax[0].set_ylim(0,)


#%%
tmp = np.hstack([np.abs(DDV@v.T).flatten(), [0,0]])
tmp1 = tmp.reshape(-1, 4000).mean(1)
plt.plot(ts[::4000][:-1], np.diff(tmp1), '-o')
#%%
gs = fig.add_gridspec(1, 3, wspace=0.2, hspace=0.2, top=0.30, bottom=0.05, left=0.05, right=0.46)
ax_bottom = [fig.add_subplot(gsi) for gsi in gs]
ax_bottom[0].hist(np.diff(np.diff(E_V.mean(1))), bins=100, range=(-0.1, 0.1), density=True)
ax_bottom[1].hist(np.diff(curW2E.mean(1)), bins=100, range=(-0.1, 0.1), density=True)
ax_bottom[2].hist(np.diff(curP2E.mean(1)), bins=100, range=(-0.01, 0.01), density=True)
x = np.linspace(-0.1, 0.1, 100)
std = estimate_std(E_fr, dt=bm.get_dt(), rate_I=I_fr, type='E')
print('estimated rec std: ', std)
p = norm.pdf(x, 0, std)
ax_bottom[0].plot(x, p, '--r', linewidth=2, label='theory')
ax_bottom[1].plot(x, p, '--r', linewidth=2, label='theory')
x = np.linspace(-0.01, 0.01, 100)
std = np.sqrt(0.05 * 1.0 * bm.get_dt()/E_spike.shape[1])
print('estimated ext std: ', std)
p = norm.pdf(x, 0, std)
ax_bottom[2].plot(x, p, '--r', linewidth=2, label='theory')
ax_bottom[0].legend()
ax_bottom[0].set_xlabel(r'$\Delta^2 V_i$')
ax_bottom[1].set_xlabel(r'$\Delta^2 V_i^\mathrm{rec}$')
ax_bottom[2].set_xlabel(r'$\Delta^2 V_i^\mathrm{ext}$')
ax_bottom[0].set_ylabel('Density')
ax_bottom[1].set_ylabel('Density')
ax_bottom[2].set_ylabel('Density')

gs = fig.add_gridspec(1, 3, wspace=0.2, hspace=0.2, top=0.30, bottom=0.05, left=0.52, right=0.9)
ax_bottom = [fig.add_subplot(gsi) for gsi in gs]
ax_bottom[0].hist(np.diff(np.diff(I_V.mean(1))), bins=100, range=(-0.1, 0.1), density=True)
ax_bottom[1].hist(np.diff(curW2I.mean(1)), bins=100, range=(-0.1, 0.1), density=True)
ax_bottom[2].hist(np.diff(curP2I.mean(1)), bins=100, range=(-0.01, 0.01), density=True)
x = np.linspace(-0.1, 0.1, 100)
std = estimate_std(I_fr, dt=bm.get_dt(), rate_I=I_fr, type='I')
print('estimated rec std: ', std)
p = norm.pdf(x, 0, std)
ax_bottom[0].plot(x, p, '--r', linewidth=2, label='theory')
ax_bottom[1].plot(x, p, '--r', linewidth=2, label='theory')
x = np.linspace(-0.01, 0.01, 100)
std = np.sqrt(0.05 * 0.8 * bm.get_dt()/I_spike.shape[1])
print('estimated ext std: ', std)
p = norm.pdf(x, 0, std)
ax_bottom[2].plot(x, p, '--r', linewidth=2, label='theory')
ax_bottom[0].legend()
ax_bottom[0].set_xlabel(r'$\Delta^2 V_i$')
ax_bottom[1].set_xlabel(r'$\Delta^2 V_i^\mathrm{rec}$')
ax_bottom[2].set_xlabel(r'$\Delta^2 V_i^\mathrm{ext}$')
ax_bottom[0].set_ylabel('Density')
ax_bottom[1].set_ylabel('Density')
ax_bottom[2].set_ylabel('Density')

fig.text(0.01, 0.95, 'a', fontsize=24, fontweight='bold')
fig.text(0.01, 0.32, 'b', fontsize=24, fontweight='bold')
fig.text(0.47, 0.95, 'c', fontsize=24, fontweight='bold')
fig.text(0.47, 0.32, 'd', fontsize=24, fontweight='bold')
fig.savefig(path / 'figures' / 'fig1_EI32k_raster.pdf', dpi=300, bbox_inches='tight')










#%%  more tests
bp.visualize.raster_plot(
    ts=indices*bm.get_dt(),
    sp_matrix=E_spike,
    title='E neuron spikes',
    xlabel='Time (ms)',
    ylabel='Neuron index',
    show=True,
)
print(E_spike.sum()/25600/(indices.shape[0]*bm.get_dt()))
print(I_spike.sum()/3200/(indices.shape[0]*bm.get_dt()))
#%%
DDV = np.diff(np.diff(E_V, axis=0), axis=0).mean(1)
# DDV = np.diff(np.diff(I_V, axis=0), axis=0).mean(1)
# DDV = np.hstack((DDV_E, DDV_I)).mean(1)
#%%
plt.hist(DDV, bins=100, density=True, range=(-0.1, 0.1))
mu, std = norm.fit(DDV)
print('fitted std: ', std)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
# plt.plot(x, p, 'k--', linewidth=2)
# rate = (E_spike.sum()+I_spike.sum())/(
#     E_spike.shape[1]+I_spike.shape[1])/(indices.shape[0]*bm.get_dt())
rate_E = E_spike.sum()/(E_spike.shape[1])/(indices.shape[0]*bm.get_dt())
rate_I = I_spike.sum()/(I_spike.shape[1])/(indices.shape[0]*bm.get_dt())
print(rate_E, rate_I)
# p = 0.01
# std = np.sqrt(p*20.46*2*bm.get_dt()*rate_E)
# std = estimate_std(rate_E, dt=bm.get_dt(), rate_I=rate_I, type='I')
# print('estimated std: ', std)
# p = norm.pdf(x, 0, std)
# plt.plot(x, p, 'r', linewidth=2)
std = estimate_std(rate_E, dt=bm.get_dt(), rate_I=rate_I, type='E')
plt.plot(x, norm.pdf(x, 0, std), 'C1', linewidth=2, label='E')
std = estimate_std(rate_E, dt=bm.get_dt(), rate_I=rate_I, type='I')
plt.plot(x, norm.pdf(x, 0, std), 'C2', linewidth=2, label='I')
std = estimate_std(rate_E, dt=bm.get_dt(), rate_I=rate_I, type='total')
plt.plot(x, norm.pdf(x, 0, std), 'C3', linewidth=2, label='total')
plt.legend()

# %%


# %%
class LIFNet_EI(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float, poisson_seed=0,
        delay:float=None, method='exp_auto'):
        super().__init__()
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=0., R=1.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            )
        num_e = int(num_neurons*0.8)
        num_i = int(num_neurons*0.2)
        w_e2e =  1.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_e2i =  1.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2e = -2.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8/bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  0.8/bm.sqrt(K)  # excitatory synaptic weight
        freqs = mu*K

        self.E = bp.dyn.LifRef(num_e, V_th=1.0, **pars)
        self.I = bp.dyn.LifRef(num_i, V_th=1.0, **pars)

        # synapses
        conn_mat = np.random.rand(num_e, num_e) < (K/num_e)
        pre_ids, post_ids = np.nonzero(conn_mat)
        comm_e2e = get_IJcomm(pre_ids, post_ids, num_e, num_e, w_e2e)
        self.E2E = bp.dyn.FullProjDelta(self.E, delay, comm_e2e, self.E, out_label='W')

        conn_mat = np.random.rand(num_e, num_i) < (K/num_e)
        pre_ids, post_ids = np.nonzero(conn_mat)
        comm_e2i = get_IJcomm(pre_ids, post_ids, num_e, num_i, w_e2i)
        self.E2I = bp.dyn.FullProjDelta(self.E, delay, comm_e2i, self.I, out_label='W')

        conn_mat = np.random.rand(num_i, num_e) < (K/num_i)
        pre_ids, post_ids = np.nonzero(conn_mat)
        comm_i2e = get_IJcomm(pre_ids, post_ids, num_i, num_e, w_i2e)
        self.I2E = bp.dyn.FullProjDelta(self.I, delay, comm_i2e, self.E, out_label='W')

        conn_mat = np.random.rand(num_i, num_i) < (K/num_i)
        pre_ids, post_ids = np.nonzero(conn_mat)
        comm_i2i = get_IJcomm(pre_ids, post_ids, num_i, num_i, w_i2i)
        self.I2I = bp.dyn.FullProjDelta(self.I, delay, comm_i2i, self.I, out_label='W')

        # connectioni from noise neurons to excitatory and inhibitory neurons
        self.Poi_E = bp.dyn.PoissonGroup(num_e, freqs=freqs, seed=poisson_seed*1000)
        self.Poi_I = bp.dyn.PoissonGroup(num_i, freqs=freqs, seed=poisson_seed*1000)
        self.Poi2E = bp.dyn.FullProjDelta(self.Poi_E, 0.0, get_IJcomm(np.arange(num_e), np.arange(num_e), num_e, num_e, f2e), self.E, out_label='P')
        self.Poi2I = bp.dyn.FullProjDelta(self.Poi_I, 0.0, get_IJcomm(np.arange(num_i), np.arange(num_i), num_i, num_i, f2i), self.I, out_label='P')
    
    def update(self):
        self.Poi_E()
        self.Poi_I()
        self.Poi2E()
        self.Poi2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
        return self.E.V, self.I.V, self.E.spike, self.I.spike


class LIFNet_pureI(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float, poisson_seed=0,
        delay:float=None, method='exp_auto'):
        super().__init__()
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=0., R=20.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            )
        num_i = int(num_neurons)
        w_i2i = -1.0/bm.sqrt(K)  # inhibitory synaptic weight
        f2i   =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        freqs = mu*K

        self.I = bp.dyn.LifRef(num_i, V_th=1.0, **pars)

        # synapses
        conn_mat = np.random.rand(num_i, num_i) < (K/num_i)
        pre_ids, post_ids = np.nonzero(conn_mat)
        comm_i2i = get_IJcomm(pre_ids, post_ids, num_i, num_i, w_i2i)
        self.I2I = bp.dyn.FullProjDelta(self.I, delay, comm_i2i, self.I, out_label='W')

        # connectioni from noise neurons to excitatory and inhibitory neurons
        self.Poi_I = bp.dyn.PoissonGroup(num_i, freqs=freqs, seed=poisson_seed*1000)
        self.Poi2I = bp.dyn.FullProjDelta(self.Poi_I, 0.0, get_IJcomm(np.arange(num_i), np.arange(num_i), num_i, num_i, f2i), self.I, out_label='P')
    
    def update(self):
        self.Poi_I()
        self.Poi2I()
        self.I2I()
        self.I()
        curW2I = self.I.sum_delta_inputs(label='W')
        curP2I = self.I.sum_delta_inputs(label='P')
        return curW2I, curP2I, self.I.V, self.I.spike
    
    def save_neu_state(self):
        return {
            'I': self.I.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.I.V = state_dict['I']
        return self

#%% run model
model = LIFNet_EI(num_neurons=4000, K=40, mu=10, method='euler')
bm.set_dt(0.02)
indices = np.arange(10000)
E_V, I_V, E_spike, I_spike = bm.for_loop(
    model.step_run, indices, progress_bar=True)
#%%
bp.visualize.raster_plot(
    ts=indices*bm.get_dt(),
    sp_matrix=E_spike,
    title='E neuron spikes',
    xlabel='Time (ms)',
    ylabel='Neuron index',
    show=True,
)
print(E_spike.sum()/3200/(indices.shape[0]*bm.get_dt()))
print(I_spike.sum()/800/(indices.shape[0]*bm.get_dt()))
#%%
DDV_E = np.diff(np.diff(E_V, axis=0), axis=0)
DDV_I = np.diff(np.diff(I_V, axis=0), axis=0)
DDV = np.hstack((DDV_E, DDV_I)).mean(1)[100:]
#%%
from scipy.stats import norm
plt.hist(DDV, bins=100, density=True, range=(-0.1, 0.1))
mu, std = norm.fit(DDV)
print('fitted std: ', std)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k--', linewidth=2)
# rate = (E_spike.sum()+I_spike.sum())/(
#     E_spike.shape[1]+I_spike.shape[1])/(indices.shape[0]*bm.get_dt())
rate_E = E_spike.sum()/(E_spike.shape[1])/(indices.shape[0]*bm.get_dt())
rate_I = I_spike.sum()/(I_spike.shape[1])/(indices.shape[0]*bm.get_dt())
print('rate_E: ', rate_E)
print('rate_I: ', rate_I)
p = 0.01
# std = np.sqrt(p*6.088*2*bm.get_dt()*rate_E*4)
std = np.sqrt(p*20.46*2*bm.get_dt()*rate_E)
# 20.46*0.01
# std = np.sqrt(p*2*bm.get_dt()*rate_I)
print('estimated std: ', std)
p = norm.pdf(x, 0, std)
plt.plot(x, p, 'r', linewidth=2)
# %%
