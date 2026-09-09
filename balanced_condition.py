# %%
from common import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import brainpy as bp
import brainpy.math as bm
bm.set_platform('gpu')
print(bp.__version__)
from EINet import get_IJcomm, get_comms_fromfile

from scipy.stats import norm
plt.rcParams.update({'axes.spines.top': False,
                 'axes.spines.right': False,
                 'axes.labelsize': 16,
                 'xtick.labelsize': 12,
                 'ytick.labelsize': 12,
                 })

class LIFNet(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float,
        conn_path:Union[str, PosixPath], poisson_seed=0,
        weight_path:Union[str, PosixPath]=None,
        delay:float=None, method='exp_auto'):
        super().__init__()
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=2.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            )
        num_e = int(num_neurons*0.8)
        num_i = int(num_neurons*0.2)
        w_e2e =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8/bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  0.8/bm.sqrt(K)  # excitatory synaptic weight
        freqs = mu*K

        self.E = bp.dyn.LifRef(num_e, V_th=1.0, **pars)
        self.I = bp.dyn.LifRef(num_i, V_th=0.7, **pars)

        # synapses
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, num_e, num_i, w_e2e, w_e2i, w_i2e, w_i2i, weight_path
            )
        print(comm_e2e.weight)
        self.E2E = bp.dyn.FullProjDelta(self.E, delay, comm_e2e, self.E, out_label='E')
        self.E2I = bp.dyn.FullProjDelta(self.E, delay, comm_e2i, self.I, out_label='E')
        self.I2E = bp.dyn.FullProjDelta(self.I, delay, comm_i2e, self.E, out_label='I')
        self.I2I = bp.dyn.FullProjDelta(self.I, delay, comm_i2i, self.I, out_label='I')

        # connectioni from noise neurons to excitatory and inhibitory neurons
        self.Poi_E = bp.dyn.PoissonGroup(num_e, freqs=freqs, seed=poisson_seed)
        self.Poi_I = bp.dyn.PoissonGroup(num_i, freqs=freqs, seed=poisson_seed*1000)
        self.Poi2E = bp.dyn.FullProjDelta(self.Poi_E, 0.0, get_IJcomm(np.arange(num_e), np.arange(num_e), num_e, num_e, f2e), self.E, out_label='E')
        self.Poi2I = bp.dyn.FullProjDelta(self.Poi_I, 0.0, get_IJcomm(np.arange(num_i), np.arange(num_i), num_i, num_i, f2i), self.I, out_label='E')
    
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
        curE2E = self.E.sum_delta_inputs(label='E')
        curI2E = self.E.sum_delta_inputs(label='I')
        curE2I = self.I.sum_delta_inputs(label='E')
        curI2I = self.I.sum_delta_inputs(label='I')
        return curE2E, curI2E, curE2I, curI2I
    
    def save_neu_state(self):
        return {
            'E': self.E.V.to_numpy(),
            'I': self.I.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E']
        self.I.V = state_dict['I']
        return self

from scipy.ndimage import gaussian_filter1d
from functools import partial
gf_window = 100
gf = partial(gaussian_filter1d, sigma=gf_window)
path = Path(__file__).parents[0]
data_path = path / 'N4000'

#%% run model
bm.set_dt(0.02)
warmup_time = 20         # ms
simulation_time = 1000   # ms
curE2E_list, curE2I_list, curI2E_list, curI2I_list = [], [], [], []
state = None
conn_path = data_path / f"connect_matrix-p=0.020-s0.npy"
weight_paths = [data_path / f"connect_matrix-p=0.020-s0-d1.0-w{i:d}.npy" for i in range(2)]
weight_paths[0] = None
for weight_path in weight_paths:
    model = LIFNet(num_neurons=4000, K=40, mu=50, conn_path=conn_path, poisson_seed=0, weight_path=weight_path)
    if state is None:
        indices = np.arange(int((warmup_time+simulation_time)/bm.get_dt()))
        curE2E, curI2E, curE2I, curI2I = bm.for_loop(
            model.step_run, indices, progress_bar=True)
        curE2E = curE2E[int(warmup_time/bm.get_dt()):, :]
        curE2I = curE2I[int(warmup_time/bm.get_dt()):, :]
        curI2E = curI2E[int(warmup_time/bm.get_dt()):, :]
        curI2I = curI2I[int(warmup_time/bm.get_dt()):, :]
        state = model.save_neu_state()
    else:
        model.load_neu_state(state)
        indices = np.arange(int(simulation_time/bm.get_dt()))
        curE2E, curI2E, curE2I, curI2I = bm.for_loop(
            model.step_run, indices, progress_bar=True)

    curE2E_list.append(curE2E)
    curE2I_list.append(curE2I)
    curI2E_list.append(curI2E)
    curI2I_list.append(curI2I)

ts = np.arange(int(len(weight_paths)*simulation_time/bm.get_dt())) * bm.get_dt()
curE2E = np.concatenate(curE2E_list, axis=0)
curE2I = np.concatenate(curE2I_list, axis=0)
curI2E = np.concatenate(curI2E_list, axis=0)
curI2I = np.concatenate(curI2I_list, axis=0)
#%%
fig, ax = plt.subplots(1,2, figsize=(16, 4), sharex=True, sharey=True)
ax[0].plot(ts, gf(curE2E[:, 0]), label='Excitatory inputs')
ax[0].plot(ts, gf(curI2E[:, 0]), label='Inhibitory inputs')
ax[0].plot(ts, gf(curE2E[:, 0] + curI2E[:, 0]), label='Total inputs')
ax[0].axhline(np.mean(curE2E[:, 0] + curI2E[:, 0]), color='C3', label='Total mean inputs')
ax[1].plot(ts, gf(curE2I[:, 0]), label='Excitatory inputs')
ax[1].plot(ts, gf(curI2I[:, 0]), label='Inhibitory inputs')
ax[1].plot(ts, gf(curE2I[:, 0] + curI2I[:, 0]), label='Total inputs')
ax[1].axhline(np.mean(curE2I[:, 0] + curI2I[:, 0]), color='C3', label='Total mean inputs')
for axi in ax:
    axi.axhline(0, ls='--', color='k', lw=1)
    axi.legend()
    axi.set_xlabel('Time (ms)')
    axi.set_ylabel('Input currents')
# ax[0].set_xlim(490,610)

#%%
fig, ax = plt.subplots(4, 1, figsize=(16, 8), 
                       gridspec_kw={'hspace': 0.2, 'top': 0.95, 'bottom': 0.4, 'left': 0.05},
                       sharex=True, sharey=True)
# v_ = np.random.randn(DDV_W.shape[1])
# v_rec = v_/np.linalg.norm(v_)
# ax[0].plot(ts[1:-1], np.abs(DDV.mean(1)), label='V')
# ax[1].plot(ts[1:-1], np.abs(DDV_ion.mean(1)), label='ion')
# ax[2].plot(ts[1:], np.abs(DDV_P.mean(1)), label='ext')
# ax[3].plot(ts[1:], np.abs(DDV_W.mean(1)), label='rec')
# ax[0].plot(ts[1:-1], np.abs(DDV@v_rec.T), label='V')
# ax[1].plot(ts[1:-1], np.abs(DDV_ion@v_rec.T), label='ion')
# ax[2].plot(ts[1:], np.abs(DDV_P@v_rec.T), label='ext')
# ax[3].plot(ts[1:], np.abs(DDV_W@v_rec.T), label='rec')
ax[0].plot(ts[1:-1], np.abs(DDV@v.T), label='V')
ax[0].axhline(np.abs(DDV@v.T)[:50000].std(), color='C1', lw=2, ls='--')
ax[0].axhline(np.abs(DDV@v.T)[50000:].std(), color='C2', lw=2, ls='--')
ax[1].plot(ts[1:-1], np.abs(DDV_ion@v.T), label='ion')
ax[2].plot(ts[1:], np.abs(DDV_P@v.T), label='ext')
ax[3].plot(ts[1:], np.abs(DDV_W@v.T), label='rec')
for axi in ax:
    axi.fill_between(
        ts[:int(scan_range/0.02)], 0, axi.get_ylim()[1], color='C0', alpha=0.2, lw=0)
ax[0].set_ylabel(r'$|\langle\Delta^2 v_i\rangle_i|$')
ax[0].set_ylabel(r'$|\langle\Delta^2 v_i\rangle_i|$')
ax[1].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{ion}\rangle_i|$')
ax[2].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{ext}\rangle_i|$')
ax[3].set_ylabel(r'$|\langle\Delta^2 v_i^\mathrm{rec}\rangle_i|$')
ax[3].set_xlabel('Time (ms)')
# ax[0].set_xlim(0,2000)
ax[0].set_xlim(990,1010)
ax[0].set_ylim(0, 0.4)
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
