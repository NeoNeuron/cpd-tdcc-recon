# %%
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='./tmp/')
argparser.add_argument('--num', type=int, default=100)
argparser.add_argument('--K', type=int, default=40)
argparser.add_argument('--mu', type=int, default=50)
argparser.add_argument('--delay', type=float, default=0.0)
# argparser.add_argument('--conn_seed', type=int, default=0)
argparser.add_argument('--T_total', type=float, default=4e4)
argparser.add_argument('--n_processes', type=int, default=40)
argparser.add_argument('--cuda', type=str, default='')
argparser.add_argument('--conn_mode', type=str, choices=['FixedPre', 'FixedPost', 'random'], default='FixedPre')
argparser.add_argument('--regen_conn', type=bool, default=False)
argparser.add_argument('--regen_weight', type=bool, default=False)
argparser.add_argument('--record_v', type=bool, default=False)
args = argparser.parse_args()

import os
if args.cuda == '':
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['JAX_PLATFORMS'] = 'gpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
import brainpy as bp
import brainpy.math as bm
import pdif.utils as utils
if args.cuda == '':
    bm.set_platform('cpu')
else:
    bm.set_platform('gpu')
import numpy as np
import uuid
from EINet import get_conn_matrix, get_IJcomm

def get_comms_fromfile(
    conn_path:str, N:int, w_max:float, weight_path:np.ndarray=None):
    _conn_e2e = np.load(conn_path)
    if weight_path is not None:
        _w_e2e = np.load(weight_path)
        w_max = w_max * _w_e2e
    comm = get_IJcomm(_conn_e2e[0], _conn_e2e[1], N, N, w_max)
    return comm

class LIFNet(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float,
        conn_path, poisson_seed=0,
        weight_path=None,
        delay:float=None, method='exp_auto'):
        super().__init__()
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=2.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            )
        w_e2e =  0.02  # excitatory synaptic weight
        f2e   =  0.064  # excitatory synaptic weight
        freqs = mu*10

        self.E = bp.dyn.LifRef(num_neurons, V_th=1.0, **pars)

        # synapses
        comm_e2e = get_comms_fromfile(conn_path, num_neurons, w_e2e, weight_path)
        
        self.E2E = bp.dyn.FullProjDelta(self.E, delay, comm_e2e, self.E)

        self.Poi_E = bp.dyn.PoissonGroup(num_neurons, freqs=freqs, seed=poisson_seed)
        self.Poi2E = bp.dyn.FullProjDelta(self.Poi_E, 0.0, get_IJcomm(np.arange(num_neurons), np.arange(num_neurons), num_neurons, num_neurons, f2e), self.E)
    
    def update(self):
        self.Poi_E()
        self.Poi2E()
        self.E2E()
        self.E()
        # curE2E = self.E.sum_delta_inputs(label='E')
        # curI2E = self.E.sum_delta_inputs(label='I')
        # return curE2E, curI2E

    
    def save_neu_state(self):
        return {
            'E': self.E.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E']
        return self

from pathlib import Path
folder = Path(args.path)
folder.mkdir(parents=True, exist_ok=True)

N = args.num
K = args.K
mu = args.mu
T_total = args.T_total
delay = args.delay
regen_conn = args.regen_conn
regen_weight = args.regen_weight

fname = f"EINet-K={K:d}mu={mu:d}delay{delay:.1f}conn={args.conn_mode:s}_T={T_total:.2e}_spike_train.dat"
vfname = f"EINet-K={K:d}mu={mu:d}delay{delay:.1f}conn={args.conn_mode:s}_T={T_total:.2e}_voltage.dat"
# generate and save sparse conn_matrix
conn_path = folder / f"connect_matrix-{args.conn_mode:s}-p={K*1./N:.3f}.npy"
if regen_conn or not conn_path.exists():
    if args.conn_mode == 'random':
        conn_mat = (np.random.rand(N, N) < K*1./N) * (~np.eye(N, dtype=bool))
        conn_mat = np.asarray(np.nonzero(conn_mat))
    else:
        conn_mat = get_conn_matrix(N, N, K, mode=args.conn_mode, seed=100, include_self=False)
    np.save(conn_path, conn_mat)
else:
    conn_mat = np.load(conn_path)

strength_deviation = 1.0
weight_path = folder / f"connect_matrix-{args.conn_mode:s}-p={K*1./N:.3f}-s0-d{strength_deviation:.1f}.npy"
num_connections = conn_mat.shape[1]
if regen_weight or not weight_path.exists():
    np.random.seed(42)
    weight_mat = np.random.uniform(
        low=1-strength_deviation, high=1+strength_deviation, size=num_connections)
    # chunking negative connections
    weight_mat = np.maximum(weight_mat, 0.0)
    np.save(weight_path, weight_mat)

# define all parameter values need to explore
# import ray
# from ray.util.multiprocessing import Pool
# if not ray.is_initialized():
#     ray.init()
from multiprocessing import Pool
# num_total_cpus = ray.cluster_resources()['CPU']
# n_cpus_per_process = 4       # number of cpus to use for each process
n_processes = args.n_processes
 #int(num_total_cpus/n_cpus_per_process)     # number of processes to use
T_single = T_total/n_processes
# define all parameter values need to explore
record_v = args.record_v
#%%
def run_model(poisson_seed:int,):
    dt = 0.02
    init_state=None
    save_state=False
    model = LIFNet(N, K, mu, conn_path, poisson_seed, weight_path=weight_path, delay=delay)
    monitors = {'E.spike': model.E.spike, }
    if record_v:
        monitors.update({'E.V': model.E.V, })
        runner = bp.DSRunner(model, monitors=monitors, dt=dt, memory_efficient=False)
    else:
        runner = bp.DSRunner(model, monitors=monitors, dt=dt, memory_efficient=False)
    if init_state is None:
        runner.run(5.)
        runner.reset_state()
    else:
        model = model.load_neu_state(init_state)
    n_epoch = 1
    uid = str(uuid.uuid1())
    spk_fname = f"tmp{uid}_spike_train.dat"
    vol_fname = f"tmp{uid}_voltage.dat"
    for i in range(n_epoch):
        runner.run(T_single/n_epoch)
        print(f"{poisson_seed:d} th copy: E {runner.mon['E.spike'].sum()/N/T_single*1e3*n_epoch:f} Hz")
        spike_time = utils.prepare_save_spikes(runner.mon.ts, runner.mon['E.spike']).T

        if record_v:
            voltage = np.hstack((runner.mon.ts.reshape(-1,1), runner.mon['E.V']))
        if i == 0:
            if record_v:
                utils.save2bin(folder/spk_fname, spike_time, verbose=True)
                utils.save2bin(folder/vol_fname, voltage[::1], verbose=True)
            else:
                utils.save2bin(folder/spk_fname, spike_time, verbose=True)
        else:
            if record_v:
                utils.save2bin(folder/spk_fname, spike_time, 'ab', verbose=False)
                utils.save2bin(folder/vol_fname, voltage[::1], 'ab', verbose=False)
            else:
                utils.save2bin(folder/spk_fname, spike_time, 'ab', verbose=False)
    runner.mon = None        # clear cached memory in monitors
    # runner._monitors = None  # clear cached memory in monitors
    if record_v:
        if save_state:
            return spk_fname, vol_fname, model.save_neu_state()
        else:
            return spk_fname, vol_fname
    else:
        if save_state:
            return spk_fname, model.save_neu_state()
        else:
            return spk_fname
#%%
# mu_list = np.repeat([mu/2, mu], len(n_processes/2))
conn_id_list = np.arange(n_processes)
# pool = Pool(ray_address="auto",
#             processes=n_processes,
#             ray_remote_args={"num_cpus": n_cpus_per_process})
pool = Pool(n_processes)
processes = [pool.apply_async(run_model, [i,]) for i in conn_id_list]
pool.close()
pool.join()
results = [r.get() for r in processes]

# save data
for i in range(n_processes):
    mode = 'wb' if i==0 else 'ab'
    # concatenate spike trains
    if record_v:
        results_buffer = np.fromfile(folder / results[i][0], dtype=float).reshape(-1, 2)
    else:
        results_buffer = np.fromfile(folder / results[i], dtype=float).reshape(-1, 2)
    results_buffer[:,0] += i*T_single
    utils.save2bin(folder/fname, results_buffer, mode, verbose=True if i==0 else False)
    # concatenate voltage
    if record_v:
        results_buffer = np.fromfile(folder / results[i][1], dtype=float).reshape(-1, 1+N)
        results_buffer[:,0] += i*T_single
        utils.save2bin(folder/vfname, results_buffer, mode, verbose=True if i==0 else False)
    # clear tmp files
for i in range(n_processes):
    if record_v:
        (folder / results[i][0]).unlink()
        (folder / results[i][1]).unlink()
    else:
        (folder / results[i]).unlink()
# %%
