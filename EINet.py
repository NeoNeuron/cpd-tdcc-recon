# ## References
# 
# 1. van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. Science (New York, N.Y.), 274(5293), 1724–1726. https://doi.org/10.1126/science.274.5293.1724
# %%
import brainpy as bp
import brainpy.math as bm
import jax
from pathlib import PosixPath
from typing import Union
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def get_conn_matrix(num_pre:int, num_post:int, K:int,
                    mode:str='FixedPre', seed:int=0,
                    include_self:bool=True, sparse:bool=True):
    np.random.seed(seed)
    if sparse:
        conn_mat = None
        if mode == 'FixedPre':
            conn_mat = [np.vstack(
                (
                    np.random.choice(num_pre, size=K, replace=False),
                    i*np.ones(K, dtype=int)
                    )
                ) for i in range(num_post)]
        elif mode == 'FixedPost':
            conn_mat = [np.vstack(
                (
                    i*np.ones(K, dtype=int),
                    np.random.choice(num_post, size=K, replace=False)
                    )
                ) for i in range(num_pre)]
        else:
            raise NotImplementedError(f"mode {mode} not implemented!")
        conn_mat = np.hstack(conn_mat)
        if not include_self and num_pre == num_post:
            conn_mat = conn_mat[:, conn_mat[0]!=conn_mat[1]]
    else:
        if mode == 'FixedPre':
            conn_mat = np.zeros((num_pre, num_post), dtype=bool)
            for i in range(num_post):
                conn_mat[np.random.choice(num_pre, K, replace=False), i] = True
        elif mode == 'FixedPost':
            conn_mat = np.zeros((num_pre, num_post), dtype=bool)
            for i in range(num_pre):
                conn_mat[i, np.random.choice(num_post, K, replace=False)] = True
        else:
            raise NotImplementedError(f"mode {mode} not implemented!")
        if not include_self and num_pre == num_post:
            conn_mat[np.eye(num_pre, dtype=bool)] = False
    return conn_mat


def get_IJcomm(pre_ids, post_ids, pre_size, post_size, weight):
    conn = bp.connect.IJConn(pre_ids, post_ids)
    conn = conn(pre_size=pre_size, post_size=post_size)
    return bp.dnn.EventCSRLinear(conn, np.ones_like(pre_ids)*weight)

def get_comms_fromfile(conn_path:str, num_e:int, num_i:int,
                       w_e2e:float, w_e2i:float, w_i2e:float, w_i2i:float):
    conn_mat = np.load(conn_path)
    _conn_e2e = conn_mat[:, (conn_mat[0]< num_e) * (conn_mat[1]< num_e)]
    _conn_e2i = conn_mat[:, (conn_mat[0]< num_e) * (conn_mat[1]>=num_e)]
    _conn_e2i[1] -= num_e
    _conn_i2e = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]< num_e)]
    _conn_i2e[0] -= num_e
    _conn_i2i = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]>=num_e)]
    _conn_i2i -= num_e
    comm_e2e = get_IJcomm(_conn_e2e[0], _conn_e2e[1], num_e, num_e, w_e2e)
    comm_e2i = get_IJcomm(_conn_e2i[0], _conn_e2i[1], num_e, num_i, w_e2i)
    comm_i2e = get_IJcomm(_conn_i2e[0], _conn_i2e[1], num_i, num_e, w_i2e)
    comm_i2i = get_IJcomm(_conn_i2i[0], _conn_i2i[1], num_i, num_i, w_i2i)
    return comm_e2e, comm_e2i, comm_i2e, comm_i2i


class Exponential(bp.Projection):
  def __init__(self, pre, post, comm, delay=None, tau=5., E=0., **kwargs):
    super().__init__()
    self.proj = bp.dyn.ProjAlignPostMg2(
        pre=pre,
        delay=delay, 
        comm=comm,
        syn=bp.dyn.Expon.desc(size=post.num, tau=tau, sharding=[bm.sharding.NEU_AXIS]),
        out=bp.dyn.COBA.desc(E=E),
        post=post,
        **kwargs,
    )


class LIFNet(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float,
        conn_path:Union[str, PosixPath], poisson_seed=0,
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
            conn_path, num_e, num_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        self.E2E = bp.dyn.FullProjDelta(self.E, delay, comm_e2e, self.E)
        self.E2I = bp.dyn.FullProjDelta(self.E, delay, comm_e2i, self.I)
        self.I2E = bp.dyn.FullProjDelta(self.I, delay, comm_i2e, self.E)
        self.I2I = bp.dyn.FullProjDelta(self.I, delay, comm_i2i, self.I)

        # connectioni from noise neurons to excitatory and inhibitory neurons
        self.Poi_E = bp.dyn.PoissonGroup(num_e, freqs=freqs, seed=poisson_seed)
        self.Poi_I = bp.dyn.PoissonGroup(num_i, freqs=freqs, seed=poisson_seed*1000)
        self.Poi2E = bp.dyn.FullProjDelta(self.Poi_E, 0.0, get_IJcomm(np.arange(num_e), np.arange(num_e), num_e, num_e, f2e), self.E)
        self.Poi2I = bp.dyn.FullProjDelta(self.Poi_I, 0.0, get_IJcomm(np.arange(num_i), np.arange(num_i), num_i, num_i, f2i), self.I)
    
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
        # curE2E = self.E.sum_delta_inputs(label='E')
        # curI2E = self.E.sum_delta_inputs(label='I')
        # curE2I = self.I.sum_delta_inputs(label='E')
        # curI2I = self.I.sum_delta_inputs(label='I')
        # return curE2E, curI2E, curE2I, curI2I

    
    def save_neu_state(self):
        return {
            'E': self.E.V.to_numpy(),
            'I': self.I.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E']
        self.I.V = state_dict['I']
        return self


class LIFNet_monitor(bp.Network):
    def __init__(self,
        num_neurons:int, K:int, mu:float,
        conn_path:Union[str, PosixPath], poisson_seed=0,
        delay:float=None, method='exp_auto'):
        super().__init__()
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=0.,
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
        self.I = bp.dyn.LifRef(num_i, V_th=1.0, **pars)

        # synapses
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, num_e, num_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        self.E2E = bp.dyn.FullProjDelta(self.E, delay, comm_e2e, self.E, out_label='W')
        self.E2I = bp.dyn.FullProjDelta(self.E, delay, comm_e2i, self.I, out_label='W')
        self.I2E = bp.dyn.FullProjDelta(self.I, delay, comm_i2e, self.E, out_label='W')
        self.I2I = bp.dyn.FullProjDelta(self.I, delay, comm_i2i, self.I, out_label='W')

        # connectioni from noise neurons to excitatory and inhibitory neurons
        self.Poi_E = bp.dyn.PoissonGroup(num_e, freqs=freqs, seed=poisson_seed)
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
        curW2E = self.E.sum_delta_inputs(label='W')
        curP2E = self.E.sum_delta_inputs(label='P')
        curW2I = self.I.sum_delta_inputs(label='W')
        curP2I = self.I.sum_delta_inputs(label='P')
        return curW2E, curP2E, curW2I, curP2I, self.E.V, self.I.V, self.E.spike, self.I.spike
    
    def save_neu_state(self):
        return {
            'E': self.E.V.to_numpy(),
            'I': self.I.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E']
        self.I.V = state_dict['I']
        return self


class HHNet(bp.Network):
    def __init__(self, num_neurons:int, K:int, mu:float, conn_path:str, poisson_seed:int=0):
        super().__init__()
        # define parameters following the EI balance scaling
        n_e = int(num_neurons*4/5)
        n_i = int(num_neurons*1/5)
        # w_e2e = 0.050/bm.sqrt(K)  # excitatory synaptic weight
        # w_e2i = 0.060/bm.sqrt(K)  # excitatory synaptic weight
        # w_i2e = 0.665/bm.sqrt(K)  # inhibitory synaptic weight
        # w_i2i = 0.67/bm.sqrt(K)   # inhibitory synaptic weight
        # f2e   = 0.090/bm.sqrt(K)  # excitatory synaptic weight
        # f2i   = 0.090/bm.sqrt(K)  # excitatory synaptic weight
        # freqs = mu*K
        w_e2e =  1.0*10/bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0*10/bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0*10/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8*10/bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.0*10/bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  0.8*10/bm.sqrt(K)  # excitatory synaptic weight
        freqs = mu*K

        # define neuronal populations
        self.E  = bp.dyn.HH(size=n_e)
        self.I  = bp.dyn.HH(size=n_i)
        self.PE = bp.dyn.PoissonGroup(n_e, freqs=freqs, seed=poisson_seed)
        self.PI = bp.dyn.PoissonGroup(n_i, freqs=freqs, seed=poisson_seed+100)

        # define recurrent connections
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, n_e, n_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        
        self.E2E = bp.dyn.FullProjDelta(self.E, None, comm_e2e, self.E, out_label='E')
        self.E2I = bp.dyn.FullProjDelta(self.E, None, comm_e2i, self.I, out_label='E')
        self.I2E = bp.dyn.FullProjDelta(self.I, None, comm_i2e, self.E, out_label='I')
        self.I2I = bp.dyn.FullProjDelta(self.I, None, comm_i2i, self.I, out_label='I')
        # self.E2E = Exponential(self.E, self.E, comm_e2e, out_label='E')
        # self.E2I = Exponential(self.E, self.I, comm_e2i, out_label='E')
        # self.I2E = Exponential(self.I, self.E, comm_i2e, tau=10., E=-80, out_label='I')
        # self.I2I = Exponential(self.I, self.I, comm_i2i, tau=10., E=-80, out_label='I')

        # define feedforward connections
        p2e_comm = get_IJcomm(np.arange(n_e), np.arange(n_e), n_e, n_e, weight=f2e)
        p2i_comm = get_IJcomm(np.arange(n_i), np.arange(n_i), n_i, n_i, weight=f2i)
        # self.P2E = Exponential(self.PE, self.E, p2e_comm, out_label='E')  # synaptic projection
        # self.P2I = Exponential(self.PI, self.I, p2i_comm, out_label='E')  # synaptic projection
        self.P2E = bp.dyn.FullProjDelta(self.PE, None, p2e_comm, self.E, out_label='E')
        self.P2I = bp.dyn.FullProjDelta(self.PI, None, p2i_comm, self.I, out_label='E')
    
    def update(self):
        self.PE()
        self.PI()
        self.P2E()
        self.P2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
    #    curE2E = self.E.sum_delta_inputs(label='E')
    #    curI2E = self.E.sum_delta_inputs(label='I')
    #    curE2I = self.I.sum_delta_inputs(label='E')
    #    curI2I = self.I.sum_delta_inputs(label='I')
    #    curT2E = self.E.sum_delta_inputs()
    #    return curE2E, curI2E, curE2I, curI2I, curT2E
    
    def save_neu_state(self):
        return {
            'E.V': self.E.V.to_numpy(),
            'E.m': self.E.m.to_numpy(),
            'E.h': self.E.h.to_numpy(),
            'E.n': self.E.n.to_numpy(),
            'I.V': self.I.V.to_numpy(),
            'I.m': self.I.m.to_numpy(),
            'I.h': self.I.h.to_numpy(),
            'I.n': self.I.n.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E.V']
        self.E.m = state_dict['E.m']
        self.E.h = state_dict['E.h']
        self.E.n = state_dict['E.n']
        self.I.V = state_dict['I.V']
        self.I.m = state_dict['I.m']
        self.I.h = state_dict['I.h']
        self.I.n = state_dict['I.n']
        return self


class MLNet(bp.Network):
    def __init__(self, num_neurons:int, K:int, mu:float, conn_path:str, poisson_seed:int=0):
        super().__init__()
        # define parameters following the EI balance scaling
        n_e = int(num_neurons*4/5)
        n_i = int(num_neurons*1/5)
        w_e2e =  1.0 * 200 / bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0 * 200 / bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0 * 200 / bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8 * 200 / bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.0 * 200 / bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  0.8 * 200 / bm.sqrt(K)  # excitatory synaptic weight
        mu_e  =  1.0 * mu * K
        mu_i  =  1.0 * mu * K

        # define neuronal populations
        self.E  = bp.dyn.MorrisLecar(size=n_e)
        self.I  = bp.dyn.MorrisLecar(size=n_i)
        self.PE = bp.dyn.PoissonGroup(n_e, freqs=mu_e, seed=poisson_seed)
        self.PI = bp.dyn.PoissonGroup(n_i, freqs=mu_i, seed=poisson_seed+100)

        # define recurrent connections
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, n_e, n_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        
        self.E2E = bp.dyn.FullProjDelta(self.E, None, comm_e2e, self.E, out_label='E')
        self.E2I = bp.dyn.FullProjDelta(self.E, None, comm_e2i, self.I, out_label='E')
        self.I2E = bp.dyn.FullProjDelta(self.I, None, comm_i2e, self.E, out_label='I')
        self.I2I = bp.dyn.FullProjDelta(self.I, None, comm_i2i, self.I, out_label='I')
        # self.E2E = Exponential(self.E, self.E, comm_e2e, out_label='E')
        # self.E2I = Exponential(self.E, self.I, comm_e2i, out_label='E')
        # self.I2E = Exponential(self.I, self.E, comm_i2e, tau=10., E=-80, out_label='I')
        # self.I2I = Exponential(self.I, self.I, comm_i2i, tau=10., E=-80, out_label='I')

        # define feedforward connections
        p2e_comm = get_IJcomm(np.arange(n_e), np.arange(n_e), n_e, n_e, weight=f2e)
        p2i_comm = get_IJcomm(np.arange(n_i), np.arange(n_i), n_i, n_i, weight=f2i)
        # self.P2E = Exponential(self.PE, self.E, p2e_comm, out_label='E')  # synaptic projection
        # self.P2I = Exponential(self.PI, self.I, p2i_comm, out_label='E')  # synaptic projection
        self.P2E = bp.dyn.FullProjDelta(self.PE, None, p2e_comm, self.E, out_label='E')
        self.P2I = bp.dyn.FullProjDelta(self.PI, None, p2i_comm, self.I, out_label='E')
    
    def update(self):
        self.PE()
        self.PI()
        self.P2E()
        self.P2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
    #    curE2E = self.E.sum_delta_inputs(label='E')
    #    curI2E = self.E.sum_delta_inputs(label='I')
    #    curE2I = self.I.sum_delta_inputs(label='E')
    #    curI2I = self.I.sum_delta_inputs(label='I')
    #    curT2E = self.E.sum_delta_inputs()
    #    return curE2E, curI2E, curE2I, curI2I, curT2E
    
    def save_neu_state(self):
        return {
            'E.V': self.E.V.to_numpy(),
            'E.W': self.E.W.to_numpy(),
            'I.V': self.I.V.to_numpy(),
            'I.W': self.I.W.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E.V']
        self.E.W = state_dict['E.W']
        self.I.V = state_dict['I.V']
        self.I.W = state_dict['I.W']
        return self


class IzhNet(bp.Network):
    def __init__(self, num_neurons:int, K:int, mu:float, conn_path:str, poisson_seed:int=0):
        super().__init__()
        # define parameters following the EI balance scaling
        n_e = int(num_neurons*4/5)
        n_i = int(num_neurons*1/5)
        w_e2e =  1.0 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.8 * 30.0 / bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -2.4 * 30.0 / bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.8 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  1.4 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        mu_e  =  1.0 * mu * K
        mu_i  =  1.0 * mu * K

        # define neuronal populations
        self.E  = bp.dyn.Izhikevich(size=n_e)
        self.I  = bp.dyn.Izhikevich(size=n_i)
        self.PE = bp.dyn.PoissonGroup(n_e, freqs=mu_e, seed=poisson_seed)
        self.PI = bp.dyn.PoissonGroup(n_i, freqs=mu_i, seed=poisson_seed+100)

        # define recurrent connections
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, n_e, n_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        
        self.E2E = bp.dyn.FullProjDelta(self.E, None, comm_e2e, self.E, out_label='E')
        self.E2I = bp.dyn.FullProjDelta(self.E, None, comm_e2i, self.I, out_label='E')
        self.I2E = bp.dyn.FullProjDelta(self.I, None, comm_i2e, self.E, out_label='I')
        self.I2I = bp.dyn.FullProjDelta(self.I, None, comm_i2i, self.I, out_label='I')
        # self.E2E = Exponential(self.E, self.E, comm_e2e, out_label='E')
        # self.E2I = Exponential(self.E, self.I, comm_e2i, out_label='E')
        # self.I2E = Exponential(self.I, self.E, comm_i2e, tau=10., E=-80, out_label='I')
        # self.I2I = Exponential(self.I, self.I, comm_i2i, tau=10., E=-80, out_label='I')

        # define feedforward connections
        p2e_comm = get_IJcomm(np.arange(n_e), np.arange(n_e), n_e, n_e, weight=f2e)
        p2i_comm = get_IJcomm(np.arange(n_i), np.arange(n_i), n_i, n_i, weight=f2i)
        # self.P2E = Exponential(self.PE, self.E, p2e_comm, out_label='E')  # synaptic projection
        # self.P2I = Exponential(self.PI, self.I, p2i_comm, out_label='E')  # synaptic projection
        self.P2E = bp.dyn.FullProjDelta(self.PE, None, p2e_comm, self.E, out_label='E')
        self.P2I = bp.dyn.FullProjDelta(self.PI, None, p2i_comm, self.I, out_label='E')
    
    def update(self):
        self.PE()
        self.PI()
        self.P2E()
        self.P2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
    #    curE2E = self.E.sum_delta_inputs(label='E')
    #    curI2E = self.E.sum_delta_inputs(label='I')
    #    curE2I = self.I.sum_delta_inputs(label='E')
    #    curI2I = self.I.sum_delta_inputs(label='I')
    #    curT2E = self.E.sum_delta_inputs()
    #    return curE2E, curI2E, curE2I, curI2I, curT2E
    
    def save_neu_state(self):
        return {
            'E.V': self.E.V.to_numpy(),
            'E.u': self.E.u.to_numpy(),
            'I.V': self.I.V.to_numpy(),
            'I.u': self.I.u.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E.V']
        self.E.u = state_dict['E.u']
        self.I.V = state_dict['I.V']
        self.I.u = state_dict['I.u']
        return self

class IzhNet_monitor(bp.Network):
    def __init__(self, num_neurons:int, K:int, mu:float, conn_path:str, poisson_seed:int=0):
        super().__init__()
        # define parameters following the EI balance scaling
        n_e = int(num_neurons*4/5)
        n_i = int(num_neurons*1/5)
        w_e2e =  1.0 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.8 * 30.0 / bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -2.4 * 30.0 / bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.8 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  1.4 * 30.0 / bm.sqrt(K)  # excitatory synaptic weight
        mu_e  =  1.0 * mu * K
        mu_i  =  1.0 * mu * K

        # define neuronal populations
        self.E  = bp.dyn.Izhikevich(size=n_e)
        self.I  = bp.dyn.Izhikevich(size=n_i)
        self.PE = bp.dyn.PoissonGroup(n_e, freqs=mu_e, seed=poisson_seed)
        self.PI = bp.dyn.PoissonGroup(n_i, freqs=mu_i, seed=poisson_seed+100)

        # define recurrent connections
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, n_e, n_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        
        self.E2E = bp.dyn.FullProjDelta(self.E, None, comm_e2e, self.E, out_label='W')
        self.E2I = bp.dyn.FullProjDelta(self.E, None, comm_e2i, self.I, out_label='W')
        self.I2E = bp.dyn.FullProjDelta(self.I, None, comm_i2e, self.E, out_label='W')
        self.I2I = bp.dyn.FullProjDelta(self.I, None, comm_i2i, self.I, out_label='W')

        # define feedforward connections
        p2e_comm = get_IJcomm(np.arange(n_e), np.arange(n_e), n_e, n_e, weight=f2e)
        p2i_comm = get_IJcomm(np.arange(n_i), np.arange(n_i), n_i, n_i, weight=f2i)
        self.P2E = bp.dyn.FullProjDelta(self.PE, None, p2e_comm, self.E, out_label='P')
        self.P2I = bp.dyn.FullProjDelta(self.PI, None, p2i_comm, self.I, out_label='P')
    
    def update(self):
        self.PE()
        self.PI()
        self.P2E()
        self.P2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
        curW2E = self.E.sum_delta_inputs(label='W')
        curP2E = self.E.sum_delta_inputs(label='P')
        curW2I = self.I.sum_delta_inputs(label='W')
        curP2I = self.I.sum_delta_inputs(label='P')
        return curW2E, curP2E, curW2I, curP2I, self.E.V, self.I.V, self.E.spike, self.I.spike
    
    def save_neu_state(self):
        return {
            'E.V': self.E.V.to_numpy(),
            'E.u': self.E.u.to_numpy(),
            'I.V': self.I.V.to_numpy(),
            'I.u': self.I.u.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E.V']
        self.E.u = state_dict['E.u']
        self.I.V = state_dict['I.V']
        self.I.u = state_dict['I.u']
        return self

class QIFNet(bp.Network):
    def __init__(self, num_neurons:int, K:int, mu:float, conn_path:str, poisson_seed:int=0):
        super().__init__()
        # define parameters following the EI balance scaling
        n_e = int(num_neurons*4/5)
        n_i = int(num_neurons*1/5)
        w_e2e =  1.0 * 20.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0 * 20.0 / bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0 * 20.0 / bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8 * 20.0 / bm.sqrt(K)  # inhibitory synaptic weight
        f2e   =  1.0 * 20.0 / bm.sqrt(K)  # excitatory synaptic weight
        f2i   =  0.8 * 20.0 / bm.sqrt(K)  # excitatory synaptic weight
        mu_e  =  1.0 * mu * K
        mu_i  =  1.0 * mu * K

        # define neuronal populations
        self.E  = bp.dyn.QuaIF(size=n_e, tau=20.)
        self.I  = bp.dyn.QuaIF(size=n_i, tau=20.)
        self.PE = bp.dyn.PoissonGroup(n_e, freqs=mu_e, seed=poisson_seed)
        self.PI = bp.dyn.PoissonGroup(n_i, freqs=mu_i, seed=poisson_seed+100)

        # define recurrent connections
        comm_e2e, comm_e2i, comm_i2e, comm_i2i = get_comms_fromfile(
            conn_path, n_e, n_i, w_e2e, w_e2i, w_i2e, w_i2i
            )
        
        self.E2E = bp.dyn.FullProjDelta(self.E, None, comm_e2e, self.E, out_label='E')
        self.E2I = bp.dyn.FullProjDelta(self.E, None, comm_e2i, self.I, out_label='E')
        self.I2E = bp.dyn.FullProjDelta(self.I, None, comm_i2e, self.E, out_label='I')
        self.I2I = bp.dyn.FullProjDelta(self.I, None, comm_i2i, self.I, out_label='I')
        # self.E2E = Exponential(self.E, self.E, comm_e2e, out_label='E')
        # self.E2I = Exponential(self.E, self.I, comm_e2i, out_label='E')
        # self.I2E = Exponential(self.I, self.E, comm_i2e, tau=10., E=-80, out_label='I')
        # self.I2I = Exponential(self.I, self.I, comm_i2i, tau=10., E=-80, out_label='I')

        # define feedforward connections
        p2e_comm = get_IJcomm(np.arange(n_e), np.arange(n_e), n_e, n_e, weight=f2e)
        p2i_comm = get_IJcomm(np.arange(n_i), np.arange(n_i), n_i, n_i, weight=f2i)
        # self.P2E = Exponential(self.PE, self.E, p2e_comm, out_label='E')  # synaptic projection
        # self.P2I = Exponential(self.PI, self.I, p2i_comm, out_label='E')  # synaptic projection
        self.P2E = bp.dyn.FullProjDelta(self.PE, None, p2e_comm, self.E, out_label='E')
        self.P2I = bp.dyn.FullProjDelta(self.PI, None, p2i_comm, self.I, out_label='E')
    
    def update(self):
        self.PE()
        self.PI()
        self.P2E()
        self.P2I()
        self.E2E()
        self.E2I()
        self.I2E()
        self.I2I()
        self.E()
        self.I()
    #    curE2E = self.E.sum_delta_inputs(label='E')
    #    curI2E = self.E.sum_delta_inputs(label='I')
    #    curE2I = self.I.sum_delta_inputs(label='E')
    #    curI2I = self.I.sum_delta_inputs(label='I')
    #    curT2E = self.E.sum_delta_inputs()
    #    return curE2E, curI2E, curE2I, curI2I, curT2E
    
    def save_neu_state(self):
        return {
            'E.V': self.E.V.to_numpy(),
            'I.V': self.I.V.to_numpy(),
            }
    
    def load_neu_state(self, state_dict:dict):
        self.E.V = state_dict['E.V']
        self.I.V = state_dict['I.V']
        return self




# ====================================
# deprecated code 
_default_g_max = dict(type='homo', value=1., prob=0.1, seed=123)
_default_uniform = dict(type='uniform', w_low=0.1, w_high=1., prob=0.1, seed=123)
_default_normal = dict(type='normal', w_mu=0.1, w_sigma=1., prob=0.1, seed=123)

class Delta_old(bp.synapses.TwoEndConn):
  def __init__(
      self,
      pre: bp.NeuGroup,
      post: bp.NeuGroup,
      output: bp.synapses.SynOut = bp.synouts.CUBA(target_var='V'),
      g_max_par=_default_g_max,
      delay_step=None,
      post_ref_key: str=None,
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(pre, post, None, output=output, name=name, mode=mode)
    self.post_ref_key = post_ref_key
    if post_ref_key:
        self.check_post_attrs(post_ref_key)
    self.g_max_par = g_max_par
    self.delay_step = self.register_delay(f"{self.pre.name}.spike", delay_step, self.pre.spike)

  def update(self):
    pre_spike = self.get_delay_data(f"{self.pre.name}.spike", self.delay_step)
    if self.g_max_par['type'] == 'homo':
        f = lambda s: bm.jitconn.event_mv_prob_homo(s,
                                                    self.g_max_par['value'],
                                                    conn_prob=self.g_max_par['prob'],
                                                    shape=(self.pre.num, self.post.num),
                                                    seed=self.g_max_par['seed'],
                                                    transpose=True)
    elif self.g_max_par['type'] == 'uniform':
        f = lambda s: bm.jitconn.event_mv_prob_uniform(s,
                                                       w_low=self.g_max_par['w_low'],
                                                       w_high=self.g_max_par['w_high'],
                                                       conn_prob=self.g_max_par['prob'],
                                                       shape=(self.pre.num, self.post.num),
                                                       seed=self.g_max_par['seed'],
                                                       transpose=True)
    elif self.g_max_par['type'] == 'normal':
        f = lambda s: bm.jitconn.event_mv_prob_normal(s,
                                                      w_mu=self.g_max_par['w_mu'],
                                                      w_sigma=self.g_max_par['w_sigma'],
                                                      conn_prob=self.g_max_par['prob'],
                                                      shape=(self.pre.num, self.post.num),
                                                      seed=self.g_max_par['seed'],
                                                      transpose=True)
    else:
        raise ValueError
    if isinstance(self.mode, bm.BatchingMode):
        f = jax.vmap(f)
    post_vs = f(pre_spike)
    return self.output(post_vs)

#%%