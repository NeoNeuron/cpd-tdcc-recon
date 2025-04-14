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
# %%
class EINet(bp.Network):
    def __init__(self, num_e:int, num_i:int, K:int,
                 mu0:float, conn_path:Union[str, PosixPath],
                 poisson_seed=0, delay_step:int=0,
                 method='exp_auto', **kwargs):
        super(EINet, self).__init__(**kwargs)
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=2., R=20.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            mode = bm.NonBatchingMode(), ref_var=True,
            )
        E = bp.neurons.LIF(num_e, input_var=False, V_th=1.0, **pars)
        I = bp.neurons.LIF(num_i, input_var=False, V_th=0.7, **pars)
        self.num_e = num_e
        self.num_i = num_i

        # synapses
        w_e2e =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8/bm.sqrt(K)  # inhibitory synaptic weight

        sparse = True
        syn_kws = dict(delay_step=delay_step, post_ref_key='refractory')
        if sparse:
            conn_mat = np.load(conn_path)
            _conn = conn_mat[:, (conn_mat[0]<num_e) * (conn_mat[1]<num_e)]
            self.E2E = bp.synapses.Delta(E, E, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_e2e, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]<num_e) * (conn_mat[1]>=num_e)]
            _conn[1] -= num_e
            self.E2I = bp.synapses.Delta(E, I, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_e2i, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]<num_e)]
            _conn[0] -= num_e
            self.I2E = bp.synapses.Delta(I, E, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_i2e, **syn_kws)
            _conn = conn_mat[:, (conn_mat[0]>=num_e) * (conn_mat[1]>=num_e)]
            _conn -= num_e
            self.I2I = bp.synapses.Delta(I, I, bp.conn.IJConn(_conn[0], _conn[1]), g_max=w_i2i, **syn_kws)
            _conn = None
            conn_mat = None
        else:
            conn_mat = np.load(conn_path)
            self.E2E = bp.synapses.Delta(E, E, bp.conn.MatConn(conn_mat[:num_e, :num_e]), g_max=w_e2e, **syn_kws)
            self.E2I = bp.synapses.Delta(E, I, bp.conn.MatConn(conn_mat[:num_e, num_e:]), g_max=w_e2i, **syn_kws)
            self.I2E = bp.synapses.Delta(I, E, bp.conn.MatConn(conn_mat[num_e:, :num_e]), g_max=w_i2e, **syn_kws)
            self.I2I = bp.synapses.Delta(I, I, bp.conn.MatConn(conn_mat[num_e:, num_e:]), g_max=w_i2i, **syn_kws)
            conn_mat = None

        # connectioni from noise neurons to excitatory and inhibitory neurons
        f2e = 1.0/bm.sqrt(K)  # excitatory synaptic weight
        f2i = 0.8/bm.sqrt(K)  # excitatory synaptic weight
        # self.ffwd_E = bp.synapses.PoissonInput(E.V, 1, freq=mu0*K, weight=f2e, mode=bm.NonBatchingMode())
        # self.ffwd_I = bp.synapses.PoissonInput(I.V, 1, freq=mu0*K, weight=f2i, mode=bm.NonBatchingMode())
        self.ffwd_E = bp.neurons.PoissonGroup(num_e, freqs=mu0*K, seed=poisson_seed)
        self.ffwd_I = bp.neurons.PoissonGroup(num_i, freqs=mu0*K, seed=poisson_seed*1000)
        self.ffwd2E = bp.synapses.Delta(self.ffwd_E, E, bp.conn.One2One(), g_max=f2e, post_ref_key='refractory')
        self.ffwd2I = bp.synapses.Delta(self.ffwd_I, I, bp.conn.One2One(), g_max=f2i, post_ref_key='refractory')

        self.E = E
        self.I = I
        self.FE = self.ffwd_E
        self.FI = self.ffwd_I
        self.w_e2e =  1.0/bm.sqrt(K)
        self.w_e2i =  1.0/bm.sqrt(K)
        self.w_i2e = -2.0/bm.sqrt(K)
        self.w_i2i = -1.8/bm.sqrt(K)
        self.f2e   =  1.0/bm.sqrt(K)
        self.f2i   =  0.8/bm.sqrt(K)
    
    def save_state(self):
        return {
            'E': self.E.V.to_numpy(),
            'I': self.I.V.to_numpy(),
            }
    
    def load_state(self, state_dict:dict):
        self.E.V = state_dict['E']
        self.I.V = state_dict['I']
        return self

_default_g_max = dict(type='homo', value=1., prob=0.1, seed=123)
_default_uniform = dict(type='uniform', w_low=0.1, w_high=1., prob=0.1, seed=123)
_default_normal = dict(type='normal', w_mu=0.1, w_sigma=1., prob=0.1, seed=123)

class Delta(bp.synapses.TwoEndConn):
  def __init__(
      self,
      pre: bp.NeuGroup,
      post: bp.NeuGroup,
      output: bp.SynOut = bp.synouts.CUBA(target_var='V'),
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

class EINet_jit(bp.Network):
    def __init__(self, num_e:int, num_i:int, K:int,
                 mu0:float,
                 poisson_seed=0, delay_step:int=0,
                 method='exp_auto', **kwargs):
        super(EINet_jit, self).__init__(**kwargs)
        pars = dict(
            V_rest=0., V_reset=0., tau=20., tau_ref=2., R=20.,
            method=method, V_initializer=bp.init.Uniform(0., 0.7),
            mode = bm.NonBatchingMode(), ref_var=True,
            )
        E = bp.neurons.LIF(num_e, input_var=False, V_th=1.0, **pars)
        I = bp.neurons.LIF(num_i, input_var=False, V_th=0.7, **pars)
        self.num_e = num_e
        self.num_i = num_i

        # synapses
        w_e2e =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_e2i =  1.0/bm.sqrt(K)  # excitatory synaptic weight
        w_i2e = -2.0/bm.sqrt(K)  # inhibitory synaptic weight
        w_i2i = -1.8/bm.sqrt(K)  # inhibitory synaptic weight

        syn_kws = dict(delay_step=delay_step, post_ref_key='refractory')
        self.E2E = Delta(E, E, g_max_par=dict(type='homo', value=w_e2e, prob=K/num_e, seed=12), **syn_kws)
        self.E2I = Delta(E, I, g_max_par=dict(type='homo', value=w_e2i, prob=K/num_e, seed=123), **syn_kws)
        self.I2E = Delta(I, E, g_max_par=dict(type='homo', value=w_i2e, prob=K/num_i, seed=1234), **syn_kws)
        self.I2I = Delta(I, I, g_max_par=dict(type='homo', value=w_i2i, prob=K/num_i, seed=12345), **syn_kws)

        # connectioni from noise neurons to excitatory and inhibitory neurons
        f2e = 1.0/bm.sqrt(K)  # excitatory synaptic weight
        f2i = 0.8/bm.sqrt(K)  # excitatory synaptic weight
        # self.ffwd_E = bp.synapses.PoissonInput(E.V, 1, freq=mu0*K, weight=f2e, mode=bm.NonBatchingMode())
        # self.ffwd_I = bp.synapses.PoissonInput(I.V, 1, freq=mu0*K, weight=f2i, mode=bm.NonBatchingMode())
        self.ffwd_E = bp.neurons.PoissonGroup(num_e, freqs=mu0*K, seed=poisson_seed, name='P4E')
        self.ffwd_I = bp.neurons.PoissonGroup(num_i, freqs=mu0*K, seed=poisson_seed*1000, name='P4I')
        self.ffwd2E = bp.synapses.Delta(self.ffwd_E, E, bp.conn.One2One(), g_max=f2e, post_ref_key='refractory', name='P2E')
        self.ffwd2I = bp.synapses.Delta(self.ffwd_I, I, bp.conn.One2One(), g_max=f2i, post_ref_key='refractory', name='P2I')

        self.E = E
        self.I = I
        self.FE = self.ffwd_E
        self.FI = self.ffwd_I
        self.w_e2e =  1.0/bm.sqrt(K)
        self.w_e2i =  1.0/bm.sqrt(K)
        self.w_i2e = -2.0/bm.sqrt(K)
        self.w_i2i = -1.8/bm.sqrt(K)
        self.f2e   =  1.0/bm.sqrt(K)
        self.f2i   =  0.8/bm.sqrt(K)
#%%