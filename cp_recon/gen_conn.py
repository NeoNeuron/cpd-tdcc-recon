# ## References
# 
# 1. van Vreeswijk, C., & Sompolinsky, H. (1996). Chaos in neuronal networks with balanced excitatory and inhibitory activity. Science (New York, N.Y.), 274(5293), 1724–1726. https://doi.org/10.1126/science.274.5293.1724
# %%
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--path', type=str, default='./tmp/')
argparser.add_argument('--network', type=str, default='LIFNet')
argparser.add_argument('--num', type=int, default=4000)
argparser.add_argument('--K', type=int, default=40)
argparser.add_argument('--mu', type=int, default=50)
# argparser.add_argument('--delay', type=float, default=0.0)
# argparser.add_argument('--conn_seed', type=int, default=0)
argparser.add_argument('--strength_deviation', type=float, default=0.5)
argparser.add_argument('--T_total', type=float, default=4e4)
argparser.add_argument('--n_chunk', type=int, default=50)
# argparser.add_argument('--cuda', type=str, default='')
argparser.add_argument('--regen_conn', type=bool, default=False)
argparser.add_argument('--regen_weight', type=bool, default=False)
argparser.add_argument('--regen_mask', type=bool, default=False)
argparser.add_argument('--record_v', type=bool, default=True)
argparser.add_argument('--num_sampled', type=int, default=200)
args = argparser.parse_args()

import brainpy as bp
import brainpy.math as bm
import pdif.utils as utils
if args.cuda == '':
    bm.set_platform('cpu')
else:
    bm.set_platform('gpu')
import numpy as np
import pickle as pkl
import uuid
from EINet import *
from pathlib import Path
folder = Path(args.path)
folder.mkdir(parents=True, exist_ok=True)
NUM_E_SAMPLED = int(args.num_sampled*0.8)
NUM_I_SAMPLED = int(args.num_sampled*0.2)

N = args.num
K = args.K
num_e = int(args.num * 0.8)
num_i = int(args.num * 0.2)
mu = args.mu
regen_conn = args.regen_conn
regen_weight = args.regen_weight
regen_mask = args.regen_mask
strength_deviation = args.strength_deviation

# generate and save sparse conn_matrix
conn_path = folder / f"connect_matrix-p={K*2./N:.3f}-s0.npy"
mask_file = folder / f'connect_matrix-p={K*2./N:.3f}-s0_mask.npy'
if regen_conn or not conn_path.exists():
    conn_E2E = get_conn_matrix(num_e, num_e, K, mode='FixedPre', seed=100, include_self=False)
    conn_E2I = get_conn_matrix(num_e, num_i, K, mode='FixedPre', seed=200)
    conn_I2E = get_conn_matrix(num_i, num_e, K, mode='FixedPre', seed=300)
    conn_I2I = get_conn_matrix(num_i, num_i, K, mode='FixedPre', seed=400, include_self=False)
    conn_E2I[1] += num_e
    conn_I2E[0] += num_e
    conn_I2I += num_e
    reference_conn_mat = np.hstack([conn_E2E, conn_E2I, conn_I2E, conn_I2I])
    conn_E2E = conn_E2I = conn_I2E = conn_I2I = None
    np.save(conn_path, reference_conn_mat)
    # conn_mat = None
else:
    reference_conn_mat = np.load(conn_path)

rewire_seeds = np.arange(2, 11, 2).astype(int)
rewire_percentage = np.arange(0.2, 1.1, 0.2)
for seed_, rewire_p in zip(rewire_seeds, rewire_percentage):
    np.random.seed(seed_)
    new_conn = []
    for post_i in range(N):
        pre_i_pool = reference_conn_mat[0, reference_conn_mat[1] == post_i]
        exist_e_pre = pre_i_pool[pre_i_pool <  num_e]
        exist_i_pre = pre_i_pool[pre_i_pool >= num_e]
        uncon_e_pre = np.setdiff1d(np.arange(num_e),         np.hstack([exist_e_pre, [post_i]]))
        uncon_i_pre = np.setdiff1d(np.arange(num_i) + num_e, np.hstack([exist_i_pre, [post_i]]))
        num_new_e_pre = int(np.round(len(exist_e_pre) * rewire_p))
        num_new_i_pre = int(np.round(len(exist_i_pre) * rewire_p))
        new_e_pre = np.random.choice(uncon_e_pre, num_new_e_pre, replace=False)
        new_i_pre = np.random.choice(uncon_i_pre, num_new_i_pre, replace=False)
        old_e_pre = np.random.choice(exist_e_pre, len(exist_e_pre)-num_new_e_pre, replace=False)
        old_i_pre = np.random.choice(exist_i_pre, len(exist_i_pre)-num_new_i_pre, replace=False)
        all_pre = np.hstack([old_e_pre, old_i_pre, new_e_pre, new_i_pre])
        new_conn.append(np.vstack([all_pre, np.ones_like(all_pre) * post_i]))
    new_conn = np.hstack(new_conn)
    np.save(folder / f"connect_matrix-p={K*2./N:.3f}-s0-rewire{rewire_p:.1f}.npy", new_conn)

# if regen_mask or not mask_file.exists():
#     if NUM_E_SAMPLED > num_e:
#         conn_mat_mask = [
#             (conn_mat[0] < num_e)*(conn_mat[1] < num_e),    # E2E
#             (conn_mat[0] < num_e)*(conn_mat[1] >= num_e),   # E2I
#             (conn_mat[0] >= num_e)*(conn_mat[1] < num_e),   # I2E
#             (conn_mat[0] >= num_e)*(conn_mat[1] >= num_e),  # I2I
#             ]
#         full_conn_mats = [
#             np.eye(num_e, dtype=bool),
#             np.zeros((num_e,N-num_e), dtype=bool),
#             np.zeros((N-num_e,num_e), dtype=bool),
#             np.eye(N-num_e, dtype=bool),
#             ]
#         offsets = [(0,0), (0, num_e), (num_e, 0), (num_e, num_e)]
#     else:
#         conn_mat_mask = [
#             (conn_mat[0] < NUM_E_SAMPLED)*(conn_mat[1] < NUM_E_SAMPLED),        # E2E
#             (conn_mat[0] < NUM_E_SAMPLED)*(conn_mat[1] >= N-NUM_I_SAMPLED),     # E2I
#             (conn_mat[0] >= N-NUM_I_SAMPLED)*(conn_mat[1] < NUM_E_SAMPLED),     # I2E
#             (conn_mat[0] >= N-NUM_I_SAMPLED)*(conn_mat[1] >= N-NUM_I_SAMPLED),  # I2I
#             ]
#         full_conn_mats = [
#             np.eye(NUM_E_SAMPLED, dtype=bool),
#             np.zeros((NUM_E_SAMPLED,NUM_I_SAMPLED), dtype=bool),
#             np.zeros((NUM_I_SAMPLED,NUM_E_SAMPLED), dtype=bool),
#             np.eye(NUM_I_SAMPLED, dtype=bool),
#             ]
#         offsets = [(0,0), (0, N-NUM_I_SAMPLED), (N-NUM_I_SAMPLED, 0), (N-NUM_I_SAMPLED, N-NUM_I_SAMPLED)]
#     for mask in conn_mat_mask:
#         print('Number of connected pairs:', np.sum(mask))
#     print('Generating mask file')
#     # pick connected pairs
#     mask_pairs = []
#     for mask, full_mat, offset in zip(conn_mat_mask, full_conn_mats, offsets):
#         num_pairs = int(np.sum(mask))
#         _conn_mat_sparse = conn_mat[:, mask]
#         _conn_mat_sparse -= np.asarray(offset).reshape(2,1)
#         full_mat[_conn_mat_sparse[0].astype(int), _conn_mat_sparse[1].astype(int)] = True
#         unconn_pairs = np.asarray(np.nonzero(~full_mat))
#         # sample pairs
#         unconn_ids = np.random.choice(unconn_pairs.shape[1], num_pairs, replace=False)
#         mask_pairs.append(np.hstack((_conn_mat_sparse, unconn_pairs[:, unconn_ids])) + np.asarray(offset).reshape(2,1))
#     np.save(mask_file, np.hstack(mask_pairs))
# else:
#     mask_pairs = np.load(mask_file)

# if hasattr(weight_seeds, '__len__'):
#     weight_paths = [folder / f"connect_matrix-p={K*2./N:.3f}-s0-d{strength_deviation:.1f}-w{seed_:d}.npy" for seed_ in weight_seeds]
# else:
#     weight_paths = [folder / f"connect_matrix-p={K*2./N:.3f}-s0-d{strength_deviation:.1f}-w{weight_seeds:d}.npy", ]

# for seed_, weight_path in zip(weight_seeds, weight_paths): 
#     num_connections = reference_conn_mat.shape[1]
#     if regen_weight or not weight_path.exists():
#         np.random.seed(seed_)
#         weight_mat = np.random.uniform(
#             low=1-strength_deviation, high=1+strength_deviation, size=num_connections)
#         # chunking negative connections
#         weight_mat = np.maximum(weight_mat, 0.0)
#         np.save(weight_path, weight_mat)

# fname = f"{args.network:s}-K={K:d}mu={mu:d}s0w{strength_deviation:.1f}_T={T_total:.2e}_spike_train.dat"
# vfname = f"{args.network:s}-K={K:d}mu={mu:d}s0w{strength_deviation:.1f}_T={T_total:.2e}_voltage.dat"

# T_single = T_total/len(weight_seeds)
# define all parameter values need to explore
#%%