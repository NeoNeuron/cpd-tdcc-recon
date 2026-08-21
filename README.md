# CPD-TDCC: change-point-aware network reconstruction

Code for a paper (planned submission to *Philosophical Transactions of the Royal Society B*) proposing
**CPD-TDCC** — Change-Point Detection + Time-Delayed Correlation-Coefficient — a pipeline for reconstructing
network connectivity when the underlying structure (topology or coupling strength) changes partway through a
recording, rather than staying fixed for the whole trial as most causal-connectivity methods assume.

All example networks are excitatory/inhibitory **balanced-state** networks, demonstrated across multiple neuron
model types:
- LIF, Izhikevich, and QIF, all via `EINet.py`'s `LIFNet`/`LIFNet_monitor`, `IzhNet_monitor`, `QIFNet_monitor`
  (GPU-accelerated via `brainpy`/`jax`)
- Hodgkin-Huxley, via a Brian2-based simulation implemented directly inside
  `cp_recon/57 - 0 + 0 变点检测（四）.ipynb` ("change-point detection (4)") — used to probe a CPD failure case

## What's here

- `cp_recon/` — all paper code:
  - `common.py` — shared utilities, **and the CPD algorithm itself**: `TCD`, `TCD_Ftest` (an F-test-based
    change-point detector), `square_windowed_mean`, `square_windowed_Fstats`. Also re-exports `causal4.utils`
    (as `c4u`) and `causal4.Causality.CausalityEstimator`, so any script doing `from common import *` gets both
    transitively. The TDCC half of the pipeline comes entirely from the external `causal4` package (see
    Dependencies below) — this repo only supplies the change-point-detection half.
  - `gen_conn.py` — generates connectivity matrices (CLI, `--num`/`-K`/`--mu` etc.)
  - `recon_LIF.py` — LIF network reconstruction driver
  - `balanced_condition.py` / `balanced_condition_vary_W.py` — E/I balance exploration (varying weight
    heterogeneity in the latter)
  - `test_LIF_U_EE.py` — LIF parameter-sweep script (`multiprocessing.Pool`)
  - `test_QIF_adj_change_detection.py` — QIF/Izhikevich adjacency-change-detection experiment (GPU-bound, sets
    `CUDA_VISIBLE_DEVICES`)
  - `fig1.py` … `fig5_TCD.py` — the paper's figures (`fig5_TCD.py` is the main CPD/TCD_Ftest change-point figure,
    varying coupling heterogeneity)
  - Two Chinese-named development notebooks documenting the CPD/F-test method's origin: `变点检测-F检验.ipynb`
    ("change-point detection – F-test") and `57 - 0 + 0 变点检测（四）.ipynb` ("change-point detection (4)")
- `EINet.py` — this repo's only simulation dependency (pure-Python, `brainpy`/`jax`-based E-I network simulator).
  No C++ simulators or build step are needed for this paper at all.

## Dependencies

```bash
pip install -r requirements.txt
```

This installs [`causal4`](https://github.com/NeoNeuron/causal4-core) (the shared causality-estimation package —
`pip install` builds its C++ backend automatically, no separate `make` step) plus the CPU-side Python
dependencies. `EINet.py`'s GPU packages (`jax`, `brainpy`) need a custom index/CUDA suffix that plain
`pip install -r` can't resolve — install them separately per the commented instructions in `requirements.txt`.

## Known gaps (pre-existing, not introduced by extracting this repo)

- **`cp_recon/fig2.py` needs `schematics.png` and `DDV_SVD.png`** in its working directory. Neither file exists
  anywhere in the original monorepo's git history — they're presumably hand-drawn schematics that were never
  committed. `fig2.py` cannot run until these are sourced from wherever they were originally created.
- **No simulation/connectivity/voltage data is committed.** `gen_conn.py`, `recon_LIF.py`, and the balanced-network
  scripts write into `N4000/`, `N32000/`, and `N4000-2-normal/` (siblings of `cp_recon/`, git-ignored) — these
  must be regenerated locally before running the figure scripts. Parameters are hardcoded per-script rather than
  config-driven, e.g. `recon_LIF.py` expects `K=40, mu=50` connectivity matrices at `p=0.020` density
  (`connect_matrix-p=0.020-s{seed}.npy`) — check the individual script for the exact values it expects before
  regenerating data.
