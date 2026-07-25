#%%
from common import *
path = Path(__file__).parents[1]
data_path = path / 'N4000'

fname = 'LIFNet-K=40mu=50_T=4.00e+05'
vol_data = c4u.fetch_voltage(data_path / (fname+'_voltage.dat'), 200, (0, 4e5))
ts = vol_data[1:-1, 0].copy()
DDV = np.diff(np.diff(vol_data[:, 1:], axis=0), axis=0)
DDV_mean = np.abs(DDV.mean(1))
#%%
import brainpy as bp
conn_seed = [0,1,2,3]
conn_paths = [data_path / f"connect_matrix-p=0.020-s{seed_:d}.npy" for seed_ in conn_seed]
mask_files = [data_path / f'connect_matrix-p=0.020-s{seed_:d}_mask.npy' for seed_ in conn_seed]
from EINet import LIFNet
state = None
dt = 0.02
T_single = 1e3
vol_list = []
for i, conn_path in enumerate(conn_paths):
    model = LIFNet(4000, 40, 50, conn_path, i)
    monitors = {'E.V': model.E.V, 'I.V': model.I.V,}
    runner = bp.DSRunner(model, monitors=monitors, dt=dt, t0=i*T_single)
    if state is None:
        runner.run(5.)
        runner.reset_state()
    else:
        model = model.load_neu_state(state)
    runner.run(T_single)
    vol_list.append(np.hstack(
        (runner.mon.ts.reshape(-1,1), runner.mon['E.V'], runner.mon['I.V'])))
    runner.mon = None        # clear cached memory in monitors
    runner._monitors = None  # clear cached memory in monitors
    state = model.save_neu_state()
voltage = np.vstack(vol_list)

#%%
ts = voltage[1:-1, 0].copy()
DDV = np.diff(np.diff(voltage[:, 1:], axis=0), axis=0)
DDV_mean = np.abs(DDV.mean(1))

#%%
_,_,v = svds(DDV[-30000:], k=1, which='SM')
ts[30000], v.shape
#%%
# 假设 data 是原始序列
window_size = 10000
rolling_mean = np.convolve((DDV[::10]@v.T).flatten(), np.ones(window_size)/window_size, mode='same')
delta_mean = np.abs(np.diff(rolling_mean))

change_point = np.argmax(delta_mean)  # 最大变化位置
print(change_point)
plt.plot(ts[::10], rolling_mean, color='C0', lw=2)
# plt.plot(ts[::10], (DDV[::10]@v.T).flatten(), color='C0', lw=2)
# plt.axvline(x=ts[::10][change_point], color='r', linestyle='--')  
#%% # Load the voltage data
from scipy.ndimage import gaussian_filter1d
#%
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(2, 1, left=0.1, right=0.9, top=0.9, bottom=0.5)
axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
datas = [
    DDV_mean[::1],
    (DDV[::1]@v.T).flatten(),
]
for axi, data in zip(axs, datas):
    axi.plot(ts[::1], data, color='C0', lw=2)
    axi.set_ylim(0)
    axi.set_xlim(0,4e3)
    axi.set_xlabel('Time (ms)', fontsize=14)
    axi.set_ylabel(r'$|\langle\Delta^2 V_i(t)\rangle|$')
    axi.ticklabel_format(style='sci', scilimits=(0,0), axis='x')


gs = fig.add_gridspec(1, 1, left=0.1, right=0.9, top=0.18, bottom=0.1)
ax = fig.add_subplot(gs[0, 0])
ax.fill_between([0,1], -0.5, 1, color='C0', alpha=0.4, lw=0)
ax.fill_between([1,2], -0.5, 1, color='C1', alpha=0.4, lw=0)
ax.fill_between([2,3], -0.5, 1, color='C2', alpha=0.4, lw=0)
ax.fill_between([3,4], -0.5, 1, color='C3', alpha=0.4, lw=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xlim(0,4)
ax.set_ylim(-0.5, 1)
ax.set_xlabel('Time', fontsize=14)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])