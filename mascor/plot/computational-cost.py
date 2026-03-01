import os, pickle, json
import numpy as np
import matplotlib.pyplot as plt

Ns = [10, 20, 50,
      100, 200, 500,
      1000, 2000, 5000,
      10000]

def _safe_get_time(payload):
    """compute_wall_s가 없으면 e2e_wall_s로 폴백"""
    if not isinstance(payload, dict):
        return np.nan
    return payload.get("compute_wall_s", payload.get("e2e_wall_s", np.nan))

def _extract_curve(d, section_key, device, Ns):
    """
    d[section_key][str(n)]에서 시간을 꺼내 Ns 순서대로 리스트 반환
    section_key:
      - Transformer: "S2_transformer"
      - MILP: "S3_milp_parallel"
    """
    sec = d.get(section_key, {})
    if device == 'gpu':
        sec = sec[device]
    ys, xs = [], []
    for n in Ns:
        k = str(n)
        if k in sec:
            t = _safe_get_time(sec[k])
            if t is not None:
                xs.append(n)
                ys.append(t)
    return np.array(xs), np.array(ys, dtype=float)

def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)
    
MILP_data = []
core_list = [1, 4, 8, 12, 16]
for core in core_list:
    path = f"running-cost/compcost_results_workers_{core}.pkl"
    data = load_pkl(path) 
    MILP_data.append(_extract_curve(data, "S3_milp_parallel", "cpu", Ns))
PT_data = load_pkl("running-cost/compcost_results_gpu.pkl")
x_pt, y_pt = _extract_curve(PT_data, "S2_transformer", "gpu", Ns)
#%%
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.legend_handler import HandlerLine2D
rcParams['font.family'] = 'Arial'
rcParams['font.sans-serif'] = ['Arial']

# blue-pallete
start = np.array([0x8E/255, 0xC9/255, 0xFF/255])  
end   = np.array([0x00/255, 0x47/255, 0xB3/255])
marker_size = 7.5
colors = [
    start + (end - start) * (i / (len(core_list)-1))
    for i in range(len(core_list))
]

#viridis-pallete
#cmap = cm.get_cmap("viridis")
#n = len(core_list)
#colors = [cmap(i / (n-1)) for i in range(n)]

plt.figure(figsize=(7,5))
plt.plot(x_pt, y_pt, "-o", markeredgecolor="black",
         markeredgewidth=0.6, markersize=marker_size*1.2, color="#EF0000", label="Proposed model (GPU)")


for idx, (x, y) in enumerate(MILP_data):
    plt.plot(
        x, y,
        marker="s",
        markeredgecolor="black",
        markeredgewidth=0.6,
        color=colors[idx], 
        markersize=marker_size,
        label=f"Gurobi (CPU core {core_list[idx]})"
    )
plt.xlabel('Problem number', fontsize=14)
plt.ylabel("Solution time (s)", fontsize=14)
plt.tick_params(axis='both', length=6, width=1.2, labelsize = 13)
plt.ylim(1e0, 1e4)
#plt.xscale("log")
plt.yscale("log")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize = 11, frameon = True, edgecolor = "black", framealpha = 0.8,)
plt.tight_layout()
plt.savefig("plot/supporting/compute-time.pdf", dpi = 400)
plt.show()
plt.close()
#%%
import math
(x1,y1) = MILP_data[0]
(x16,y16) = MILP_data[-1]
for i in range(len(y16)):
    order = math.log10(y1[i]/y16[i])
    print(f"order of magnitude at {x16[i]}", order)
    
for i in range(len(y16)):
    order = math.log10(y16[i]/y_pt[i])
    print(f"order of magnitude at {x16[i]}", order)


