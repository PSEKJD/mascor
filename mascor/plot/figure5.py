# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 21:20:11 2026

@author: com
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  
import pickle
from utils.env import ptx_env_single
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib as mpl
country_list = {
    'France': ['Dunkirk'],
    #'Denmark': ['Skive', 'Fredericia'],
    #'Germany': ['Weener']
    }
import json
from utils.env import ptx_env_stack
import numpy as np
import matplotlib.pyplot as plt
from utils.env.ptx_env_stack import *
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from utils.helper import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def loading_idx(country, region, min_diff = 0.02):
    log_dir = './optimization/result'
    save_path = os.path.join(log_dir,
                     f'{country}/{region}/iter_100_history_pfss_0.5_sample_size_1000.pkl')
    with open(save_path, "rb") as f:
        history_dict = pickle.load(f)
        
    des_list, obj_list, con_list = [], [], []
    K = len(history_dict)
    for i in range(K):
        step = history_dict[f"step-{i}"]
        des = torch.as_tensor(step["des"], dtype=torch.float32)
        pfss = torch.as_tensor(step["pfss"], dtype=torch.float32)
        con = (pfss - 0.5).unsqueeze(1)
        lcox = torch.as_tensor(step["mu-lcox[$/kg]"],     dtype=torch.float32)
        ctg  = torch.as_tensor(step["mu-ctg[ton/month]"], dtype=torch.float32)
        obj  = torch.stack([-lcox, -ctg/100], dim=1)
        des_list.append(des); obj_list.append(obj); con_list.append(con)
    des_set = torch.cat(des_list, dim=0)
    obj_set = torch.cat(obj_list, dim=0)
    con_set = torch.cat(con_list, dim=0)
    is_feas   = (con_set <= 0).all(-1)
    feas_des  = des_set[is_feas]
    feas_obj  = obj_set[is_feas]
    feas_con  = con_set[is_feas]
    pareto_m  = is_non_dominated(feas_obj)
    pareto_obj = feas_obj[pareto_m]
    keep_mask = min_diff_filter_on_lcox_min(-pareto_obj, min_diff = min_diff)
    idx_list = keep_mask.nonzero(as_tuple=True)[0].tolist()
    return idx_list

def loading_performance(country, region, min_diff = 0.02, data_type = "test"):
    log_dir = './optimization/result'
    if data_type == "test":
        save_template = os.path.join(log_dir,f'{country}/{region}/uq_results_pfss_0.5_test_idx{{idx}}.pkl')
    else:
        save_template = os.path.join(log_dir,f'{country}/{region}/uq_results_pfss_0.5_idx{{idx}}.pkl')
    env_list = []
    idx_list = loading_idx(country, region, min_diff = min_diff)
    for idx in idx_list:
        save_path = save_template.format(idx=idx)
        with open(save_path, "rb") as file:
            env = pickle.load(file)
        env_list.append(env)
    num_points = len(env_list)
    lcox_list = np.zeros((num_points, 1000))
    ctg_list = np.zeros((num_points, 1000))
    des_list = np.zeros((num_points, 4))
    pfss_list = np.zeros((num_points, 1))

    for i in range(num_points):
        env = env_list[i]
        mu_LCOX, var_LCOX, LCOX = env.LCOX.mean(), env.LCOX.var(), env.LCOX
        ctg = np.sum(env.CO2_emit.detach().cpu().numpy(), axis = 1)
        pf = len(np.where(ctg>0)[0])/len(ctg)
        mu_ctg = np.mean(ctg)
        var_ctg = np.var(ctg)
        X_flow = env.X_flow #kg
        #LCOX[np.where(LCOX<0)] = 0
        lcox_list[i, :] = LCOX
        ctg_list[i, :] = ctg
        des_list[i, :] = np.array([env.X_flow, env.PEM_ratio, env.ESS_cap, env.LH2_cap])
        pfss_list[i,:] = pf

    mu_lcox = np.mean(lcox_list, axis=1)
    idx = np.argsort(mu_lcox)
    
    return lcox_list[idx], ctg_list[idx], des_list[idx], pfss_list[idx], np.array(idx_list)[idx], env_list, idx

#%% Figure 5 (a)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
country = "Germany"
region = "Weener"

log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
num = 1000
se_ctg_pred_list = np.zeros(shape = (num ,576))
se_ctg_list = np.zeros(shape = (num ,576))
se_rtg_list = np.zeros(shape = (num ,576))
se_renew_list = np.zeros(shape = (num ,576))
se_prob_vio_list = np.zeros(shape = (num ,576))
se_vio_acc_list = np.zeros(shape = (num ,576))
se_cum_list = np.zeros(shape = (num ,576))

pr_ctg_pred_list = np.zeros(shape = (num,576))
pr_ctg_list = np.zeros(shape = (num,576))
pr_rtg_list = np.zeros(shape = (num,576))
pr_renew_list = np.zeros(shape = (num,576))
pr_prob_vio_list = np.zeros(shape = (num,576))
pr_vio_acc_list = np.zeros(shape = (num ,576))
pr_cum_list = np.zeros(shape = (num,576))

# epoch times (24 points)
epoch_times = np.arange(24, 577, 24)  # [24,48,...,576]
K = len(epoch_times)  # 24
T = 576

def epoch_uncertainty_from_forecast(forecast_24_20_576):
    """
    forecast: (24,20,576) where k-th forecast is issued at t_k=24*(k+1)
    Uncertainty at epoch k:
        mean_{tau=t_k..T} std_scenario( forecast[k, :, tau] )
    returns: (24,)
    """
    fcst = np.asarray(forecast_24_20_576, dtype=float)
    assert fcst.shape == (24, 50, 576), f"forecast shape must be (24,20,576), got {fcst.shape}"

    unc = np.zeros(24, dtype=float)
    for k, t_k in enumerate(epoch_times):
        start = t_k - 1  # python index
        # std across 20 scenarios at each absolute future time
        std_future = np.std(fcst[k, :, start:], axis=0)  # (576-start,)
        unc[k] = std_future.mean()  # scalar
    return unc

# -------------------------
# collect uncertainty for SE / PR across idx
# -------------------------
N = 1000
se_unc_epoch = np.full((N, K), np.nan, dtype=float)
pr_unc_epoch = np.full((N, K), np.nan, dtype=float)

scaler_path = os.path.join('./dataset/', f'{country}/{region}/oracle_dataset_c_fax_fix_sample_50000')
with open(os.path.join(scaler_path, 'scaler_package.pkl'), "rb") as file: 
    scaler_package = pickle.load(file)
co2_scaler = scaler_package['co2']
reward_scaler = scaler_package['reward']
rtg_scaler = scaler_package['rtg']
ctg_scaler = scaler_package['ctg']
ctg_mu  = ctg_scaler.mean_[0]
ctg_std = ctg_scaler.scale_[0]

for idx in range(1000):
    if region == "Dunkirk" or region == "Skive":
        save_path = os.path.join(log_dir, f"operation_results_PT_solver_SE1_idx_{idx}.pkl")
    else:
        save_path = os.path.join(log_dir, f"operation_results_PT_solver_SE1_idx_{idx}_noise_infer_True.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    se_ctg_list[idx] = (env.ctg*ctg_std + ctg_mu)[:,0]
    se_rtg_list[idx] = np.cumsum(env.cost_list)/1e6
    se_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    se_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled  * ctg_std + ctg_mu
    se_ctg_pred_list[idx] = mu_ctg
    std_ctg = std_ctg_scaled * ctg_std
    z = (-cum_co2 - mu_ctg) / std_ctg
    se_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,)
    
    final_vio = (cum_co2[-1] > 0)   # episode-level ground truth (bool)
    pred_pos = se_prob_vio_list[idx] > 0.5
    pred_neg = se_prob_vio_list[idx] <= 0.5
    tp = pred_pos & final_vio
    tn = pred_neg & (~final_vio)
    se_vio_acc_list[idx] = (tp | tn).astype(int)
    
for idx in range(1000):
    if region == "Dunkirk" or region == "Skive":
        save_path = os.path.join(log_dir, f"operation_results_PT_solver_PRN_idx_{idx}.pkl")
    else:
        save_path = os.path.join(log_dir, f"operation_results_PT_solver_PRN_idx_{idx}_noise_infer_True.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    pr_ctg_list[idx] = (env.ctg*ctg_std + ctg_mu)[:,0]
    pr_rtg_list[idx] = np.cumsum(env.cost_list)/1e6
    pr_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    pr_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled  * ctg_std + ctg_mu
    pr_ctg_pred_list[idx] = mu_ctg
    std_ctg = std_ctg_scaled * ctg_std
    z = (-cum_co2 - mu_ctg) / std_ctg
    pr_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,) 
    
    final_vio = (cum_co2[-1] > 0)   # episode-level ground truth (bool)
    pred_pos = pr_prob_vio_list[idx] > 0.5
    pred_neg = pr_prob_vio_list[idx] <= 0.5
    tp = pred_pos & final_vio
    tn = pred_neg & (~final_vio)
    pr_vio_acc_list[idx] = (tp | tn).astype(int)
    
#%% Figure 5 (a)
import numpy as np
import matplotlib.pyplot as plt

T = se_cum_list.shape[1]
t = np.arange(T)

lcox_tr, ctg_tr, des_tr, pfss_tr, _, env_tr, idx_tr = loading_performance(country, region, data_type="train")
env_tr_min = env_tr[idx_tr[0]]
env_tr_max = env_tr[idx_tr[-1]]

# -------------------------
# mean / std across 1000 runs
# -------------------------
se_rtg_mu  = np.mean(se_rtg_list, axis=0)
se_rtg_std = np.std(se_rtg_list, axis=0)

pr_rtg_mu  = np.mean(pr_rtg_list, axis=0)
pr_rtg_std = np.std(pr_rtg_list, axis=0)

se_cum_mu  = np.mean(se_cum_list, axis=0)
se_cum_std = np.std(se_cum_list, axis=0)

pr_cum_mu  = np.mean(pr_cum_list, axis=0)
pr_cum_std = np.std(pr_cum_list, axis=0)

se_p_mu  = np.mean(se_prob_vio_list, axis=0)
se_p_std = np.std(se_prob_vio_list, axis=0)

pr_p_mu  = np.mean(pr_prob_vio_list, axis=0)
pr_p_std = np.std(pr_prob_vio_list, axis=0)

se_unc_mu  = np.nanmean(se_unc_epoch, axis=0)  # (24,)
se_unc_std = np.nanstd(se_unc_epoch, axis=0)

pr_unc_mu  = np.nanmean(pr_unc_epoch, axis=0)
pr_unc_std = np.nanstd(pr_unc_epoch, axis=0)

se_vio_acc = np.mean(se_vio_acc_list, axis = 0)
pr_vio_acc = np.mean(pr_vio_acc_list, axis = 0)
# -------------------------
# plotting
# -------------------------
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True

fig, axes = plt.subplots(
    3, 1,
    figsize=(5, 6.6),
    sharex=True,
    gridspec_kw={"height_ratios": [2.0, 2.0, 1.2], "hspace": 0.15}
)

ax0, ax1, ax2 = axes

# =========================
# (a) RTG
# =========================
ax0.plot(t, se_rtg_mu, lw=2.0, color="red", label="SE1 (mean)")
ax0.fill_between(
    t, se_rtg_mu - se_rtg_std, se_rtg_mu + se_rtg_std,
    color="red", alpha=0.18, linewidth=0
)

if region == "Dunkirk":
    prn_color = "#C00000"
else:
    prn_color = "#00994C"
ax0.plot(t, pr_rtg_mu, lw=2.0, color= prn_color, label="PRN (mean)")
ax0.fill_between(
    t, pr_rtg_mu - pr_rtg_std, pr_rtg_mu + pr_rtg_std,
    color= prn_color, alpha=0.18, linewidth=0
)
ax0.set_ylim(0, 1.2)
#ax0.set_ylabel("RTG")
#ax0.set_title("(a) Return-to-go (mean ± std)")
#ax0.legend(frameon=False, loc="best")

# =========================
# (b) Cumulative CO2
# =========================
ax1.plot(t, se_cum_mu, lw=2.0, color="red", label="SE1 (mean)")
ax1.fill_between(
    t, se_cum_mu - se_cum_std, se_cum_mu + se_cum_std,
    color="red", alpha=0.18, linewidth=0
)

ax1.plot(t, pr_cum_mu, lw=2.0, color=prn_color, label="PRN (mean)")
ax1.fill_between(
    t, pr_cum_mu - pr_cum_std, pr_cum_mu + pr_cum_std,
    color=prn_color, alpha=0.18, linewidth=0
)
ax1.set_ylim(-400, 400)
ax1.axhline(0.0, ls=":", lw=1.2, color="black", alpha=0.6)
#ax1.set_ylabel("Cumulative CO$_2$")
#ax1.set_title("(b) Cumulative CO$_2$ (mean ± std)")

# =========================
# (c) Violation probability
# =========================
ax2.plot(t, se_vio_acc, lw=1.8, color="red")
ax2.plot(t, pr_vio_acc, lw=1.8, color=prn_color)
ax2.set_ylim(0.0, 1.0)
ax0.set_xlim(0.0, 576)
#ax2.set_ylabel("Violation probability")
#ax2.set_xlabel("Time step (hour)")
#ax2.set_title("(c) Predicted violation probability (mean ± std)")
ax0.tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
ax1.tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
ax2.tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
plt.savefig(f"./plot/{region}_online_operation_evolution.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.close()
#%% Figure 5 (b)
from scipy.stats import gaussian_kde
def performance_extraction(env_path_list):
    log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
    lcox = np.zeros(1000)
    co2 = np.zeros(1000)
    for idx in range(1000):
        save_path = os.path.join(log_dir, env_path_list[idx])
        with open(save_path, "rb") as f:
            env = pickle.load(f)
        lcox[idx] = env.LCOX
        co2[idx] = np.sum(env.CO2_emit)
        a = env.wind_forecast
    
    return lcox, co2

def performance_comparison(country, region, des_type):
    if region == "Dunkirk" or region == "Skive":
        env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}.pkl" for idx in range(1000)]
    else:
        env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}_noise_infer_True.pkl" for idx in range(1000)]
    per_gan = performance_extraction(env_path_list)
    
    return per_gan

def ecdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    y = np.arange(1, x.size + 1) / x.size
    return x, y

per_gan_se = performance_comparison(country, region, des_type = "SE1")
per_gan_pr = performance_comparison(country, region, des_type = "PRN")

lcox_tr, ctg_tr, des_tr, pfss_tr, _, env_tr, idx_tr = loading_performance(country, region, data_type="train")
env_tr_min = env_tr[idx_tr[0]]
env_tr_max = env_tr[idx_tr[-1]]

def kde_curve(x, gridsize=400, cut=3.0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    kde = gaussian_kde(x)
    pad = cut * np.std(x) if np.std(x) > 0 else 0.0
    grid = np.linspace(x.min() - pad, x.max() + pad, gridsize)
    dens = kde(grid)
    return grid, dens

# LCOX
lcox_se_true = per_gan_se[0]          # GAN inferred (SE)
lcox_pr_true = per_gan_pr[0]          # GAN inferred (PR)
lcox_se_real = env_tr_min.LCOX        # real (train extreme min)
lcox_pr_real = env_tr_max.LCOX        # real (train extreme max)

# CO2
co2_se_true = per_gan_se[1]
co2_pr_true = per_gan_pr[1]
co2_se_real = env_tr_min.CO2
co2_pr_real = env_tr_max.CO2
#%%
def fill_pos_area(ax, x, y, color, alpha=0.1):
    x = np.asarray(x)
    y = np.asarray(y)
    mask = x >= 0
    if np.any(mask):
        ax.fill_between(x[mask], y[mask], 1,
                        color=color, alpha=alpha)
        
fig, axes = plt.subplots(2, 1, figsize=(5, 6), gridspec_kw={"hspace": 0.4}, sharex=False)

# -------------------------------------------------
# (1) LCOX distribution (KDE 그대로)
# -------------------------------------------------
x, y = kde_curve(lcox_se_true)
axes[0].plot(x, y, lw=2, color="red", alpha=0.5, ls="--")
x, y = kde_curve(lcox_se_real)
axes[0].plot(x, y, lw=2, color="red")

x, y = kde_curve(lcox_pr_true)
axes[0].plot(x, y, lw=2, color = prn_color, alpha=0.5, ls="--")
x, y = kde_curve(lcox_pr_real)
axes[0].plot(x, y, lw=2, color = prn_color)
axes[0].grid(alpha=0.25)
axes[0].set_xlim(-10, 10)
# -------------------------------------------------
# (2) CO2 cumulative (ECDF)
# -------------------------------------------------
# SE true (GAN) dashed
x, y = ecdf(co2_se_true)
axes[1].plot(x, y, lw=2, color="red", alpha=0.5, ls="--")
# SE real solid
x, y = ecdf(co2_se_real)
axes[1].plot(x, y, lw=2, color="red")

# PR true (GAN) dashed
x, y = ecdf(co2_pr_true)
axes[1].plot(x, y, lw=2, color=prn_color, alpha=0.5, ls="--")
# PR real solid
x, y = ecdf(co2_pr_real)
axes[1].plot(x, y, lw=2, color=prn_color)

# x=0 reference line
axes[1].axvline(0, color="black", ls="--", lw=2, alpha=0.8)

# y=F(0) reference (optional)
# axes[1].axhline(0.5, color="k", lw=0.8, alpha=0.3)

#axes[1].set_ylabel("Cumulative probability (CDF)")
#axes[1].set_xlabel("CO$_2$ emission")
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.25)
axes[1].set_xlim(-400,1000)
# ---- annotate P(CO2>0) = 1 - F(0) (아주 간단히) ----
def p_pos(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return np.mean(x > 0)

txt = (
    f"SE: P(CO$_2$>0) = {p_pos(co2_se_true):.2f} / {p_pos(co2_se_real):.2f}\n"
    f"PR: P(CO$_2$>0) = {p_pos(co2_pr_true):.2f} / {p_pos(co2_pr_real):.2f}"
)

x, y = ecdf(co2_se_true)
fill_pos_area(axes[1], x, y, color="red")

# SE real
x, y = ecdf(co2_se_real)
fill_pos_area(axes[1], x, y, color="red")

# PR true
x, y = ecdf(co2_pr_true)
fill_pos_area(axes[1], x, y, color=prn_color)

# PR real
x, y = ecdf(co2_pr_real)
fill_pos_area(axes[1], x, y, color=prn_color)

axes[0].tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
axes[1].tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
plt.tight_layout()
plt.savefig(f"./plot/{region}_online_operation_performance_validation.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
#%% Figure 5 (c) histogram for optimality-gap and co2-reduction compared with MILP
def performance_extraction(env_path_list):
    log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
    profit = np.zeros(1000)
    co2 = np.zeros(1000)
    for idx in range(1000):
        save_path = os.path.join(log_dir, env_path_list[idx])
        with open(save_path, "rb") as f:
            env = pickle.load(f)
        profit[idx] = np.sum(env.cost_list)
        co2[idx] = np.sum(env.CO2_emit)
        a = env.wind_forecast
    
    return profit, co2

def performance_comparison(country, region, des_type):
    if region == "Dunkirk" or region == "Skive":
        env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}.pkl" for idx in range(1000)]
    else:
        env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}_noise_infer_True.pkl" for idx in range(1000)]
    per_gan = performance_extraction(env_path_list)
    return per_gan
per_gan_se = performance_comparison(country, region, des_type = "SE1")
per_gan_pr = performance_comparison(country, region, des_type = "PRN")
log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
global_solution = np.load(os.path.join(log_dir, "global_solution.npy"))
se_opt_gap = (global_solution[0,:,0]-per_gan_se[0])/np.abs(global_solution[0,:,0])*100
se_co2_reduct = (-per_gan_se[1] + global_solution[0,:,1])
pr_opt_gap = (global_solution[1,:,0]-per_gan_pr[0])/np.abs(global_solution[1,:,0])*100
pr_co2_reduct = (-per_gan_pr[1] + global_solution[1,:,1])

se_gap_mean, se_gap_std = np.mean(se_opt_gap), np.std(se_opt_gap)*0.1
pr_gap_mean, pr_gap_std = np.mean(pr_opt_gap), np.std(pr_opt_gap)*0.1

se_co2_mean, se_co2_std = np.mean(se_co2_reduct), np.std(se_co2_reduct)*0.1
pr_co2_mean, pr_co2_std = np.mean(pr_co2_reduct), np.std(pr_co2_reduct)*0.1
#%%
fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
ax2 = ax1.twinx()

# x positions (IMPORTANT)
x_gap = np.array([0, 1])        # SE-gap, PR-gap
x_co2 = np.array([3, 4])        # SE-CO2, PR-CO2
width = 0.6

# ---- Left axis: Optimality gap ----
ax1.bar(
    x_gap,
    [se_gap_mean, pr_gap_mean],
    yerr=[se_gap_std, pr_gap_std],
    width=width,
    capsize=4,
    color=["red", prn_color],
    alpha=1,
)

# ---- Right axis: CO2 reduction ----
ax2.bar(
    x_co2,
    [se_co2_mean, pr_co2_mean],
    yerr=[se_co2_std, pr_co2_std],
    width=width,
    capsize=4,
    color=["red", prn_color],
    alpha=1,
)

# =========================
# x-axis formatting
# =========================
ax1.set_xticks([0, 1, 3, 4])
ax1.set_xticklabels(
    [],
    rotation=0)
ax1.set_ylim(0, 25)
ax2.set_ylim(-500, 2500)
# visual separator between gap / CO2

# =========================
# labels
# =========================

ax1.grid(axis="y", alpha=0.3)
ax1.axvline(2, color="black", ls="--", lw=1.1, alpha=0.8)

ax1.tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
ax2.tick_params(axis='both', which='major',
       direction='out', length=5, width=1.1, labelsize=10.5)
plt.tight_layout()
plt.savefig(f"./plot/{region}_online_operation_optimality_gap.png", dpi=600, bbox_inches='tight')
plt.tight_layout()
plt.show()
plt.close()

#%% Figure 5 Supplementary co-joint distributional comparison on LCOX & CO2 w/wo gan
from scipy.stats import gaussian_kde
def performance_extraction(env_path_list):
    log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
    lcox = np.zeros(100)
    co2 = np.zeros(100)
    for idx in range(100):
        save_path = os.path.join(log_dir, env_path_list[idx])
        with open(save_path, "rb") as f:
            env = pickle.load(f)
        lcox[idx] = env.LCOX
        co2[idx] = np.sum(env.CO2_emit)
        a = env.wind_forecast
    
    return lcox, co2

def performance_comparison(country, region, des_type):
    env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}.pkl" for idx in range(100)]
    per_gan = performance_extraction(env_path_list)
    
    env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}_noise_infer_False.pkl" for idx in range(100)]
    per = performance_extraction(env_path_list)
    
    return per_gan, per

country = "France"
region = "Dunkirk"
per_gan_se, per_se = performance_comparison(country, region, des_type = "SE1")
per_gan_pr, per_pr = performance_comparison(country, region, des_type = "PRN")
lcox_tr, ctg_tr, des_tr, pfss_tr, _, env_tr, idx_tr = loading_performance(country, region, data_type="train")

# --- plotting parameters ---
line_width  = 0.7
label_size  = 12
tick_length = 5
plt.rcParams['font.family'] = 'Arial'

# create 2×1 layout: top scatter (3× height), bottom profiles (1× height)
fig, (ax_main) = plt.subplots(
    1, 1,
    figsize=(7.0, 6.0),
)

ax_main.scatter(
    per_se[0], per_se[1],
    marker='o', edgecolor='k',
    alpha=0.6, s=40
)

ax_main.scatter(
    per_pr[0], per_pr[1],
    marker='s', edgecolor='k',
    alpha=0.6, s=40
)

# --- kde for inferred distribution ---
env_tr_min = env_tr[idx_tr[0]]
env_tr_max = env_tr[idx_tr[-1]]
tr_min_xy = np.vstack([env_tr_min.LCOX, env_tr_min.CO2])
tr_max_xy = np.vstack([env_tr_max.LCOX, env_tr_max.CO2])
kde_min = gaussian_kde(tr_min_xy)
kde_max = gaussian_kde(tr_max_xy)

# 3) 그릴 grid 생성 (축 한계 그대로 사용)
ax_main.set_ylim(-1000, 1500)
ax_main.set_xlim(-4.0, 4.0)
xmin, xmax = ax_main.get_xlim()
ymin, ymax = ax_main.get_ylim()
X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])

# 4) 밀도 계산 후 2D 배열로 reshape
Z_min   = np.reshape(kde_min(positions), X.shape)
Z_max = np.reshape(kde_max(positions), X.shape)

# 5) 원하는 밀도 레벨(예: 전체 중 상위 10% 영역) 계산
level_min   = np.percentile(Z_min, 90)
level_max = np.percentile(Z_max, 90)

ax_main.contourf(
X, Y, Z_min,
levels=[level_min, Z_min.max()],   # level_free 이상만
colors="red",        
alpha=0.15                          # 투명도
)
if region == "Dunkirk":
    colors = "#C00000"
else:
    colors = '#00994C'
ax_main.contourf(
X, Y, Z_max,
levels=[level_max, Z_max.max()],
colors=colors,
alpha=0.15
)

plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
ax = plt.gca()
ax.tick_params(axis='both', which='major',
       direction='out', length=12, width=1.5, labelsize=20)
ax.tick_params(axis='both', which='minor',
       direction='out', length=6, width=1.5)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

ax_main.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# save & show
plt.tight_layout()
plt.show()
plt.close()
#%% Figure 5 (a)
def moving_average_1d(x, w=24):
    # same length 유지 (edge는 반사 padding)
    pad = w // 2
    xpad = np.pad(x, (pad, w-1-pad), mode="reflect")
    kernel = np.ones(w) / w
    return np.convolve(xpad, kernel, mode="valid")

idx = 8
country = "France"
region = "Dunkirk"
log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
save_path = os.path.join(log_dir, f"operation_results_PT_solver_SE1_idx_{idx}.pkl")
with open(save_path, "rb") as f:
    env = pickle.load(f)

forecast = env.wind_forecast
ground = env.real_wind_speed
time_indices = 12
start = time_indices * 24
x_fc = np.arange(start-1, 576)

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), sharey=True)
ax.plot(ground, color="red", linewidth=1.5, label="Ground-truth")

w = 12  # 24시간 window

for i in range(20):
    y = forecast[time_indices, i, :]          # 길이 576
    y_smooth = moving_average_1d(y, w=w)      # 길이 576 유지
    #y_smooth[start] = ground[start]
    ax.plot(
        x_fc,
        y_smooth[start-1:],
        color="blue",
        alpha=0.3,
        linewidth=1.5
    )

ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.5, labelsize=10)
plt.tight_layout()
plt.savefig("./plot/scenario_generation.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
#%%
from scipy.stats import norm
country = "France"
region = "Dunkirk"
log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
se_ctg_pred_list = np.zeros(shape = (100,576))
se_ctg_list = np.zeros(shape = (100,576))
se_rtg_list = np.zeros(shape = (100,576))
se_renew_list = np.zeros(shape = (100,576))
se_prob_vio_list = np.zeros(shape = (100,576))
se_cum_list = np.zeros(shape = (100,576))

pr_ctg_pred_list = np.zeros(shape = (100,576))
pr_ctg_list = np.zeros(shape = (100,576))
pr_rtg_list = np.zeros(shape = (100,576))
pr_renew_list = np.zeros(shape = (100,576))
pr_prob_vio_list = np.zeros(shape = (100,576))
pr_cum_list = np.zeros(shape = (100,576))

scaler_path = os.path.join('./dataset/', f'{country}/{region}/oracle_dataset_c_fax_fix_sample_50000')
with open(os.path.join(scaler_path, 'scaler_package.pkl'), "rb") as file: 
    scaler_package = pickle.load(file)
co2_scaler = scaler_package['co2']
reward_scaler = scaler_package['reward']
rtg_scaler = scaler_package['rtg']
ctg_scaler = scaler_package['ctg']
ctg_mu  = ctg_scaler.mean_[0]
ctg_std = ctg_scaler.scale_[0]

for idx in range(100):
    save_path = os.path.join(log_dir, f"operation_results_PT_solver_SE1_idx_{idx}.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    se_ctg_list[idx] = (env.ctg*ctg_std + ctg_mu)[:,0]
    se_rtg_list[idx] = env.rtg[:,0]
    se_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    se_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled  * ctg_std + ctg_mu
    se_ctg_pred_list[idx] = mu_ctg
    std_ctg = std_ctg_scaled * ctg_std
    z = (-cum_co2 - mu_ctg) / std_ctg
    se_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,)
    
for idx in range(100):
    save_path = os.path.join(log_dir, f"operation_results_PT_solver_PRN_idx_{idx}.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    pr_ctg_list[idx] = (env.ctg*ctg_std + ctg_mu)[:,0]
    pr_rtg_list[idx] = env.rtg[:,0]
    pr_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    pr_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled  * ctg_std + ctg_mu
    pr_ctg_pred_list[idx] = mu_ctg
    std_ctg = std_ctg_scaled * ctg_std
    z = (-cum_co2 - mu_ctg) / std_ctg
    pr_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,)  

#pick best & worst scenario
se_idx = np.argmax(se_ctg_list[:,0])
pr_idx = np.argmin(pr_ctg_list[:,0])

t = np.arange(len(se_ctg_list[0, :]))
plt.rcParams["axes.spines.top"] = True
plt.rcParams["axes.spines.right"] = True
fig, axes = plt.subplots(
    2, 1,
    figsize=(6.5, 5.2),
    sharex=True,
    gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.1}
)

# =========================
# (a) Cumulative CO2
# =========================
ax0 = axes[0]

ax0.plot(t, se_cum_list[se_idx], lw=2.0, color="red", label="SE1")
ax0.plot(t, pr_cum_list[pr_idx], lw=2.0, color="#00994C", label="PRN")

ax0.axhline(0.0, ls=":", lw=1.3, color="black", alpha=0.6)
#ax0.set_ylabel("Cumulative CO$_2$")
#ax0.legend(frameon=False, loc="upper left")
ax0.tick_params(axis="x", which="both", bottom=False)

# =========================
# (b) Violation probability
# =========================
ax1 = axes[1]

ax1.plot(t, se_prob_vio_list[se_idx], lw=1.8, color="red", label="SE1")
ax1.plot(t, pr_prob_vio_list[pr_idx], lw=1.8, color="#00994C", label="PRN")

ax1.set_ylim(0.0, 1.05)
#ax1.set_ylabel("Violation probability")
#ax1.set_xlabel("Time step (hour)")
#ax1.legend(frameon=False, loc="upper right")

# =========================
# panel labels (optional, Nature-style)
# =========================

plt.tight_layout()
plt.show()

np.where(se_ctg_list[:,0]>0)[0].shape, np.where(pr_ctg_list[:,0]>0)[0].shape 
#%%
def crps_gaussian(mu, sigma, y):
    """
    mu, sigma, y : array-like (broadcast 가능)
    return: CRPS value
    """
    sigma = np.maximum(sigma, 1e-6)  # 안정화
    z = (y - mu) / sigma
    return sigma * (
        z * (2 * norm.cdf(z) - 1)
        + 2 * norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
country = "Denmark"
region = "Fredericia"
log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
se_ctg_pred_list = np.zeros(shape = (100,576,2))
se_ctg_list = np.zeros(shape = (100,576))
se_rtg_list = np.zeros(shape = (100,576))
se_renew_list = np.zeros(shape = (100,576))
se_wind_scenario_list = np.zeros(shape = (100,24, 20, 576))
se_real_wind_list = np.zeros(shape = (100, 576))
se_prob_vio_list = np.zeros(shape = (100,576))
se_cum_list = np.zeros(shape = (100,576))

pr_ctg_pred_list = np.zeros(shape = (100,576,2))
pr_ctg_list = np.zeros(shape = (100,576))
pr_rtg_list = np.zeros(shape = (100,576))
pr_renew_list = np.zeros(shape = (100,576))
pr_prob_vio_list = np.zeros(shape = (100,576))
pr_cum_list = np.zeros(shape = (100,576))

scaler_path = os.path.join('./dataset/', f'{country}/{region}/oracle_dataset_c_fax_fix_sample_50000')
with open(os.path.join(scaler_path, 'scaler_package.pkl'), "rb") as file: 
    scaler_package = pickle.load(file)
co2_scaler = scaler_package['co2']
reward_scaler = scaler_package['reward']
rtg_scaler = scaler_package['rtg']
ctg_scaler = scaler_package['ctg']
ctg_mu  = ctg_scaler.mean_[0]
ctg_std = ctg_scaler.scale_[0]

for idx in range(100):
    save_path = os.path.join(log_dir, f"operation_results_PT_solver_SE1_idx_{idx}.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    se_ctg_list[idx] = (env.ctg)[:,0]
    se_rtg_list[idx] = np.cumsum(env.cost_list)/1e6
    se_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    se_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled
    se_ctg_pred_list[idx,:,0] = mu_ctg
    std_ctg = std_ctg_scaled 
    z = (-cum_co2 - mu_ctg) / std_ctg
    se_ctg_pred_list[idx,:,1] = std_ctg
    se_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,)
    se_wind_scenario_list[idx] = env.wind_forecast
    se_real_wind_list[idx] = env.real_wind_speed
    
for idx in range(100):
    save_path = os.path.join(log_dir, f"operation_results_PT_solver_PRN_idx_{idx}.pkl")
    with open(save_path, "rb") as f:
        env = pickle.load(f)
    pr_ctg_list[idx] = (env.ctg)[:,0]
    pr_rtg_list[idx] = np.cumsum(env.cost_list)/1e6
    pr_renew_list[idx] = env.renewable
    
    #constraint-violation probability
    cum_co2 = np.cumsum(env.co2_list)
    pr_cum_list[idx] = cum_co2
    pred_ctg_dist = env.pred_ctg_dist
    mu_ctg_scaled  = pred_ctg_dist[:, 0]
    std_ctg_scaled = pred_ctg_dist[:, 1]
    std_ctg_scaled = np.clip(std_ctg_scaled, 1e-6, None)
    mu_ctg  = mu_ctg_scaled  * ctg_std + ctg_mu
    pr_ctg_pred_list[idx,:,0] = mu_ctg_scaled 
    pr_ctg_pred_list[idx,:,1] = std_ctg_scaled
    std_ctg = std_ctg_scaled * ctg_std
    z = (-cum_co2 - mu_ctg) / std_ctg
    pr_prob_vio_list[idx] = 1-norm.cdf(z)   # (T,)  

def corr_matrix(X, Y):
    """
    X: (N, Tx), Y: (N, Ty)
    returns Corr: (Tx, Ty) where Corr[i,j]=corr(X[:,i], Y[:,j])
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert X.shape[0] == Y.shape[0], "N (samples) must match"

    # center
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    Yc = Y - np.nanmean(Y, axis=0, keepdims=True)

    # std
    Xs = np.nanstd(Xc, axis=0, ddof=1)
    Ys = np.nanstd(Yc, axis=0, ddof=1)

    # avoid division by zero
    Xs = np.where(Xs < 1e-12, np.nan, Xs)
    Ys = np.where(Ys < 1e-12, np.nan, Ys)

    # covariance (Tx, Ty)
    # Using nanmean assumes no NaNs; if you have NaNs, this is a best-effort
    cov = (Xc.T @ Yc) / (X.shape[0] - 1)

    # correlation
    Corr = cov / (Xs[:, None] * Ys[None, :])
    return Corr

# -----------------------------
# Inputs
# -----------------------------
cum  = np.asarray(pr_ctg_list)        # (100,576)
prob = np.asarray(pr_ctg_pred_list)   # (100,576,2)

mu    = prob[..., 0]
sigma = prob[..., 1]

# CRPS 계산 → (100,576)
crps = crps_gaussian(mu, sigma, cum)

#c_ctg_renew = corr_matrix(cum, prob)         # (576,576)
fig, ax = plt.subplots(figsize=(3.5, 3.0))
cmap = plt.get_cmap("bwr")
im = ax.imshow(
    crps,          
    cmap="bwr",
    vmin=0, vmax=0.24,      
    aspect="auto",
    interpolation="nearest",
    origin="lower"        
)
plt.plot(crps.mean(axis=0))
plt.plot(crps.mean(axis=0)-crps.std(axis=0))
plt.plot(crps.mean(axis=0)+crps.std(axis=0))
#ax.set_xlabel("Renew time index")
#ax.set_ylabel("CTG time index")
#ax.set_title("Pearson correlation: CTG vs Renewable")

#cbar = fig.colorbar(im, ax=ax)
#cbar.set_label("Pearson correlation")

plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

fake = np.asarray(se_wind_scenario_list[8])  # (100, 24, 20, 576)

# (A) 어떤 fake를 uncertainty로 볼지 선택
# 옵션 1) 전체 다 합쳐서(100*24*20개) 시간별 분포
samples = fake.reshape(-1, fake.shape[-1])   # (100*24*20, 576)

# 옵션 2) 특정 time_indices(예: 12)에서의 시나리오들만 보고 싶으면:
    
time_indices = 12
samples = fake[time_indices, :, :].reshape(-1, 576)  # (100*20, 576)

# (B) y축(bin) 범위 설정 (robust하게 1~99% 분위로 자르면 outlier에 덜 흔들림)
ymin = np.quantile(samples, 0.01)
ymax = np.quantile(samples, 0.99)
bins = 80
edges = np.linspace(ymin, ymax, bins + 1)
centers = 0.5 * (edges[:-1] + edges[1:])

# (C) 시간별로 1D histogram 쌓아서 (bins, 576) 행렬 만들기
H = np.zeros((bins, samples.shape[1]), dtype=float)  # (bins, 576)

for t in range(samples.shape[1]):
    hist, _ = np.histogram(samples[:, t], bins=edges, density=True)  # density=True -> 확률밀도
    H[:, t] = hist

# (선택) 보기 좋게 로그 스케일 (밝은 곳만 너무 튀는 것 방지)
H_plot = np.log1p(H)

# (D) plot
plt.rcParams['font.family'] = 'Arial'
fig, ax = plt.subplots(figsize=(7.5, 3.2))

im = ax.imshow(
    H_plot,
    origin="lower",
    aspect="auto",
    extent=[0, samples.shape[1]-1, ymin, ymax],  # x: time, y: value
)

ax.set_xlabel("Time (hour index)", fontsize=12)
ax.set_ylabel("Wind speed (or forecast value)", fontsize=12)
ax.set_title("Fake scenario uncertainty (density heatmap)", fontsize=12)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("log(1 + density)", fontsize=11)

ax.tick_params(axis='both', which='major', direction='out', length=4, width=1.2, labelsize=10)
plt.tight_layout()
plt.show()
plt.close()
#%%
country = "France"
region = "Dunkirk"
def crps_gaussian(mu, sigma, y):
    sigma = np.maximum(sigma, 1e-6)
    z = (y - mu) / sigma
    return sigma * (z * (2 * norm.cdf(z) - 1)+ 2 * norm.pdf(z)- 1 / np.sqrt(np.pi))

def goal_crps_calculation(env_path_list):
    log_dir = f'./optimization/result/{country}/{region}/online_operation_validation'
    ctg_dist_pred = np.zeros(shape = (100,576,2))
    rtg_dist_pred = np.zeros(shape = (100,576,2))
    ctg = np.zeros(shape = (100,576))
    rtg = np.zeros(shape = (100,576))
    
    for idx in range(100):
        save_path = os.path.join(log_dir, env_path_list[idx])
        with open(save_path, "rb") as f:
            env = pickle.load(f)
        ctg[idx] = (env.ctg)[:,0]
        rtg[idx] = (env.rtg)[:,0]
        ctg_dist_pred[idx] = env.pred_ctg_dist
        rtg_dist_pred[idx] = env.pred_rtg_dist
    
    ctg_crps = crps_gaussian(ctg_dist_pred[:,:,0], ctg_dist_pred[:,:,1], ctg)
    rtg_crps = crps_gaussian(rtg_dist_pred[:,:,0], rtg_dist_pred[:,:,1], rtg)
    
    return ctg_crps, rtg_crps

def crps_comparison(country, region, des_type):
    env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}.pkl" for idx in range(100)]
    crps_gan = goal_crps_calculation(env_path_list)
    
    env_path_list = [f"operation_results_PT_solver_{des_type}_idx_{idx}_noise_infer_False.pkl" for idx in range(100)]
    crps = goal_crps_calculation(env_path_list)
    
    return crps_gan, crps

def summarize_crps(crps):  
    mean = np.mean(crps, axis=0)
    std  = np.std(crps, axis=0)
    return mean, std

crps_gan_se, crps_se = crps_comparison("Denmark", "Skive", "SE1")
crps_gan_pr, crps_pr = crps_comparison("Denmark", "Skive", "PRN")

time = np.arange(crps_se[0].shape[1])

fig, axes = plt.subplots(2, 1, figsize=(7.0, 4.5), sharex=True)

# =========================
# RTG
# =========================
crps_gan_se[0].shape
rtg_mean_gan, rtg_std_gan = summarize_crps(np.vstack((crps_gan_se[1], crps_gan_pr[1])))
rtg_mean, rtg_std = summarize_crps(np.vstack((crps_se[1], crps_pr[1])))

axes[0].plot(time, rtg_mean_gan, lw=2, label="With GAN", color="tab:blue")
axes[0].fill_between(
    time,
    rtg_mean_gan - 0.1 * rtg_std_gan,
    rtg_mean_gan + 0.1 * rtg_std_gan,
    color="tab:blue",
    alpha=0.25,
)

axes[0].plot(time, rtg_mean, lw=2, label="Without GAN", color="tab:red")
axes[0].fill_between(
    time,
    rtg_mean - 0.1 * rtg_std,
    rtg_mean + 0.1 * rtg_std,
    color="tab:red",
    alpha=0.25,
)

axes[0].legend(frameon=False)


# =========================
# CTG
# =========================
ctg_mean_gan, ctg_std_gan = summarize_crps(np.vstack((crps_gan_se[0], crps_gan_pr[0])))
ctg_mean, ctg_std = summarize_crps(np.vstack((crps_se[0], crps_pr[0])))

axes[1].plot(time, ctg_mean_gan, lw=2, label="With GAN", color="tab:blue")
axes[1].fill_between(
    time,
    ctg_mean_gan - 0.1 * ctg_std_gan,
    ctg_mean_gan + 0.1 * ctg_std_gan,
    color="tab:blue",
    alpha=0.25,
)

axes[1].plot(time, ctg_mean, lw=2, label="Without GAN", color="tab:red")
axes[1].fill_between(
    time,
    ctg_mean - 0.1 * ctg_std,
    ctg_mean + 0.1 * ctg_std,
    color="tab:red",
    alpha=0.25,
)

for ax in axes:
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
plt.close()