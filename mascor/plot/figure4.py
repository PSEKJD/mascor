import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  
import pickle
import json
from utils.env import ptx_env_stack
import numpy as np
import matplotlib.pyplot as plt
from utils.env.ptx_env_stack import *
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
from utils.helper import *

def loading_idx(country, region, min_diff = 0.02):
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

# Load and data-preprocess
country_list = {
    'France': ['Dunkirk'],
    'Denmark': ['Skive', 'Fredericia'],
    'Germany': ['Weener']}

log_dir = './optimization/result'
for country in country_list:
    for region in country_list[country]:
        save_path = os.path.join(log_dir,f'{country}/{region}/pareto_validation_test_dataset_gan_epoch_15000.json')
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # filling the uq result of train-pareto & validation-pareto
        lcox_tr, ctg_tr, des_tr, pfss_tr, _, _, _ = loading_performance(country, region, data_type="train")
        lcox_te, ctg_te, des_te, pfss_te, _, _, _ = loading_performance(country, region, data_type="test")
        
        po_tr = np.array([lcox_tr.mean(axis=1), 
                          ctg_tr.mean(axis=1)]).transpose()
        po_te = np.array([lcox_te.mean(axis=1), 
                          ctg_te.mean(axis=1)]).transpose()
        
        do_tr = data['train']['dominant'] 
        do_te = data['test']['dominant']
        do_tr = np.array([[do_tr[f"des-{i}"]["mu-LCOX(s-1000)[$/kg]"] for i in range(len(do_tr))],
                          [do_tr[f"des-{i}"]["mu-ctg(s-1000)[ton/month]"]*100 for i in range(len(do_tr))]]).transpose()
        do_te = np.array([[do_te[f"des-{i}"]["mu-LCOX(s-10000)[$/kg]"] for i in range(len(do_te))],
                          [do_te[f"des-{i}"]["mu-ctg(s-10000)[ton/month]"] for i in range(len(do_te))]]).transpose()
        
        tr_idx = np.argsort(po_tr[:,0])
        des_tr = des_tr[tr_idx]
        knee_tr = np.where(des_tr[:,1]<0.1)[0]
        
        te_idx = np.argsort(po_te[:,0])
        des_te = des_te[te_idx]
        knee_te = np.where(des_te[:,1]<0.1)[0]
        alpha = 0.5
        plt.figure(figsize=(7.0, 6.0))
        plt.scatter(do_tr[:,0], do_tr[:,1], s=80, color='gray', alpha=alpha)
        plt.scatter(do_te[:,0], do_te[:,1], s=80, color='gray')
           
        if knee_tr.size == 0:
            plt.plot(po_tr[tr_idx,0], po_tr[tr_idx,1], '--o',color='red', linewidth=2.5, alpha = alpha,
                     markersize=11, markerfacecolor='white')
            plt.plot(po_te[te_idx,0], po_te[te_idx,1], '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx_tr = knee_tr[0]
            knee_idx_te = knee_te[0]
            plt.plot(po_tr[tr_idx,0][:knee_idx_tr+1], po_tr[tr_idx,1][:knee_idx_tr+1], '--o',color='red', linewidth=2.5, alpha = alpha,
                     markersize=10, markerfacecolor='white')
            plt.plot(po_tr[tr_idx,0][knee_idx_tr:], po_tr[tr_idx,1][knee_idx_tr:], '--s',color='#00994C', linewidth=2.5, alpha = alpha,
                     markersize=10, markerfacecolor='white')
            
            plt.plot(po_te[te_idx,0][:knee_idx_te+1], po_te[te_idx,1][:knee_idx_te+1], '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
            plt.plot(po_te[te_idx,0][knee_idx_te:], po_te[te_idx,1][knee_idx_te:], '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.0, 1.4)
        plt.ylim(-200, 150)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        #plt.savefig(f"./plot/{region}_pareto_curve_validation.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%%
for country in country_list:
    for region in country_list[country]:
        save_path = os.path.join(log_dir,f'{country}/{region}/pareto_validation_test_dataset_gan_epoch_15000.json')
        with open(save_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # filling the uq result of train-pareto & validation-pareto
        lcox_tr, ctg_tr, des_tr, pfss_tr, _, _, _ = loading_performance(country, region, min_diff= 0.0, data_type="train")
        lcox_te, ctg_te, des_te, pfss_te, _, _, _ = loading_performance(country, region, min_diff= 0.0, data_type="test")
        
        po_te = np.array([lcox_te.mean(axis=1), 
                          ctg_te.mean(axis=1)]).transpose()
        
        do_tr = data['train']['dominant'] 
        do_te = data['test']['dominant']
        do_tr = np.array([[do_tr[f"des-{i}"]["mu-LCOX(s-1000)[$/kg]"] for i in range(len(do_tr))],
                          [do_tr[f"des-{i}"]["mu-ctg(s-1000)[ton/month]"]*100 for i in range(len(do_tr))]]).transpose()
        do_te = np.array([[do_te[f"des-{i}"]["mu-LCOX(s-10000)[$/kg]"] for i in range(len(do_te))],
                          [do_te[f"des-{i}"]["mu-ctg(s-10000)[ton/month]"] for i in range(len(do_te))]]).transpose()
        
        te_idx = np.argsort(po_te[:,0])
        des_te = des_te[te_idx]
        knee_te = np.where(des_te[:,1]<0.1)[0]
        
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
        pareto_m  = is_non_dominated(feas_obj)
        pareto_obj = feas_obj[pareto_m]
        pareto_des = feas_des[pareto_m]
        po = pareto_obj.detach().cpu().numpy()
        idx = np.argsort(-po[:, 0])
        des = pareto_des.detach().cpu().numpy()
        des = des[idx]
        knee_point = np.where(des[:,2]<0.1)[0]
        
        alpha = 0.35
        plt.figure(figsize=(7.0, 6.0))
        plt.scatter(do_tr[:,0], do_tr[:,1], s=80, color='gray', alpha=alpha)
        plt.scatter(do_te[:,0], do_te[:,1], s=80, color='gray')
           
        if knee_point.size == 0:
            plt.plot(-po[idx,0], -po[idx,1]*100, '--o',color='red', linewidth=2.5, alpha = alpha,
                     markersize=11, markerfacecolor='white')
            plt.plot(po_te[te_idx,0], po_te[te_idx,1], '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx = knee_point[0]
            plt.plot(-po[idx,0][:knee_idx+1], -po[idx,1][:knee_idx+1]*100, '--o',color='red', linewidth=2.5, alpha = alpha,
                     markersize=10, markerfacecolor='white')

            plt.plot(-po[idx,0][knee_idx:], -po[idx,1][knee_idx:]*100, '--s',color='#00994C', linewidth=2.5, alpha = alpha,
                     markersize=10, markerfacecolor='white')
            
            knee_idx_te = knee_te[0]
            plt.plot(po_te[te_idx,0][:knee_idx_te+1], po_te[te_idx,1][:knee_idx_te+1], '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
            plt.plot(po_te[te_idx,0][knee_idx_te:], po_te[te_idx,1][knee_idx_te:], '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.0, 1.4)
        plt.ylim(-200, 150)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"./plot/{region}_pareto_curve_validation.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%%
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
from matplotlib.colors import LinearSegmentedColormap
def find_longest_calm(power_series, calm_threshold=0.05):
    
    calm_indices = np.where(power_series <= calm_threshold)[0]
    if len(calm_indices) == 0:
        return 0, []

    calm_spells = []
    start = prev = calm_indices[0]
    for idx in calm_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            calm_spells.append((start, prev))
            start = prev = idx
    calm_spells.append((start, prev))

    longest = max(calm_spells, key=lambda x: x[1] - x[0])
    longest_duration = longest[1] - longest[0] + 1
    return longest_duration, calm_spells

def weather_labelling(dataset):
    size = len(dataset)
    calm_info = [] 
    for i in range(size):
        longest_calm, calm_spells = find_longest_calm(dataset[i], calm_threshold=base_load/50000) #50,000 kW scale
        calm_info.append((i, longest_calm))
   
    # Ascending order depending on calm length
    calm_info_sorted = sorted(calm_info, key=lambda x: x[1])
    sorted_indices = [idx for idx, _ in calm_info_sorted]
    data_stack_sorted = dataset[sorted_indices]
    
    sampled_indices = np.linspace(0, size - 1, size, dtype=int)
    data_stack_sampled = data_stack_sorted[sampled_indices]
    calm_info_sampled = [calm_info_sorted[i] for i in sampled_indices]
     
    return data_stack_sampled, calm_info_sampled

prob_error = []
for country in country_list:
    for region in country_list[country]:
        lcox_tr, ctg_tr, des_tr, pfss_tr, _, env_tr, idx_tr = loading_performance(country, region, data_type="train")
        lcox_te, ctg_te, des_te, pfss_te, _, env_te, idx_te = loading_performance(country, region, data_type="test")
        SP_H2 = 55.7
        X_H2  = 0.19576
        P_X   = 0.65702
        env_te_min = env_te[idx_tr[0]]
        env_te_max = env_te[idx_tr[-1]]
        base_load_min = env_te_min.X_flow*(P_X + X_H2 * SP_H2)
        base_load_max = env_te_max.X_flow*(P_X + X_H2 * SP_H2)
        
        print("*"*50)
        print(f"Target-region {region}")
        print(f"SE1-PF-infer: {pfss_tr[idx_tr[0]]} & SE1-PF-true: {pfss_te[idx_te[0]]}")
        print(f"PRN-PF-infer: {pfss_tr[idx_tr[-1]]} & PRN-PF-true: {pfss_te[idx_te[-1]]}")
        prob_error.append(np.abs(pfss_tr[idx_tr[0]]-pfss_te[idx_te[0]]))
        prob_error.append(np.abs(pfss_tr[idx_tr[-1]]-pfss_te[idx_te[-1]]))
        
        # label by drought length
        #dataset_min, calm_min = weather_labelling(env_te_min.SMP.detach().cpu().numpy())
        #dataset_max, calm_max = weather_labelling(env_te_max.renewable.detach().cpu().numpy())
        dataset_min = env_te_min.renewable.detach().cpu().numpy()
        renew_mu_min = dataset_min.mean(axis=1)
        dataset_max = env_te_max.renewable.detach().cpu().numpy()
        renew_mu_max = dataset_max.mean(axis=1)
        renew_min, renew_max = np.min([renew_mu_min, renew_mu_max]), np.max([renew_mu_min, renew_mu_max])

        emit_min = env_te_min.CO2
        LCOX_min = env_te_min.LCOX
        emit_max = env_te_max.CO2
        LCOX_max = env_te_max.LCOX

        color_list = [
            (64/255, 42/255, 180/255),
            (35/255, 160/255, 229/255),
            (87/255, 204/255, 122/255),
            (240/255, 186/255, 54/255),
            (247/255, 245/255, 27/255)
        ]
        custom_cmap = LinearSegmentedColormap.from_list("custom_rgb_cmap", color_list, N=256)
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
    
        # --- main scatter plot ---
        norms_min = (renew_mu_min-renew_min)/(renew_max-renew_min)
        norms_max = (renew_mu_max-renew_min)/(renew_max-renew_min)
        ax_main.scatter(
            LCOX_min[::3], emit_min[::3],
            c=norms_min[::3], cmap=custom_cmap,
            marker='o', edgecolor='k',
            alpha=0.6, s=40
        )
        ax_main.scatter(
            LCOX_max[::3], emit_max[::3],
            c=norms_max[::3], cmap=custom_cmap,
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
        plt.savefig(
            f"./plot/{region}_extreme_des_validation.png",
            dpi=600, bbox_inches='tight')
        plt.tight_layout()
        plt.show()
        plt.close()
#%%
np.mean(prob_error)
#%%
plt.rcParams['font.family'] = 'Arial'
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", color_list, N=256)

dummy_data = np.linspace(0, 1, 256).reshape(1, -1)  # 1행짜리 gradient

vmin_real = 5#renew_min/1000
vmax_real =  40#renew_max/1000

# dummy normalized data
dummy_data = np.linspace(0, 1, 256).reshape(1, -1)

fig, ax = plt.subplots(figsize=(4.5, 3.0))
img = ax.imshow(dummy_data, cmap=custom_cmap, aspect='auto')
ax.set_visible(False)

cbar = fig.colorbar(
    img, orientation='horizontal',
    ax=ax, fraction=1.0, pad=0.8, aspect=10
)

# tick 위치는 normalized space
ticks_norm = [0.0, 0.5, 1.0]
# tick label은 실제 값
ticks_real = [
    vmin_real,
    0.5 * (vmin_real + vmax_real),
    vmax_real
]

cbar.set_ticks(ticks_norm)
cbar.set_ticklabels([f"{v:.1f}" for v in ticks_real])

cbar.ax.tick_params(labelsize=15, length=15, width=1.5)
plt.savefig("./plot/colorbar_only.png", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()
#%%
import pandas as pd
color_list = [
    (64/255, 42/255, 180/255),
    (35/255, 160/255, 229/255),
    (87/255, 204/255, 122/255),
    (240/255, 186/255, 54/255),
    (247/255, 245/255, 27/255)
]
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_rgb_cmap", color_list, N=256
)

def heatmap_plot(env, region, env_type, option = "avg"):
    renew = env.renewable.detach().cpu().numpy()
    grid = env.SMP.detach().cpu().numpy()
    renew_dlt = np.abs(renew[:,1:]-renew[:,:-1])
    grid_dlt = np.abs(grid[:,1:]-grid[:,:-1])
    dispatch = env.P_to_G.detach().cpu().numpy()
    h2_sell = env.H2_to_market.detach().cpu().numpy()
    bu = env.SOC_profile.detach().cpu().numpy()
    hu = env.L_H2_profile.detach().cpu().numpy()
    bu_utill = np.abs(bu[:,1:]-bu[:,:-1])
    h2_utill = np.abs(hu[:,1:]-hu[:,:-1])
    lcox = env.LCOX
    co2 = env.CO2 
    
    if option == "avg":
        data = {
            "renew": renew.mean(axis=1),
            "renew-dlt": renew_dlt.mean(axis=1),
            "grid": grid.mean(axis=1),
            "grid-dlt": grid_dlt.mean(axis=1),
            "dispatch": dispatch.mean(axis=1),
            "h2 sell": h2_sell.sum(axis=1),
            "BU": bu_utill.mean(axis=1),
            "HU": h2_utill.mean(axis=1),
            "LCOX": lcox,
            "CO2": co2}

    df = pd.DataFrame(data)
    x_vars = ["renew", "renew-dlt", "grid", "grid-dlt"]
    #y_vars = ["dispatch", "h2 sell", "BU", "HU", "LCOX", "CO2"]
    y_vars = ["LCOX", "CO2", "dispatch", "h2 sell", "BU", "HU"]
    corr_rect = pd.DataFrame(
        index=y_vars,
        columns=x_vars,
        dtype=float
    )
    for y in y_vars:
        for x in x_vars:
            corr_rect.loc[y, x] = df[y].corr(df[x], method="pearson")
    # =========================
    # Plot heatmap
    # =========================
    plt.figure(figsize=(9, 2.0))
    im = plt.imshow(
        corr_rect.values,
        cmap="bwr",     # -1: blue, 0: white, +1: red
        vmin=-1,
        vmax=1
    )

    plt.xticks(range(len(x_vars)), x_vars, rotation=30, ha="right")
    plt.yticks(range(len(y_vars)), y_vars)

    #cbar = plt.colorbar(im)
    #cbar.set_label("Pearson correlation coefficient")
    #plt.title(f"{region}: env-type of {env_type}", fontsize=13)
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(f"./plot/{region}_{env_type}_operation_heat_map.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

for country in country_list:
    for region in country_list[country]:
        lcox_te, ctg_te, des_te, pfss_te, _, env_te, idx_te = loading_performance(country, region, data_type="test")
        heatmap_plot(env_te[idx_te[0]], region, env_type = "SE1")
        heatmap_plot(env_te[idx_te[-1]], region, env_type = "PRN")
#%% 
plt.rcParams['font.family'] = 'Arial'

dummy_data = np.linspace(-1, 1, 256).reshape(1, -1)  # 1행짜리 gradient

fig, ax = plt.subplots(figsize=(4.5, 3.0))
img = ax.imshow(dummy_data, cmap="bwr", aspect='auto')
ax.set_visible(False)

cbar = fig.colorbar(
    img, orientation='horizontal',
    ax=ax, fraction=1.0, pad=0.8, aspect=10
)

# tick 위치는 normalized space
ticks_norm = [-1.0, 0.0, 1.0]

cbar.set_ticks(ticks_norm)
cbar.set_ticklabels([f"{v:.1f}" for v in ticks_norm])

cbar.ax.tick_params(labelsize=15, length=15, width=1.5)
plt.savefig("./plot/colorbar_operation_only.png", dpi=600, bbox_inches='tight', transparent=True)
plt.show()
plt.close()

