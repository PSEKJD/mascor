#%% Figure 1 (a) geographical map
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 10),
                       subplot_kw={"projection": ccrs.PlateCarree()})

# Coastlines & borders
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)
ax.add_feature(cfeature.OCEAN, facecolor="#eef4f8")

# Set extent
ax.set_extent([1, 10.2, 50, 57.3], crs=ccrs.PlateCarree())

xticks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
yticks = [50, 51, 52, 53, 54, 55, 56, 57]

ax.set_xticks(xticks, crs=ccrs.PlateCarree())
ax.set_yticks(yticks, crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter(number_format='.0f', degree_symbol='°')
lat_formatter = LatitudeFormatter(number_format='.0f', degree_symbol='°')

ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

ax.tick_params(
    axis='both',
    direction='out',   
    length=8,         
    width=1.2,
    labelsize=18
)
gl = ax.gridlines(
    crs=ccrs.PlateCarree(),
    draw_labels=False,     
    linewidth=0.0,
    color='gray',
    alpha=0.5,
    linestyle='--'
)
plt.savefig("./plot/figure_1_a.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
#%% Figure 1 (a) renew, grid price mean
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.env.ptx_env_stack import *
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import FormatStrFormatter
# Load and preprocess data into data_package
country_list = {
    'France': {'Dunkirk': {"num": 28}},
    'Denmark': {'Skive': {"num": 24}, 'Fredericia': {"num": 34}},
    'Germany': {'Weener':{"num": 21}}
    #'Germany': {'Wunsiedel': {}, 'Weener': {}}
}

data_package = {}

for target_country in country_list:
    for region in country_list[target_country]:
        log_dir = './optimization/result'
        save_template = os.path.join(log_dir,f'{target_country}/{region}/uq_results_pfss_0.5_idx{{idx}}.pkl')
        num = country_list[target_country][region]["num"]
        env_list = []
        for idx in range(1, num + 1):
            save_path = save_template.format(idx=idx)
            print(save_path)
            with open(save_path, "rb") as file:
                env = pickle.load(file)
            env_list.append(env)
        short = {'Dunkirk': 'DU', 'Skive': 'SK', 'Fredericia': 'FR', 'Weener': 'WE'}[region]
        name = f"{short}"
        data_package[name] = {'env':  env_list}

regions = ['DU', 'SK', 'FR', 'WE']
renewable_list = []
price_list = []

for region in regions:
    env_list = data_package[region]["env"]
    for idx, env in enumerate(env_list):
        if idx == 0:
            renew = env.renewable.detach().cpu().numpy()/1000 #MW
            price = env.SMP.detach().cpu().numpy()*1000 #$/MWh
        else:
            renew = np.concatenate((renew, env.renewable.detach().cpu().numpy()/1000), axis = 0)
            price = np.concatenate((price, env.SMP.detach().cpu().numpy()*1000), axis = 0)
    renewable_list.append(renew)
    price_list.append(price)
#%% supporting fig for renew-grid distribution
state_list = ['Dunkirk', 'Skive', 'Fredericia', 'Weener']
line_width = 0.7
label_size = 12
tick_length = 5
statics = []

for i in range(len(renewable_list)):
    state = state_list[i]    
    #renew = renewable_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    #price = price_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    
    renew = renewable_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    price = price_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    #price = np.abs(profile[:,:,1:]-profile[:,:,:-1]).mean(axis = -1).reshape(-1)[:10000]
    
    statics.append([np.mean(renew), np.var(renew),
                    np.mean(price), np.var(price)])
    
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    # Density plot (KDE)
    sns.kdeplot(x=renew, y=price, fill=True, cmap="viridis", levels=10, alpha=1.0, ax=ax)

    # Scatter overlay
    #ax.scatter(renew, price, s=2, color='gray', alpha=0.1)

    # Quadrant coloring
    colors = [
        # First regime: high PEMEC
        (0/255, 102/255, 204/255),    # Deep blue (dominant first regime)
        (102/255, 178/255, 255/255),  # Light blue (second case in first regime)

        # Second regime: low PEMEC
        (0/255, 153/255, 76/255),     # Deep green (dominant second regime)
        (153/255, 230/255, 179/255),  # Pale green (second case in second regime)
    ] 
    
    # Axis formatting
    ax.set_xlim(-10, 70)
    ax.set_ylim(-10, 70)
    #ax.set_xlabel("Nitrogen application rate (kg ha$^{-1}$)")
    #ax.set_ylabel("Yield (ton ha$^{-1}$)")
    #ax.set_title("Terai of Nepal")
    ax.grid(True, which='both', linestyle=':', linewidth=line_width/5, color='gray', alpha=0.5)
    ax.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
    ax.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=tick_length/3*2, width=line_width)
    #if i == 3:
        #cbar.set_label("Density level", fontsize=label_size)
        #cbar = plt.colorbar(kde.collections[0], ax=ax)
        #cbar.set_label("Density", fontsize=label_size)     
        #ticks = np.linspace(cbar.vmin, cbar.vmax, 5)
        #cbar.set_ticks(ticks)
        #cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        #cbar.ax.tick_params(labelsize=label_size-1)
    plt.tight_layout()
    plt.savefig(f"./plot/{state}_renew_price_distribution.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
#%% supporting fig for renew-grid's fluctuation distribution
from matplotlib.ticker import FormatStrFormatter
state_list = ['Dunkirk', 'Skive', 'Fredericia', 'Weener']
line_width = 0.7
label_size = 12
tick_length = 5
df_statics = []
for i in range(len(renewable_list)):
    state = state_list[i]    
    #renew = renewable_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    #price = price_list[i].reshape(-1,24,24).mean(axis = -1).reshape(-1)[:10000]
    
    renew = renewable_list[i].reshape(-1,24,24)
    renew = np.abs(renew[:,:,1:]-renew[:,:,:-1]).mean(axis = -1).reshape(-1)[:10000]
    
    price = price_list[i].reshape(-1,24,24)
    price = np.abs(price[:,:,1:]-price[:,:,:-1]).mean(axis = -1).reshape(-1)[:10000]
    
    df_statics.append([np.mean(renew), np.var(renew),
                    np.mean(price), np.var(price)])
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    # Density plot (KDE)
    kde = sns.kdeplot(x=renew, y=price, fill=True, cmap="viridis", levels=10, alpha=1.0, ax=ax)

    # Scatter overlay
    #ax.scatter(renew, price, s=2, color='gray', alpha=0.1)

    # Quadrant coloring
    colors = [
        # First regime: high PEMEC
        (0/255, 102/255, 204/255),    # Deep blue (dominant first regime)
        (102/255, 178/255, 255/255),  # Light blue (second case in first regime)

        # Second regime: low PEMEC
        (0/255, 153/255, 76/255),     # Deep green (dominant second regime)
        (153/255, 230/255, 179/255),  # Pale green (second case in second regime)
    ] 
    
    # Axis formatting
    ax.set_xlim(-2, 7)
    ax.set_ylim(-2, 7)
    #ax.set_xlabel("Nitrogen application rate (kg ha$^{-1}$)")
    #ax.set_ylabel("Yield (ton ha$^{-1}$)")
    #ax.set_title("Terai of Nepal")
    ax.grid(True, which='both', linestyle=':', linewidth=line_width/5, color='gray', alpha=0.5)
    ax.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
    ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
    ax.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=tick_length/3*2, width=line_width)
    plt.tight_layout()
    #if i == 3:
    #    cbar.set_label("Density level", fontsize=label_size)
    #    cbar = plt.colorbar(kde.collections[0], ax=ax)
    #    #cbar.set_label("Density", fontsize=label_size)     
    #    ticks = np.linspace(cbar.vmin, cbar.vmax, 5)
    #    cbar.set_ticks(ticks)
    #    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    #    cbar.ax.tick_params(labelsize=label_size-1)
    plt.savefig(f"./plot/{state}_renew_price_delta_distribution.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
#%% figure 1 (c) pareto-curve with (b) design information 
from utils.helper import *
import copy
import json
from matplotlib.ticker import AutoMinorLocator

pareto_obj_supp = []
pareto_des_supp = []
pareto_des_energy = []
pareto_des_energy_percentage = []
SP_H2 = 55.7
X_H2 = 0.19576
P_X = 0.65702

log_dir = './optimization/result'
for country in country_list:
    for region in country_list[country]:
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

        # Feasible → Pareto / Dominated
        is_feas   = (con_set <= 0).all(-1)
        feas_des  = des_set[is_feas]
        feas_obj  = obj_set[is_feas]
        feas_con  = con_set[is_feas]
        pareto_m  = is_non_dominated(feas_obj)
        pareto_des = feas_des[pareto_m]
        pareto_obj = feas_obj[pareto_m]
        pareto_con = feas_con[pareto_m]
        dom_des   = feas_des[~pareto_m]
        dom_obj   = feas_obj[~pareto_m]
        dom_con = feas_con[~pareto_m]
        
        po = pareto_obj.detach().cpu().numpy()
        idx = np.argsort(-po[:, 0])
        des = pareto_des.detach().cpu().numpy()
        des = des[idx]
        knee_point = np.where(des[:,2]<0.1)[0]
        
        scale_min = 5000
        scale_max = 25000
        SP_H2 = 55.7
        X_H2 = 0.19576
        P_X = 0.65702
        X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
        LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
        ESS_cap_range = np.array([scale_min, scale_max*4])
        des = des[:,[3,2,1,0]] #X-flow, PEM-ratio, BESS, LH2
        pareto_des_supp.append(des) 
        des_energy = np.zeros(shape = (len(des),4))
        des_energy[:,0] = des[:,0]*(P_X+X_H2*SP_H2)
        des_energy[:,2] = des[:,2]*0.3
        des_energy[:,3] = des[:,3]*SP_H2
        PEM_P_cap_min = des[:,0]*X_H2*SP_H2
        PEM_P_cap_max = des[:,3]*SP_H2 + PEM_P_cap_min
        des_energy[:,1] = des[:,1]*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        total_cap = des_energy.sum(axis=1)          
        des_energy_pct = des_energy.copy()      
        des_energy_pct = des_energy_pct / total_cap.reshape(-1, 1)
        des_energy_final = np.hstack([des_energy_pct, 
                                      total_cap.reshape(-1, 1)/1000
                                      ])
        
        pareto_des_energy.append(des_energy_final.transpose())

        do = dom_obj.detach().cpu().numpy()

        plt.figure(figsize=(7.5, 4.0))
        #plt.scatter(-do[:,0], -do[:,1]*100, s=30, color='gray', alpha=0.30)
        if knee_point.size == 0:
            plt.plot(-po[idx,0], des_energy[:,0]/1000, '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx = knee_point[0]
            plt.plot(-po[idx,0][:knee_idx+1], des_energy[:,0][:knee_idx+1]/1000, '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')

            plt.plot(-po[idx,0][knee_idx:], des_energy[:,0][knee_idx:]/1000, '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.2, 1.4)
        plt.ylim(4, 12)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"./plot/{region}_meoh_curve.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%%
for country in country_list:
    for region in country_list[country]:
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

        # Feasible → Pareto / Dominated
        is_feas   = (con_set <= 0).all(-1)
        feas_des  = des_set[is_feas]
        feas_obj  = obj_set[is_feas]
        feas_con  = con_set[is_feas]
        pareto_m  = is_non_dominated(feas_obj)
        pareto_des = feas_des[pareto_m]
        pareto_obj = feas_obj[pareto_m]
        pareto_con = feas_con[pareto_m]
        dom_des   = feas_des[~pareto_m]
        dom_obj   = feas_obj[~pareto_m]
        dom_con = feas_con[~pareto_m]
        
        po = pareto_obj.detach().cpu().numpy()
        idx = np.argsort(-po[:, 0])
        des = pareto_des.detach().cpu().numpy()
        des = des[idx]
        knee_point = np.where(des[:,2]<0.1)[0]
        
        scale_min = 5000
        scale_max = 25000
        SP_H2 = 55.7
        X_H2 = 0.19576
        P_X = 0.65702
        X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
        LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
        ESS_cap_range = np.array([scale_min, scale_max*4])
        des = des[:,[3,2,1,0]] #X-flow, PEM-ratio, BESS, LH2
        pareto_des_supp.append(des) 
        des_energy = np.zeros(shape = (len(des),4))
        des_energy[:,0] = des[:,0]*(P_X+X_H2*SP_H2)
        des_energy[:,2] = des[:,2]*0.3
        des_energy[:,3] = des[:,3]*SP_H2
        PEM_P_cap_min = des[:,0]*X_H2*SP_H2
        PEM_P_cap_max = des[:,3]*SP_H2 + PEM_P_cap_min
        des_energy[:,1] = des[:,1]*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        total_cap = des_energy.sum(axis=1)          
        des_energy_pct = des_energy.copy()      
        des_energy_pct = des_energy_pct / total_cap.reshape(-1, 1) * 100
        des_energy_final = np.hstack([des_energy_pct, 
                                      total_cap.reshape(-1, 1)/1000
                                      ])
        
        pareto_des_energy.append(des_energy_final)

        do = dom_obj.detach().cpu().numpy()

        plt.figure(figsize=(7.5, 3.5))
        #plt.scatter(-do[:,0], -do[:,1]*100, s=30, color='gray', alpha=0.30)
        if knee_point.size == 0:
            plt.plot(-po[idx,0], des_energy[:,1]/1000, '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx = knee_point[0]
            plt.plot(-po[idx,0][:knee_idx+1], des_energy[:,1][:knee_idx+1]/1000, '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')

            plt.plot(-po[idx,0][knee_idx:], des_energy[:,1][knee_idx:]/1000, '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.2, 1.4)
        plt.ylim(-5, 85)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"./plot/{region}_pemec_curve.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%%
for country in country_list:
    for region in country_list[country]:
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

        # Feasible → Pareto / Dominated
        is_feas   = (con_set <= 0).all(-1)
        feas_des  = des_set[is_feas]
        feas_obj  = obj_set[is_feas]
        feas_con  = con_set[is_feas]
        pareto_m  = is_non_dominated(feas_obj)
        pareto_des = feas_des[pareto_m]
        pareto_obj = feas_obj[pareto_m]
        pareto_con = feas_con[pareto_m]
        dom_des   = feas_des[~pareto_m]
        dom_obj   = feas_obj[~pareto_m]
        dom_con = feas_con[~pareto_m]
        
        po = pareto_obj.detach().cpu().numpy()
        idx = np.argsort(-po[:, 0])
        des = pareto_des.detach().cpu().numpy()
        des = des[idx]
        knee_point = np.where(des[:,2]<0.1)[0]
        
        scale_min = 5000
        scale_max = 25000
        SP_H2 = 55.7
        X_H2 = 0.19576
        P_X = 0.65702
        X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
        LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
        ESS_cap_range = np.array([scale_min, scale_max*4])
        des = des[:,[3,2,1,0]] #X-flow, PEM-ratio, BESS, LH2
        pareto_des_supp.append(des) 
        des_energy = np.zeros(shape = (len(des),4))
        des_energy[:,0] = des[:,0]*(P_X+X_H2*SP_H2)
        des_energy[:,2] = des[:,2]
        des_energy[:,3] = des[:,3]*SP_H2
        PEM_P_cap_min = des[:,0]*X_H2*SP_H2
        PEM_P_cap_max = des[:,3]*SP_H2 + PEM_P_cap_min
        des_energy[:,1] = des[:,1]*(PEM_P_cap_max-PEM_P_cap_min)# + PEM_P_cap_min
        total_cap = des_energy.sum(axis=1)          
        des_energy_pct = des_energy.copy()/1000      
        #des_energy_pct = des_energy_pct / total_cap.reshape(-1, 1) * 100
        des_energy_final = np.hstack([des_energy_pct, 
                                      total_cap.reshape(-1, 1)/1000
                                      ])
        
        pareto_des_energy.append(des_energy_final)

        do = dom_obj.detach().cpu().numpy()

        plt.figure(figsize=(7.5, 4.0))
        #plt.scatter(-do[:,0], -do[:,1]*100, s=30, color='gray', alpha=0.30)
        if knee_point.size == 0:
            plt.plot(-po[idx,0], des_energy[:,2:4].sum(axis=1)/1000, '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx = knee_point[0]
            plt.plot(-po[idx,0][:knee_idx+1], des_energy[:,2:4].sum(axis=1)[:knee_idx+1]/1000, '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')

            plt.plot(-po[idx,0][knee_idx:], des_energy[:,2:4].sum(axis=1)[knee_idx:]/1000, '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.2, 1.4)
        plt.ylim(-5, 200)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"./plot/{region}_ees_curve.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%% figure 1 (d) pareto-curve with (b) design information 
from utils.helper import *
import copy
import json
from matplotlib.ticker import AutoMinorLocator

pareto_obj_supp = []
pareto_des_supp = []
pareto_des_energy = []
pareto_des_energy_percentage = []
SP_H2 = 55.7
X_H2 = 0.19576
P_X = 0.65702

log_dir = './optimization/result'
for country in country_list:
    for region in country_list[country]:
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

        # Feasible → Pareto / Dominated
        is_feas   = (con_set <= 0).all(-1)
        feas_des  = des_set[is_feas]
        feas_obj  = obj_set[is_feas]
        feas_con  = con_set[is_feas]
        pareto_m  = is_non_dominated(feas_obj)
        pareto_des = feas_des[pareto_m]
        pareto_obj = feas_obj[pareto_m]
        pareto_con = feas_con[pareto_m]
        dom_des   = feas_des[~pareto_m]
        dom_obj   = feas_obj[~pareto_m]
        dom_con = feas_con[~pareto_m]
        
        po = pareto_obj.detach().cpu().numpy()
        idx = np.argsort(-po[:, 0])
        des = pareto_des.detach().cpu().numpy()
        des = des[idx]
        knee_point = np.where(des[:,2]<0.1)[0]
        
        scale_min = 5000
        scale_max = 25000
        SP_H2 = 55.7
        X_H2 = 0.19576
        P_X = 0.65702
        X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
        LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
        ESS_cap_range = np.array([scale_min, scale_max*4])
        des = des[:,[3,2,1,0]] #X-flow, PEM-ratio, BESS, LH2
        pareto_des_supp.append(des) 
        des_energy = np.zeros(shape = (len(des),4))
        des_energy[:,0] = des[:,0]*(P_X+X_H2*SP_H2)
        des_energy[:,2] = des[:,2]*0.3
        des_energy[:,3] = des[:,3]*SP_H2
        PEM_P_cap_min = des[:,0]*X_H2*SP_H2
        PEM_P_cap_max = des[:,3]*SP_H2 + PEM_P_cap_min
        des_energy[:,1] = des[:,1]*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        total_cap = des_energy.sum(axis=1)          
        des_energy_pct = des_energy.copy()      
        des_energy_pct = des_energy_pct / total_cap.reshape(-1, 1) * 100
        des_energy_final = np.hstack([des_energy_pct, 
                                      total_cap.reshape(-1, 1)/1000
                                      ])
        
        pareto_des_energy.append(des_energy_final)

        do = dom_obj.detach().cpu().numpy()

        plt.figure(figsize=(7.5, 6.0))
        plt.scatter(-do[:,0], -do[:,1]*100, s=30, color='gray', alpha=0.30)
        if knee_point.size == 0:
            plt.plot(-po[idx,0], -po[idx,1]*100, '-o',color='red', linewidth=2.5,
                     markersize=11, markerfacecolor='white')
        else:
            knee_idx = knee_point[0]
            plt.plot(-po[idx,0][:knee_idx+1], -po[idx,1][:knee_idx+1]*100, '-o',color='red', linewidth=2.5,
                     markersize=10, markerfacecolor='white')

            plt.plot(-po[idx,0][knee_idx:], -po[idx,1][knee_idx:]*100, '-s',color='#00994C', linewidth=2.5,
                     markersize=10, markerfacecolor='white')
        
        plt.grid(ls='--', lw=0.5, color='gray', alpha=0.4)
        plt.xlim(0.2, 1.4)
        plt.ylim(-200, 50)
        ax = plt.gca()
        ax.tick_params(axis='both', which='major',
               direction='out', length=12, width=1.5, labelsize=20)
        ax.tick_params(axis='both', which='minor',
               direction='out', length=6, width=1.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        plt.tight_layout()
        plt.savefig(f"./plot/{region}_pareto_curve.png", dpi=600, bbox_inches='tight')
        plt.show()
        plt.close()
#%% Supplementary figure (two-panel distribution (histogram)+DES heat map)
import seaborn as sns
import cmcrameri.cm as cmc
from matplotlib.ticker import AutoMinorLocator
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

country_list = {
    'France': {'Dunkirk': {}},
    'Denmark': {'Skive': {}, 'Fredericia': {}},
    'Germany': {'Weener': {}}
}
nature_colors = ["#EF0000", "#336699", "#FEC211", "#3BC371", "#666699", "#999999", "#91D1C2"]

def loading_idx(country, region):
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
    keep_mask = min_diff_filter_on_lcox_min(-pareto_obj, min_diff = 0.02)
    idx_list = keep_mask.nonzero(as_tuple=True)[0].tolist()
    return idx_list

def loading_performance(country, region):
    #country = "France"
    #region = "Dunkirk"
    log_dir = './optimization/result'
    save_template = os.path.join(log_dir,f'{country}/{region}/uq_results_pfss_0.5_idx{{idx}}.pkl')
    env_list = []
    idx_list = loading_idx(country, region)
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
        ctg_list[i, :] = ctg/(X_flow*576/1000)
        des_list[i, :] = np.array([env.X_flow, env.PEM_ratio, env.ESS_cap, env.LH2_cap])
        pfss_list[i,:] = pf

    mu_lcox = np.mean(lcox_list, axis=1)
    idx = np.argsort(mu_lcox)
    
    return lcox_list[idx], ctg_list[idx], des_list[idx], pfss_list[idx], np.array(idx_list)[idx]


def build_regionwise_lcox_ctg():
    lcox_by_region = {}
    ctg_by_region = {}
    pfss_by_region = {}
    color_idx = 0
    for country, regions in country_list.items():
        for region in regions:
            region_key = f"{country}-{region}"
            try:
                lcox_list, ctg_list,_, pfss_list, _ = loading_performance(country, region)
                lcox_by_region[region_key] = []
                ctg_by_region[region_key] = []
                pfss_by_region[region_key] = []
                for i in range(lcox_list.shape[0]):
                    label = f"{region_key}-{i+1}"
                    color = nature_colors[color_idx % len(nature_colors)]
                    lcox_by_region[region_key].append((label, lcox_list[i], color))
                    ctg_by_region[region_key].append((label, ctg_list[i], color))
                    pfss_by_region[region_key].append((label, pfss_list[i], color))
                color_idx += 1
            except Exception as e:
                print(f"[Error] {region_key}: {e}")
    return lcox_by_region, ctg_by_region, pfss_by_region

def flatten_with_spacing(region_dict, base_colors, region_spacing=0.7):
    flat_data = []
    x_position = 0
    color_list = []
    for i, (region, entries) in enumerate(region_dict.items()):
        region_color = base_colors[i % len(base_colors)]
        for entry in entries:
            label, samples, _ = entry
            flat_data.append((x_position, label, samples, region_color))
            color_list.append(region_color)
            x_position += 1

        x_position += region_spacing  # add space between regions
    return flat_data, color_list

def plot_lcox_nature_style_final_with_spacing(flat_data, flat_data_base, ax, y_range=None, value_name="LCOM ($/kg)"):
    
    x_ticks, all_samples, colors, means = [], [], [], []
    all_samples_base = []

    for x_pos, label, samples, color in flat_data:
        x_ticks.append(x_pos)
        all_samples.append(samples)
        colors.append(color)
        means.append(np.mean(samples))

    for x_pos, label, samples, color in flat_data_base:
        all_samples_base.append(samples)

    # Boxplot
    box_positions = [x - 0.2 for x in x_ticks]
    bp = ax.boxplot(
        all_samples,
        positions=box_positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        whis=[5, 95],
        medianprops=dict(color='black', linewidth=0),
        whiskerprops=dict(color='black', linewidth=0.6),
        capprops=dict(color='black', linewidth=0.6)
    )

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(1.0)
        patch.set_edgecolor('none')

    # Scatter
    neg_mask_list = []
    for xi, samples in zip(x_ticks, all_samples_base):
        samples = np.array(samples)
        neg_mask = samples > 0
        neg_mask_list.append(neg_mask)

    for i, (xi, samples) in enumerate(zip(x_ticks, all_samples)):
        xs = np.random.normal(loc=xi + 0.2, scale=0.04, size=len(samples))
        samples = np.array(samples)
        neg_mask = neg_mask_list[i]
        pos_mask = ~neg_mask
        ax.scatter(xs[neg_mask], samples[neg_mask], color='black', alpha=0.3, s=0.3, edgecolor='none')
        ax.scatter(xs[pos_mask], samples[pos_mask], color='black', alpha=0.3, s=0.3, edgecolor='none')

    # Mean lines
    for xi_box, mean in zip(box_positions, means):
        ax.hlines(mean, xi_box - 0.25, xi_box + 0.25, color='blue', linewidth=0.7, zorder=5)

    # Axis
    ax.set_xticks([])
    ax.set_xlim(-1.0, max(x_ticks) + 1.0)
    ax.grid(False)
    ax.tick_params(axis='y', which='both', direction='out', length=3, width=0.6, labelsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=2, width=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if y_range is not None:
        ax.set_ylim(y_range)

    #ax.set_ylabel(value_name, fontsize=8)
    
def plot_pfss_bar_panel(flat_data, pfss_values, ax):
    x_ticks = [x for x, _, _, _ in flat_data]
    bar_positions = [x - 0.2 for x in x_ticks]
    pfss_values = np.array(pfss_values).flatten()
    bar_colors = [color for _, _, _, color in flat_data]  # extract color per design

    ax.bar(bar_positions, pfss_values, width=0.5, color=bar_colors, alpha=0.5)
    ax.set_ylim(0, 1.0)
    #ax.set_ylabel("P(CTG > 0)", fontsize=8)
    ax.set_xticks([])
    ax.tick_params(axis='y', which='both', direction='out', length=3, width=0.6, labelsize=8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.tick_params(axis='y', which='minor', length=2, width=0.5)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#%% Data-loading
lcox_by_region, ctg_by_region, pfss_by_region = build_regionwise_lcox_ctg()
base_colors = ["#F23030", "#0060CA", "#FECE42", "#6BD28A", "#3BC371"]
#%%
flat_lcox, color_list = flatten_with_spacing(lcox_by_region, base_colors, region_spacing=0.3)
flat_ctg, region_color_list = flatten_with_spacing(ctg_by_region, base_colors, region_spacing=0.3)
flat_pfss = [pf[1] for region in pfss_by_region.values() for pf in region]
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(10, 5.8))
gs = gridspec.GridSpec(3, 1, height_ratios=[0.6, 1.8, 1.8])  # Adjust height ratios as needed

# No gap between top (pfss) and middle (LCOM)
gs.update(hspace=0.3)

# Axes
ax0 = fig.add_subplot(gs[0])  # pfss bar
ax1 = fig.add_subplot(gs[1], sharex=ax0)  # LCOM
ax2 = fig.add_subplot(gs[2], sharex=ax0)  # CTG

plot_pfss_bar_panel(flat_lcox, flat_pfss, ax=ax0)

plot_lcox_nature_style_final_with_spacing(
    flat_lcox, flat_ctg, ax=ax1, y_range=(-3.0, 3), value_name="LCOM ($/kg)"
)

plot_lcox_nature_style_final_with_spacing(
    flat_ctg, flat_ctg, ax=ax2, y_range=(-2.0, 2.0),
    value_name=r"Carbon intensity (kgCO₂/kgMeOH)"
)
#axes[1].set_title("(b)", loc='left', fontsize=8, pad=2)
plt.subplots_adjust(hspace=1.0)  # More space between panels
plt.tight_layout()
plt.savefig("./plot/pareto_uq_result.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
#%% DES-heat map
from matplotlib.colors import LinearSegmentedColormap
output_dir='./plot'
color_list = [
    (64/255, 42/255, 180/255),
    (35/255, 160/255, 229/255),
    (87/255, 204/255, 122/255),
    (240/255, 186/255, 54/255),
    (247/255, 245/255, 27/255)
]
# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_rgb_cmap', color_list, N=256
)
des_pack = []
scale_min = 5000
scale_max = 25000
SP_H2 = 55.7
X_H2 = 0.19576
P_X = 0.65702
X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
ESS_cap_range = np.array([scale_min, scale_max*4])

lb = np.array([X_flow_range[0], 0, ESS_cap_range[0], LH2_cap_range[0], ])
ub = np.array([X_flow_range[1], 1, ESS_cap_range[1], LH2_cap_range[1], ])
idx_pack = []
for country, regions in country_list.items():
    for region in regions:
        try:
            _, _, des_list, _, idx_list = loading_performance(country, region)
            des_list = (des_list - lb) / (ub - lb)
            #des_pack.append(des_list)
            # Transpose to shape (4, n_designs) for heatmap
            data = des_list.T
            fig, ax = plt.subplots(figsize=(max(4, des_list.shape[0] * 0.4), 4.5))
            sns.heatmap(
                data,
                cmap=custom_cmap,  
                vmin=0, vmax=1,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax
            )
            idx_pack.append(idx_list)
            #ax.set_yticklabels(['X_flow', 'ESS', 'LH2', 'PEM'], rotation=0, fontsize=8)
            #ax.tick_params(left=True, bottom=False)
            #ax.set_aspect('equal')
            plt.tight_layout()
            save_name = f"{country}_{region}_heatmap.png".replace(" ", "_")
            plt.savefig(os.path.join(output_dir, save_name), dpi=600, bbox_inches='tight')
            plt.show()
            plt.close()
            print(f"[Saved] {save_name}")
        except Exception as e:
            print(f"[Error] {country}-{region}: {e}")
#%%
from matplotlib.colors import LinearSegmentedColormap
output_dir='./plot'
color_list = [
    (64/255, 42/255, 180/255),
    (35/255, 160/255, 229/255),
    (87/255, 204/255, 122/255),
    (240/255, 186/255, 54/255),
    (247/255, 245/255, 27/255)
]
# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_rgb_cmap', color_list, N=256
)
for i in range(4):
    des_list = pareto_des_energy[i][:,:-1]/100
    data = des_list.T
    fig, ax = plt.subplots(figsize=(max(4, des_list.shape[0] * 0.4), 4.5))
    sns.heatmap(
                data,
                cmap=custom_cmap,  
                vmin=0, vmax=1,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=ax
    )
    plt.tight_layout()
    #save_name = f"{country}_{region}_heatmap.png".replace(" ", "_")
    #plt.savefig(os.path.join(output_dir, save_name), dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
            
#%%   
des_pack_unnormal = [] #kg/hr, MW, BESS-MW, CH2-kg
des_pack_unnormal_MW = []
for i in range(len(des_pack)):
    des_list = des_pack[i]*(ub-lb) + lb
    P_X = 0.65702 
    X_H2 = 0.19576
    SP_H2 = 55.7

    PEM_P_cap_min = des_list[:,0]*X_H2*SP_H2
    PEM_P_cap_max = des_list[:,-1]*SP_H2 + PEM_P_cap_min
    des_list[:,1] = des_list[:,1]*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min

    des_list[:,1:3] = des_list[:,1:3]/1000 #kw-->MW    
    des_pack_unnormal.append(des_list)
    
    des_list_MW = np.copy(des_list)
    des_list_MW[:,0] = des_list[:,0]*(P_X+X_H2*SP_H2)/1000 #kW-->MW
    des_list_MW[:,3] = des_list[:,3]*(SP_H2)/1000 #kW-->MW
    des_pack_unnormal_MW.append(des_list_MW)