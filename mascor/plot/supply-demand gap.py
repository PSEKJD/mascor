import os
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.env.ptx_env_stack import *
import seaborn as sns

# Load and preprocess data into data_package
country_list = {
    'France': {'Dunkirk': {}},
    'Denmark': {'Skive': {}, 'Fredericia': {}},
    'Germany': {'Weener':{}}
    #'Germany': {'Wunsiedel': {}, 'Weener': {}}
}

data_package = {}

for target_country in country_list:
    for region in country_list[target_country]:
        log_dir = './optimization/result'
        #if region == 'Dunkirk':
        #    save_path = os.path.join(log_dir, f'{target_country}/{region}/uq_results_random_pfss_0.5.pkl')
        #else:
        #    save_path = os.path.join(log_dir, f'{target_country}/{region}/uq_results_random_pfss_0.05.pkl')
        save_path = os.path.join(log_dir, f'{target_country}/{region}/uq_results_random_pfss_0.5.pkl')
        with open(save_path, "rb") as file:
            env_list = pickle.load(file)
        # Region label
        short = {'Dunkirk': 'DU', 'Skive': 'SK', 'Fredericia': 'FR', 'Weener': 'WE'}[region]
        name = f"{short}"
        data_package[name] = {'env':  env_list}
#%%
def extract_power_usage(env):
    demand = env.X_flow_P_cap + env.X_flow*env.X_H2*env.SP_H2
    supply = env.renewable
    gap = supply-demand
    gap[gap < 0] = 0
    #gap = gap.reshape(-1, 24, 24).mean(axis=1)
    
    PtG = env.P_to_G
    PtG[PtG<0] = 0
    #PtG = PtG.reshape(-1,24,24).mean(axis=1)

    HU = (env.L_H2_profile[:,1:] - env.L_H2_profile[:, :-1]) * env.SP_H2
    HS = np.copy(-HU)
    HU[HU < 0] = 0
    HS[HS < 0] = 0
    DH = (env.SOC_profile[:,1:] - env.SOC_profile[:, :-1]) * env.ESS_eff
    CH = np.copy(-DH)
    DH[DH < 0] = 0
    CH[CH > 0] = 0
    HM = env.H2_to_market*env.SP_H2
    #HU, HS, DH, CH = (HU.reshape(-1,24,24).mean(axis=1), HS.reshape(-1,24,24).mean(axis=1), 
    #                  DH.reshape(-1,24,24).mean(axis=1), CH.reshape(-1,24,24).mean(axis=1))
    #HM = HM.reshape(-1,24,24).mean(axis = 1)
    return (gap, PtG, HU, HS, DH, CH, HM)

def keep_index_env(dataset, min_mean_diff=0.01):
    num_points = len(dataset['env'])
    lcox_list = np.zeros((num_points, 1000))
    ctg_list = np.zeros((num_points, 1000))
    des_list = np.zeros((num_points, 4))
    pfss_list = np.zeros((num_points, 1))

    for i in range(num_points):
        env = dataset['env'][i]
        mu_LCOX, var_LCOX, LCOX = env.LCOX_calculation()
        ctg = np.sum(env.CO2_emit, axis = 1)
        pf = len(np.where(ctg>0)[0])/len(ctg)
        mu_ctg = np.mean(ctg)
        var_ctg = np.var(ctg)
        X_flow = env.X_flow #kg
        LCOX[np.where(LCOX<0)] = 0
        lcox_list[i, :] = LCOX
        ctg_list[i, :] = ctg/(X_flow*576/1000)
        des_list[i, :] = np.array([env.X_flow, env.PEM_ratio, env.ESS_cap, env.LH2_cap])
        pfss_list[i,:] = pf

        #PEM_P_cap_min = env.X_flow_range[0]*env.X_H2*env.SP_H2
        #PEM_P_cap_max = env.LH2_cap_range[1]*env.SP_H2 + env.X_flow_range[1]*env.X_H2*env.SP_H2
        #lb = np.array([env.X_flow_range[0], PEM_P_cap_min, env.ESS_cap_range[0], env.LH2_cap_range[0], ])
        #ub = np.array([env.X_flow_range[1], PEM_P_cap_max, env.ESS_cap_range[1], env.LH2_cap_range[1], ])
        lb = np.array([env.X_flow_range[0], 0, env.ESS_cap_range[0], env.LH2_cap_range[0], ])
        ub = np.array([env.X_flow_range[1], 1, env.ESS_cap_range[1], env.LH2_cap_range[1], ])
    
    des_list = (des_list - lb) / (ub - lb)
    mu_lcox = np.mean(lcox_list, axis=1)
    keep_indices = [0]
    for i in range(1, len(mu_lcox)):
        if np.min(np.abs(mu_lcox[i] - mu_lcox[keep_indices])) > min_mean_diff:
            keep_indices.append(i)
    return keep_indices, des_list, lcox_list.mean(axis = 1), ctg_list.mean(axis=1), pfss_list[:,0]

regions = ['DU', 'SK', 'FR', 'WE']
cases = ['Economic', 'Middle', 'Environment']
gap_list = []
PtG_list = []
ST_list = []
HM_list = []

for region in regions:
    dataset = data_package[region]
    env_indicies, _,_,_,_ = keep_index_env(dataset)
    for i in [0,0.5,-1]: #first & last design case
        if i == 0.5:
            idx = int(len(env_indicies)/2)
        else:
            idx = i
        env = dataset['env'][env_indicies[idx]]
        (gap, PtG, HU, HS, DH, CH, HM) = extract_power_usage(env)
        gap_list.append(gap.mean(axis = 1)/1000)
        PtG_list.append(PtG.mean(axis = 1)/1000)
        HM_list.append(HM.mean(axis = 1)/1000)
#%% Plot box plot of renewable power usage
base_y = np.arange(len(regions))*len(cases)
positions = []
gap = 0.6
for y in base_y:
    positions.append(y - gap)  # High Profit
    positions.append(y)  # High Profit
    positions.append(y + gap)  # Low CO2

# Create figure
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_axes([0.0, 0.1, 0.5, 0.85])  # left subplot
ax2 = fig.add_axes([0.5, 0.1, 0.5, 0.85])  # right subplot

# Shared visible x-axis limit
visible_xlim = (0, 20)

# Boxplot 1: Renewable Power to Grid
bp1 = ax1.boxplot(PtG_list, positions=positions, widths=0.4, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=0.6),
                  capprops=dict(color='black', linewidth=0.6))

# Boxplot 2: Hydrogen Production
bp2 = ax2.boxplot(HM_list, positions=positions, widths=0.4, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=0.6),
                  capprops=dict(color='black', linewidth=0.6))

# Colors
colors = ['tab:grey', 'tab:blue', 'tab:green'] * len(regions)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

# Middle dotted line
fig.lines.append(plt.Line2D([0.5, 0.5], [0.1, 0.95],
                            transform=fig.transFigure,
                            linestyle='--', color='gray', linewidth=1.2))

# Ax1: Renewable to grid
ax1.set_yticks(base_y)
ax1.set_yticklabels(regions)
ax1.invert_yaxis()
ax1.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax1.set_xlim(visible_xlim)
ax1.set_xbound(lower=0, upper=22.5)  # Internal span
xticks_ax1 = ax1.get_xticks()
ax1.set_xticks(xticks_ax1[:-1]) 

# Ax2: Hydrogen production
ax2.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax2.set_yticks([])
ax2.tick_params(labelleft=False, left=False)
ax2.spines['left'].set_visible(False)
ax2.set_xlim(visible_xlim)
ax2.set_xbound(lower=-2.5, upper=20)  # Internal span
ax2.invert_yaxis()
xticks_ax2 = ax2.get_xticks()
ax2.set_xticks(xticks_ax2[1:]) 

plt.show()
plt.close()
#%% Load dataset and select one case
dataset = data_package['SK']
env_indicies, des_list, mu_lcox, mu_ctg, pfss = keep_index_env(dataset)
des_list = des_list[env_indicies]
mu_lcox = mu_lcox[env_indicies]
mu_ctg = mu_ctg[env_indicies]
pfss = pfss[env_indicies]
#%% Plotting renewable power usage distribution


#%%
def extract_emission_and_storage(env):
    demand = env.X_flow_P_cap + env.X_flow*env.X_H2*env.SP_H2
    supply = env.renewable
    gap = supply-demand
    #gap[gap > 0] = 0
    gap[gap < 0] = 0
    gap = gap.reshape(-1, 24, 24).mean(axis=1)
    
    PtG = env.P_to_G
    PtG[PtG<0] = 0
    PtG = PtG.reshape(-1,24,24).mean(axis=1)

    HU = (env.L_H2_profile[:,1:] - env.L_H2_profile[:, :-1]) * env.SP_H2
    HS = np.copy(-HU)
    HU[HU < 0] = 0
    #HS[HS < 0] = 0
    DH = (env.SOC_profile[:,1:] - env.SOC_profile[:, :-1]) * env.ESS_eff
    CH = np.copy(-DH)
    DH[DH < 0] = 0
    #CH[CH > 0] = 0
    HM = env.H2_to_market*env.SP_H2
    
    HU, HS, DH, CH = (HU.reshape(-1,24,24).mean(axis=1), HS.reshape(-1,24,24).mean(axis=1), 
                      DH.reshape(-1,24,24).mean(axis=1), CH.reshape(-1,24,24).mean(axis=1))
    HM = HM.reshape(-1,24,24).mean(axis = 1)
    return (gap.mean(axis=0), gap.std(axis=0),
            DH.mean(axis=0), DH.std(axis=0), #util-part
            HU.mean(axis=0), HU.std(axis=0), #util-part
            HM.mean(axis = 0), HM.std(axis=0),
            PtG.mean(axis = 0), PtG.std(axis=0))

for i in env_indicies:
    env1 = dataset['env'][i]

    # Extract values
    (emission_1_mean, emission_1_std, 
     bess_storage_1_mean, bess_storage_1_std,  
     h2_storage_1_mean, h2_storage_1_std,
     h2_market_1_mean, h2_market_1_std,
     PtG_mean, PtG_std) = extract_emission_and_storage(env1)
    threshold_1 = env1.X_flow * env1.X_CO2/env1.emission_factor

    # Plot
    plt.figure(figsize=(7, 4))

    # Emission band
    plt.fill_between(np.arange(24),
                     emission_1_mean - emission_1_std * 1.5,
                     emission_1_mean + emission_1_std * 1.5,
                     color='red', alpha=0.2)
    plt.plot(emission_1_mean, color='red', linewidth=2.0, label='Emissions')
    plt.axhline(y=threshold_1, color='red', linestyle='--', linewidth=1.5, label='Threshold')

    # Storage utilization band
    plt.fill_between(np.arange(24),
                     bess_storage_1_mean - bess_storage_1_std* 1.5,
                     bess_storage_1_mean + bess_storage_1_std* 1.5,
                     color='blue', alpha=0.2)
    plt.plot(bess_storage_1_mean, color='blue', linestyle='-', linewidth=2.0, label='BESS Utilization')
    
    plt.fill_between(np.arange(24),
                     h2_storage_1_mean - h2_storage_1_std* 1.5,
                     h2_storage_1_mean + h2_storage_1_std* 1.5,
                     color='green', alpha=0.2)
    plt.plot(h2_storage_1_mean, color='green', linestyle='-', linewidth=2.0, label='H2 Utilization')
    
    plt.fill_between(np.arange(24),
                     h2_market_1_mean - h2_market_1_std* 1.5,
                     h2_market_1_mean + h2_market_1_std* 1.5,
                     color='purple', alpha=0.2)
    plt.plot(h2_market_1_mean, color='purple', linestyle='-', linewidth=2.0, label='H2 market')
    
    plt.fill_between(np.arange(24),
                     PtG_mean - PtG_std* 1.5,
                     PtG_mean + PtG_std* 1.5,
                     color='grey', alpha=0.2)
    plt.plot(PtG_mean, color='grey', linestyle='-', linewidth=2.0, label='H2 market')

    # Formatting
    plt.xlabel("Hour of the Day")
    plt.ylabel("CO₂ Emissions / Storage Use")
    plt.title("Hourly Emissions & Storage Utilization (Region: DU-1, Single Case)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 23)
    #plt.ylim(0, 5000)
    plt.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.show()
    plt.close()


#%% env-check
env = dataset['env'][0]
env.LH2_cap_range, env.LH2_cap, env.PEM_ratio, env.PEM_P_cap/env.SP_H2, env.X_flow*env.X_H2
action = env.action_acc

#%%
i = 0
env = dataset['env'][i]
for i in range(10):
    plt.plot(env.SOC_profile[i])
plt.show()
plt.close()

i = 0
env = dataset['env'][i]
for i in range(10):
    plt.plot(env.L_H2_profile[i])
plt.show()
plt.close()

env.LH2_cap

i = 10
env = dataset['env'][i]
for i in range(10):
    plt.plot(env.H2_to_market[i,:])
plt.show()
plt.close()

i = 1
env = dataset['env'][i]
for i in range(10):
    plt.plot(env.P_to_G[i,:100])
plt.show()
plt.close()
i = 1
env = dataset['env'][i]
for i in range(10):
    plt.plot(env.CO2_emit[i,:100])
plt.show()
plt.close()
