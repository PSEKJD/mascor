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
#%%
def extract_power_usage(env):
    demand = env.X_flow*env.P_X + env.X_flow*env.X_H2*env.SP_H2
    supply = env.renewable.detach().cpu().numpy()
    gap = supply-demand
    gap[gap < 0] = 0
    #gap = gap.reshape(-1, 24, 24).mean(axis=1)
    
    PtG = env.P_to_G.detach().cpu().numpy()
    GtP = -np.copy(PtG)
    PtG[PtG<0] = 0
    GtP[GtP<0] = 0
    #PtG = PtG.reshape(-1,24,24).mean(axis=1)

    HU = np.abs((env.L_H2_profile[:,1:].detach().cpu().numpy() 
                 - env.L_H2_profile[:, :-1].detach().cpu().numpy())) * env.SP_H2
    #HS = np.copy(HU)
    #HU[HU < 0] = 0
    #HS[HS < 0] = 0
    BU = np.abs((env.SOC_profile[:,1:].detach().cpu().numpy() 
                 - env.SOC_profile[:, :-1].detach().cpu().numpy())) * env.ESS_eff
    #CH = np.copy(DH)
    #DH[DH < 0] = 0
    #CH[CH < 0] = 0
    HM = env.H2_to_market.detach().cpu().numpy()*env.SP_H2
    #HU, HS, DH, CH = (HU.reshape(-1,24,24).mean(axis=1), HS.reshape(-1,24,24).mean(axis=1), 
    #                  DH.reshape(-1,24,24).mean(axis=1), CH.reshape(-1,24,24).mean(axis=1))
    #HM = HM.reshape(-1,24,24).mean(axis = 1)
    DM = env.X_flow*env.P_X + env.X_flow*env.X_H2*env.SP_H2
    
    return (gap, PtG, GtP, HU, BU, HM,  DM)

def keep_index_env(region, dataset, min_mean_diff=0.00):
    num_points = len(dataset['env'])
    lcox_list = np.zeros((num_points, 1000))
    ctg_list = np.zeros((num_points, 1000))
    des_list = np.zeros((num_points, 4))
    pfss_list = np.zeros((num_points, 1))
    demand_list = np.zeros((num_points, 1))
    
    for i in range(num_points):
        env = dataset['env'][i]
        mu_LCOX, var_LCOX, LCOX = env.LCOX.mean(), env.LCOX.var(), env.LCOX
        ctg = np.sum(env.CO2_emit.detach().cpu().numpy(), axis = 1)
        pf = len(np.where(ctg>0)[0])/len(ctg)
        mu_ctg = np.mean(ctg)
        var_ctg = np.var(ctg)
        X_flow = env.X_flow #kg
        LCOX[np.where(LCOX<0)] = 0
        lcox_list[i, :] = LCOX
        ctg_list[i, :] = ctg/(X_flow*576/1000)
        des_list[i, :] = np.array([env.X_flow, env.PEM_ratio, env.ESS_cap, env.LH2_cap])
        pfss_list[i,:] = pf

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
        base_demand = env.X_flow*P_X + env.X_flow*env.X_H2*env.SP_H2
        demand_list[i,0] = base_demand
            
    des_list = (des_list - lb) / (ub - lb)
    mu_lcox = np.mean(lcox_list, axis=1)
    
    keep_indices = [0]
    for i in range(1, len(mu_lcox)):
        if np.min(np.abs(mu_lcox[i] - mu_lcox[keep_indices])) > min_mean_diff:
            keep_indices.append(i)
    
    keep_indices_sorted = sorted(keep_indices, key=lambda x: mu_lcox[x])
        
    return keep_indices_sorted, des_list, lcox_list.mean(axis = 1), ctg_list.mean(axis=1), pfss_list[:,0], demand_list
#%%
regions = ['DU', 'SK', 'FR', 'WE']
cases = ['regime1-1', 'regime1-2', 'regime2-1', 'regime2-2']
gap_list = []
PtG_list = []
GtP_list = []
ST_list = []
HM_list = []
HU_list = []
BU_list = []
demand_list = []
LCOX_list = []
ctg_list = []
pf_list = []
GtP_threshold_list = []
renewable_list = []
price_list = []
des_pack = []
for region in regions:
    dataset = data_package[region]
    env_indicies, des_list,_,_,_, _ = keep_index_env(region, dataset)
    if region == 'DU':
        case_idx = [0, -1, None, None]
    else:
        first_regime = np.where(des_list[env_indicies,1]>0.2)[0][[0,-1]]
        second_regime = np.where(des_list[env_indicies,1]<0.05)[0][[0,-1]]
        case_idx = np.concatenate((first_regime, second_regime))
    
    des_array = np.zeros(shape = (len(case_idx),4))
    for i, idx in enumerate(case_idx): #first & last design case
        if idx == None:
            gap_list.append([])
            PtG_list.append([])
            GtP_list.append([])
            HM_list.append([])
            HU_list.append([])
            BU_list.append([])
            demand_list.append([])
            GtP_threshold_list.append([])
        
        else:
            env = dataset['env'][env_indicies[idx]]
            mu_LCOX, var_LCOX, LCOX = env.LCOX.mean(), env.LCOX.var(), env.LCOX
            ctg = np.sum(env.CO2_emit.detach().cpu().numpy(), axis = 1)
            mu_ctg = np.mean(ctg)
            pf = len(np.where(ctg>0)[0])/len(ctg)
            LCOX_list.append(mu_LCOX)
            ctg_list.append(mu_ctg)
            pf_list.append(pf)
            
            base_capture = env.X_flow*env.X_CO2/env.emission_factor/1000
            
            print(f"{region}: LCOX {mu_LCOX}, ctg {mu_ctg}, pf {pf}")
            (gap, PtG, GtP, HU, BU, HM, demand) = extract_power_usage(env)
            gap_list.append(gap.mean(axis = 1)/1000)
            PtG_list.append(PtG.mean(axis = 1)/1000)
            GtP_list.append(GtP.mean(axis = 1)/1000)
            HM_list.append(HM.mean(axis = 1)/1000)
            HU_list.append(HU.mean(axis = 1)/1000)
            BU_list.append(BU.mean(axis = 1)/1000)
            demand_list.append(demand/1000)
            GtP_threshold_list.append(base_capture)
            
            des_array[i, :] = [env.X_flow, env.PEM_ratio, env.ESS_cap, env.LH2_cap]
            
    scale_min = 5000
    scale_max = 25000
    SP_H2 = 55.7
    X_H2 = 0.19576
    P_X = 0.65702
    X_flow_range = np.array([scale_min/(P_X+X_H2*SP_H2), scale_max/(P_X+X_H2*SP_H2)])
    LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
    ESS_cap_range = np.array([scale_min, scale_max*4])
    des_energy = np.zeros(shape = (len(des_array),4))
    des_energy[:,0] = des_array[:,0]*(P_X+X_H2*SP_H2)
    des_energy[:,2] = des_array[:,2]*0.3
    des_energy[:,3] = des_array[:,3]*SP_H2
    PEM_P_cap_min = des_array[:,0]*X_H2*SP_H2
    PEM_P_cap_max = des_array[:,3]*SP_H2 + PEM_P_cap_min
    des_energy[:,1] = des_array[:,1]*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
    total_cap = des_energy.sum(axis=1)          
    des_energy_pct = des_energy.copy()      
    des_energy_pct = des_energy_pct / total_cap.reshape(-1, 1) * 100
    des_energy_final = np.hstack([des_energy_pct, 
                                  total_cap.reshape(-1, 1)/1000
                                  ])
    des_pack.append(des_energy_final)
            #else:
                #renew = np.concatenate((renew, env.renewable/1000), axis = 0)
               # price = np.concatenate((renew, env.SMP*1000), axis = 0)
#%% Plot box plot of surplus renew & Power to grid & HM 
base_y = np.arange(len(regions))*len(cases)
positions = []
gap = 1.0
tick_gap = 1.25
for y in base_y:
    positions.append(y - 1.5 * gap)  # Regime 1 - Case 1
    positions.append(y - 0.5 * gap)  # Regime 1 - Case 2
    positions.append(y + 0.5 * gap)  # Regime 2 - Case 1
    positions.append(y + 1.5 * gap)  # Regime 2 - Case 2

# Create figure
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(16, 5))
ax0 = fig.add_axes([0.0,    0.1, 0.4, 0.85])
ax1 = fig.add_axes([0.4,  0.1, 0.4/(25+tick_gap)*(25+tick_gap*2), 0.85])
ax2 = fig.add_axes([0.4+0.4/(25+tick_gap)*(25+tick_gap*2),  0.1, 0.4, 0.85])

# Shared visible x-axis limit
box_width = 0.45
line_width = 1.0
label_size = 20
tick_length = 5

# Boxplot 0: Renewable production
bp0 = ax0.boxplot(gap_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 90],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))
# Boxplot 1: Renewable Power to Grid
bp1 = ax1.boxplot(PtG_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))

# Boxplot 2: Hydrogen Production
bp2 = ax2.boxplot(HM_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))

# Colors
colors = [
    # First regime: high PEMEC
    (192/255, 0/255, 0/255),    # Deep red (dominant first regime)
    (255/255, 33/255, 33/255),  # Light red (second case in first regime)

    # Second regime: low PEMEC
    (0/255, 153/255, 76/255),     # Deep green (dominant second regime)
    (153/255, 230/255, 179/255),  # Pale green (second case in second regime)
] * len(regions)

#colors = ['tab:grey', 'tab:blue', 'tab:green'] * len(regions)
for patch, color in zip(bp0['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

# Middle dotted line
fig.lines.append(plt.Line2D([0.4, 0.4], [0.1, 0.95],
                            transform=fig.transFigure,
                            linestyle='--', color='gray', linewidth=line_width))
fig.lines.append(plt.Line2D([0.4+0.4/(25+tick_gap)*(25+tick_gap*2),
                             0.4+0.4/(25+tick_gap)*(25+tick_gap*2)], [0.1, 0.95],
                            transform=fig.transFigure,
                            linestyle='--', color='gray', linewidth=line_width))

for y in base_y[1:]:
    ax0.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)
    ax1.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)
    ax2.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)

# Ax1: Renewable to grid
visible_xlim = (0, 30)
ax0.set_yticks(base_y)
ax0.set_yticklabels([])
ax0.grid(False)
ax0.invert_yaxis()
#ax1.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax0.set_xlim(visible_xlim)
ax0.set_xbound(lower=0, upper=30+tick_gap)  # Internal span
xticks_ax0 = ax0.get_xticks()
ax0.set_xticks(xticks_ax0[:-1]) 
ax0.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
ax0.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax0.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)    
ax0.spines['right'].set_visible(False)

# Ax1: Renewable to grid
visible_xlim = (0, 30)
ax1.set_yticks(base_y)
ax1.set_yticklabels([])
ax1.tick_params(labelleft=False, left=False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(False)
ax1.invert_yaxis()
ax1.set_xlim(visible_xlim)
ax1.set_xbound(lower=-tick_gap, upper=30+tick_gap)  # Internal span
xticks_ax1 = ax1.get_xticks()
visible_xlim = (0, 30)
shared_xticks = np.arange(0, 35, 5.0) 
ax1.set_xticks(shared_xticks)  
ax1.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)

# Ax2: Hydrogen production
ax2.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax2.set_yticks([])
ax2.tick_params(labelleft=False, left=False)
ax2.spines['left'].set_visible(False)
ax2.set_xlim(visible_xlim)
ax2.grid(False)
ax2.set_xbound(lower=-tick_gap, upper=30.0)  # Internal span
xticks_ax2 = ax2.get_xticks()
ax2.set_xticks(xticks_ax2[1:]) 
ax2.invert_yaxis()
ax2.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
ax2.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig("./plot/surplus_three_panel.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()
#%% Plot box plot of GtP & BESS & Hydrogen tank utilization
base_y = np.arange(len(regions))*len(cases)
positions = []
gap = 1.0
tick_gap = 1.25
for y in base_y:
    positions.append(y - 1.5 * gap)  # Regime 1 - Case 1
    positions.append(y - 0.5 * gap)  # Regime 1 - Case 2
    positions.append(y + 0.5 * gap)  # Regime 2 - Case 1
    positions.append(y + 1.5 * gap)  # Regime 2 - Case 2

# Create figure
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(16, 5))
ax0 = fig.add_axes([0.0,    0.1, 0.4, 0.85])
ax1 = fig.add_axes([0.4,  0.1, 0.4/(20+tick_gap)*(20+tick_gap*2), 0.85])
ax2 = fig.add_axes([0.4+0.4/(20+tick_gap)*(20+tick_gap*2),  0.1, 0.4, 0.85])
# Shared visible x-axis limit
box_width = 0.45
line_width = 1.0
label_size = 20
tick_length = 5

# Boxplot 0: Grid import
bp0 = ax0.boxplot(GtP_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))

# Boxplot 1: BESS utilization
bp1 = ax1.boxplot(BU_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))

# Boxplot 2: Hydrogen utilization
bp2 = ax2.boxplot(HU_list, positions=positions, widths=box_width, vert=False,
                  patch_artist=True, showfliers=False, whis=[5, 95],
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black', linewidth=line_width),
                  capprops=dict(color='black', linewidth=line_width))

# Colors
colors = [
    # First regime: high PEMEC
    (240/255, 34/255, 34/255),    # Deep red (dominant first regime)
    (248/255, 150/255, 150/255),  # Light red (second case in first regime)

    # Second regime: low PEMEC
    (0/255, 153/255, 76/255),     # Deep green (dominant second regime)
    (153/255, 230/255, 179/255),  # Pale green (second case in second regime)
] * len(regions)

#colors = ['tab:grey', 'tab:blue', 'tab:green'] * len(regions)
for patch, color in zip(bp0['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

# Middle dotted line
fig.lines.append(plt.Line2D([0.4, 0.4], [0.1, 0.95],
                            transform=fig.transFigure,
                            linestyle='--', color='gray', linewidth=line_width))
fig.lines.append(plt.Line2D([0.4+0.4/(20+tick_gap)*(20+tick_gap*2),
                             0.4+0.4/(20+tick_gap)*(20+tick_gap*2)], [0.1, 0.95],
                            transform=fig.transFigure,
                            linestyle='--', color='gray', linewidth=line_width))

for y in base_y[1:]:
    ax0.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)
    ax1.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)
    ax2.axhline(y=y - len(cases)/2, color='gray', linestyle='--', linewidth=line_width)
    
#line_color = "#0066CC"
#for xpos, yval in zip(GtP_threshold_list, positions):
#    ax0.plot([xpos, xpos], [yval - 0.3, yval + 0.3], color=line_color, linewidth=line_width*3)


# Ax0: Grid power import
#visible_xlim = (0, 20)
ax0.set_yticks(base_y)
ax0.set_yticklabels([])
ax0.grid(False)
ax0.invert_yaxis()
#ax1.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax0.set_xlim((0, 4.5))
#ax0.set_xbound(lower=0, upper=20+tick_gap)  # Internal span
xticks_ax0 = ax0.get_xticks()
ax0.set_xticks(xticks_ax0[:-1]) 
ax0.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax0.xaxis.set_minor_locator(AutoMinorLocator(4))
ax0.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax0.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax0.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)    
ax0.spines['right'].set_visible(False)

# Ax1: BESS utilization
#visible_xlim = (0, 20)
ax1.set_yticks(base_y)
ax1.set_yticklabels([])
ax1.tick_params(labelleft=False, left=False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(False)
ax1.invert_yaxis()
ax1.set_xlim((-0.05, 1.2))
#ax1.set_xbound(lower=0, upper=0.5)  # Internal span
xticks_ax1 = ax1.get_xticks()
ax1.set_xticks(xticks_ax1[1:-1])
#visible_xlim = (0, 20)
#shared_xticks = np.arange(0, 21, 2.5) 
#ax1.set_xticks(shared_xticks)  
ax1.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
ax1.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax1.tick_params(axis='y', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)

# Ax2: Hydrogen utilization
ax2.grid(True, axis='x', linestyle=':', linewidth=0.7)
ax2.set_yticks([])
ax2.tick_params(labelleft=False, left=False)
ax2.spines['left'].set_visible(False)
ax2.set_xlim((-0.05, 2.0))
ax2.grid(False)
xticks_ax2 = ax2.get_xticks()
ax2.set_xticks(xticks_ax2[1:]) 
ax2.invert_yaxis()
ax2.tick_params(axis='x', which='both', direction='out', length=tick_length, width=line_width, labelsize=label_size)
ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
ax2.tick_params(axis='x', which='minor', length=tick_length/3*2, width=line_width)
ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
plt.savefig("./plot/storage_three_panel.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close

#%% Figure 3 a part: des & scale
def des_bar_plot(des_data, region):
    # Step 1. Remove NaN rows
    valid_mask = ~np.isnan(des_data[:,0])
    des_data = des_data[valid_mask]

    # Step 2. Define labels based on region
    if region == 'DU':
        labels = ["SE LC", "SE LE"]
    else:
        labels = ["SE LC", "SE LE", "PE LC", "PE LE"]

    # Step 3. Adjust label length to match the number of valid rows
    labels = labels[:len(des_data)]

    # Step 4. Extract data (4 components + total scale)
    data = des_data[:, :4]      # shape (n_designs, 4)
    totals = des_data[:, 4]     # shape (n_designs,)

    x = np.arange(len(labels))
    
    if len(labels) == 2:
        fig, ax1 = plt.subplots(figsize=(3.5,3.5))
    else:
        fig, ax1 = plt.subplots(figsize=(7,3.5))
        
    # ---------- Left axis: Stacked bar ----------
    bottom = np.zeros(len(labels))
    colors = ["#FAB2B2", "#6BD28A", "#FECE42", "#8EB4E3"]  # MeOH, PEMEC, BESS, CHT
    bar_width = 0.5

    # Loop over 4 components (not over designs!)
    for comp_idx in range(4):
        ax1.bar(
            x,
            data[:, comp_idx],
            bottom=bottom,
            width=bar_width,
            color=colors[comp_idx],
        )
        bottom += data[:, comp_idx]

    # ---------- Right axis: Total system load ----------
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        totals,
        '--o',
        color='black',
        linewidth=2,
        markersize=10,
        label='Total load'
    )
    ax2.set_ylim([10, 300])

    # ---------- Ticks & labels ----------
    ax1.set_xticks(x)
    ax1.set_xticklabels([])

    # ---------- Tick style ----------
    ax1.tick_params(axis='both', which='major',
                   direction='out', length=12, width=1.5, labelsize=20)
    
    
    ax2.tick_params(axis='both', which='major',
                   direction='out', length=12, width=1.5, labelsize=20)
    

    plt.tight_layout()
    plt.savefig(f"./plot/des_se_pe_bar_{region}.png", dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

for i in range(4):
    des_bar_plot(des_pack[i], regions[i])
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

all_comp = []       # (N, 4)
all_total = []      # (N,)
all_labels = []     # x축 라벨
x_positions = []    # 실제 x좌표
region_ranges = []  # (start_idx, end_idx) for each region

gap_size = 1.0
current_x = 0.0

for region, des_data in zip(regions, des_pack):
    # NaN row 제거
    mask = ~np.isnan(des_data[:, 0])
    d = des_data[mask]

    # region별 label
    if region == 'DU':
        labels = ["SE LC", "SE LE"]
    else:
        labels = ["SE LC", "SE LE", "PE LC", "PE LE"]
    labels = labels[:len(d)]

    start_idx = len(all_comp)  # 이 region이 시작되는 데이터 index

    for i in range(len(d)):
        all_comp.append(d[i, :4])
        all_total.append(d[i, 4])
        all_labels.append(f"{region}-{labels[i]}")
        x_positions.append(current_x)
        current_x += 1.0

    end_idx = len(all_comp)    # 이 region이 끝나는 데이터 index
    region_ranges.append((start_idx, end_idx))

    # region 사이 gap
    current_x += gap_size

all_comp = np.array(all_comp)     # (N,4)
all_total = np.array(all_total)   # (N,)
x_positions = np.array(x_positions)
N = len(all_total)

fig, ax1 = plt.subplots(figsize=(14, 3.1))

# ---------- Stacked bar ----------
bottom = np.zeros(N)
colors = ["#BA7AE6", "#6BD28A", "#FECE42", "#8EB4E3"]
bar_width = 0.6

for comp_i in range(4):
    ax1.bar(
        x_positions,
        all_comp[:, comp_i],
        bottom=bottom,
        width=bar_width,
        color=colors[comp_i]
    )
    bottom += all_comp[:, comp_i]

# ---------- 오른쪽 축: energy (region별로 끊어서 plot) ----------
ax2 = ax1.twinx()
for (start_idx, end_idx) in region_ranges:
    xs = x_positions[start_idx:end_idx]
    ys = all_total[start_idx:end_idx]
    ax2.plot(xs, ys, '--o', color='black', linewidth=2.5, markersize=10)

# ---------- region 경계선 (gap 가운데에 세로선) ----------
for (start_idx, end_idx), (next_start, _) in zip(region_ranges[:-1], region_ranges[1:]):
    x_last = x_positions[end_idx - 1]
    x_next = x_positions[next_start]
    x_mid = 0.5 * (x_last + x_next)
    ax1.axvline(x_mid, color='gray', linestyle='--', linewidth=1)

# ---------- x축 라벨 ----------
ax1.set_xticks(x_positions)
ax1.set_xticklabels([], ha='right')

# ---------- 스타일 ----------
ax1.tick_params(axis='both', direction='out', length=10, width=1.5, labelsize = 15)
ax2.tick_params(axis='y',   direction='out', length=10, width=1.5, labelsize = 15)


plt.tight_layout()
plt.savefig(f"./plot/des_se_pe_bar.png", dpi=600, bbox_inches='tight')
plt.show()
plt.close()


    