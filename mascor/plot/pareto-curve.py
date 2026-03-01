# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:33:34 2025

@author: USER
"""
import os
# Set the environment variable before importing other libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pickle
import sys
import matplotlib
import numpy as np
import pickle
import os
import torch
import matplotlib.pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated

# Define country and region data
country_list = {}    
country_list['France'] = {}
country_list['France']['Dunkirk'] = {}
#country_list['Norway']['Porsgrunn'] = {}

# Color map for regions
region_colors = {
    'Dunkirk': 'r',    # Red for Dunkirk (France)
}

all_pareto_obj = []
all_pareto_des = []
all_colors = []

# Iterate over countries and regions
for target_country in country_list:
    for region in country_list[target_country]:
        log_dir =  os.path.join('D:/Na-research-dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}/optimization'.format(country = target_country, 
                                                                                                              region = region,
                                                                                                              option = 'c_fax_fix',
                                                                                                              sample = 50000))
        save_path = os.path.join(log_dir, 'iter_{iteration}_history_pfss_{pfss}_sample_size_{size}.pkl'.format(iteration = 100,
                                                                                                           pfss = 0.50,
                                                                                                           size = 1000))
        
        with open(save_path, "rb") as file:
            history_dict = pickle.load(file)
        
        # Collect data from history
        for i in range(len(history_dict)):
            des = np.array(history_dict["step-{}".format(i)]['des'])

            obj = np.array((history_dict["step-{}".format(i)]['mu-lcox[$/kg]'], history_dict["step-{}".format(i)]['var-lcox'])).transpose()
            con = (np.array(history_dict["step-{}".format(i)]['pfss'])-0.01).reshape(-1,1)
            
            # Stack data from different steps
            if i == 0:
                des_set, obj_set, con_set = des, obj, con
            else:
                des_set, obj_set, con_set = np.vstack((des_set, des)), np.vstack((obj_set, obj)), np.vstack((con_set, con))
        
        # Convert to tensors
        des_set, obj_set, con_set = torch.tensor(des_set), torch.tensor(obj_set), torch.tensor(con_set)
        
        # Find feasible solutions
        is_feas = (con_set <= 0).all(dim=-1)
        feas_obj, feas_des = obj_set[is_feas], des_set[is_feas]
        
        # If there are feasible solutions, extract Pareto points
        if feas_obj.shape[0] > 0:
            pareto_mask = is_non_dominated(feas_obj)
            pareto_obj, pareto_des = feas_obj[pareto_mask], feas_des[pareto_mask]
            print(f'{region} - pareto points: ', pareto_obj.shape[0])
        else:
            assert False, f"No feasible solution from RBDO in {region}"

        # Store Pareto points and regions for plotting
        all_pareto_obj.append(pareto_obj)
        all_pareto_des.append(pareto_des)
        all_colors.extend([region_colors[region]] * pareto_obj.shape[0])  # Assign region color
    
        del des, obj, con, des_set, obj_set, con_set, is_feas

# Convert lists to arrays
all_pareto_obj = np.vstack(all_pareto_obj)
all_pareto_des = np.vstack(all_pareto_des)
all_colors = np.array(all_colors)

# Plot the Pareto points
plt.figure(figsize=(10, 6))
for i, region in enumerate(region_colors):
    region_mask = all_colors == region_colors[region]
    plt.scatter(all_pareto_obj[region_mask, 0], all_pareto_obj[region_mask, 1], 
                label=region, color=region_colors[region], alpha=0.7)

plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Pareto Front of All Regions')
plt.legend()
plt.grid(True)
plt.show()