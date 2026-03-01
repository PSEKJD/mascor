import argparse
import numpy as np
import time
import warnings
from scipy.stats import qmc
warnings.filterwarnings("ignore", category=FutureWarning)
from pyomo.environ import * 
from mascor.models import generator
from mascor.utils.gan_data_loader import *
from torch.utils.data import DataLoader
from mascor.utils.env import env_single
from mascor.utils.planning_utils import *
from mascor.solvers import global_solver as solver
import os
import pickle
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument(
'--target-country', type=str, default="France", help="target country",
)

parser.add_argument(
    "--region", type=str, default="Dunkirk", help="target region"
)

parser.add_argument(
    '--data-type', type=str, default="wind", help="datatype",
)

parser.add_argument(
    "--device", type=str, default="cuda:0", help="device"
)

parser.add_argument(
    "--pre-train-epoch", type = int,default= 1000,help="Epoch",
)

parser.add_argument(
    "--gp-weight", type = float, default= 20.0, help="env config: grid penalty",
)

parser.add_argument(
    "--sample-size", type = int, default= 50000, help="oracle dataset size",
)

parser.add_argument(
    "--design-option", type = str, default= 'c_fax_fix', help="whether fixing c-tax or not",
)

parser.add_argument(
    "--save-freq", type = int, default= 100, help="save frequency of oracle dataset",
)
# %%
if __name__ == "__main__":
    args = parser.parse_args()    
    #Loading and instantiate dataloader
    dataset = Dataset(args.target_country, args.region, uni_seq = 24, max_seq = 24*24, data_type = 'wind-ele', flag='train')
    data_loader = DataLoader(dataset,batch_size=64, shuffle=True, drop_last=True) #1200 by 1 by 24 by 24 should be feded
    del dataset
    
    # Use GPU if available.
    device = torch.device(args.device if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")
    
    #Loading the pre-trained netG
    REPO_ROOT = Path(__file__).resolve().parents[2]
    CHEKPOINT_DIR = str(REPO_ROOT / "dataset")
    if "ele" in args.data_type:
        ch_dim = 2
    else:
        ch_dim = 1
    netG = generator(ch_dim = ch_dim, nz = 205).to(device)
    save_path = os.path.join(CHEKPOINT_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(country = args.target_country, 
                                                                                                          region = args.region, data_type = args.data_type,
                                                                                                          gp = args.gp_weight))
    
    # Pre-trained netG load
    checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_{epoch}'.format(epoch = 15000))      
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    print(f"Checkpoint loaded successfully from {checkpoint_path}")
    netG.load_state_dict(checkpoint['netG'])
    netG.eval()
    del save_path
    
    # Scenario generation
    save_path = os.path.join(CHEKPOINT_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country, 
                                                                                                              region = args.region,
                                                                                                              option = args.design_option,
                                                                                                              sample = args.sample_size))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
        
    # Design search space
    scale_min = 5000 # 5MW
    scale_max = 25000 # 25MW
    P_X = 0.65702 
    X_H2 = 0.19576
    SP_H2 = 55.7
    X_flow_range = np.array([scale_min/(P_X + X_H2*SP_H2), scale_max/(P_X + X_H2*SP_H2)])
    LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4]) # 5MW to 100MW
    ESS_cap_range = np.array([scale_min, scale_max*4]) # 5MW to 100MW
    PEM_ratio_range = np.array([0, 1])
    c_tax_range = np.array([0.10, 132.12])

    # c_tax in different country
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
    # Storing scenario generation and design spec only once
    if os.path.exists(os.path.join(save_path, 'noise_seed{}.npy'.format(torch_seed))):
        noise = np.load(os.path.join(save_path, 'noise_seed{}.npy'.format(torch_seed)))
        wind_scenario = np.load(os.path.join(save_path, 'wind_scenario_seed{}.npy'.format(torch_seed)))
        price_scenario = np.load(os.path.join(save_path, 'price_scenario_seed{}.npy'.format(torch_seed)))
        design_sample = np.load(os.path.join(save_path, 'design_spec_seed{}.npy'.format(torch_seed)))
        del netG, checkpoint
    else:
        noise, wind_scenario, price_scenario = scenario_generation(netG, args.sample_size, data_loader, args)
        np.save(os.path.join(save_path, 'noise_seed{}.npy'.format(torch_seed)), noise)
        np.save(os.path.join(save_path, 'wind_scenario_seed{}.npy'.format(torch_seed)), wind_scenario)
        np.save(os.path.join(save_path, 'price_scenario_seed{}.npy'.format(torch_seed)), price_scenario)
        
        #design sample
        sampler = qmc.LatinHypercube(d=5, seed=torch_seed)
        sample = sampler.random(n=args.sample_size)
        l_bounds = [X_flow_range[0], LH2_cap_range[0], ESS_cap_range[0], PEM_ratio_range[0], c_tax_range[0]] #X_flow, H2_cap, ESS_cap, PEM_ratio, c_tax
        u_bounds = [X_flow_range[1], LH2_cap_range[1], ESS_cap_range[1], PEM_ratio_range[1],c_tax_range[1]] #X_flow, H2_cap, ESS_cap, PEM_ratio, c_tax
        design_sample = qmc.scale(sample, l_bounds, u_bounds)
        
        if 'fix' in args.design_option:
            design_sample[:,-1] = c_tax_list[args.target_country]
        else:
            pass
        np.save(os.path.join(save_path, 'design_spec_seed{}.npy'.format(torch_seed)), design_sample)
        del netG, checkpoint
    
    #Trajectory dataset
    if os.path.exists(os.path.join(save_path, 'data_package.pkl')):
        with open(os.path.join(save_path, 'data_package.pkl'), "rb") as file: 
          data_package = pickle.load(file)
    else:
        data_package = {}
        data_package['episode-id'] = []
        data_package['state-stack'] = []
        data_package['action-stack'] = []
        data_package['design-spec'] = []
        data_package['cum-reward-stack'] = []
        data_package['reward-stack'] = []
        data_package['cum-co2-stack'] = []
        data_package['co2-stack'] = []
        data_package['co2-scale'] = []
        data_package['converge-idx'] = [] #1 convergence, 0 nonconvergence
        data_package['noise'] = [] 
        
    env_config = {}
    env_config['scale'] = 50000 #50MW
    env_config['op_period'] = wind_scenario.shape[1]
    env_config['max-SMP'] = data_loader.dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = data_loader.dataset.price_scale.data_min_[0]
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['flatten'] = True
    env_config['renew-split'] = False
    env_config['fw'] = 1
    del data_loader
#%%     
    start = time.time()
    for i in range(args.sample_size):
        if i in data_package['episode-id']:
            pass
        else:
            env_config['X-flow'] = design_sample[i,0]
            env_config['LH2-cap'] = design_sample[i,1]
            env_config['ESS-cap'] = design_sample[i,2]
            env_config['PEM-ratio'] = design_sample[i,3]
            env_config['c-tax'] = design_sample[i,4]
            env_config['SOC-init'] = env_config['ESS-cap']*0.1
            env_config['L-H2-init'] = 0 
            
            global_solver = solver(env_config)
            global_solver.solver_instance(renewable = wind_scenario[i]*env_config['scale'], SMP =price_scenario[i], option=True)
            global_result = global_solver.solve_planning()
            
            if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                conv_idx = 1
            else:
                global_solver.solver_instance(renewable = wind_scenario[i]*env_config['scale'], SMP =price_scenario[i], option=False)
                global_result = global_solver.solve_planning()
                if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                    conv_idx = 0
            state_list, action_list = offline_data_processing(global_solver)
            error, cum_reward, reward, cum_co2, co2, test_env = optimal_planning(env_config, wind_scenario[i]*env_config['scale'], price_scenario[i], state_list, action_list,
                                                                                                                   global_solver, env_class=env_single)
            
            #Store trajectory
            data_package['episode-id'].append(i)
            data_package['state-stack'].append(state_list[:,:4])
            data_package['action-stack'].append(action_list) #[[-1,1], [0,1], [0,1], [0,1]]
            data_package['design-spec'].append(state_list[:,4:])
            data_package['cum-reward-stack'].append(cum_reward) #un-normalize
            data_package['reward-stack'].append(reward) #un-normalize
            data_package['cum-co2-stack'].append(cum_co2) #un-normalize
            data_package['co2-stack'].append(co2) #un-normalize
            data_package['co2-scale'].append(np.array(test_env.co2_emit_scale()))
            data_package['noise'].append(noise[i])
            data_package['converge-idx'].append(conv_idx)
            
            if i%args.save_freq == 0 and (i!=0):
                with open(os.path.join(save_path, 'data_package.pkl'), "wb") as file: 
                    pickle.dump(data_package, file)
    