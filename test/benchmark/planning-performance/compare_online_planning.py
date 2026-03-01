'''
Usage of:
    - Online planning of different solvers
'''
# %% Import libraries
import argparse
from scipy.stats import qmc
import time
import pickle
import copy
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pyomo.environ import *
from mascor.models import generator, discriminator 
from mascor.utils.gan_data_loader import *
from mascor import solvers
from mascor.utils.planning_utils import *
from mascor.utils.env import env_single, env_rl
from pathlib import Path

# Seed fixing
seed = 2026
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
parser = argparse.ArgumentParser()
# Common parsing part
parser.add_argument('--target-country', type=str, default="France", help="target country",)
parser.add_argument("--region", type=str, default="Dunkirk", help="target region")
parser.add_argument("--sample-size", type = int, default= 50000, help="oracle dataset size",)
parser.add_argument("--design-option", type = str, default= 'c_fax_fix', help="whether fixing c-tax or not",)
parser.add_argument("--op-period", type = int, default= 576, help="Planning period",)
parser.add_argument("--device", type = str, default= 'cuda:0', help="Device for solver-running",)
parser.add_argument('--policy-type', type=str, default="GLOBAL", help="policy-types",)
parser.add_argument("--prob-fail", type = float, default= 0.01, help="failure probability",)
# GAN-dataset
parser.add_argument('--flag', type=str, default="test", help="datatype",)

# DRL & BC solver
parser.add_argument("--obs-length", type = int, default= 24, help="Observation length of profile data",)
parser.add_argument("--flatten", action="store_true", help="Flatten the dimension of renewable and grid profile",)

# PT-solver
parser.add_argument("--candidate-num", type=int, default= 100, help="candidate size for action inference", )
parser.add_argument("--d-lambda", type=float, default= 0, help="lambda for discriminator", )
parser.add_argument("--infer-step", type=int, default=24, help="scenario size", )
parser.add_argument("--infer-action", type=str, default="mu",help="action-inference based on mu value of normal dist", )
parser.add_argument("--z-type", type = str, default= "mv", help="z_type for ctg & rtg prediction",)

# Episode part
parser.add_argument("--scenario-size", type = int, default= 1000, help="scenario size",)
parser.add_argument("--resample-scenario", action='store_true', help="re-sample design-point & episodic scenario",)
parser.add_argument("--seed", type = int, default= 2026, help = "seed for episode sampling regeneration")

def get_bounds_from_percentile(global_range, lower_percentile, upper_percentile):
    return (
        global_range[0] + lower_percentile * (global_range[1] - global_range[0]),
        global_range[0] + upper_percentile * (global_range[1] - global_range[0])
    )

def design_sampling(case: str, seed = 2025, sample_size = 250):
    assert case in ['Base', 'Buffered', 'Responsive', 'Max-product'], "Invalid case name"
    # Design search space
    scale_min = 5000 # 1MW-->5MW
    scale_max = 25000 # 25MW
    P_X = 0.65702 
    X_H2 = 0.19576
    SP_H2 = 55.7
    X_flow_range = np.array([scale_min/(P_X + X_H2*SP_H2), scale_max/(P_X + X_H2*SP_H2)])
    LH2_cap_range = np.array([X_flow_range[0]*X_H2, X_flow_range[1]*X_H2])
    ESS_cap_range = np.array([scale_min, scale_max])
    PEM_ratio_range = np.array([0, 1])
    c_tax_range = np.array([0.10, 132.12])
    # c_tax in different country
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}

    #---Custom bounds by case---
    if case == 'Base':
        LH2_bounds = get_bounds_from_percentile(LH2_cap_range, 0.4, 0.6)
        ESS_bounds = get_bounds_from_percentile(ESS_cap_range, 0.4, 0.6)
        PEM_bounds = get_bounds_from_percentile(PEM_ratio_range, 0.4, 0.6)
        X_flow_bounds = get_bounds_from_percentile(X_flow_range, 0.4, 0.6)

    elif case == 'Buffered':
        LH2_bounds = get_bounds_from_percentile(LH2_cap_range, 0.4, 0.8)
        ESS_bounds = get_bounds_from_percentile(ESS_cap_range, 0.4, 0.8)
        PEM_bounds = get_bounds_from_percentile(PEM_ratio_range, 0.0, 0.2)
        X_flow_bounds = get_bounds_from_percentile(X_flow_range, 0.0, 0.2)

    elif case == 'Responsive':
        LH2_bounds = get_bounds_from_percentile(LH2_cap_range, 0.0, 0.2)
        ESS_bounds = get_bounds_from_percentile(ESS_cap_range, 0.0, 0.2)
        PEM_bounds = get_bounds_from_percentile(PEM_ratio_range, 0.4, 0.6)
        X_flow_bounds = get_bounds_from_percentile(X_flow_range, 0.4, 0.6)
    
    elif case == 'Max-product':
        LH2_bounds = get_bounds_from_percentile(LH2_cap_range, 0.4, 0.6)
        ESS_bounds = get_bounds_from_percentile(ESS_cap_range, 0.4, 0.6)
        PEM_bounds = get_bounds_from_percentile(PEM_ratio_range, 0.6, 1.0)
        X_flow_bounds = get_bounds_from_percentile(X_flow_range, 0.6, 1.0)
        
    sampler = qmc.LatinHypercube(d=4, seed=seed)
    sample = sampler.random(n=sample_size)
    design_sample = qmc.scale(sample, [LH2_bounds[0], ESS_bounds[0], PEM_bounds[0], X_flow_bounds[0]],
                                      [LH2_bounds[1], ESS_bounds[1], PEM_bounds[1], X_flow_bounds[1]])
    return design_sample

def loading_gan(args):
    save_epoch = {}
    save_epoch['France/Dunkirk'] = 15000
    save_epoch['France/Alpes-de-Haute-Provence'] = 15000
    save_epoch['Denmark/Skive'] = 15000
    save_epoch['Denmark/Fredericia'] = 15000
    save_epoch['Germany/Wunsiedel'] = 15000
    save_epoch['Germany/Weener'] = 150000
    save_epoch['Norway/Porsgrunn'] = 15000
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATASET_DIR = str(REPO_ROOT / "dataset")
    save_path = os.path.join(DATASET_DIR, '{country}/{region}/checkpoint_gan/{data_type}_{gp}'.format(country = args.target_country, 
                                                                                                                region = args.region, data_type = 'wind',
                                                                                                                gp = 20.0))
    checkpoint_path = os.path.join(save_path, 'model_mmd_True_epoch_{epoch}'.format(epoch = save_epoch[args.target_country+'/'+ args.region]))      
    state_dict = torch.load(checkpoint_path, map_location=args.device)   
    netG = generator(ch_dim = 1, nz = 205).to(args.device)
    netD = discriminator(ch_dim = 1).to(args.device)
    netG.load_state_dict(state_dict['netG'])
    netD.load_state_dict(state_dict['discriminator'])
    netG.eval()
    netD.eval()
    del save_path, checkpoint_path, state_dict
    return netG, netD

def loading_solver(args, device, solver_type, dataset):
    solver_args = copy.deepcopy(args)
    solver_args.device = device
    if solver_type == 'BC':
        solver_args.flatten = False
        solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.drl_solver(solver_args, mode = 'bc', env_class= env_rl)
        
    elif solver_type == 'DRL':
        solver_args.flatten = False
        solver_args.bc_support = False
        solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.drl_solver(solver_args, mode = 'drl', env_class= env_rl)
        
    elif solver_type == 'DRL+BC':
        solver_args.flatten = False
        solver_args.bc_support = True
        solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.drl_solver(solver_args, mode = 'drl', env_class= env_rl)
    
    # DT & ST share same solver & policy
    elif solver_type == 'DT':
        solver_args.critic = None
        solver_args.des_token = False
        solver_args.z_token = False
        netG, netD = None, None
        solver_args.z_type = 'mv'
        solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.st_solver(solver_args, netG, netD, env_class= env_single) 
        
    elif solver_type == 'ST':
        solver_args.critic = True
        solver_args.des_token = False
        solver_args.z_token = False
        netG, netD = None, None
        solver_args.z_type = 'mv'
        if solver_args.infer_action == "mu":
            solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.st_solver(solver_args, netG, netD, env_class= env_single) 
    
    elif solver_type == 'ST-des-token':
        solver_args.critic = True
        solver_args.des_token = True
        solver_args.z_token = False
        netG, netD = None, None
        solver_args.z_type = 'mv'
        if solver_args.infer_action == "mu":
            solver_args.candidate_num = 1 #forcing buffer-cap = 1
        solver = solvers.st_solver(solver_args, netG, netD) 
        
    elif solver_type == 'PT':
        solver_args.critic = True
        solver_args.des_token = True
        solver_args.z_token = True
        solver_args.z_type = 'mv'
        solver_args.prob_fail = 0.5
        netG, netD = loading_gan(solver_args)
        policy = solvers.pt_policy(solver_args)
        solver = solvers.pt_solver(solver_args, env_single, policy, dataset, netG, netD)
    return solver, solver_args

def wind_power_function(Wind_speed):
    # Turbine model: G-3120
    cutin_speed = 1.5  # [m/s]
    rated_speed = 12  # [m/s]
    cutoff_speed = 25  # [m/s]
    # Wind_speed data is collectd from 50m
    Wind_speed = Wind_speed * (80 / 50) ** (1 / 7)
    idx_zero = Wind_speed <= cutin_speed
    idx_rated = (cutin_speed < Wind_speed) & (Wind_speed <= rated_speed)
    idx_cutoff = (rated_speed < Wind_speed) & (Wind_speed <= cutoff_speed)
    idx_zero_cutoff = (Wind_speed > cutoff_speed)
    Wind_speed[idx_zero] = 0
    Wind_speed[idx_rated] = (Wind_speed[idx_rated] ** 3 - cutin_speed ** 3) / (rated_speed ** 3 - cutin_speed ** 3)
    Wind_speed[idx_cutoff] = 1
    Wind_speed[idx_zero_cutoff] = 0
    return Wind_speed  # Capacity fator =[0,1]
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    env_config = {}
    env_config['scale'] = 50000 #50MW
    env_config['op-period'] = args.op_period
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['flatten'] = args.flatten
    env_config['fw'] = 1
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
    env_config['c-tax'] = c_tax_list[args.target_country]
    dataset = Dataset(args.target_country, args.region, uni_seq = 24, max_seq = args.op_period, data_type = 'wind-ele', flag=args.flag)
    env_config['max-SMP'] = dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = dataset.price_scale.data_min_[0]
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['obs-length'] = 24
    env_config['n-worker'] = args.scenario_size
    args.env_config = env_config

    # Dataset-loading 
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "online_planning" / f"{args.flag}_dataset"
    log_dir.mkdir(parents=True, exist_ok=True)
    des_path = log_dir / f'design_point_{args.seed}.npy'
    episode_path = log_dir / f'episode_seed_{args.seed}.pkl'
    if not args.resample_scenario:    
        design_point = np.load(des_path)
        with open(episode_path, "rb") as f:
            episode_list = pickle.load(f)
    else:    
        design_case = ['Base', 'Buffered', 'Responsive', 'Max-product']
        design_point = np.array([design_sampling(seed=args.seed, sample_size=250, case=case) for case in design_case]).reshape(-1,4)
        np.save(des_path, design_point)
        
        g = torch.Generator()
        g.manual_seed(args.seed)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, generator=g)
        episode_list = []
        for i, (wind_scaled, grid_scaled) in enumerate(data_loader):
            start = time.time()
            wind_speed = data_loader.dataset.weather_scale.inverse_transform(wind_scaled.reshape(-1,1)).reshape(-1)
            grid_price = data_loader.dataset.price_scale.inverse_transform(grid_scaled.reshape(-1,1)).reshape(-1)
            episode = (wind_speed, grid_price, torch.as_tensor(wind_scaled[0]))
            episode_list.append(episode)
            if i == args.scenario_size:
                break
        with open(episode_path, "wb") as f:
            pickle.dump(episode_list, f)
    #%% Solver-setting
    if args.policy_type == 'GLOBAL':
        pass
    else:
        solver, args = loading_solver(args, device = args.device, solver_type = args.policy_type, dataset = dataset)
    # %% run solvers
    if args.policy_type != "GLOBAL":
        env_list = []
        print(f"Length of full-scenari list: {len(episode_list)}")
        for idx, episode in enumerate(episode_list):
            if idx == args.scenario_size:
                break
            start = time.time()
            des = torch.as_tensor(design_point[idx])
            env = solver.planning(des, episode)    
            env_list.append(env)
            end = time.time()
            profit = np.sum(env.cost_list)
            co2 = np.sum(env.CO2_emit)
            print(f"step {idx} at des {np.round(des.cpu().detach().numpy(),2)}:")
            print(f"Profit (k$/month) = {profit/1000:.2f} CO2 (ton/month) = {co2:.2f}, compute-time = {end-start:.2f}")
            if idx%10 == 0 or idx == args.scenario_size-1:
                if args.policy_type == "PT":
                    with open(os.path.join(log_dir, f'PT_candidate_{args.candidate_num}_lambda_{args.d_lambda}_solution_seed_{args.seed}'),'wb') as f:
                        pickle.dump(env_list, f)
                else:
                    with open(os.path.join(log_dir, f'{args.policy_type}_{args.candidate_num}_infer_action_{args.infer_action}_solution_seed_{args.seed}'),'wb') as f:
                        pickle.dump(env_list, f)
    else:
        GLOBAL_result = np.zeros(shape = (1000,3))
        print(f"Length of full-scenari list: {len(episode_list)}")
        for idx, episode in enumerate(episode_list):
            if idx == args.scenario_size:
                break
            des = design_point[idx]
            wind_speed, price, _ = episode
            renewable = wind_power_function(wind_speed)*env_config['scale']
            env_config['LH2-cap'] = des[0]
            env_config['ESS-cap'] = des[1]
            env_config['PEM-ratio'] = des[2]
            env_config['X-flow'] = des[3]
            env_config['SOC-init'] = env_config['ESS-cap']*0.1
            env_config['L-H2-init'] = 0
            global_solver = GLOBAL_solver.solver(env_config)
            global_solver.solver_instance(renewable = renewable, SMP = price, option=True)
            global_result = global_solver.solve_planning()
            if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                conv_idx = 1
            else:
                global_solver.solver_instance(renewable = renewable, SMP = price, option=False)
                global_result = global_solver.solve_planning()
                if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                    conv_idx = 0
            GLOBAL_result[idx,:] = [global_solver.profit, global_solver.CO2_emit, conv_idx]
            np.save(log_dir / f"global_solution_seed_{args.seed}.npy", GLOBAL_result)