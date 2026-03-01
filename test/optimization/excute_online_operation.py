import argparse
import time
import pickle
import copy
import random
import numpy as np
import torch
from pyomo.environ import * 
from torch.utils.data import DataLoader
from botorch.utils.multi_objective.pareto import is_non_dominated
from mascor.models import generator, discriminator
from mascor.utils.gan_data_loader import Dataset
from mascor.solvers import global_solver, pt_solver, pt_policy
from mascor.utils.planning_utils import *
from mascor.utils.env import env_single 
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
# GAN-dataset
parser.add_argument('--flag', type=str, default="test", help="datatype",)
# PT-solver
parser.add_argument("--candidate-num", type=int, default= 20, help="candidate size for action inference", )
parser.add_argument("--d-lambda", type=float, default= 0, help="lambda for discriminator", )
parser.add_argument("--noise-infer", action='store_true', help="re-sample design-point & episodic scenario",)
parser.add_argument("--infer-step", type=int, default=24, help="scenario size", )
parser.add_argument("--infer-action", type=str, default="mu",help="action-inference based on mu value of normal dist", )
parser.add_argument("--z-type", type = str, default= "mv", help="z_type for ctg & rtg prediction",)
# Episode part
parser.add_argument("--scenario-size", type = int, default= 50, help="scenario size",)
parser.add_argument("--resample-scenario", action='store_true', help="re-sample design-point & episodic scenario",)
parser.add_argument("--seed", type = int, default= 2026, help = "seed for episode sampling regeneration")

def loading_gan(args):
    save_epoch = {}
    save_epoch['France/Dunkirk'] = 15000
    save_epoch['France/Alpes-de-Haute-Provence'] = 15000
    save_epoch['Denmark/Skive'] = 15000
    save_epoch['Denmark/Fredericia'] = 15000
    save_epoch['Germany/Wunsiedel'] = 15000
    save_epoch['Germany/Weener'] = 150000
    save_epoch['Norway/Porsgrunn'] = 15000
    REPO_ROOT = Path(__file__).resolve().parents[2]
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

def loading_solver(args, device, dataset):
    solver_args = copy.deepcopy(args)
    solver_args.device = device
    solver_args.critic = True
    solver_args.des_token = True
    solver_args.z_token = True
    solver_args.z_type = 'mv'
    solver_args.prob_fail = 0.5
    netG, netD = loading_gan(solver_args)
    policy = pt_policy(solver_args)
    solver = pt_solver(solver_args, env_single, policy, dataset, netG, netD)
    return solver, solver_args

def wind_power_function(wind_speed: torch.Tensor) -> torch.Tensor:
    factor = (80.0 / 50.0) ** (1.0 / 7.0)
    w = wind_speed * factor
    cutin, rated, cutoff = 1.5, 12.0, 25.0
    denom = (rated ** 3) - (cutin ** 3)
    p = (w ** 3 - cutin ** 3) / denom
    p = p.clamp_(0.0, 1.0)
    p = torch.where(w > cutoff, torch.zeros((), dtype=p.dtype, device=p.device), p)
    return p

def wind_power_function_np(Wind_speed):
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

def online_operation_saving(des, episode_list, des_type, args):
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}/online_operation_validation"
    if os.path.isdir(log_dir)==False:
        os.makedirs(log_dir)
    save_path = os.path.join(log_dir, "operation_results.pkl")
    print(f"Starting online operation of {des_type} design with {len(episode_list[0])} validation-scenario")
    wind_speed, grid_price, wind_scaled = episode_list
    for idx in range(len(wind_speed)):
        episode = (wind_speed[idx], grid_price[idx], wind_scaled[idx])
        start = time.time()
        target_des = torch.as_tensor(des)
        print(f"{args.noise_infer}")
        env = solver.planning(target_des, episode, noise_infer=args.noise_infer)    
        end = time.time()
        profit = np.sum(env.cost_list)
        co2 = np.sum(env.CO2_emit)
        print(f"step {idx} at des {np.round(target_des.cpu().detach().numpy(),2)}:")
        print(f"Profit (k$/month) = {profit/1000:.2f} CO2 (ton/month) = {co2:.2f}, compute-time = {end-start:.2f}")
        #if idx%10 == 0 or idx == args.scenario_size-1:
        env_path = save_path.replace(".pkl", f"_PT_solver_{des_type}_idx_{idx}_noise_infer_{args.noise_infer}.pkl")
        with open(env_path, "wb") as f:
            pickle.dump(env, f)
        del env
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    train_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type='wind-ele',
                            flag='train')
    test_dataset = Dataset(args.target_country, args.region, uni_seq=24, max_seq=24 * 24, data_type='wind-ele',
                           flag="test")
    env_config = {}
    env_config['scale'] = 50000 #50MW
    env_config['op-period'] = args.op_period
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['flatten'] = True
    env_config['fw'] = 1
    c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
    env_config['c-tax'] = c_tax_list[args.target_country]
    env_config['max-SMP'] = train_dataset.price_scale.data_max_[0]
    env_config['min-SMP'] = train_dataset.price_scale.data_min_[0]
    env_config['max-c-tax'] = 132.12
    env_config['min-c-tax'] = 0.10
    env_config['obs-length'] = 24
    env_config['n-worker'] = args.scenario_size
    args.env_config = env_config

    # GAN & Solver loading
    if args.policy_type == 'GLOBAL':
        pass
    else:
        solver, args = loading_solver(args, device = args.device, dataset = train_dataset)

    # Validation-dataset loading
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}/online_operation_validation"
    log_dir.mkdir(parents=True, exist_ok=True)
    episode_path = os.path.join(log_dir, f'validation_{args.scenario_size}_scenario_seed_{args.seed}.pkl')
    if not args.resample_scenario:    
        with open(episode_path, "rb") as f:
            episode_list = pickle.load(f)
    else:
        print(f"Real historial dataset is re-collected at {args.target_country}/{args.region}")
        print(f"Overall length of historical data are {test_dataset.__len__()}")
        num_iter = 1
        loader = DataLoader(test_dataset, batch_size=args.scenario_size, shuffle=True)
        weather_scenario = []
        it = iter(loader)
        for i in range(num_iter):
            weather_data, _ = next(it)
            weather_scenario.append(weather_data)
        weather_scenario = torch.cat(weather_scenario, dim=0).to(device = args.device)
        wind_speed_scaled = torch.clamp(weather_scenario, min=0)
        weather_scenario = weather_scenario * (train_dataset.weather_scale.data_max_[0] 
                                               - train_dataset.weather_scale.data_min_[0]) + train_dataset.weather_scale.data_min_[0]
        backup_si = np.random.randint(0, train_dataset.__len__(), size=args.scenario_size)
        price_np = np.array([train_dataset.price_scaled[idx:idx + train_dataset.max_seq, 0] for idx in backup_si])
        price_scenario = train_dataset.price_scale.inverse_transform(price_np)
        price_scenario = torch.tensor(price_scenario, dtype=torch.float32, device=args.device)    
        episode_list = (weather_scenario.detach().cpu().clone().numpy(), 
                   price_scenario.detach().cpu().clone().numpy(), 
                   wind_speed_scaled.detach().cpu().clone())
        with open(episode_path, "wb") as f:
            pickle.dump(episode_list, f)
    # loading-SE1 & PRN design idx
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}"
    save_path = os.path.join(log_dir,'iter_100_history_pfss_0.5_sample_size_1000.pkl')
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
    print("*"*100)
    print(f"Idx-{idx[0]} & Idx-{idx[-1]} selected as SE1 & PRN for online-operation")
    print(f"SE1 des: {des[idx[0]]}")
    print(f"PRN des: {des[idx[-1]]}")
    se_des = des[idx[0]]
    pr_des = des[idx[-1]]
    del des_set, obj_set, con_set, feas_des, feas_obj, pareto_m, pareto_des, pareto_obj
    #%% Online-operation
    if args.policy_type != "GLOBAL":
        online_operation_saving(se_des, episode_list, "SE1", args)
        online_operation_saving(pr_des, episode_list, "PRN", args)
    else:
        BASE_DIR = Path(__file__).resolve().parent 
        log_dir = BASE_DIR / "result" / f"{args.target_country}/{args.region}/online_operation_validation"
        save_path = os.path.join(log_dir, "global_solution.npy")
        GLOBAL_result = np.zeros(shape = (2, args.scenario_size,3))
        print(f"Starting online operation of SE1 design with {len(episode_list[0])} validation-scenario")
        wind_speed, grid_price, wind_scaled = episode_list
        for idx in range(len(wind_speed)):
            des = np.copy(se_des)
            renewable = wind_power_function_np(wind_speed[idx])*env_config['scale']
            env_config['LH2-cap'] = des[0]
            env_config['ESS-cap'] = des[1]
            env_config['PEM-ratio'] = des[2]
            env_config['X-flow'] = des[3]
            env_config['SOC-init'] = env_config['ESS-cap']*0.1
            env_config['L-H2-init'] = 0
            solver = global_solver(env_config)
            solver.solver_instance(renewable = renewable, SMP = grid_price[idx], option=True)
            global_result = solver.solve_planning()
            if (solver.results.solver.status == SolverStatus.ok) and (solver.results.solver.termination_condition == TerminationCondition.optimal):
                conv_idx = 1
            else:
                solver.solver_instance(renewable = renewable, SMP = grid_price[idx], option=False)
                global_result = solver.solve_planning()
                if (solver.results.solver.status == SolverStatus.ok) and (solver.results.solver.termination_condition == TerminationCondition.optimal):
                    conv_idx = 0
            GLOBAL_result[0,idx,:] = [solver.profit, solver.CO2_emit, conv_idx]
            np.save(save_path, GLOBAL_result)
        print(f"Starting online operation of PRN design with {len(episode_list[0])} validation-scenario")
        for idx in range(len(wind_speed)):
            des = np.copy(pr_des)
            renewable = wind_power_function_np(wind_speed[idx])*env_config['scale']
            env_config['LH2-cap'] = des[0]
            env_config['ESS-cap'] = des[1]
            env_config['PEM-ratio'] = des[2]
            env_config['X-flow'] = des[3]
            env_config['SOC-init'] = env_config['ESS-cap']*0.1
            env_config['L-H2-init'] = 0
            solver = global_solver(env_config)
            solver.solver_instance(renewable = renewable, SMP = grid_price[idx], option=True)
            global_result = solver.solve_planning()
            if (solver.results.solver.status == SolverStatus.ok) and (solver.results.solver.termination_condition == TerminationCondition.optimal):
                conv_idx = 1
            else:
                solver.solver_instance(renewable = renewable, SMP = grid_price[idx], option=False)
                global_result = solver.solve_planning()
                if (solver.results.solver.status == SolverStatus.ok) and (solver.results.solver.termination_condition == TerminationCondition.optimal):
                    conv_idx = 0
            GLOBAL_result[1,idx,:] = [solver.profit, solver.CO2_emit, conv_idx]
            np.save(save_path, GLOBAL_result)