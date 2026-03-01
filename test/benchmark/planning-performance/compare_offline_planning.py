# %% Import libraries
import argparse
import pickle
import copy
from scipy.stats import qmc
from torch.utils.data import DataLoader
from pyomo.environ import *
from mascor.models import generator, discriminator 
from mascor.utils.gan_data_loader import *
from mascor import solvers
from mascor.utils.planning_utils import *
from mascor.utils.env import env_stack, env_rl_stack 
from mascor.optimization import uq_problem, uq_drl_problem
from pathlib import Path

parser = argparse.ArgumentParser()
# Common parsing part
parser.add_argument('--target-country', type=str, default="France", help="target country",)
parser.add_argument("--region", type=str, default="Dunkirk", help="target region")
parser.add_argument("--sample-size", type = int, default= 50000, help="oracle dataset size",)
parser.add_argument("--design-option", type = str, default= 'c_fax_fix', help="whether fixing c-tax or not",)
parser.add_argument("--op-period", type = int, default= 576, help="Planning period",)
parser.add_argument("--device", type = str, default= 'cuda:0', help="Device for solver-running",)

# GAN-dataset
parser.add_argument('--flag', type=str, default="test", help="datatype",)

# DRL & BC solver
parser.add_argument("--obs-length", type = int, default= 24, help="Observation length of profile data",)
parser.add_argument("--flatten", action="store_true", help="Flatten the dimension of renewable and grid profile",)

# PT-solver
parser.add_argument("--prob-fail", type = float, default= 0.01, help="failure probability",)
parser.add_argument("--candidate-num", type = int, default= 100, help="candidate size for action inference",)
parser.add_argument('--policy-type', type=str, default="GLOBAL", help="policy-types",)
parser.add_argument("--z-type", type = str, default= "mv", help="z_type for ctg & rtg prediction",)
parser.add_argument("--infer_action", type=str, default="mu",
                    help="action-inference based on mu value of normal dist", )

# Episode part
parser.add_argument("--scenario-size", type = int, default= 100, help="scenario size",)

def get_bounds_from_percentile(global_range, lower_percentile, upper_percentile):
    return (
        global_range[0] + lower_percentile * (global_range[1] - global_range[0]),
        global_range[0] + upper_percentile * (global_range[1] - global_range[0])
    )

def design_sampling(case: str, seed = 2025, sample_size = 1000):
    assert case in ['Base', 'Buffered', 'Responsive', 'Max-product'], "Invalid case name"
    # Design search space
    scale_min = 5000 # 5MW
    scale_max = 25000 # 25MW
    P_X = 0.65702 
    X_H2 = 0.19576
    SP_H2 = 55.7
    X_flow_range = np.array([scale_min/(P_X + X_H2*SP_H2), scale_max/(P_X + X_H2*SP_H2)])
    LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4])
    ESS_cap_range = np.array([scale_min, scale_max*4])
    PEM_ratio_range = np.array([0, 1])
    c_tax_range = np.array([0.10, 132.12])
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

def loading_policy(args, device, policy_type):
    policy_args = copy.deepcopy(args)
    policy_args.device = device
    if policy_type == 'BC':
        #policy_args.flatten = True
        policy = solvers.drl_policy(policy_args, mode = 'bc')
        
    elif policy_type == 'DRL':
        #policy_args.flatten = True
        policy_args.bc_support = False
        policy = solvers.drl_policy(policy_args, mode = 'drl')
        
    elif policy_type == 'DRL+BC':
        #policy_args.flatten = True
        policy_args.bc_support = True
        policy = solvers.drl_policy(policy_args, mode = 'drl')
    
    # DT & ST share same solver & policy
    elif policy_type == 'DT':
        policy_args.critic = None
        policy_args.des_token = False
        policy_args.z_token = False
        policy = solvers.pt_policy(policy_args) 
        
    elif policy_type == 'ST':
        policy_args.critic = True
        policy_args.des_token = False
        policy_args.z_token = False
        #policy_args.z_type = 'default'
        policy = solvers.pt_policy(policy_args) 
    
    elif policy_type == 'ST-des-token':
        policy_args.critic = True
        policy_args.des_token = True
        policy_args.z_token = False
        #policy_args.z_type = 'default'
        policy = solvers.pt_policy(policy_args) 
        
    elif policy_type == 'PT':
        policy_args.critic = True
        policy_args.des_token = True
        policy_args.z_token = True
        policy = solvers.pt_policy(policy_args)
        
    return policy
        
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
    
    # Sampling design in case by case
    design_case = ['Base', 'Buffered', 'Responsive', 'Max-product']
    design_point = np.array([design_sampling(seed=2025, sample_size=25, case=case) for case in design_case])

    # GAN-loading
    netG, netD = loading_gan(args)
    
    # Dataset-loading & noise inference
    BASE_DIR = Path(__file__).resolve().parent 
    log_dir = BASE_DIR / "offline_planning" / f"{args.flag}_dataset"
    log_dir.mkdir(parents=True, exist_ok=True)
    if os.path.exists(os.path.join(log_dir, 'episode_seed_{}_z_type_{}.pkl'.format(2025, args.z_type))):
        with open(os.path.join(log_dir, 'episode_seed_{}_z_type_{}.pkl'.format(2025, args.z_type)), "rb") as f:
            episode = pickle.load(f)
    else:
        g = torch.Generator()
        g.manual_seed(2025)
        data_loader = DataLoader(dataset, batch_size= args.scenario_size, shuffle=True, drop_last=True, generator=g)    
        wind_scaled, grid_scaled = next(iter(data_loader))
        wind_speed = data_loader.dataset.weather_scale.inverse_transform(wind_scaled.reshape(-1,1)).reshape(-1, 576)
        grid_price = data_loader.dataset.price_scale.inverse_transform(grid_scaled.reshape(-1,1)).reshape(-1, 576)
        
        # Noise inference
        noise = torch.randn(args.scenario_size, 205, device = args.device, dtype=torch.float32).requires_grad_()
        mse_erorr = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([noise], lr=1e-3)
        lam = 0.01
        z_infer_num = 10000
        real_img = wind_scaled.clone().detach().float().to(args.device).reshape(-1, 1, 24,24)
        for j in range(z_infer_num):
            optimizer.zero_grad()
            fake_img = netG(noise)
            #masking only historical
            mse_loss = mse_erorr(fake_img, real_img)
            loss = mse_loss - lam*torch.mean(netD(fake_img))
            loss.backward()
            optimizer.step()
            mse_loss_mean = torch.mean((fake_img-real_img)**2)
            if mse_loss_mean.item()<0.003:
                print("\n" + "*" * 50)
                print(f"MSE: {mse_loss.item():.6f}")
                print("*" * 50 + "\n")
                break
        if args.z_type == 'default':
            pass#episode = (wind_speed, grid_price, noise.detach().cpu())
        elif args.z_type =='mv':
            renew = wind_power_function(wind_speed)
            window_size = 24
            renew_df = pd.DataFrame(renew)
            rolling_mean_renew = renew_df.rolling(window=window_size, axis=1).mean().values[:,23:] #first 24 is NaN
            noise = torch.tensor(rolling_mean_renew[:,::24], dtype=torch.float32) #dowm-sample sample x 24
            episode = (noise, renew, grid_price)
        with open(os.path.join(log_dir, 'episode_seed_{}_z_type_{}.pkl'.format(2025, args.z_type)), "wb") as f:
            pickle.dump(episode, f)

    #%% solver setting
    if args.policy_type == 'GLOBAL':
        pass
    else:
        policy = loading_policy(args, device = args.device, policy_type = args.policy_type)  
        args.uq_sample_size = args.scenario_size
        if args.policy_type in ['DRL', 'BC', 'DRL+BC']:
            problem = uq_drl_problem(args, env_rl_stack, policy)
        else:
            problem = uq_problem(args, env_stack, policy, dataset)
    #%% solver running
    if not args.policy_type == "GLOBAL":
        ENV_PACK = []
        for i in range(len(design_case)):
            env_list = []
            for j in range(len(design_point[i])):
                if args.policy_type in ['BC', 'DRL', 'DRL+BC']:
                    env = problem.planning(design_point[i][j], netG, episode, dataset, loop_idx=j)
                else:
                    _, _, _, _, _, env = problem.planning(torch.tensor(design_point[i][j]), netG, dataset, dataset, episode, mode = "heavy")
                env_list.append(env)
            ENV_PACK.append(env_list)
        if args.policy_type in ['BC', 'DRL', 'DRL+BC']:
            with open(os.path.join(log_dir, '{}_solution'.format(args.policy_type)),'wb') as f:
                pickle.dump(ENV_PACK, f)
        else:
            with open(os.path.join(log_dir, '{}_solution_z_type_{}'.format(args.policy_type, args.z_type)),'wb') as f:
                pickle.dump(ENV_PACK, f)
    else:
        GLOBAL_result = np.zeros(shape = (4, 25, 100, 3)) #profit & co2-emission
        print(f"Length of full-scenari list: {len(episode[0])}")
        wind_speed_list, price_list, _ = episode
        for i in range(4):
            for j in range(25):
                for k in range(args.scenario_size):
                    des = design_point[i,j]
                    wind_speed = wind_speed_list[k]
                    price = price_list[k]
                    renewable = wind_power_function(wind_speed)*env_config['scale']
                    env_config['LH2-cap'] = des[0]
                    env_config['ESS-cap'] = des[1]
                    env_config['PEM-ratio'] = des[2]
                    env_config['X-flow'] = des[3]
                    env_config['SOC-init'] = env_config['ESS-cap']*0.1
                    env_config['L-H2-init'] = 0
                    print(env_config)
                    global_solver = solvers.global_solver(env_config)
                    global_solver.solver_instance(renewable = renewable, SMP = price, option=True)
                    global_result = global_solver.solve_planning()
                    if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                        conv_idx = 1
                    else:
                        global_solver.solver_instance(renewable = renewable, SMP = price, option=False)
                        global_result = global_solver.solve_planning()
                        if (global_solver.results.solver.status == SolverStatus.ok) and (global_solver.results.solver.termination_condition == TerminationCondition.optimal):
                            conv_idx = 0
                    GLOBAL_result[i,j,k,:] = [global_solver.profit, global_solver.CO2_emit, conv_idx]
                    np.save(os.path.join(log_dir, "global_solution.npy"), GLOBAL_result)
        
    
        
    
    
   
   
