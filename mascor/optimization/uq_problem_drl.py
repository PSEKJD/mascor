import torch
import numpy as np
from botorch.utils.transforms import normalize, unnormalize
import time

class UQ_Problem():
    def __init__(self, args, env_class, policy):        
        self.args = args
        # Problem bounds & parameter setting
        self.scale_min = 5000 # 5MW
        self.scale_max = 25000 # 25MW
        self.P_X = 0.65702 
        self.X_H2 = 0.19576
        self.SP_H2 = 55.7
        self.X_flow_range = np.array([self.scale_min/(self.P_X + self.X_H2*self.SP_H2), self.scale_max/(self.P_X + self.X_H2*self.SP_H2)])
        self.LH2_cap_range = np.array([self.scale_min/self.SP_H2, self.scale_max/self.SP_H2*4])
        self.ESS_cap_range = np.array([self.scale_min, self.scale_max*4])
        self.PEM_ratio_range = np.array([0, 1])
        self.c_tax_range = np.array([0.10, 132.12])
        # c_tax in different country
        self.c_tax_list = {'France':47.96, 'Denmark':28.10, 'Germany': 48.39, 'Norway':107.78}
        
        self.lb = torch.tensor([self.c_tax_range[0], self.LH2_cap_range[0], self.ESS_cap_range[0], 0, self.X_flow_range[0]], dtype = torch.float64) 
        self.ub = torch.tensor([self.c_tax_range[0], self.LH2_cap_range[1], self.ESS_cap_range[1], 1, self.X_flow_range[1]], dtype = torch.float64) 
        
        if 'c_fax_fix' in self.args.design_option:
            self.lb = self.lb[1:] #c_tax is excluded
            self.ub = self.ub[1:] #c_tax is excluded
            
        self.bounds = torch.stack((self.lb, self.ub))
        self.normalize = normalize
        self.num_objectives = 2
        self.num_con = 1
        self.limit_state = 0
        self.prob_failure = args.prob_fail 
        self.ref_point = torch.tensor(np.zeros(shape = self.num_objectives), dtype=torch.float)
        
        #solver, buffer, env register
        self.policy = policy
        self.env_class = env_class
        
    @torch.no_grad()    
    def planning(self, x, netG, episode, dataset, loop_idx = None):
        self.design_config_setting(x)
        if episode is not None:
            _, renewable, grid = episode
        else:
            _, renewable, grid = self.scenario_sampling(netG, dataset) #renew [0,1], grid [un-norm]
        renewable = self.args.env_config['scale']*(renewable)
            
        # Padding renewable & grid profile to matching state obs
        renewable = np.concatenate((renewable[:,:self.args.env_config['obs-length']-1], renewable), axis = 1)
        grid = np.concatenate((grid[:,:self.args.env_config['obs-length']-1], grid), axis = 1)
        #env-reset
        env = self.env_class(self.args.env_config)
        states, _ = env.reset(renewable, grid)
        del renewable, grid
        done = False
        is_terminal_record = []
            
        start = time.time()
        while not done:
            action = self.policy.compute_actions(states)
            states, reward, co2, done, _ , _ = env.step(action)
   
        if env.step_count == env.renewable.shape[1]-(self.args.env_config['obs-length']-1):
            end = time.time()
            mu_LCOX, var_LCOX = env.LCOX_calculation()
            ctg = np.sum(env.CO2_emit_scaled, axis = 1)
            lb, ub = env.co2_emit_scale()
            limit = (self.limit_state-env.op_period*lb)/(ub-lb)
            pfss = len(np.where(ctg>limit)[0])/len(ctg)
            mu_ctg = np.mean(ctg*(ub-lb) + env.op_period*lb)
            var_ctg = np.var(ctg*(ub-lb) + env.op_period*lb)
            des = normalize(torch.tensor(x, dtype=torch.float64), self.bounds)
            print(f"step {loop_idx} at des {np.round(des.cpu().detach().numpy(),2)}:")
            print(f"E[LCOX] = {mu_LCOX:.2f}, Var[LCOX] = {var_LCOX:.2f}, E[CTG] = {mu_ctg:.2f}, Var[CTG] = {var_ctg:.2f}, pfss = {np.mean(pfss):.2f}, compute-time = {end-start:.2f}")
                     
        return env
                                       
    def design_config_setting(self, x):
        if 'c_fax_fix' in self.args.design_option:
            self.args.env_config['LH2-cap'] = x[0]
            self.args.env_config['ESS-cap'] = x[1]
            self.args.env_config['PEM-ratio'] = x[2]
            self.args.env_config['X-flow'] = x[3]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap']*0.1
        else:
            self.args.env_config['c-tax'] = x[0]
            self.args.env_config['LH2-cap'] = x[1]
            self.args.env_config['ESS-cap'] = x[2]
            self.args.env_config['PEM-ratio'] = x[3]
            self.args.env_config['X-flow'] = x[4]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap']*0.1
    
    def scenario_sampling(self, netG, dataset):
        noise = torch.randn(self.args.scenario_size, 205, device = 'cpu')
        weather_scenario = netG(noise).detach().cpu().numpy().reshape(-1,24*24)
        weather_scenario[np.where(weather_scenario<0)] = 0
        weather_scenario = dataset.weather_scale.inverse_transform(weather_scenario)
        weather_scenario = weather_scenario.reshape(self.args.scenario_size, 576)
        wind_power_scenario = self.wind_power_function(weather_scenario)
        
        #price data
        si = np.random.randint(0, dataset.__len__(), size = self.args.scenario_size)
        price_scenario = np.array([dataset.price_scaled[idx:idx + dataset.max_seq] for idx in si])[:,:,0] #N x 576 x 1
        price_scenario = dataset.price_scale.inverse_transform(price_scenario)
        
        return noise, wind_power_scenario, price_scenario
    
    def wind_power_function(self, Wind_speed):

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