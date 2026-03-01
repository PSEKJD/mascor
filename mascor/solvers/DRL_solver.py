# Import libraries
import torch
import numpy as np
from mascor.solvers import drl_policy
from botorch.utils.transforms import normalize

class solver(): #only for online planning
    def __init__(self, args, mode = None, env_class = None):        
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
        self.limit_state = 0
        self.prob_failure = args.prob_fail 
        
        self.policy = drl_policy(self.args, mode)
        self.env_class = env_class

    def planning(self, des, episode, save_option = True):
        self.design_config_setting(des.cpu().detach().numpy().astype(np.float64))
        wind_speed, grid, wind_speed_scaled = episode
        renewable = self.args.env_config['scale']*(self.wind_power_function(wind_speed))
        
        # Padding renewable & grid profile to matching state obs
        renewable = np.concatenate((renewable[:self.args.env_config['obs-length']-1], renewable))
        grid = np.concatenate((grid[:self.args.env_config['obs-length']-1], grid))
                                   
        #env-reset
        env = self.env_class(self.args.env_config)
        states, _ = env.reset(renewable, grid)
        del renewable, grid
        done = False
        while not done:
            action = self.policy.compute_actions(np.expand_dims(states, axis = 0))
            #real-env-step
            states, reward, done, _ , _ = env.step(action[0])
        return env
        
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
    
    def design_config_setting(self, x):
        if 'c_fax_fix' in self.args.design_option:
           self.args.env_config['LH2-cap'] = x[0]
           self.args.env_config['ESS-cap'] = x[1]
           self.args.env_config['PEM-ratio'] = x[2]
           self.args.env_config['X-flow'] = x[3]
           self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap']*0.1
           self.args.env_config['L-H2-init'] = 0
        else:
           self.args.env_config['c-tax'] = x[0]
           self.args.env_config['LH2-cap'] = x[1]
           self.args.env_config['ESS-cap'] = x[2]
           self.args.env_config['PEM-ratio'] = x[3]
           self.args.env_config['X-flow'] = x[4]
           self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap']*0.1
           self.args.env_config['L-H2-init'] = 0
        