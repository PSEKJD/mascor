import numpy as np
import gymnasium as gym
import math
import random
import os

class PTX_env(gym.Env): 

    def __init__(self, config = None):
        # Operating parameter------------------------------------------------------------------------------------------------------------------------------------
        self.SP_H2 = 55.7               # specific power consumption for H2 production via PEMEC [kW/kgH2/h]
        self.SPC_H2 = 55.7+3.03         # specific power consumption for H2 production and compression [kW/kgH2/h]
        self.FC = 22.28                 # specific power generation from H2 via fuel cell [kW/kgH2/h]
        
        # BESS 
        self.ESS_eff = 0.95             # discharge and charge efficiency
        self.self_dh = 0.05/(30*24)     # self-discharge efficiency 
        self.SOC_lb = 0.1
        self.SOC_up = 0.9
        
        # MeOH conversion unit
        self.X_H2 = 0.19576         # specific H2 consumption for "X" production，[kgH2/s / kgX/s]
        self.X_CO2 = 1.435802       # specific CO2 consumption for "X" production，[kgCO2/s / kgX/s]
        self.P_X = 0.65702      # specific power consumption for "X" production，kW/kg/h, X is methanol
        
        # Cost parameters
        # Renewable power
        self.CAP_solar = 740 # [$/kW]
        self.CAP_wind = 1250 # [$/kW]
        self.OPEX_solar = 12.6 # [$/kW]
        self.OPEX_wind = 25.0 #$ [/kW]
        
        # Hydrogen buffer
        self.CAP_PEM = 600              # PEM cost [$/kW]
        self.CAP_FC = 170               # Fuel cell cost [$/kW]
        self.CAP_H2 = 751700            # $/tonne
        self.H2_price = 5               # H2 sale price [$/kg] (Ref: powermag.com/blog/hydrogen-prices-skyrocket-over-2021-amid-tight-power-and-gas-supply/)
        
        # BESS
        self.CAPEX_BESS = 236.5         # $/kW
        
        # MeOH conversion unit
        self.C_CO2 = 50                 # CO2 purchase cost [$/tonne]
        self.emission_factor = 0.5 #kg/kWh
        
        # Design configuration range------------------------------------------------------------------------------------------------------------------------------------
        self.scale = config.get('scale', 50000) # [kW]
        self.scale_min = 5000 #5MW
        self.scale_max = 25000 #25MW
        
        self.X_flow_range = np.array([self.scale_min/(self.P_X + self.X_H2*self.SP_H2), self.scale_max/(self.P_X + self.X_H2*self.SP_H2)])
        self.X_flow_P_cap_range = self.X_flow_range*self.P_X # [kWh]
        self.LH2_cap_range = np.array([self.scale_min/self.SP_H2, self.scale_max/self.SP_H2*4])
        self.ESS_cap_range = np.array([self.scale_min, self.scale_max*4])
        self.ESS_P_cap_range = self.ESS_cap_range*0.3
        self.des_lb = np.array([self.LH2_cap_range[0], self.ESS_cap_range[0], 0, self.X_flow_range[0]])
        self.des_ub = np.array([self.LH2_cap_range[1], self.ESS_cap_range[1], 1, self.X_flow_range[1]])
        
        # Design configuration
        self.c_tax = config.get('c-tax', 10)
        self.fw = config.get('fw', 1)
        self.co2_option = config.get('co2-option', 'strict')
        self.X_flow = config.get('X-flow', 1000)
        self.X_flow_P_cap = self.X_flow * self.P_X #kWh 
        self.LH2_cap = config.get('LH2-cap', 400)
        self.ESS_cap = config.get('ESS-cap', 25000)
        self.ESS_P_cap = self.ESS_cap * 0.3
        PEM_P_cap_min = self.X_flow*self.X_H2*self.SP_H2
        PEM_P_cap_max = self.LH2_cap*self.SP_H2 + PEM_P_cap_min
        self.PEM_ratio = config.get('PEM-ratio', 1)
        self.PEM_P_cap = self.PEM_ratio*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        self.max_SMP = config.get('max-SMP', 1.0)
        self.min_SMP = config.get('min-SMP', 0.0)
        self.max_c_tax = config.get('max-c-tax', 1.0)
        self.min_c_tax = config.get('min-c-tax', 1.0)
        self.des = np.array([self.LH2_cap, self.ESS_cap, self.PEM_ratio, self.X_flow])
        
        # State and action space------------------------------------------------------------------------------------------------------------------------------------
        self.state_flatten = config.get('flatten', True)
        self.obs_length = config.get('obs-length', 24)
        self.op_period = config.get('op-period', 720)
      
        #Initialize list ans state------------------------------------------------------------------------------------------------------------------------------------
        self.penalty_weight = config.get('penalty-weigth', 0)
        self.cost_weight = config.get('cost-weigth', 1)
        self.grid_penalty_weigth = config.get('grid-penlaty-weigth', 0)        
        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.AWE_acc = []
        self.grid_acc = []
        self.L_H2 = np.zeros(1)
        self.X_acc = []
        self.SOC =  np.zeros(1)
        self.penalty = 0
        self.step_count = 0
        self.n_worker = config.get('n-worker', 100)
        self.L_H2 = np.zeros(self.n_worker)
        self.SOC =  np.zeros(self.n_worker)
        self.L_H2[:] = config.get('L-H2-init', 0) 
        self.SOC[:] = config.get('SOC-init', 0.1)
          
    def _RESET(self, renewable, SMP, seed=None, options=None):
        random.seed(seed)
        np.random.seed(seed)
        
        # History initialize------------------------------------------------------------------------------------------------------------------------------------ 
        self.step_count = 0
        self.action_acc = []
        self.ESS_charge = []
        self.ESS_discharge = []
        self.AWE_acc = []
        self.grid_acc = []        
        self.X_acc = []        
        self.penalty = 0
        self.normalized_cost_list = []
        self.normalized_grid_penalty_list = []
        self.cost_list = []
        self.production_cost_list = []
        self.penalty_list = []
        self.reward_list = []
        self.step_reward = 0
        self.total_reward = 0
        self.step_count = 0
        self.L_H2_init = self.L_H2
        self.SOC_init = self.SOC
  
        self.renewable = renewable
        self.SMP = SMP
        print(f"E[renew] = {np.mean(self.renewable):.2f}, Var[renew] = {np.var(self.renewable):.2f}, E[price] = {np.mean(self.SMP):.2f}, Var[price] = {np.var(self.SMP):.2f}")
       
        # Initialize state
        if self.state_flatten:
            self.state = np.zeros(shape = (self.n_worker, self.obs_length*2 + 2 + 4),dtype = np.float32) 
            self.state[:, :self.obs_length] = self.renewable[:, self.step_count:self.step_count+self.obs_length]/self.scale
            self.state[:, self.obs_length:self.obs_length*2] = (self.SMP[:, self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
            
            
            self.state[:, self.obs_length*2:] = np.concatenate((np.stack((self.SOC / self.ESS_cap, self.L_H2 / self.LH2_cap), axis=1),  # shape: (n_worker, 2)
                                                                np.tile((self.des - self.des_lb) / (self.des_ub - self.des_lb),(self.n_worker, 1))  # shape: (n_worker, 4)
                                                                ),axis=1)
        else:
            self.state = np.zeros(shape = (self.n_worker, 2, self.obs_length + 2 + 4, 1), dtype = np.float32)
            self.state[:, 0,:self.obs_length, 0] = self.renewable[:,self.step_count:self.step_count+self.obs_length]/self.scale
            self.state[:, 1,:self.obs_length, 0] = (self.SMP[:, self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
            info = np.concatenate((np.stack((self.SOC / self.ESS_cap, self.L_H2 / self.LH2_cap), axis=1),
                                                             np.tile((self.des- self.des_lb)/(self.des_ub- self.des_lb),(self.n_worker,1))), axis = -1)
            info = np.expand_dims(info, axis=1)  # (100, 1, 6)
            info = np.repeat(info, 2, axis=1)    # (100, 2, 6)
            self.state[:, :,self.obs_length:, 0] = info
                              
        self.P_to_G = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))
        self.TAOM = np.zeros(shape = (self.n_worker,self.renewable.shape[1]-self.obs_length+1))
        self.SOC_profile = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+2))
        self.SOC_profile[:,self.step_count] = self.SOC      
        self.L_H2_profile = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+2))
        self.L_H2_profile[:,self.step_count] = self.L_H2      
        self.X_profile = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))
        self.ptx_CO2_list = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))
        self.P_consum_profile = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+2))
        self.error_profile = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+2, 3))
        self.H2_to_market = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))
        
        # Scaling factor
        self.ESS_penalty_factor = 1/self.ESS_P_cap
        self.PEM_penalty_factor = 1/self.PEM_P_cap 
        self.CO2_emit = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))
        self.CO2_emit_scaled = np.zeros(shape = (self.n_worker, self.renewable.shape[1]-self.obs_length+1))

        self.cost_factor = self.cost_scale() #min, max
        self.emit_scale = self.co2_emit_scale() #min, max
                
        return self.state, {}        

    def _STEP(self, action): #Action = 'ratio' of  ESS power & AWE power  & LH2 utilization & H2 split_fraction to market
        
        self.ESS_penalty = 0
        self.PEM_penalty = 0
        self.action_acc.extend([action])
        ESS_action = action[:,0]*self.ESS_P_cap
        AWE_action = action[:,1]*self.PEM_P_cap
        LH2_util = action[:,2]*self.LH2_cap
        split =  action[:,3] 

        X_load = self.X_flow_P_cap        

        #Mass balance
        idx1 = np.where(LH2_util>self.L_H2)
        LH2_util[idx1] = self.L_H2[idx1]
        H_mis = X_load/self.P_X*self.X_H2 - LH2_util 
        idx2 = np.where(H_mis>0)
        self.L_H2[idx2] = self.L_H2[idx2]-LH2_util[idx2]
        idx3 = np.where(H_mis<=0)
        self.L_H2[idx3] -= X_load/self.P_X*self.X_H2
        H_mis[idx3] = 0
        
        # Power balance
        P_ptx = X_load
        ptx_H2 = H_mis*self.SP_H2
        ptx_CO2 = X_load/self.P_X*self.X_CO2         
        P_consum = P_ptx + ptx_H2
        P_mis = self.renewable[:,self.step_count+self.obs_length-1] - P_consum
        L_H2_prev = self.L_H2
        SOC_prev = self.SOC
        self.P_to_G[:,self.step_count] += P_mis
        
        # Available action option
        # ESS: charging or discharging
        # AWE: hydrogen storage or not (*As further step, fuel cell can be considered) 
        # ESS action
        ESS_action = self.ESS_masking(ESS_action)
        self.P_to_G[:, self.step_count] += -ESS_action
        
        idx4 = np.where(AWE_action<0)
        AWE_action[idx4] = 0
        idx5 = np.where(H_mis*self.SP_H2>AWE_action)
        AWE_action[idx5] = H_mis[idx5]*self.SP_H2
        
        H2_produce = AWE_action/self.SP_H2 - H_mis
        idx6 = np.where(H2_produce<0)
        H2_produce[idx6] = 0

        H2_to_sell = split*H2_produce
        H2_to_storage = (1-split)*H2_produce
        idx7 = np.where(H2_to_storage + self.L_H2 >= self.LH2_cap)
        H2_to_storage[idx7] = self.LH2_cap-self.L_H2[idx7]        
        H2_to_sell = H2_produce - H2_to_storage
        self.P_to_G[:, self.step_count] += -H2_produce*self.SP_H2-H2_to_storage*(self.SPC_H2-self.SP_H2)
        self.L_H2 += H2_to_storage
        
        self.TBOM = 0 #ESS_store * 30
        self.H = (H_mis + H2_to_storage + H2_to_sell)#kg 
        self.TAOM[:, self.step_count] = 10.11 * 0.012 * self.H + 0.0019 * 2.96 * self.H + 0.11 * 0.012 * self.H + 0.00029 * 0.33 * self.H
        
        # Tracking net CO2 emission
        if self.co2_option == 'strict':
            idx8 = np.where(self.P_to_G[:,self.step_count]<0)
            self.CO2_emit[idx8,self.step_count] = -self.P_to_G[idx8,self.step_count]/1000*self.emission_factor - ptx_CO2/1000 #ton/hr
            idx9 = np.where(self.P_to_G[:,self.step_count]>=0)
            self.CO2_emit[idx9,self.step_count] = - ptx_CO2/1000 #ton/hr
        else:
            self.CO2_emit[:,self.step_count] = -self.P_to_G[:,self.step_count]/1000*self.emission_factor - ptx_CO2/1000 #ton/hr
        
        self.CO2_emit_scaled[:,self.step_count] = (self.CO2_emit[:,self.step_count]-self.emit_scale[0])/(self.emit_scale[1]-self.emit_scale[0])
         
        if self.step_count == self.renewable.shape[1]-self.obs_length:
            self.X_profile[:, self.step_count] = X_load/self.P_X   
            self.H2_to_market[:, self.step_count] += H2_to_sell                  
            reward = self.cost_calculation(ptx_CO2)
            self.cost_list.append(self.cost_calculation(ptx_CO2))
            self.reward_list.append(reward)
            
            self.step_count += 1
            self.P_consum_profile[:,self.step_count] = P_consum
            self.SOC_profile[:,self.step_count] = self.SOC
            self.L_H2_profile[:,self.step_count] = self.L_H2
            done = True  
            
        else:
            self.X_profile[:, self.step_count] = X_load/self.P_X   
            self.H2_to_market[:, self.step_count] += H2_to_sell
            reward = self.cost_calculation(ptx_CO2)                       
            self.cost_list.append(self.cost_calculation(ptx_CO2))
            self.reward_list.append(reward) 
            
            self.step_count += 1
            self.P_consum_profile[:,self.step_count] = P_consum             
            self.SOC_profile[:,self.step_count] = self.SOC
            self.L_H2_profile[:,self.step_count] = self.L_H2
            
            # Initialize state
            if self.state_flatten:
                self.state = np.zeros(shape = (self.n_worker, self.obs_length*2 + 2 + 4),dtype = np.float32) 
                self.state[:, :self.obs_length] = self.renewable[:, self.step_count:self.step_count+self.obs_length]/self.scale
                self.state[:, self.obs_length:self.obs_length*2] = (self.SMP[:, self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
                
                
                self.state[:, self.obs_length*2:] = np.concatenate((np.stack((self.SOC / self.ESS_cap, self.L_H2 / self.LH2_cap), axis=1),  # shape: (n_worker, 2)
                                                                    np.tile((self.des - self.des_lb) / (self.des_ub - self.des_lb),(self.n_worker, 1))  # shape: (n_worker, 4)
                                                                    ),axis=1)
            else:
                self.state = np.zeros(shape = (self.n_worker, 2, self.obs_length + 2 + 4, 1), dtype = np.float32)
                self.state[:, 0,:self.obs_length, 0] = self.renewable[:,self.step_count:self.step_count+self.obs_length]/self.scale
                self.state[:, 1,:self.obs_length, 0] = (self.SMP[:, self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
                info = np.concatenate((np.stack((self.SOC / self.ESS_cap, self.L_H2 / self.LH2_cap), axis=1),
                                                                 np.tile((self.des- self.des_lb)/(self.des_ub- self.des_lb),(self.n_worker,1))), axis = -1)
                info = np.expand_dims(info, axis=1)  # (100, 1, 6)
                info = np.repeat(info, 2, axis=1)    # (100, 2, 6)
                self.state[:, :,self.obs_length:, 0] = info
            done = False 
        
        return self.state, reward, self.CO2_emit[:,self.step_count-1], done, False, {}
    
    def cost_calculation(self, ptx_CO2):
        
       profit = self.P_to_G[:,self.step_count]*self.SMP[:,self.step_count+self.obs_length-1] + self.H2_to_market[:,self.step_count]*self.H2_price - self.TBOM - self.TAOM[:,self.step_count]
       #Carbon tax
       carbon_tax = -self.c_tax*self.emission_factor*self.P_to_G[:,self.step_count]/1000
       idx_zero = np.where(self.P_to_G[:, self.step_count]>=0)
       carbon_tax[idx_zero] = 0
       
       return profit-carbon_tax
    
    def LCOX_calculation(self, mu_profit = None, var_profit = None): 
        ii = 0.08 # interest rate
        N = 25 # plant life, years
        CRF =  ii * ((ii+1) ** N) / ((ii+1) ** N - 1)
        CAP_gen = self.CAP_solar * self.scale * (1-self.fw) + self.CAP_wind * self.scale * self.fw        
        OPEX_gen = self.OPEX_solar * self.scale * (1-self.fw) + self.OPEX_wind * self.scale * self.fw
        CAP_hydrogen = self.CAP_H2*self.LH2_cap/1000
        CAP_electrolyzer = (self.PEM_P_cap)*self.CAP_PEM
        CAP_distillation  = self.distillation_cost()
        BESS_cos = self.ESS_cap*self.CAPEX_BESS
        CAP_total = CAP_gen + CAP_hydrogen + CAP_electrolyzer + CAP_distillation + BESS_cos  
        ptx_CO2 = self.X_flow_P_cap/self.P_X*self.X_CO2        
        C_ptx = 8600 * ptx_CO2 / 1000 * (
                    0.204 * (math.log10(ptx_CO2 * 8.6)) ** 4 - 4.819 * (math.log10(ptx_CO2 * 8.6)) ** 3 + 43.02 * (
                math.log10(ptx_CO2 * 8.6)) ** 2 - 175.9 * (math.log10(ptx_CO2 * 8.6)) + 1014.14 * self.C_CO2 / 1000 + 332.22)
        
        if mu_profit is not None and var_profit is not None:
            mu_OPEX = OPEX_gen + C_ptx + mu_profit/576*8600
            X_flow_total = self.X_flow*8600
            mu_LCOX = (mu_OPEX+CAP_total*CRF)/(X_flow_total/1000)
            var_LCOX = var_profit*((1/576*8600)/(X_flow_total/1000))**2
        else:
            OPEX_total = OPEX_gen + C_ptx - np.sum(self.P_to_G*self.SMP[:,self.obs_length-1:]+self.H2_to_market*self.H2_price-self.TAOM, axis = 1)/(self.op_period)*8600
            carbon_tax = (-np.sum(self.P_to_G*(self.P_to_G<0),axis=1)*self.emission_factor/1000*self.c_tax)/self.op_period*8600
            X_flow_total = self.X_flow*8600
            LCOX = (OPEX_total+carbon_tax+CAP_total*CRF)/(X_flow_total/1000)
            mu_LCOX = np.mean(LCOX)
            var_LCOX = np.var(LCOX)
        
        return mu_LCOX/1000, var_LCOX/1e6 #$/kg
           
    def cost_scale(self):
    
        profit_min = 0#-(self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*(self.max_SMP+self.min_SMP)/3 #- ((self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*self.emission_factor/1000)*self.c_tax
        profit_max = (self.scale-self.X_flow_P_cap)*(self.max_SMP+self.min_SMP)/2
        
        return profit_min, profit_max
    
    def co2_emit_scale(self):
        
        emit_max = (self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*self.emission_factor - self.X_flow*self.X_CO2
        emit_min = - self.X_flow*self.X_CO2
        
        return emit_min/1000, emit_max/1000 #kg ->ton
        
    def ESS_masking(self, ESS_action):   
        # Overcharging
        ch_cond = ESS_action>=0
        ovch_cond =  (ESS_action*self.ESS_eff + self.SOC>self.ESS_cap*self.SOC_up)
        idx_ovch = np.where(np.logical_and(ch_cond, ovch_cond))
        ESS_action[idx_ovch] = (self.ESS_cap*self.SOC_up-self.SOC[idx_ovch])/self.ESS_eff
        self.SOC[idx_ovch] = self.ESS_cap*self.SOC_up
        
        #Charging
        idx_ch = np.where(np.logical_and(ch_cond, np.logical_not(ovch_cond)))
        self.SOC[idx_ch] += ESS_action[idx_ch]*self.ESS_eff
               
        # Overdischarging
        dch_cond = ESS_action<0
        odch_cond = (ESS_action/self.ESS_eff + self.SOC<self.ESS_cap*self.SOC_lb)
        idx_odch = np.where(np.logical_and(dch_cond, odch_cond))
        ESS_action[idx_odch] = (-self.SOC[idx_odch] + self.ESS_cap*self.SOC_lb)*self.ESS_eff
        self.SOC[idx_odch] = self.ESS_cap*self.SOC_lb
        
        #Discharging
        idx_dh = np.where(np.logical_and(dch_cond, np.logical_not(odch_cond)))
        self.SOC[idx_dh] += ESS_action[idx_dh]/self.ESS_eff
        
        #self discharge
        self.SOC = self.SOC*(1-self.self_dh)
        self.SOC[np.where(self.SOC<0)] = 0
        return ESS_action
     
    def step(self, action):
        
        return self._STEP(action)
    
    def reset(self, renewable, SMP, seed=None, options=None):
        return self._RESET(renewable, SMP, seed=None, options=None)
    
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

    def distillation_cost(self):
        
        # Column diameter
        D = ((4/3.14/0.761) *(self.X_flow/32) *2 *22.4 * (64+273)/273 *1 * 1/3600)**0.5
   
        # Column length
        L = 0.61 * 38 + 4.27
   
        # Column vessel cost
        CC = 17640 * D**1.066 * L**0.802
   
        # Tray cost
        TC = 229 * D**1.55 *38
   
        # Heat exchanger cost
        ConC = 7296 * (1063* self.X_flow/96872.7)**0.65
        ExC = 7296 * (3109* self.X_flow/96872.7)**0.65
   
        # Compressor cost
        cmpC = 5840 * (23238.8* self.X_flow/96872.7)**0.82
   
        CAPEX = CC+ TC +ConC + ExC + cmpC
        
        return CAPEX
    
    