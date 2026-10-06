import numpy as np
import gymnasium as gym
import math
import random

class PTX_env(gym.Env): 

    def __init__(self, config = None):
        
        # Operating parameter------------------------------------------------------------------------------------------------------------------------------------
        # Hydrogen buffer
        self.SP_H2 = 55.7               # specific power consumption for H2 production via PEMEC [kW/kgH2/h]
        self.SPC_H2 = 55.7+3.03         # specific power consumption for H2 production and compression [kW/kgH2/h]
        self.FC = 22.28                 # specific power generation from H2 via fuel cell [kW/kgH2/h]
        # Due to the large energy loss between power and hydrogen, FC is excluded
        
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
        self.country = config.get('country', 'France')
        self.region = config.get('region', 'Dunkirk')
        self.scale = config.get('scale', 50000) # [kW]
        self.c_tax = config.get('c-tax', 10)
        self.fw = config.get('fw', 1)
        self.co2_option = config.get('co2-option', 'strict')
        self.X_flow = config.get('X-flow', 1000)
        self.X_flow_P_cap = self.X_flow * self.P_X #kWh 
        self.LH2_cap = config.get('LH2-cap', 400)
        self.ESS_cap = config.get('ESS-cap', 25000)
        self.ESS_P_cap = self.ESS_cap * 0.3
        self.fw = config.get('fw')
        PEM_P_cap_min = self.X_flow*self.X_H2*self.SP_H2
        PEM_P_cap_max = self.LH2_cap*self.SP_H2 + PEM_P_cap_min
        self.PEM_ratio = config.get('PEM-ratio', 1)
        self.PEM_P_cap = self.PEM_ratio*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        self.c_tax = config.get('c-tax', 10)
        self.co2_option = config.get('c-option', 'strict') #track co2 capture and emission
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
  
        #Sampling scenario & des-----------------------------------------------------------------------------------------------------------------------------------
        self.L_H2[0] = 0
        self.SOC[0] = self.ESS_cap*0.1
        self.L_H2_init = self.L_H2[0]
        self.SOC_init = self.SOC[0]
        # Scailing factor
        self.emit_scale = self.co2_emit_scale() #min, max
        # self.cost_factor = self.cost_scale() #min, max
        self.cost_factor = (-4000, 4000)
        print('Current design configuration: X_flow ({X_flow}) / H2_cap ({H2_cap}) / ESS_cap ({ESS_cap}) / PEM_P_cap ({PEM_P_cap}) / c-tax ({c_tax})'.format(X_flow = self.X_flow, H2_cap = self.LH2_cap, 
                                                                                                                                           ESS_cap = self.ESS_cap, PEM_P_cap = self.PEM_P_cap, c_tax = self.c_tax))
        
        self.renewable = renewable
        self.SMP = SMP
        print(f"E[renew] = {np.mean(self.renewable[23:]):.2f}, Var[renew] = {np.var(self.renewable[23:]):.2f}, E[price] = {np.mean(self.SMP[23:]):.2f}, Var[price] = {np.var(self.SMP[23:]):.2f}")
       
        # Initialize state
        if self.state_flatten:
            self.state = np.zeros(shape = (self.obs_length*2 + 2 + 4),dtype = np.float32) 
            self.state[:self.obs_length] = self.renewable[self.step_count:self.step_count+self.obs_length]/self.scale
            self.state[self.obs_length:self.obs_length*2] = (self.SMP[self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
            self.state[self.obs_length*2:] = np.concatenate((np.array([self.SOC[0]/self.ESS_cap, self.L_H2[0]/self.LH2_cap]),
                                                             (self.des- self.des_lb)/(self.des_ub- self.des_lb)))
        else:
            self.state = np.zeros(shape = (2, self.obs_length + 2 + 4, 1), dtype = np.float32)
            self.state[0,:self.obs_length, 0] = self.renewable[self.step_count:self.step_count+self.obs_length]/self.scale
            self.state[1,:self.obs_length, 0] = (self.SMP[self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
            self.state[:,self.obs_length:, 0] = np.concatenate((np.array([self.SOC[0]/self.ESS_cap, self.L_H2[0]/self.LH2_cap]),
                                                             (self.des- self.des_lb)/(self.des_ub- self.des_lb)))
                              
        self.P_to_G = np.zeros(len(self.renewable)-self.obs_length+1)
        self.SOC_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.SOC_profile[self.step_count] = self.SOC[0]      
        self.L_H2_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.L_H2_profile[self.step_count] = self.L_H2[0]        
        self.X_profile = np.zeros(len(self.renewable)-self.obs_length+1)
        self.ptx_CO2_list = np.zeros(len(self.renewable)-self.obs_length+1)
        self.P_consum_profile = np.zeros(len(self.renewable)-self.obs_length+2)
        self.error_profile = np.zeros(shape = (len(self.renewable)-self.obs_length+2, 3))
        self.H2_to_market = np.zeros(len(self.renewable)-self.obs_length+1)
        
        # Scaling factor
        self.ESS_penalty_factor = 1/self.ESS_P_cap
        self.PEM_penalty_factor = 1/self.PEM_P_cap 
        self.CO2_emit = np.zeros(len(self.renewable)-self.obs_length+1)
        self.CO2_emit_scaled = np.zeros(len(self.renewable)-self.obs_length+1)

        self.cost_factor = self.cost_scale() #min, max
        self.emit_scale = self.co2_emit_scale() #min, max
                
        return self.state, {}        

    def _STEP(self, action): #Action = 'ratio' of  ESS power & AWE power  & LH2 utilization & H2 split_fraction to market
        
        self.ESS_penalty = 0
        self.PEM_penalty = 0
        self.action_acc.extend([action])
        
        ESS_action = action[0]*self.ESS_P_cap
        AWE_action = action[1]*self.PEM_P_cap
        LH2_util = action[2]*self.LH2_cap
        split =  action[3] 
        
        X_load = self.X_flow_P_cap
        
        #Mass balance
        if LH2_util>self.L_H2[0]: 
            self.PEM_penalty += (LH2_util-self.L_H2[0])*self.SP_H2
            LH2_util = self.L_H2[0] 
        else:
            pass 
        
        H_mis = X_load/self.P_X*self.X_H2 - LH2_util 
        
        if H_mis >= 0:
            self.L_H2[0] = self.L_H2[0]-LH2_util
        else:
            self.PEM_penalty += (LH2_util- X_load/self.P_X*self.X_H2)*self.SP_H2
            self.L_H2[0] -= X_load/self.P_X*self.X_H2
            H_mis = 0
            
        # Power balance
        P_ptx = X_load
        ptx_H2 = H_mis*self.SP_H2
        ptx_CO2 = X_load/self.P_X*self.X_CO2         
        P_consum = P_ptx + ptx_H2
        P_mis = self.renewable[self.step_count+self.obs_length-1] - P_consum
                
        L_H2_prev = self.L_H2[0]
        SOC_prev = self.SOC[0]
        
        self.P_to_G[self.step_count] += P_mis
        
        # Available action option
        # ESS: charging or discharging
        # AWE: hydrogen storage or not (*As further step, fuel cell can be considered) 
        # ESS action
        
        ESS_action = self.ESS_masking(ESS_action)
        self.P_to_G[self.step_count] += -ESS_action
        
        if AWE_action<0:
            self.PEM_penalty += -AWE_action
            AWE_action = 0
            
        if H_mis>0:           
            if H_mis*self.SP_H2>AWE_action:
                self.PEM_penalty +=  (H_mis*self.SP_H2 - AWE_action)
                AWE_action = H_mis*self.SP_H2
            else:
                pass
        
        H2_produce = AWE_action/self.SP_H2 - H_mis
        
        if H2_produce<0:
            self.PEM_penalty += (H_mis-AWE_action/self.SP_H2)*self.SP_H2 #this part modified
            H2_produce = 0
        else: 
            pass 
         
        H2_to_sell = split*H2_produce 
        H2_to_storage = H2_produce*(1-split) 
        
        if H2_to_storage + self.L_H2[0] >= self.LH2_cap:
            self.PEM_penalty += (H2_to_storage + self.L_H2[0]-self.LH2_cap)*self.SP_H2
            H2_to_storage = self.LH2_cap-self.L_H2[0]
            
        H2_to_sell = H2_produce - H2_to_storage
        self.P_to_G[self.step_count] += -H2_produce*self.SP_H2-H2_to_storage*(self.SPC_H2-self.SP_H2)
        self.L_H2[0] += H2_to_storage
            
        if ESS_action>0:
            ESS_store = ESS_action
        else:
            ESS_store = 0
        
        self.TBOM = 0 #ESS_store * 30
        self.H = (H_mis + H2_to_storage + H2_to_sell)#kg 
        self.TAOM = 10.11 * 0.012 * self.H + 0.0019 * 2.96 * self.H + 0.11 * 0.012 * self.H + 0.00029 * 0.33 * self.H
        
        #Supply and demand
        #self.grid_supply[self.step_count] = self.P_to_G[self.step_count]
        
        # Tracking net CO2 emission
        # Before modified
        if self.co2_option == 'strict':
            if self.P_to_G[self.step_count]<0:    
                self.CO2_emit[self.step_count] = -self.P_to_G[self.step_count]/1000*self.emission_factor - ptx_CO2/1000 #ton/hr
            else:
                self.CO2_emit[self.step_count] = - ptx_CO2/1000 #ton/hr
        else:
            self.CO2_emit[self.step_count] = -self.P_to_G[self.step_count]/1000*self.emission_factor - ptx_CO2/1000 #ton/hr
        
        self.CO2_emit_scaled[self.step_count] = (self.CO2_emit[self.step_count]-self.emit_scale[0])/(self.emit_scale[1]-self.emit_scale[0])
         
        if self.step_count == len(self.renewable)-self.obs_length:
            self.X_profile[self.step_count] = X_load/self.P_X   
            self.H2_to_market[self.step_count] += H2_to_sell
            self.penalty = self.ESS_penalty*self.ESS_penalty_factor + self.PEM_penalty*self.PEM_penalty_factor
            reward = self.cost_calculation(ptx_CO2)
            self.penalty_list.append(self.penalty)
            self.normalized_cost_list.append((self.cost_calculation(ptx_CO2)-self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0]))            
            
            self.cost_list.append(self.cost_calculation(ptx_CO2))
            self.reward_list.append(reward)
            
            self.step_count += 1
            self.P_consum_profile[self.step_count] = P_consum
            self.SOC_profile[self.step_count] = self.SOC[0]
            self.L_H2_profile[self.step_count] = self.L_H2[0] 
            self.P_consum_profile[self.step_count] = P_consum 
            #self.error_profile[self.step_count] = self.energy_balance_function(self.step_count)
            #self.LCOX = self.LCOX_calculation(ptx_CO2)
            done = True  
                  
        else:
            self.X_profile[self.step_count] = X_load/self.P_X  
            self.H2_to_market[self.step_count] += H2_to_sell
            self.penalty = self.ESS_penalty*self.ESS_penalty_factor + self.PEM_penalty*self.PEM_penalty_factor                                   
            reward = self.cost_calculation(ptx_CO2)
            self.penalty_list.append(self.penalty)
            self.normalized_cost_list.append((self.cost_calculation(ptx_CO2)-self.cost_factor[0])/(self.cost_factor[1]-self.cost_factor[0]))          
            
            self.cost_list.append(self.cost_calculation(ptx_CO2))
            self.reward_list.append(reward) 
            
            self.step_count += 1
            self.P_consum_profile[self.step_count] = P_consum             
            self.L_H2_profile[self.step_count] = self.L_H2[0]
            self.SOC_profile[self.step_count] = self.SOC[0]
            self.error_profile[self.step_count] = self.energy_balance_function(P_mis, H2_produce*self.SP_H2, ESS_action, H2_to_storage*(self.SPC_H2-self.SP_H2), L_H2_prev, SOC_prev)
                                  
            # Initialize state
            if self.state_flatten:
                self.state = np.zeros(shape = (self.obs_length*2 + 2 + 4),dtype = np.float32) 
                self.state[:self.obs_length] = self.renewable[self.step_count:self.step_count+self.obs_length]/self.scale
                self.state[self.obs_length:self.obs_length*2] = (self.SMP[self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
                self.state[self.obs_length*2:] = np.concatenate((np.array([self.SOC[0]/self.ESS_cap, self.L_H2[0]/self.LH2_cap]),
                                                                 (self.des- self.des_lb)/(self.des_ub- self.des_lb)))
            else:
                self.state = np.zeros(shape = (2, self.obs_length + 2 + 4, 1), dtype = np.float32)
                self.state[0,:self.obs_length, 0] = self.renewable[self.step_count:self.step_count+self.obs_length]/self.scale
                self.state[1,:self.obs_length, 0] = (self.SMP[self.step_count:self.step_count+self.obs_length]-self.min_SMP)/(self.max_SMP-self.min_SMP)
                self.state[:,self.obs_length:, 0] = np.concatenate((np.array([self.SOC[0]/self.ESS_cap, self.L_H2[0]/self.LH2_cap]),
                                                                 (self.des- self.des_lb)/(self.des_ub- self.des_lb)))
            done = False 
        
        return self.state, reward, done, False, {}
    
    def cost_calculation(self, ptx_CO2):
        
        #Cost estimation
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
        
        profit = self.P_to_G[self.step_count]*self.SMP[self.step_count+self.obs_length-1] + self.H2_to_market[self.step_count]*self.H2_price - self.TBOM - self.TAOM
        
        #Carbon tax
        if self.P_to_G[self.step_count]<0:
            carbon_tax = -self.c_tax*self.emission_factor*self.P_to_G[self.step_count]/1000 
        else:
            carbon_tax = 0
        return profit-carbon_tax
    
    def LCOX_calculation(self, ptx_CO2):        
        #Cost estimation
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
        
        C_ptx = 8600 * ptx_CO2 / 1000 * (
                    0.204 * (math.log10(ptx_CO2 * 8.6)) ** 4 - 4.819 * (math.log10(ptx_CO2 * 8.6)) ** 3 + 43.02 * (
                math.log10(ptx_CO2 * 8.6)) ** 2 - 175.9 * (math.log10(ptx_CO2 * 8.6)) + 1014.14 * self.C_CO2 / 1000 + 332.22)
        
        OPEX_total = OPEX_gen + C_ptx - np.sum(self.P_to_G*self.SMP[self.obs_length-1:])/(self.op_period)*8600-np.sum(self.H2_to_market*self.H2_price)/(self.op_period)*8600
        
        #Carbon taxing (considering real c_tax)
        #Before modified
        tax_idx = np.where(self.P_to_G<0)
        carbon_tax = (-np.sum(self.P_to_G[tax_idx])*self.emission_factor/1000*self.c_tax)/(self.op_period)*8600
        #After modified
        #tax_idx = np.where(self.P_to_G*self.emission_factor + ptx_CO2<0) #this part revised (11.25)
        
        X_flow_total = self.X_flow*8600
  
        if (OPEX_total+carbon_tax+CAP_total*CRF)<0:
            production_cost = 0
        else:
            production_cost = (OPEX_total+carbon_tax+CAP_total*CRF)/(X_flow_total/1000)
                
        return production_cost
           
    def cost_scale(self):
    
        profit_min = 0#-(self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*(self.max_SMP+self.min_SMP)/3 #- ((self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*self.emission_factor/1000)*self.c_tax
        profit_max = (self.scale-self.X_flow_P_cap)*(self.max_SMP+self.min_SMP)/2
        
        return profit_min, profit_max
    
    def co2_emit_scale(self):
        
        emit_max = (self.X_flow_P_cap + self.ESS_P_cap + self.PEM_P_cap)*self.emission_factor - self.X_flow*self.X_CO2
        emit_min = - self.X_flow*self.X_CO2
        
        return emit_min/1000, emit_max/1000 #kg ->ton
        
    def ESS_masking(self, ESS_action):
        if ESS_action>0:
            if ESS_action*self.ESS_eff+self.SOC[0]>self.ESS_cap*self.SOC_up:
                self.ESS_penalty += ESS_action*self.ESS_eff+self.SOC[0]-self.ESS_cap*self.SOC_up
                ESS_action = (self.ESS_cap*self.SOC_up-self.SOC[0])/self.ESS_eff 
                self.SOC[0] = self.ESS_cap*self.SOC_up
                self.SOC[0] = self.SOC[0]*(1-self.self_dh)
            else:
                if (ESS_action*self.ESS_eff + self.SOC[0])*(1-self.self_dh)>self.ESS_cap*self.SOC_lb:
                    self.SOC[0] += ESS_action*self.ESS_eff
                    self.SOC[0] = self.SOC[0]*(1-self.self_dh)
                else:
                    self.ESS_penalty += (self.ESS_cap*self.SOC_lb - (ESS_action*self.ESS_eff + self.SOC[0])*(1-self.self_dh))
                    ESS_action = (self.ESS_cap*self.SOC_lb/(1-self.self_dh) - self.SOC[0])/self.ESS_eff
                    self.SOC[0] = self.ESS_cap*self.SOC_lb
        else:
            if ESS_action/self.ESS_eff+self.SOC[0] < self.ESS_cap*self.SOC_lb:
                self.ESS_penalty -= ESS_action/self.ESS_eff+self.SOC[0]
                ESS_action = (-self.SOC[0] + self.ESS_cap*self.SOC_lb)*self.ESS_eff
                self.SOC[0] = self.ESS_cap*self.SOC_lb*(1-self.self_dh)
            else:
                self.SOC[0] += ESS_action/self.ESS_eff
                self.SOC[0] = self.SOC[0]*(1-self.self_dh)
                if self.SOC[0]<0:
                    self.SOC[0] = 0
                else:
                    pass
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

    def solar_power_function(self,Solar_irradiance):
        Ht = Solar_irradiance
        H_ref = 1000  # W/m2
        idx_cutoff = Ht > H_ref
        Ht[idx_cutoff] = H_ref
        n_tot = 0.9375

        return Ht / H_ref * n_tot  # Capacity fator =[0,1]
    
    def energy_balance_function(self, P_mis, AWE_action, ESS_action, H2_storage, L_H2_prev, SOC_prev):
        # Power balance
        Input = P_mis-AWE_action-ESS_action-H2_storage
        output = self.P_to_G[self.step_count-1]
        power_error = (Input-output)/(output+1e-5)*100 #power_error>0 additional power is generated
        
        # Mass balance
        #print('Input ESS_action:', ESS_action)
        #print('SOC prev:', SOC_prev)
        #print('SOC current:', self.SOC[0])
        if ESS_action<0:
            SOC_error = ((self.SOC[0]-SOC_prev)-ESS_action/self.ESS_eff)/SOC_prev*100
        else:
            SOC_error = ((self.SOC[0]-SOC_prev)-ESS_action*self.ESS_eff)/SOC_prev*100
        
        Storage_error = (H2_storage/(self.SPC_H2-self.SP_H2)-(self.L_H2[0]-L_H2_prev))/(self.L_H2[0]-L_H2_prev+1e-5)*100
    
        return power_error, SOC_error, Storage_error
    
    