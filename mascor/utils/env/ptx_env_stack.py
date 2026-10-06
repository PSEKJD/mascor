import torch
import gymnasium as gym
import random

class PTX_env(gym.Env): 
    def __init__(self, config=None, device=None):
        super().__init__()
        config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # =================== Operating parameters ===================
        self.SP_H2 = 55.7
        self.SPC_H2 = 55.7 + 3.03
        self.FC = 22.28
        self.ESS_eff = 0.95
        self.self_dh = 0.05 / (30 * 24)
        self.SOC_lb = 0.1
        self.SOC_up = 0.9
        self.X_H2 = 0.19576
        self.X_CO2 = 1.435802
        self.P_X = 0.65702
        # =================== Cost parameters ===================
        self.CAP_solar = 740
        self.CAP_wind = 1250
        self.OPEX_solar = 12.6
        self.OPEX_wind = 25.0
        self.CAP_PEM = 600
        self.CAP_FC = 170
        self.CAP_H2 = 751700
        self.H2_price = 5
        self.CAPEX_BESS = 236.5
        self.C_CO2 = 50
        self.emission_factor = 0.5
        # =================== Design configuration ===================
        self.scale = config.get('scale', 50000)
        self.scale_min = 5000 #5MW
        self.scale_max = 25000 #25MW
        self.op_period = config.get('op-period', 576)

        self.X_flow = config.get('X-flow', 1000)
        self.LH2_cap = config.get('LH2-cap', 400)
        self.ESS_cap = config.get('ESS-cap', 25000)
        self.ESS_P_cap = self.ESS_cap * 0.3
        self.fw = config.get('fw')
        PEM_P_cap_min = self.X_flow * self.X_H2 * self.SP_H2
        PEM_P_cap_max = self.LH2_cap * self.SP_H2 + PEM_P_cap_min
        self.PEM_ratio = config.get('PEM-ratio', 1)
        self.PEM_P_cap = self.PEM_ratio * (PEM_P_cap_max - PEM_P_cap_min) + PEM_P_cap_min
        self.c_tax = config.get('c-tax', 10)
        self.co2_option = config.get('c-option', 'strict')
        # =================== State / Action spaces ===================
        self.max_SMP = config.get('max-SMP', 1.0)
        self.min_SMP = config.get('min-SMP', 0.0)
        
    # =====================================================
    def reset(self, renewable, SMP, seed=None, mode="light"):
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
        
        self.step_count = 0
        if not (renewable.shape == SMP.shape):
            raise ValueError(f"renew-{renewable.shape} & smp-{SMP.shape} size mimatch")
        self.n_worker = renewable.shape[0]
        
        self.renewable = torch.as_tensor(renewable, dtype=torch.float32, device=self.device)
        self.SMP = torch.as_tensor(SMP, dtype=torch.float32, device=self.device)
        
        self.L_H2 = torch.full((self.n_worker,), 0.0, device=self.device)
        self.SOC = torch.full((self.n_worker,), 0.1*self.ESS_cap, device=self.device)

        self.state = torch.zeros((self.n_worker, 4), dtype=torch.float32, device=self.device)
        self.state[:, 0] = self.renewable[:, self.step_count] / self.scale
        self.state[:, 1] = (self.SMP[:, self.step_count] - self.min_SMP) / (self.max_SMP - self.min_SMP)
        self.state[:, 2] = self.SOC / self.ESS_cap
        self.state[:, 3] = self.L_H2 / self.LH2_cap
        if mode == "light": #saving memory
            horizon = 1
            self.track = 0
            self.SOC_profile = None
            self.L_H2_profile = None
        else:
            horizon = self.renewable.shape[1]
            self.track = 1
            self.SOC_profile = torch.zeros((self.n_worker, horizon+1), device=self.device)
            self.L_H2_profile = torch.zeros((self.n_worker, horizon+1), device=self.device)
            self.SOC_profile[:,0] = self.SOC
            self.L_H2_profile[:,0] = self.L_H2
            
        self.P_to_G = torch.zeros((self.n_worker, horizon), device=self.device)
        self.TAOM = torch.zeros((self.n_worker, horizon), device=self.device)
        self.H2_to_market = torch.zeros((self.n_worker, horizon), device=self.device)
        self.CO2_emit = torch.zeros_like(self.P_to_G)
        # error-checking part
        # self.LH2_util_list, self.LH2_list, self.LH2_util_scaled_list, self.H2_storage_list = [], [], [], []
        # self.ESS_action_list, self.ESS_action_scaled_list = [], []
        # self.AWE_action_list = []
        # self.split_list, self.split_list_scaled = [], []
        # self.LH2_prev, self.LH2_dh, self.LH2_ch = [], [], []
        # self.SOC_prev, self.SOC_after = [], []
        # self.P_to_G_list, self.renew_list, self.P_consum_list = [], [], []
        return self.state, {}

    # =====================================================
    def step(self, action):
        ESS_action = action[:, 0] * self.ESS_P_cap
        AWE_action = action[:, 1] * self.PEM_P_cap
        LH2_util = action[:, 2] * self.LH2_cap
        split = action[:, 3]
        
        # action before-scaled
        # self.LH2_util_list.append(LH2_util)
        # self.ESS_action_list.append(ESS_action.clone().detach())
        # self.split_list.append(split)

        # mass-balance
        #self.LH2_prev.append(self.L_H2)
        X_load = self.X_flow * self.P_X
        LH2_util = torch.minimum(LH2_util, self.L_H2)
        H_mis = X_load / self.P_X * self.X_H2 - LH2_util
        H_mis_pos_mask = (H_mis>0).float()
        #self.LH2_util_scaled_list.append(H_mis_pos_mask*LH2_util + (1-H_mis_pos_mask)*X_load / self.P_X * self.X_H2)
        self.L_H2 = self.L_H2-H_mis_pos_mask*LH2_util-(1-H_mis_pos_mask)*X_load / self.P_X * self.X_H2
        #self.LH2_dh.append(self.L_H2)
        H_mis = torch.clamp(H_mis, min = 0)
    
        # power-balance
        idx = self.step_count*self.track
        P_ptx = X_load
        ptx_H2 = H_mis * self.SP_H2
        ptx_CO2 = X_load / self.P_X * self.X_CO2
        P_consum = P_ptx + ptx_H2
        self.P_to_G[:, idx] = self.renewable[:,self.step_count] - P_consum
        #self.P_to_G_list_prev.append(self.P_to_G[:, idx])
        #self.renew_list.append(self.renewable[:,self.step_count])
        #self.P_consum_list.append(P_consum)
        
        #BESS action
        #self.SOC_prev.append(self.SOC.clone().detach())
        ESS_action = self.ESS_masking(ESS_action)
        if self.SOC_profile is not None:
            self.SOC_profile[:, idx+1] = self.SOC.clone().detach()

        #self.SOC_after.append(self.SOC.clone().detach())
        #self.ESS_action_scaled_list.append(ESS_action.clone().detach())
        self.P_to_G[:, idx] = self.P_to_G[:, idx] - ESS_action

        AWE_action = torch.clamp(AWE_action, min=0)
        AWE_action = torch.maximum(AWE_action, H_mis * self.SP_H2)
        H2_produce = AWE_action / self.SP_H2 - H_mis
        H2_produce = torch.clamp(H2_produce, min=0)
        H2_to_storage = (1 - split) * H2_produce
        max_storage = self.LH2_cap - self.L_H2
        H2_to_storage = torch.minimum(H2_to_storage, max_storage)
        self.L_H2 = self.L_H2 + H2_to_storage
        if self.L_H2_profile is not None:
            self.L_H2_profile[:,idx+1] = self.L_H2.clone().detach()
        #self.H2_storage_list.append(H2_to_storage)
        #self.LH2_ch.append(self.L_H2)
        H2_to_sell = H2_produce-H2_to_storage
        self.P_to_G[:, idx] = self.P_to_G[:, idx] -H2_produce * self.SP_H2 - H2_to_storage * (self.SPC_H2 - self.SP_H2)
        #self.P_to_G_list.append(self.P_to_G[:, idx].clone().detach())
        #self.AWE_action_list.append(H2_produce * self.SP_H2 + H2_to_storage * (self.SPC_H2 - self.SP_H2))
        H_con = H_mis + H2_to_storage + H2_to_sell
        self.TAOM[:, idx] = (10.11*0.012 + 0.0019*2.96 + 0.11*0.012 + 0.00029*0.33)*H_con
        
        if self.co2_option == 'strict':
            emit_mask = (self.P_to_G[:, idx] < 0).float()
            self.CO2_emit[:, idx] = -(self.P_to_G[:, idx]/1000*self.emission_factor*emit_mask)-ptx_CO2/1000
        else:
            self.CO2_emit[:, idx] = -self.P_to_G[:, idx]/1000*self.emission_factor-ptx_CO2/ 1000
        self.H2_to_market[:, idx] = H2_to_sell 
        reward = self.cost_calculation(ptx_CO2, idx)
        done = (self.step_count == self.renewable.shape[1] - 1)
        self.step_count += 1
        if not done:
            self.state[:, 0] = self.renewable[:, self.step_count] / self.scale
            self.state[:, 1] = (self.SMP[:, self.step_count] - self.min_SMP) / (self.max_SMP - self.min_SMP)
            self.state[:, 2] = self.SOC / self.ESS_cap
            self.state[:, 3] = self.L_H2 / self.LH2_cap

        return self.state, reward, self.CO2_emit[:, idx], done, idx

    # =====================================================
    def ESS_masking(self, ESS_action):
        ch_cond = ESS_action >= 0
        ovch_cond = (ESS_action * self.ESS_eff + self.SOC > self.ESS_cap * self.SOC_up)
        idx_ovch = torch.logical_and(ch_cond, ovch_cond)
        ESS_action[idx_ovch] = (self.ESS_cap * self.SOC_up - self.SOC[idx_ovch]) / self.ESS_eff
        self.SOC[idx_ovch] = self.SOC[idx_ovch] + ESS_action[idx_ovch] * self.ESS_eff

        idx_ch = torch.logical_and(ch_cond, ~ovch_cond)
        self.SOC[idx_ch] = self.SOC[idx_ch] + ESS_action[idx_ch] * self.ESS_eff

        dch_cond = ESS_action < 0
        odch_cond = (ESS_action / self.ESS_eff + self.SOC < self.ESS_cap * self.SOC_lb)
        idx_odch = torch.logical_and(dch_cond, odch_cond)
        ESS_action[idx_odch] = (-self.SOC[idx_odch] + self.ESS_cap * self.SOC_lb) * self.ESS_eff
        self.SOC[idx_odch] = self.SOC[idx_odch] + ESS_action[idx_odch] / self.ESS_eff

        idx_dh = torch.logical_and(dch_cond, ~odch_cond)
        self.SOC[idx_dh] = self.SOC[idx_dh] + ESS_action[idx_dh] / self.ESS_eff
        
        self.SOC = self.SOC * (1 - self.self_dh)
        self.SOC = torch.clamp(self.SOC, min=0)
        return ESS_action

    # =====================================================
    def cost_calculation(self, ptx_CO2, idx):
        profit = (self.P_to_G[:, idx] * self.SMP[:, self.step_count] 
                  + self.H2_to_market[:, idx]*self.H2_price - self.TAOM[:, idx])
        taxing_mask = (self.P_to_G[:, idx] < 0).float()
        carbon_tax = -self.c_tax * self.emission_factor * self.P_to_G[:, idx] / 1000 * taxing_mask
        # print(f"P-to-G{self.P_to_G[:, idx]*self.SMP[:,self.step_count] }")
        # print(f"H2-to-market; {self.H2_to_market[:, idx]*self.H2_price}")
        # print(f"TAOM: {self.TAOM[:, idx]}")
        # print(f"C-tax: {carbon_tax}")
        return profit - carbon_tax

    def LCOX_calculation(self, mu_profit = None, var_profit = None):
        ii = torch.tensor(0.08, device=self.device)
        N = torch.tensor(25.0, device=self.device)
        CRF =  ii * ((ii+1) ** N) / ((ii+1) ** N - 1)
        CAP_gen = self.CAP_solar * self.scale * (1-self.fw) + self.CAP_wind * self.scale * self.fw        
        OPEX_gen = self.OPEX_solar * self.scale * (1-self.fw) + self.OPEX_wind * self.scale * self.fw
        CAP_hydrogen = self.CAP_H2*self.LH2_cap/1000
        CAP_electrolyzer = (self.PEM_P_cap)*self.CAP_PEM
        CAP_distillation  = self.distillation_cost()
        BESS_cos = self.ESS_cap*self.CAPEX_BESS
        CAP_total = CAP_gen + CAP_hydrogen + CAP_electrolyzer + CAP_distillation + BESS_cos  
        ptx_CO2 = torch.tensor(self.X_flow*self.X_CO2).to(dtype = torch.float32, device=self.device)
        C_ptx = 8600 * ptx_CO2 / 1000 * (
                    0.204 * (torch.log10(ptx_CO2 * 8.6)) ** 4 - 4.819 * (torch.log10(ptx_CO2 * 8.6)) ** 3 + 43.02 * (
                torch.log10(ptx_CO2 * 8.6)) ** 2 - 175.9 * (torch.log10(ptx_CO2 * 8.6)) + 1014.14 * self.C_CO2 / 1000 + 332.22)
        X_flow_total = self.X_flow*8600
        if mu_profit is not None and var_profit is not None:
            mu_OPEX = OPEX_gen + C_ptx - mu_profit/self.op_period*8600
            mu_LCOX = (mu_OPEX+CAP_total*CRF)/(X_flow_total/1000)
            var_LCOX = var_profit*((1/self.op_period*8600)/(X_flow_total/1000))**2 
            return mu_LCOX/1000, var_LCOX/1e6
        elif mu_profit is not None and var_profit is None:
            OPEX_total = OPEX_gen + C_ptx-mu_profit/self.op_period*8600
            return (OPEX_total + CAP_total*CRF)/(X_flow_total)
        else:
            raise ValueError(f"LCOX calculation impossible: mu_profit={type(mu_profit)}, var_profit={type(var_profit)}")

    def co2_emit_scale(self):
        emit_max = (self.X_flow * self.P_X + self.ESS_P_cap + self.PEM_P_cap) * self.emission_factor - self.X_flow * self.X_CO2
        emit_min = -self.X_flow * self.X_CO2
        return emit_min / 1000, emit_max / 1000 
    
    def mass_balance_error(self):
        LH2_prev, LH2_ch, LH2_dh = torch.stack(self.LH2_prev, dim = 1), torch.stack(self.LH2_ch, dim = 1), torch.stack(self.LH2_dh, dim=1)
        LH2_util_scaled = torch.stack(self.LH2_util_scaled_list, dim=1)
        error1 = torch.abs(LH2_dh-LH2_prev+LH2_util_scaled)
        
        H2_storage = torch.stack(self.H2_storage_list, dim = 1)
        error2 = torch.abs(LH2_dh + H2_storage - LH2_ch)
        return error1, error2
    
    def power_balance_error(self):
        SOC_prev, SOC_after = torch.stack(self.SOC_prev, dim = 1), torch.stack(self.SOC_after, dim = 1)/(1 - self.self_dh)
        ESS_action = torch.stack(self.ESS_action_scaled_list, dim=1)
        ch_mask = (ESS_action>=0).float()
        error1 = SOC_after-SOC_prev-ESS_action*(ch_mask*self.ESS_eff + (1-ch_mask)/self.ESS_eff)
        
        input_p = torch.stack(self.renew_list, dim = 1)
        p_to_g = torch.stack(self.P_to_G_list, dim = 1)
        p_consum = torch.stack(self.P_consum_list, dim = 1)
        awe = torch.stack(self.AWE_action_list, dim = 1)
        error2 = input_p-p_to_g-p_consum-awe-ESS_action
        
        return torch.abs(error1), torch.abs(error2)
    
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