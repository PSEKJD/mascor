import numpy as np
from pyomo.environ import * # Global solver
import math

class solver():
    def __init__(self , config):
        
        # Operating parameter------------------------------------------------------------------------------------------------------------------------------------
        # Hydrogen buffer
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
        self.H2_price = 5             # H2 sale price [$/kg] (Ref: powermag.com/blog/hydrogen-prices-skyrocket-over-2021-amid-tight-power-and-gas-supply/)
        
        # BESS
        self.CAPEX_BESS = 236.5         # $/kW
 
        # MeOH conversion unit
        self.C_CO2 = 50                 # CO2 purchase cost [$/tonne]
        self.emission_factor = 0.5 #kg/kWh
        
        # Design configuration range------------------------------------------------------------------------------------------------------------------------------------
        self.scale = config.get('scale', 50000) # [kW]
        self.op_period = config.get('op_period', 576) 
        self.X_flow = config.get('X-flow', 1000)
        self.X_flow_P_cap = self.X_flow * self.P_X #kWh 
        self.LH2_cap = config.get('LH2-cap', 400)
        self.ESS_cap = config.get('ESS-cap', 25000)
        self.ESS_P_cap = self.ESS_cap * 0.3
        PEM_P_cap_min = self.X_flow*self.X_H2*self.SP_H2
        PEM_P_cap_max = self.LH2_cap*self.SP_H2 + PEM_P_cap_min
        self.PEM_ratio = config.get('PEM-ratio', 1)
        self.PEM_P_cap = config.get('PEM-ratio', 1)*(PEM_P_cap_max-PEM_P_cap_min) + PEM_P_cap_min
        self.fw = config.get('fw', 0.5)
        self.c_tax = config.get('c-tax', 10) #$/ton        
        self.SOC_init = config.get('SOC-init', 0.0)
        self.L_H2_init = config.get('L-H2-init', 0.0)
        
        self.max_SMP = config.get('max-SMP', 1.0)
        self.min_SMP = config.get('min-SMP', 0.0)
        self.max_c_tax = config.get('max-c-tax', 1.0)
        self.min_c_tax = config.get('min-c-tax', 1.0)
        
    def solver_instance(self, renewable, SMP, option = True):
        np.random.seed(0)
        
        #Slicing renewable and grid
        self.renewable = renewable
        self.SMP = SMP
        
        #Origincal renewable and grid
        self.renewable_origin = renewable
        self.SMP_origin = SMP
        
        # Create a model
        self.model = ConcreteModel()
        self.model.Time = RangeSet(0, self.op_period-1)
        
        self.model.renewable = Param(self.model.Time, initialize={i: self.renewable[i] for i in range(len(self.renewable))}, mutable=False)
        self.model.SMP = Param(self.model.Time, initialize={i: self.SMP[i] for i in range(len(self.renewable))}, mutable=False)
        
        # Model parameters 
        # Cost parameters
        self.model.P_X = Param(initialize = self.P_X, mutable = False)      # specific power consumption for "X" production，kW/kg/h, X is methanol
        self.model.SP_H2 = Param(initialize = self.SP_H2, mutable = False)               # specific power consumption for H2 production，kW/kgH2/h
        self.model.SPC_H2 = Param(initialize = self.SPC_H2, mutable = False)         # specific power consumption for H2 production and compression，kW/kgH2/h
        self.model.X_H2 = Param(initialize = self.X_H2, mutable = False)         # specific H2 consumption for "X" production，kgH2/s / kgX/s, X is methanol
        self.model.X_CO2 = Param(initialize = self.X_CO2, mutable = False)       # specific CO2 consumption for "X" production，[kgCO2/s / kgX/s]
        self.model.H2_price = Param(initialize = self.H2_price, mutable = False)               # H2 sale price, $/kg of H2 (Ref: powermag.com/blog/hydrogen-prices-skyrocket-over-2021-amid-tight-power-and-gas-supply/)
        self.model.ESS_eff = Param(initialize = self.ESS_eff, mutable = False)
        self.model.c_tax = Param(initialize = self.c_tax, mutable = False)
        self.model.emission_factor = Param(initialize = self.emission_factor, mutable = False)
        self.model.SOC_lb = Param(initialize = self.SOC_lb, mutable = False)
        self.model.SOC_up = Param(initialize = self.SOC_up, mutable = False)
        self.model.self_dh = Param(initialize = self.self_dh, mutable = False)
        
        # Design specification
        self.model.X_flow = Param(initialize = self.X_flow, mutable = False)
        self.model.X_flow_P_cap = Param(initialize = self.X_flow_P_cap, mutable = False)
        self.model.H2_cap =  Param(initialize = self.LH2_cap, mutable = False)
        self.model.ESS_cap = Param(initialize = self.ESS_cap, mutable = False)
        self.model.ESS_P_cap = Param(initialize = self.ESS_P_cap,  mutable = False)
        self.model.PEM_P_cap = Param(initialize = self.PEM_P_cap,  mutable = False)
        
        self.model.SOC_init = Param(initialize = self.SOC_init)
        self.model.L_H2_init = Param(initialize = self.L_H2_init)
        
        # Model variables
        self.model.G_to_P = Var(self.model.Time, within=NonNegativeReals) #Renewable power transmission to grid is not permitted
        self.model.P_to_G = Var(self.model.Time, within=NonNegativeReals)
        self.model.binary_grid = Var(self.model.Time, within=Binary)

        self.model.ESS_ch = Var(self.model.Time, within=NonNegativeReals)
        self.model.binary_ch = Var(self.model.Time, within=Binary)

        self.model.PEM_X = Var(self.model.Time, within=NonNegativeReals) #only for X produciton
        self.model.PEM_storage_selling = Var(self.model.Time, within=NonNegativeReals) #only for storage and selling to market
        
        self.model.LH2_util = Var(self.model.Time, within=NonNegativeReals) #only for X produciton
        self.model.H2_to_market =  Var(self.model.Time, within=NonNegativeReals) #only for X produciton

        self.model.SOC = Var(self.model.Time, within=NonNegativeReals)
        self.model.L_H2 = Var(self.model.Time, within=NonNegativeReals)
        
        self.model.material_price = Var(self.model.Time, within=NonNegativeReals)
        
        # Model constraints
        self.model.ESS_balance = Constraint(self.model.Time, rule = self.ESS_balance)
        self.model.ch_capacity = Constraint(self.model.Time, rule = self.ch_capacity)
        self.model.ESS_capacity_up = Constraint(self.model.Time, rule = self.ESS_capacity_up)
        self.model.ESS_capacity_lb = Constraint(self.model.Time, rule = self.ESS_capacity_lb)
        self.model.H2_demand = Constraint(self.model.Time, rule = self.H2_demand)
        self.model.H2_balance = Constraint(self.model.Time, rule = self.H2_balance)
        self.model.H2_capacity = Constraint(self.model.Time, rule = self.H2_capacity)
        self.model.H2_util = Constraint(self.model.Time, rule = self.H2_util)
        self.model.H2_split = Constraint(self.model.Time, rule = self.H2_split)
        self.model.PEM_balance = Constraint(self.model.Time, rule = self.PEM_balance)
        self.model.Power_balance = Constraint(self.model.Time, rule = self.Power_balance)
        self.model.Material_price = Constraint(self.model.Time, rule = self.Material_price)
        if option:
            self.model.Negative_co2 = Constraint(rule = self.Negative_co2)
        else:
            pass
        #Objective
        self.model.obj = Objective(rule=self.obj_rule, sense=maximize)

    # Set of constraints 
    def ESS_balance(self, model, t):
        if t == 0:
            return model.SOC[t] == (model.SOC_init + (model.ESS_ch[t]*model.binary_ch[t]*model.ESS_eff - model.ESS_ch[t]*(1-model.binary_ch[t])/model.ESS_eff))*(1-model.self_dh)
        else:
            return model.SOC[t] == (model.SOC[t-1] + (model.ESS_ch[t]*model.binary_ch[t]*model.ESS_eff - model.ESS_ch[t]*(1-model.binary_ch[t])/model.ESS_eff))*(1-model.self_dh)

    def ch_capacity(self, model, t):
        return model.ESS_ch[t] <= model.ESS_P_cap

    def ESS_capacity_up(self, model, t):
        return model.SOC[t] <= model.ESS_cap*model.SOC_up
    
    def ESS_capacity_lb(self, model, t):
        return model.SOC[t] >= model.ESS_cap*model.SOC_lb

    def H2_demand(self, model, t):
        return model.X_flow_P_cap/model.P_X*model.X_H2 == model.PEM_X[t]/model.SP_H2 + model.LH2_util[t]
    
    def H2_balance(self, model, t):
        if t == 0:
            return model.L_H2[t] == model.L_H2_init-model.LH2_util[t] + model.PEM_storage_selling[t]/model.SP_H2-model.H2_to_market[t]
        else:
            return model.L_H2[t] == model.L_H2[t-1]-model.LH2_util[t] + model.PEM_storage_selling[t]/model.SP_H2-model.H2_to_market[t]

    def H2_capacity(self, model,t):
        return model.L_H2[t] <= model.H2_cap

    def H2_util(self, model, t):
        if t ==0:
            return model.LH2_util[t] <= model.L_H2_init
        else:
            return model.LH2_util[t] <= model.L_H2[t-1]

    def H2_split(self, model,t):
        return model.H2_to_market[t] <= model.PEM_storage_selling[t]/model.SP_H2

    def PEM_balance(self, model,t):
        return model.PEM_X[t] + model.PEM_storage_selling[t] <= model.PEM_P_cap

    def Power_balance(self, model,t):
        return model.X_flow_P_cap + (model.PEM_X[t] + model.PEM_storage_selling[t]) + (model.PEM_storage_selling[t]/model.SP_H2-model.H2_to_market[t])*(model.SPC_H2-model.SP_H2) + model.ESS_ch[t]*model.binary_ch[t] + model.P_to_G[t]*(1-model.binary_grid[t]) \
            == model.renewable[t] + model.ESS_ch[t]*(1-model.binary_ch[t]) + model.G_to_P[t]*model.binary_grid[t]
    
    def Material_price(self, model,t):
        return model.material_price[t] == (model.PEM_X[t] + model.PEM_storage_selling[t])/model.SP_H2*(10.11*0.012 + 0.0019*2.96 + 0.11*0.012 + 0.00029*0.3)
    
    def Negative_co2(self, model):
        return sum(model.G_to_P[t]*model.binary_grid[t]*model.emission_factor for t in model.Time) - model.X_flow*model.X_CO2*(self.op_period)<=0

    def obj_rule(self, model):
        return sum((model.P_to_G[t]*(1-model.binary_grid[t])-model.G_to_P[t]*model.binary_grid[t])*model.SMP[t] + model.H2_price*model.H2_to_market[t] - model.material_price[t] - (model.G_to_P[t]*model.binary_grid[t]*model.emission_factor)/1000*model.c_tax for t in model.Time)
    
    def solve_planning(self, algorithm = 'gurobi'):
        self.solver = SolverFactory(algorithm)  # Or another solver like 'glpk', 'scip', etc.
        self.results = self.solver.solve(self.model, tee=True)
        
        if (self.results.solver.status == SolverStatus.ok) and (self.results.solver.termination_condition == TerminationCondition.optimal):
            
            print("Optimal solution found")
            self.profit = self.model.obj.expr()            
            self.P_to_G = np.array([self.model.P_to_G[t].value for t in self.model.Time])*(1-np.array([self.model.binary_grid[t].value for t in self.model.Time]))  
            self.G_to_P = np.array([self.model.G_to_P[t].value for t in self.model.Time])*np.array([self.model.binary_grid[t].value for t in self.model.Time]) 
            self.SOC = np.array([self.model.SOC[t].value for t in self.model.Time])
            self.SOC = np.insert(self.SOC, 0, self.SOC_init)
            self.H2_to_market = np.array([self.model.H2_to_market[t].value for t in self.model.Time])
            self.L_H2 = np.array([self.model.L_H2[t].value for t in self.model.Time])
            self.L_H2 = np.insert(self.L_H2, 0, self.L_H2_init)
        
            #LCOX calculation
            ii = 0.08 # interest rate
            N = 25 # plant life, years
            ptx_CO2 = self.X_flow_P_cap/self.P_X*self.X_CO2         
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
            
            OPEX_total = OPEX_gen + C_ptx - self.profit/(self.op_period)*8600 #profit included c_tax
            X_flow_total = self.X_flow*8600
      
            if (OPEX_total + CAP_total*CRF)<0:
                self.production_cost = 0
            else:
                self.production_cost = (OPEX_total + CAP_total*CRF)/(X_flow_total/1000)
            # REP & Carbon reduction
            self.REP = (np.sum(self.renewable) - np.sum(self.P_to_G))/(np.sum(self.renewable) + np.sum(self.G_to_P-self.P_to_G))
            self.CO2_emit = np.sum(self.G_to_P)*self.emission_factor/1000 - self.X_flow*(self.op_period)*self.X_CO2/1000
        else:
            self.profit = np.nan 
            self.production_cost = np.nan
            self.REP = np.nan
            self.CO2_emit = np.nan
        
        return self.profit, self.production_cost, self.REP, self.CO2_emit
            
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