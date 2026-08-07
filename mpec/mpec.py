"""
Practical LP-follower reformulation for the multi-objective bilevel problem.

Upper level:
    minimize E[LCOX] and E[CO2]
    by choosing common design capacities x.

Lower level for each scenario:
    maximize operating profit over the full operating horizon.

For a continuous LP follower, lower-level optimality is imposed by:
    1) primal feasibility,
    2) dual feasibility (stationarity / nonnegative reduced costs), and
    3) strong duality.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import pyomo.environ as pyo

@dataclass(frozen=True)
class PlantConfig:
    # Operating parameter------------------------------------------------------------------------------------------------------------------------------------
    # Hydrogen buffer
    SP_H2: float = 55.7                 # specific power consumption for H2 production via PEMEC [kW/kgH2/h]
    SPC_H2: float = 55.7+3.03           # specific power consumption for H2 production and compression [kW/kgH2/h]
    L_H2_init: float = 0.0
    
    # BESS
    ESS_eff: float = 0.95               # discharge and charge efficiency
    self_dh: float = 0.05/(30.0*24.0) # self-discharge efficiency 
    SOC_lb: float = 0.1
    SOC_ub: float = 0.9
    SOC_init: float = 0.1
    
    # MeOH conversion unit
    X_H2: float = 0.19576               # specific H2 consumption for "X" production，[kgH2/s / kgX/s]
    X_CO2: float = 1.435802             # specific CO2 consumption for "X" production，[kgCO2/s / kgX/s]
    P_X: float = 0.65702                # specific power consumption for "X" production，kW/kg/h, X is methanol
    
    # Cost parameter------------------------------------------------------------------------------------------------------------------------------------
    # Renewable power
    CAP_solar: float = 740.0 # [$/kW]
    CAP_wind: float = 1250.0 # [$/kW]
    OPEX_solar: float = 12.6 # [$/kW]
    OPEX_wind: float = 25.0 # [$/kW]
    
    # Hydrogen buffer
    CAP_PEM: float = 600.0      # PEM cost [$/kW]
    CAP_H2: float = 751700.0    # $/tonne
    H2_price: float = 5         # H2 sale price [$/kg] (Ref: powermag.com/blog/hydrogen-prices-skyrocket-over-2021-amid-tight-power-and-gas-supply/)
    
    # BESS
    CAPEX_BESS: float = 236.5   # $/kW
    
    # MeOH conversion unit
    C_CO2: float = 50           # CO2 purchase cost [$/tonne]
    emission_factor: float = 0.5 #kg/kWh
    
    # Others
    ii: float = 0.08 # interest rate
    N: float = 25 # plant life, years
    CRF: float = ii*((ii+1)**N)/((ii+1)**N-1)
    material_cost: float =  (10.11*0.012 + 0.0019*2.96 + 0.11*0.012 + 0.00029*0.3) # $/kg of H2
    c_tax: float = 47.96 #$/ton
    annual_hours: float = 8600
    
    # Design range------------------------------------------------------------------------------------------------------------------------------------ 
    scale_min, scale_max = 5000, 25000 #kW
    scale: float = 50000 #kW
    op_period: float = 576
    X_flow_bounds: Tuple[float, float] = (scale_min/(P_X + X_H2*SP_H2), scale_max/(P_X + X_H2*SP_H2))
    LH2_cap_bounds: Tuple[float, float] = (scale_min/SP_H2, scale_max/SP_H2*4)
    ESS_cap_bounds: Tuple[float, float] = (scale_min, scale_max*4)

    PEM_P_cap_min = X_flow_bounds[0]*X_H2*SP_H2
    PEM_P_cap_max = LH2_cap_bounds[1]*SP_H2 + PEM_P_cap_min
    PEM_P_cap_bounds: Tuple[float, float] = (PEM_P_cap_min, PEM_P_cap_max)    
    
def prepare_input_data(
    renewable: np.ndarray,
    smp: np.ndarray,
    config: PlantConfig,
    scenario_limit: Optional[int] = None,
    hour_limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    ns, nt = renewable.shape
    if scenario_limit is not None:
        ns = min(ns, int(scenario_limit))
    if hour_limit is not None:
        nt = min(nt, int(hour_limit))
    if ns <= 0 or nt <= 0:
        raise ValueError("The selected scenario/hour subset is empty.")

    renewable = renewable[:ns, :nt].copy()
    smp = smp[:ns, :nt].copy()
    prob = np.full(ns, 1.0 / ns)
    return renewable, smp, prob

def build_model(renew: np.ndarray,
                smp: np.ndarray,
                config: Optional[PlantConfig] = None,
                scenario_limit: Optional[int] = None,
                hour_limit: Optional[int] = None) -> pyo.ConcreteModel:
    cfg = config or PlantConfig()
    ns, nt = renew.shape
    prob = np.full(ns, 1.0/ns)
    
    # Create a model
    m = pyo.ConcreteModel(name="LCOX_CO2_Bilevel_Strong_Duality")
    m._config = cfg
    m._n_scenarios = ns
    m._n_hours = nt
    m.S = pyo.RangeSet(0, ns - 1)
    m.T = pyo.RangeSet(0, nt - 1)
    m.prob = pyo.Param(m.S, initialize={s: float(prob[s]) for s in range(ns)})
    
    # renewable and grid
    m.renew = pyo.Param(m.S, m.T, initialize={(s, t): 
                        float(renew[s, t]) for s in range(ns) for t in range(nt)}, mutable=True)
    m.smp = pyo.Param(m.S, m.T, initialize={(s, t): 
                        float(smp[s, t]) for s in range(ns) for t in range(nt)}, mutable = True)
    
    # Variables------------------------------------------------------------------------------------------------------------------------------------ 
    # Upper variables: e-MeOH design
    m.X_flow = pyo.Var(bounds=cfg.X_flow_bounds) #kg
    m.LH2_cap = pyo.Var(bounds=cfg.LH2_cap_bounds) #kg
    m.ESS_cap = pyo.Var(bounds=cfg.ESS_cap_bounds) #kW
    m.PEM_P_cap = pyo.Var(domain = pyo.NonNegativeReals) #kW

    # Lower primal variables: e-MeOH operating
    m.G_to_P = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.P_to_G = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    
    m.ESS_ch = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.ESS_dh = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    
    m.PEM_X = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.PEM_storage_selling = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    
    m.LH2_util = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.H2_to_market = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    
    m.SOC = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.L_H2 = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)

    # Dual-variable of equality constraint
    m.lam_power = pyo.Var(m.S, m.T, domain=pyo.Reals)
    m.lam_soc = pyo.Var(m.S, m.T, domain=pyo.Reals)
    m.lam_h2_demand = pyo.Var(m.S, m.T, domain=pyo.Reals)
    m.lam_h2_balance = pyo.Var(m.S, m.T, domain=pyo.Reals)

    # Dual-variable of Inequality constraint
    m.mu_charge_cap = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_discharge_cap = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_soc_upper = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_soc_lower = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_h2_cap = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_h2_available = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_h2_market = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)
    m.mu_pem_cap = pyo.Var(m.S, m.T, domain=pyo.NonNegativeReals)

    # Aggregated var
    m.beta_x = pyo.Var(m.S,domain=pyo.Reals,)
    m.beta_ess = pyo.Var(m.S, domain=pyo.Reals,)
    m.beta_h2 = pyo.Var(m.S, domain=pyo.Reals,)
    m.beta_pem = pyo.Var(m.S, domain=pyo.Reals,)

    # Primal feasibility----------------------------------------------------------------------------------------------------------------------------
    # Primal equality constraint       
    def power_rule(mm, s, t):
        return (
            mm.X_flow*cfg.P_X
            + mm.PEM_X[s, t] + mm.PEM_storage_selling[s, t]
            + (mm.PEM_storage_selling[s, t]/cfg.SP_H2 - mm.H2_to_market[s, t])*(cfg.SPC_H2-cfg.SP_H2)
            + mm.ESS_ch[s, t]
            + mm.P_to_G[s, t]
            == mm.renew[s, t]
            + mm.ESS_dh[s,t]
            + mm.G_to_P[s,t])

    m.power_balance = pyo.Constraint(m.S, m.T, rule=power_rule)

    def soc_rule(mm, s, t):
        previous = (
            cfg.SOC_init*mm.ESS_cap
            if t == 0
            else mm.SOC[s, t - 1])
        return mm.SOC[s, t] == (1-cfg.self_dh)*(
            previous
            + cfg.ESS_eff*mm.ESS_ch[s, t]
            - mm.ESS_dh[s, t]/cfg.ESS_eff)

    m.soc_balance = pyo.Constraint(m.S, m.T, rule=soc_rule)
    
    m.h2_demand = pyo.Constraint(m.S, m.T,
                rule=lambda mm, s, t: (mm.X_flow*cfg.X_H2)
                - mm.PEM_X[s, t]/cfg.SP_H2
                - mm.LH2_util[s, t]
                == 0,)

    def h2_balance_rule(mm, s, t):
        previous = (
            cfg.L_H2_init*mm.LH2_cap
            if t == 0
            else mm.L_H2[s, t - 1])
        return (
            mm.L_H2[s, t] ==
            previous-mm.LH2_util[s, t]
            + mm.PEM_storage_selling[s, t]/cfg.SP_H2
            - mm.H2_to_market[s, t])

    m.h2_balance = pyo.Constraint(m.S, m.T, rule=h2_balance_rule)

    # Primal inequality constraint written as g(x,y) <= 0.
    m.charge_cap = pyo.Constraint(m.S, m.T,
                   rule=lambda mm, s, t: mm.ESS_ch[s, t]
                   - 0.3*mm.ESS_cap <= 0,)
    
    m.discharge_cap = pyo.Constraint(m.S, m.T,
                   rule=lambda mm, s, t: mm.ESS_dh[s, t]
                   - 0.3*mm.ESS_cap <= 0,)
    
    m.soc_upper = pyo.Constraint(m.S, m.T,
                  rule=lambda mm, s, t: mm.SOC[s, t]
                  - cfg.SOC_ub*mm.ESS_cap<= 0,)
    
    m.soc_lower = pyo.Constraint(m.S, m.T,
                  rule=lambda mm, s, t: 
                  cfg.SOC_lb*mm.ESS_cap
                  - mm.SOC[s, t]<= 0,)
        
    m.h2_cap = pyo.Constraint(m.S, m.T,
               rule=lambda mm, s, t: 
               mm.L_H2[s, t]-mm.LH2_cap<= 0,)

    def h2_available_rule(mm, s, t):
        available = (
            cfg.L_H2_init*mm.LH2_cap
            if t == 0
            else mm.L_H2[s, t - 1])
        return mm.LH2_util[s, t]-available <= 0
    
    m.h2_available = pyo.Constraint(m.S, m.T, rule=h2_available_rule)
    
    m.h2_market_limit = pyo.Constraint(m.S, m.T,
                        rule=lambda mm, s, t: mm.H2_to_market[s, t]
                        - mm.PEM_storage_selling[s, t]/cfg.SP_H2 
                        <= 0,)
    
    m.pem_cap = pyo.Constraint(m.S, m.T,
                rule=lambda mm, s, t: mm.PEM_X[s, t]
                + mm.PEM_storage_selling[s, t]
                - mm.PEM_P_cap
                <= 0,)
    
    m.pem_p_cap_lb = pyo.Constraint(
        expr=(m.PEM_P_cap
        >= m.X_flow*cfg.X_H2*cfg.SP_H2))

    m.pem_p_cap_ub = pyo.Constraint(
        expr=(m.PEM_P_cap
        <= m.X_flow*cfg.X_H2*cfg.SP_H2 + m.LH2_cap*cfg.SP_H2))
    
    # Dual feasibility----------------------------------------------------------------------------------------------------------------------------
    material_per_kwh = cfg.material_cost/cfg.SP_H2
    carbon_tax_per_kwh = cfg.c_tax*cfg.emission_factor/1000.0
    # For a minimization LP with y >= 0, each reduced cost must be >= 0.
    m.df_grid_import = pyo.Constraint(m.S, m.T,
                       rule=lambda mm, s, t: mm.smp[s, t]
                       + cfg.c_tax*cfg.emission_factor/1000.0 - mm.lam_power[s, t]>= 0,)
    
    m.df_grid_export = pyo.Constraint(m.S, m.T,
                       rule=lambda mm, s, t: -mm.smp[s, t] + mm.lam_power[s, t]>= 0,)
    
    m.df_charge = pyo.Constraint(m.S, m.T,
                  rule=lambda mm, s, t: mm.lam_power[s, t]
                  - (1-cfg.self_dh)*cfg.ESS_eff*mm.lam_soc[s, t]
                  + mm.mu_charge_cap[s, t] >= 0,)
    
    m.df_discharge = pyo.Constraint(m.S, m.T,
                     rule=lambda mm, s, t: - mm.lam_power[s, t]
                     + (1-cfg.self_dh)/cfg.ESS_eff*mm.lam_soc[s, t]
                     + mm.mu_discharge_cap[s, t] >= 0,)
    
    m.df_pem_x = pyo.Constraint(m.S, m.T,
                 rule=lambda mm, s, t: material_per_kwh
                 + mm.lam_power[s, t]
                 - mm.lam_h2_demand[s, t] / cfg.SP_H2
                 + mm.mu_pem_cap[s, t]>= 0,)
    
    m.df_pem_storage = pyo.Constraint(m.S, m.T,
                       rule=lambda mm, s, t: material_per_kwh
                       + (1 + (cfg.SPC_H2-cfg.SP_H2)/cfg.SP_H2)*mm.lam_power[s, t]
                       - mm.lam_h2_balance[s, t]/cfg.SP_H2
                       - mm.mu_h2_market[s, t] / cfg.SP_H2
                       + mm.mu_pem_cap[s, t]>= 0,)
    
    m.df_lh2_util = pyo.Constraint(m.S, m.T,
                    rule=lambda mm, s, t: -mm.lam_h2_demand[s, t]
                    + mm.lam_h2_balance[s, t]
                    + mm.mu_h2_available[s, t]>= 0,)
    
    m.df_h2_market = pyo.Constraint(m.S, m.T,
                     rule=lambda mm, s, t: -cfg.H2_price
                     - (cfg.SPC_H2-cfg.SP_H2)*mm.lam_power[s, t]
                     + mm.lam_h2_balance[s, t]
                     + mm.mu_h2_market[s, t]>= 0,)
    
    def df_soc_rule(mm, s, t):
        next_lam = ((1 - cfg.self_dh)*mm.lam_soc[s, t + 1]
        if t < nt - 1
        else 0.0)
        return (
            mm.lam_soc[s, t]
            - next_lam
            + mm.mu_soc_upper[s, t]
            - mm.mu_soc_lower[s, t]
            >= 0)

    m.df_soc = pyo.Constraint(m.S, m.T, rule=df_soc_rule,)

    def df_l_h2_rule(mm, s, t):
        next_lam = (mm.lam_h2_balance[s, t + 1]
        if t < nt - 1
        else 0.0)

        next_mu_available = (
            mm.mu_h2_available[s, t + 1]
            if t < nt - 1
            else 0.0)

        return (mm.lam_h2_balance[s, t]
                - next_lam
            + mm.mu_h2_cap[s, t]
            - next_mu_available
            >= 0)

    m.df_l_h2 = pyo.Constraint(m.S, m.T, rule=df_l_h2_rule,)
    # --------------------------- lower primal cost ---------------------------
    def lower_cost_rule(mm, s):
        return sum(
            (mm.G_to_P[s, t]-mm.P_to_G[s, t])*mm.smp[s, t]
            - cfg.H2_price*mm.H2_to_market[s, t]
            + material_per_kwh*(mm.PEM_X[s, t] + mm.PEM_storage_selling[s, t])
            + carbon_tax_per_kwh * mm.G_to_P[s, t]
            for t in mm.T)
    
    m.lower_cost = pyo.Expression(m.S, rule=lower_cost_rule)

    def beta_x_rule(mm, s):
        return mm.beta_x[s] == sum(cfg.P_X * mm.lam_power[s, t] + cfg.X_H2 * mm.lam_h2_demand[s, t] for t in mm.T)
    m.beta_x_definition = pyo.Constraint(m.S, rule=beta_x_rule,)

    def beta_ess_rule(mm, s):
        return mm.beta_ess[s] == (
                -(1 - cfg.self_dh)*cfg.SOC_init*mm.lam_soc[s, 0]
                - 0.3 * sum(mm.mu_charge_cap[s, t] for t in mm.T)
                - 0.3 * sum(mm.mu_discharge_cap[s, t] for t in mm.T)
                - cfg.SOC_ub * sum(mm.mu_soc_upper[s, t] for t in mm.T)
                + cfg.SOC_lb * sum(mm.mu_soc_lower[s, t] for t in mm.T)
                )
    m.beta_ess_definition = pyo.Constraint(m.S, rule=beta_ess_rule,)

    def beta_h2_rule(mm, s):
        return mm.beta_h2[s] == (
            -sum(mm.mu_h2_cap[s, t] for t in mm.T))
    m.beta_h2_definition = pyo.Constraint(m.S, rule=beta_h2_rule,)

    def beta_pem_rule(mm, s):
        return mm.beta_pem[s] == -sum(mm.mu_pem_cap[s, t] for t in mm.T)
    m.beta_pem_definition = pyo.Constraint(m.S, rule=beta_pem_rule,)

    def lower_dual_value_rule(mm, s):
        renewable_part = -sum(
            mm.renew[s, t]
            * mm.lam_power[s, t]
            for t in mm.T)
        return (mm.X_flow*mm.beta_x[s]
                + mm.ESS_cap*mm.beta_ess[s]
                + mm.LH2_cap*mm.beta_h2[s]
                + mm.PEM_P_cap*mm.beta_pem[s]
                + renewable_part)

    m.lower_dual_value = pyo.Expression(m.S, rule=lower_dual_value_rule,)  
    # Primal objective = dual objective enforces LP optimality.
    m.strong_duality = pyo.Constraint(m.S, rule=lambda mm, s: mm.lower_cost[s] == mm.lower_dual_value[s],)
    # ----------------------------- upper metrics -----------------------------  
    annual_factor = cfg.annual_hours / nt
    m.annualized_expected_profit = pyo.Expression(expr=-annual_factor*sum(m.prob[s]*m.lower_cost[s] for s in m.S))
    CAP_gen = cfg.CAP_wind*cfg.scale
    OPEX_gen = cfg.OPEX_wind*cfg.scale
    slope = 591.4612070635048
    intercept = 253253.24168586737
    m.distillation_CAPEX = pyo.Expression(expr=float(slope)*m.X_flow + float(intercept))
    m.total_capex = pyo.Expression(
                    expr=CAP_gen
                    + cfg.CAP_H2*m.LH2_cap/1000
                    + cfg.CAP_PEM*m.PEM_P_cap
                    + cfg.CAPEX_BESS*m.ESS_cap
                    + m.distillation_CAPEX)
    
    slope = 1304.7132364384352
    intercept = 57031.19199734369 
    m.c_ptx = pyo.Expression(expr=float(slope)*m.X_flow + float(intercept))
    m.annualized_total_cost = pyo.Expression(expr=cfg.CRF*m.total_capex
                                             + OPEX_gen + m.c_ptx 
                                             - m.annualized_expected_profit)
    
    m.scenario_annualized_profit = pyo.Expression(m.S, rule=lambda mm, s: -annual_factor * mm.lower_cost[s],)
    m.scenario_annualized_total_cost = pyo.Expression(m.S,
    rule=lambda mm, s: (
        cfg.CRF * mm.total_capex
        + OPEX_gen
        + mm.c_ptx
        - mm.scenario_annualized_profit[s]
    ),)    

    def scenario_co2_rule(mm, s):
        grid_tonne = sum((mm.G_to_P[s, t])*cfg.emission_factor/1000 for t in mm.T)
        consumed_tonne = mm.X_flow * cfg.X_CO2 * nt / 1000
        return (grid_tonne - consumed_tonne)

    m.scenario_month_CO2 = pyo.Expression(m.S, rule=scenario_co2_rule)
    m.expected_CO2 = pyo.Expression(
        expr=sum(m.prob[s]*m.scenario_month_CO2[s] for s in m.S)
    )

    # Pareto controls
    m.weight_lcox = pyo.Param(initialize=1.0, mutable=True)
    m.weight_co2 = pyo.Param(initialize=0.0, mutable=True)
    m.lcox_ref = pyo.Param(initialize=0.0, mutable=True)
    m.co2_ref = pyo.Param(initialize=0.0, mutable=True)
    m.lcox_scale = pyo.Param(initialize=1.0, mutable=True)
    m.co2_scale = pyo.Param(initialize=1.0, mutable=True)
    m.co2_epsilon = pyo.Param(initialize=0.0, mutable=True)

    print("Normalized obj is applied")
    m.objective = pyo.Objective(expr=m.annualized_total_cost, sense=pyo.minimize)
    m.co2_epsilon_constraint = pyo.Constraint(expr=m.expected_CO2 <= m.co2_epsilon)
    m.co2_epsilon_constraint.deactivate()
    return m

def acceptable(result) -> bool:
    return result.solver.termination_condition in {
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.feasible,
        pyo.TerminationCondition.maxTimeLimit,
    }

def summary(m: pyo.ConcreteModel) -> Dict[str, float]:
    return {
        "X-flow [kg/hr]": pyo.value(m.X_flow),
        "PEM_P-capacity [kW]": pyo.value(m.PEM_P_cap),
        "LH2-cap [kg]": pyo.value(m.LH2_cap),
        "BESS-cap [kWh]": pyo.value(m.ESS_cap),
        "expected_total_cost_per_year": pyo.value(m.annualized_total_cost),
        "expected_CO2_tonne_per_year": pyo.value(m.expected_CO2),
        }

def model_size(m: pyo.ConcreteModel) -> Dict[str, int]:
    return {
        "variables": sum(1 for _ in m.component_data_objects(pyo.Var, active=True)),
        "constraints": sum(
            1 for _ in m.component_data_objects(pyo.Constraint, active=True)
        ),
    }
