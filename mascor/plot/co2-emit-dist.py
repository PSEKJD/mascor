import pickle
from utils.env.ptx_env_stack import *
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
#Loading dataset and preprocessing
country_list = {}    
country_list['France'] = {}
country_list['France']['Dunkirk'] = {}
country_list['Denmark'] = {}
country_list['Denmark']['Skive'] = {}
country_list['Denmark']['Fredericia'] = {}
country_list['Germany'] = {}
country_list['Germany']['Wunsiedel'] = {}
country_list['Germany']['Weener'] = {}
country_list['Norway'] = {}
country_list['Norway']['Porsgrunn'] = {}

data_package = {}

for target_country in country_list:
    for region in country_list[target_country]:
        log_dir =  os.path.join('D:/Na-research-dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}/optimization'.format(country = target_country, 
                                                                                                                          region = region,
                                                                                                                          option = 'c_fax_fix',
                                                                                                                          sample =  50000))
        with open(os.path.join(log_dir, "uq_results_random.pkl"), "rb") as f:
            env_list = pickle.load(f)
        print('Length of env at {}'.format(region), len(env_list))
        for i, env in enumerate(env_list):
            # Input
            renewable = env.renewable
            grid = env.SMP
            
            #des
            des = np.array([env.LH2_cap, env.ESS_cap, env.PEM_ratio, env.X_flow])
            
            # Profile
            SOC_profile = env.SOC_profile[:,1:]-env.SOC_profile[:,:-1]
            CH = SOC_profile*(SOC_profile>0)
            DH = -SOC_profile*(SOC_profile<0)
            L_H2_profile = env.L_H2_profile[:,1:]-env.L_H2_profile[:,:-1]
            H2_store = L_H2_profile*(L_H2_profile>0)*env.SPC_H2
            H2_sell = env.H2_to_market*env.SP_H2
            P_to_G = env.P_to_G*(env.P_to_G>0)
            G_to_P = -env.P_to_G*(env.P_to_G<0)
            CO2_emit = env.CO2_emit
            
            # Performance (LCOX)
            ii = 0.08 # interest rate
            N = 25 # plant life, years
            CRF =  ii * ((ii+1) ** N) / ((ii+1) ** N - 1)
            CAP_gen = env.CAP_solar * env.scale * (1-env.fw) + env.CAP_wind * env.scale * env.fw        
            OPEX_gen = env.OPEX_solar * env.scale * (1-env.fw) + env.OPEX_wind * env.scale * env.fw
            CAP_hydrogen = env.CAP_H2*env.LH2_cap/1000
            CAP_electrolyzer = (env.PEM_P_cap)*env.CAP_PEM
            CAP_distillation  = env.distillation_cost()
            BESS_cos = env.ESS_cap*env.CAPEX_BESS
            CAP_total = CAP_gen + CAP_hydrogen + CAP_electrolyzer + CAP_distillation + BESS_cos  
            ptx_CO2 = env.X_flow_P_cap/env.P_X*env.X_CO2        
            C_ptx = 8600 * ptx_CO2 / 1000 * (
                        0.204 * (math.log10(ptx_CO2 * 8.6)) ** 4 - 4.819 * (math.log10(ptx_CO2 * 8.6)) ** 3 + 43.02 * (
                    math.log10(ptx_CO2 * 8.6)) ** 2 - 175.9 * (math.log10(ptx_CO2 * 8.6)) + 1014.14 * env.C_CO2 / 1000 + 332.22)
            OPEX_total = OPEX_gen + C_ptx - np.sum(env.P_to_G*env.SMP+env.H2_to_market*env.H2_price-env.TAOM, axis = 1)/(env.op_period)*8600
            carbon_tax = (-np.sum(env.P_to_G*(env.P_to_G<0),axis=1)*env.emission_factor/1000*env.c_tax)/env.op_period*8600
            X_flow_total = env.X_flow*8600
            LCOX = (OPEX_total+carbon_tax+CAP_total*CRF)/(X_flow_total/1000)
            
            if region == 'Dunkirk':
                data_package['DU-{}'.format(i+1)] = {}
                data_package['DU-{}'.format(i+1)]['des'] = des
                data_package['DU-{}'.format(i+1)]['P'] = np.mean(renewable)
                data_package['DU-{}'.format(i+1)]['Δ(P)'] = np.mean(np.abs(renewable[:,1:]-renewable[:,:-1]))
                data_package['DU-{}'.format(i+1)]['Price'] = np.mean(grid)
                data_package['DU-{}'.format(i+1)]['Δ(Price)'] = np.mean(np.abs(grid[:,1:]-grid[:,:-1]))
                data_package['DU-{}'.format(i+1)]['CH'] = np.mean(CH)
                data_package['DU-{}'.format(i+1)]['DH'] = np.mean(DH)
                data_package['DU-{}'.format(i+1)]['HS'] = np.mean(H2_store)
                data_package['DU-{}'.format(i+1)]['HM'] = np.mean(H2_sell)
                data_package['DU-{}'.format(i+1)]['PtoG'] = np.mean(P_to_G)
                data_package['DU-{}'.format(i+1)]['GtoP'] = np.mean(G_to_P)
                data_package['DU-{}'.format(i+1)]['E[LCOX]'] = np.mean(LCOX)
                data_package['DU-{}'.format(i+1)]['Var[LCOX]'] = np.var(LCOX)
                data_package['DU-{}'.format(i+1)]['CO2-emit'] = np.sum(CO2_emit, axis = 1)
            else:
                if region == 'Skive':
                    short = 'SK-{}'.format(i+1)
                elif region == 'Fredericia':
                    short = 'FR-{}'.format(i+1)
                elif region == 'Wunsiedel':
                    short = 'WU-{}'.format(i+1)
                elif region == 'Weener':
                    short = 'WE-{}'.format(i+1)
                else:
                    short = 'PO-{}'.format(i+1)
                data_package[short] = {}
                data_package[short]['des'] = des
                data_package[short]['P'] = np.mean(renewable)
                data_package[short]['Δ(P)'] = np.mean(np.abs(renewable[:,1:]-renewable[:,:-1]))
                data_package[short]['Price'] = np.mean(grid)
                data_package[short]['Δ(Price)'] = np.mean(np.abs(grid[:,1:]-grid[:,:-1]))
                data_package[short]['CH'] = np.mean(CH)
                data_package[short]['DH'] = np.mean(DH)
                data_package[short]['HS'] = np.mean(H2_store)
                data_package[short]['HM'] = np.mean(H2_sell)
                data_package[short]['PtoG'] = np.mean(P_to_G)
                data_package[short]['GtoP'] = np.mean(G_to_P)
                data_package[short]['E[LCOX]'] = np.mean(LCOX)
                data_package[short]['Var[LCOX]'] = np.var(LCOX)
                data_package[short]['CO2-emit'] = np.sum(CO2_emit, axis = 1)

#%%
region_colors = {
    'Dunkirk': 'r',    # Red for Dunkirk (France)
    'Skive': 'g',      # Green for Skive (Denmark)
    'Fredericia': 'b', # Blue for Fredericia (Denmark)
    'Wunsiedel': 'y',  # Yellow for Wunsiedel (Germany)
    'Weener': 'c'      # Cyan for Weener (Germany)
}
plt.figure(figsize=(10, 6))
for target_country in country_list:
    for region in country_list[target_country]:
        short = None
        if region == 'Dunkirk':
            short = 'DU'
        elif region == 'Skive':
            short = 'SK'
        elif region == 'Fredericia':
            short = 'FR'
        elif region == 'Wunsiedel':
            short = 'WU'
        elif region == 'Weener':
            short = 'WE'
        elif region == 'Porsgrunn':
            break
        
        # Collect 'CO2-emit' values for the region
        co2_emit_values = np.array(data_package.get(f"{short}-1")['CO2-emit'])  # Just taking the first key as an example
        for key in data_package:
            if key.startswith(short):  # Loop through all entries for the given region
                co2_emit_values = np.concatenate([co2_emit_values, np.array(data_package[key]['CO2-emit'])], axis=0)

        # Plot KDE for the region
        sns.kdeplot(co2_emit_values, label=region, color=region_colors[region], shade=True)

# Customize the plot
#plt.title('Kernel Density Estimation of CO2 Emission by Region', fontsize=16)
plt.xlabel('CO2 Emission (ton/month)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Region', fontsize = 12)
plt.grid(True)
plt.savefig('utils_figure/co2-emit-dist.png', dpi = 300)
plt.show()
plt.close()