import torch
import numpy as np
import random
torch_seed = 2024
torch.manual_seed(torch_seed)
torch.cuda.manual_seed_all(torch_seed)
np.random.seed(torch_seed)  
random.seed(torch_seed)

def scenario_generation(model, sample_size, dataloader, args):
    #wind power
    if args.data_type == 'wind-ele':
        raise ValueError("{} type is not supported".format(args.data_type))
    noise = torch.randn(sample_size, 205, device = args.device)
    weather_scenario = model(noise).detach().cpu().numpy().reshape(-1,24*24)
    weather_scenario[np.where(weather_scenario<0)] = 0
    weather_scenario = dataloader.dataset.weather_scale.inverse_transform(weather_scenario)
    weather_scenario = weather_scenario.reshape(sample_size, 576)
    wind_power_scenario = wind_power_function(weather_scenario)
    del weather_scenario
    
    #price data
    epochs = max(round(sample_size/(len(dataloader)*dataloader.batch_size)), 1)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            if i == 0 and epoch == 0:
                price_scenario = data[1][:,:,0]
            else:
                price_scenario = torch.cat((price_scenario, data[1][:,:,0]), axis = 0)
    price_scenario = price_scenario[:sample_size].detach().cpu().numpy()
    price_scenario = dataloader.dataset.price_scale.inverse_transform(price_scenario)

    return noise.detach().cpu().numpy(), wind_power_scenario, price_scenario
    
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

def offline_data_processing(solver):

    # Action 1: ESS_actcion ~ [-1,1]
    ESS_ch = np.array([solver.model.ESS_ch[t].value for t in solver.model.Time])
    binary_ch = np.array([solver.model.binary_ch[t].value for t in solver.model.Time])
    ESS_ch[np.where(binary_ch == 0)] = -ESS_ch[np.where(binary_ch == 0)]
    ESS_action = ESS_ch/solver.model.ESS_P_cap.value
    ESS_action[np.where(ESS_action>1)] = 1
    ESS_action[np.where(ESS_action<-1)] = -1
    
    # Action 2: AWE_action ~[0,1]
    PEM_X = np.array([solver.model.PEM_X[t].value for t in solver.model.Time])
    PEM_storage_selling = np.array([solver.model.PEM_storage_selling[t].value for t in solver.model.Time])
    PEM_load = PEM_X  + PEM_storage_selling
    AWE_action = PEM_load/solver.model.PEM_P_cap.value
    AWE_action[np.where(AWE_action>1)] = 1
    AWE_action[np.where(AWE_action<0)] = 0
    
    # Action 3: LH2_util ~ [0,1]
    LH2_util = np.array([solver.model.LH2_util[t].value for t in solver.model.Time])
    LH2_util = LH2_util/solver.model.H2_cap.value
    LH2_util[np.where(LH2_util>1)] = 1
    LH2_util[np.where(LH2_util<0)] = 0
    
    # Action 4: split ~ [0,1]
    H2_to_market = np.array([solver.model.H2_to_market[t].value for t in solver.model.Time])
    PEM_storage_selling = np.array([solver.model.PEM_storage_selling[t].value for t in solver.model.Time])
    H2_storage_selling = PEM_storage_selling/solver.model.SP_H2.value
    split = H2_to_market/(H2_storage_selling + 0.000001)
    split[np.where(split>1)] = 1
    
    action = np.zeros(shape = (len(ESS_action), 4))
    action[:,0] = ESS_action
    action[:,1] = AWE_action
    action[:,2] =LH2_util
    action[:,3] = split
    
    # State renew, grid, SOC, LH2
    SOC_profile = np.array([solver.model.SOC[t].value for t in solver.model.Time])
    SOC_profile = np.insert(SOC_profile, 0, solver.model.SOC_init.value)/solver.model.ESS_cap.value
    SOC_profile[np.where(SOC_profile>1)] = 1

    L_H2_profile = np.array([solver.model.L_H2[t].value for t in solver.model.Time])
    L_H2_profile = np.insert(L_H2_profile, 0, solver.model.L_H2_init.value)/solver.model.H2_cap.value
    L_H2_profile[np.where(L_H2_profile>1)] = 1
    
    renewable = solver.renewable
    SMP = solver.SMP
    
    state = np.zeros(shape=(len(ESS_action),2 + 2 + 5,)) #renew, SMP, SOC, L_H2, X_flow, LH2_cap, ESS_cap, PEM_ratio, c_tax
    
    # Design range
    scale_min = 5000 # 5MW
    scale_max = 25000 # 25MW
    P_X = 0.65702 
    X_H2 = 0.19576
    SP_H2 = 55.7
    X_flow_range = np.array([scale_min/(P_X + X_H2*SP_H2), scale_max/(P_X + X_H2*SP_H2)])
    LH2_cap_range = np.array([scale_min/SP_H2, scale_max/SP_H2*4]) # 5MW to 100MW
    ESS_cap_range = np.array([scale_min, scale_max*4]) # 5MW to 100MW
    PEM_ratio_range = np.array([0, 1])
    c_tax_range = np.array([0.10, 132.12])
    
    # Grid price range
    max_SMP = solver.max_SMP
    min_SMP = solver.min_SMP
    
    # c_tax range
    min_c_tax = solver.min_c_tax
    max_c_tax = solver.max_c_tax
    
    for i in range(len(ESS_action)):
        state[i, 0] = renewable[i]/solver.scale
        state[i, 1] = (SMP[i]-min_SMP)/(max_SMP-min_SMP)
        state[i, 2] = SOC_profile[i]
        state[i, 3] = L_H2_profile[i]
        state[i, 4] = (solver.model.c_tax.value-min_c_tax)/(max_c_tax - min_c_tax)
        state[i, 5] = (solver.model.H2_cap.value-LH2_cap_range[0])/(LH2_cap_range[1]-LH2_cap_range[0])
        state[i, 6] = (solver.model.ESS_cap.value-ESS_cap_range[0])/(ESS_cap_range[1]-ESS_cap_range[0])
        state[i, 7] = solver.PEM_ratio
        state[i, 8] =(solver.model.X_flow.value-X_flow_range[0])/(X_flow_range[1]-X_flow_range[0])
    
    return state, action

def optimal_planning(config, renewable, SMP, state_list, action_list, solver, env_class):
    #Applying historical action & compare with solver
    #Below will be removed
    #config = env_config
    #SMP = grid_scenario[i]
    #solver = global_solver
    
    test_env = env_class(config)
    done = False
    obs = test_env.reset(renewable, SMP)[0]
    obs_record = []
    reward_record = []
    is_terminal_record = []
    co2_record = []
    
    for i in range(len(state_list)):
        obs_record.append(obs)
        action = action_list[i]
        obs, reward, co2_emit, done, _, _= test_env.step(action)
        reward_record.append(reward) #un-normalize
        co2_record.append(co2_emit) #un-normalize
        is_terminal_record.append(done)

    obs_record = np.array(obs_record)

    gamma = 1
    discounted_reward = 0
    cum_reward_record = []
    cum_co2_record = []
    for reward, is_terminal in zip(reversed(reward_record), reversed(is_terminal_record)):
        if is_terminal:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        cum_reward_record.insert(0, discounted_reward)
    
    cum_reward_record = np.array(cum_reward_record)
    
    #cum CO2
    for co2_emit, is_terminal in zip(reversed(co2_record), reversed(is_terminal_record)):
        if is_terminal:
            discounted_co2 = 0
        discounted_co2 = co2_emit + (1 * discounted_co2)
        cum_co2_record.insert(0, discounted_co2)
    cum_co2_record = np.array(cum_co2_record)

    error = obs_record-state_list
    SOC_profile = solver.SOC/test_env.ESS_cap
    LH2_profile = solver.L_H2/test_env.LH2_cap
    
    return error, cum_reward_record, np.array(reward_record), cum_co2_record, np.array(co2_record), test_env

    