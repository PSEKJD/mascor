import os
import torch
import numpy as np
from mascor.utils import buffer
from botorch.utils.transforms import normalize
import math
import time
import torch.nn.functional as F
import time
from torch.utils.data import DataLoader
import torch

class UQ_Problem():
    def __init__(self, args, env_class, policy, dataset):
        self.args = args
        # Problem bounds & parameter setting
        self.scale_min = 5000  # 5MW
        self.scale_max = 25000  # 25MW
        self.P_X = 0.65702
        self.X_H2 = 0.19576
        self.SP_H2 = 55.7
        self.X_flow_range = np.array([self.scale_min / (self.P_X + self.X_H2 * self.SP_H2),
                                      self.scale_max / (self.P_X + self.X_H2 * self.SP_H2)])
        self.LH2_cap_range = np.array([self.scale_min / self.SP_H2, self.scale_max / self.SP_H2 * 4])
        self.ESS_cap_range = np.array([self.scale_min, self.scale_max * 4])
        self.PEM_ratio_range = np.array([0, 1])
        self.c_tax_range = np.array([0.10, 132.12])
        # c_tax in different country
        self.c_tax_list = {'France': 47.96, 'Denmark': 28.10, 'Germany': 48.39, 'Norway': 107.78}
        self.lb = torch.tensor(
            [self.c_tax_range[0], self.LH2_cap_range[0], self.ESS_cap_range[0], 0, self.X_flow_range[0]],
            dtype=torch.float64)
        self.ub = torch.tensor(
            [self.c_tax_range[1], self.LH2_cap_range[1], self.ESS_cap_range[1], 1, self.X_flow_range[1]],
            dtype=torch.float64)

        if 'c_fax_fix' in self.args.design_option:
            self.lb = self.lb[1:]  # c_tax is excluded
            self.ub = self.ub[1:]  # c_tax is excluded

        self.bounds = torch.stack((self.lb, self.ub)).to(device=self.args.device)
        self.normalize = normalize
        self.num_objectives = 2
        self.num_con = 1
        self.limit_state = 0
        self.prob_failure = args.prob_fail
        self.ref_point = torch.tensor(np.zeros(shape=self.num_objectives), dtype=torch.float64)

        # solver, buffer, env register
        self.policy = policy
        data_path = os.path.join('./dataset/', '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(
            country=args.target_country,
            region=args.region,
            option=args.design_option,
            sample=args.sample_size))
        self.buffer = buffer.RolloutBuffer(data_path, args.z_type, args.device)
        self.env_class = env_class
        self.infer_action = args.infer_action

        # weather-price min-max value
        self.weather_min = torch.tensor(dataset.weather_scale.data_min_, dtype=torch.float32, device=self.args.device)
        self.weather_max = torch.tensor(dataset.weather_scale.data_max_, dtype=torch.float32, device=self.args.device)
        self.price_min = torch.tensor(dataset.price_scale.data_min_, dtype=torch.float32, device=self.args.device)
        self.price_max = torch.tensor(dataset.price_scale.data_max_, dtype=torch.float32, device=self.args.device)

    @torch.no_grad()
    def planning(self, x, netG, dataset, backupdataset, episode = None, env = None, mode = "light"):
        x = x.clone().to(device = self.args.device)
        self.design_config_setting(x.cpu().detach().numpy().astype(np.float32))
        if episode is not None:
            noise, renewable, grid = episode
            renewable = self.args.env_config['scale'] * (renewable)
        else:
            print("Scenario-generation conducted") 
            noise, renewable, grid = self.scenario_sampling(netG, dataset, backupdataset)  # renew [0,1], grid [un-norm]
            renewable = self.args.env_config['scale'] * (renewable)
        if env is not None:
            states, _ = env.reset(renewable, grid, mode=mode)
        else:
            env = self.env_class(self.args.env_config)
            states, _ = env.reset(renewable, grid, mode=mode)
        del renewable, grid
        self.buffer._init(states, noise, normalize(x, self.bounds))
        done = False
        reward_list = []
        co2_list = []

        start = time.time()
        while not done:
            action = self.compute_actions()
            state, reward, co2, done, _ = env.step(action)
            reward_norm = (reward - self.buffer.reward_mu) / self.buffer.reward_std
            co2_norm = (co2 - self.buffer.co2_mu) / self.buffer.co2_std
            self.buffer.insert_data(a=action, r=reward_norm.reshape(-1, 1), co2=co2_norm.reshape(-1, 1))
            reward_list.append(reward.clone()) #clone must be used to avoid reference issue in list
            co2_list.append(co2.clone()) #clone must be used to avoid reference issue in list
            
            if self.policy.critic is None: #initial-good guess of RTG, CTG
                if env.step_count == 1: #env update step 0 --> 1 after first-env.step()
                    print("Initial RTG and CTG estimation")
                    rtg = torch.full((len(reward), 1), 5.0, device=self.args.device, dtype = torch.float32)
                    ctg = torch.full((len(co2), 1), -1.0, device=self.args.device, dtype=torch.float32)
                else:
                    rtg = next_rtg.clone().detach()
                    ctg = next_ctg.clone().detach()
            else:               
                rtg, ctg = self.compute_goals(env)
            self.buffer.insert_data(rtg=rtg.reshape(-1, 1), ctg=ctg.reshape(-1, 1))
            self.buffer.rolling_data(s=True, a=True, r=True, co2=True, rtg=True, ctg=True, t=True, mask=True)
            next_rtg = (rtg.reshape(-1, 1) * self.buffer.rtg_std + self.buffer.rtg_mu) - reward.reshape(-1, 1)
            next_rtg = (next_rtg - self.buffer.rtg_mu) / self.buffer.rtg_std
            next_ctg = (ctg.reshape(-1, 1) * self.buffer.ctg_std + self.buffer.ctg_mu) - co2.reshape(-1, 1)
            next_ctg = (next_ctg - self.buffer.ctg_mu) / self.buffer.ctg_std
            self.buffer.insert_data(s=state, mask=1, t=env.step_count, rtg=next_rtg, ctg=next_ctg)

            if env.step_count == env.renewable.shape[1]:
                end = time.time()
                reward_list = torch.stack(reward_list, dim=1)
                co2_list = torch.stack(co2_list, dim=1)
                LCOX_list = env.LCOX_calculation(mu_profit=reward_list.sum(dim=1))
                if len(co2_list) == 1:
                    return 0
                mu_LCOX, var_LCOX = LCOX_list.mean(), LCOX_list.var()
                ctg = co2_list.sum(dim=1)
                pfss = (ctg > self.limit_state).float().mean()
                mu_ctg, var_ctg = ctg.mean(), ctg.var()
                # print(f"step {i} at des {np.round(des.cpu().detach().numpy(),2)}:")
                # print(f"E[LCOX] = {mu_LCOX.item():.2f}, Var[LCOX] = {var_LCOX.item():.2f}, E[CTG] = {mu_ctg.item():.2f}, Var[CTG] = {var_ctg.item():.2f}, pfss = {pfss.item():.3f}, compute-time = {end-start:.2f}")

        return (LCOX_list.clone().detach().cpu().numpy(),
                co2_list.sum(dim=1).clone().detach().cpu().numpy(),
                mu_LCOX,
                mu_ctg,
                pfss,
                env
                )
    
    # @torch.no_grad()
    def compute_actions(self, batch_size=1000):
        batch_iter = math.ceil(self.args.scenario_size / batch_size)
        outs = []
        for j in range(batch_iter):
            with torch.inference_mode():
                s, a, r, co2, rtg, ctg, t, des, z, mask = self.buffer.batch_data(batch_size, j)
                action = self.policy.compute_actions(des, z, ctg, rtg, s, a, t, mask, mode=self.infer_action)
                outs.append(action)
        return torch.cat(outs, dim=0)

    # @torch.no_grad()
    def compute_goals(self, env, batch_size=1000):
        batch_iter = math.ceil(self.args.scenario_size / batch_size)
        mu_profit_list, std_profit_list = [], []
        mu_ctg_list, std_ctg_list = [], []
        pfss_list = []
        rtg_list, ctg_list = [], []
        
        for k in range(batch_iter):
            with torch.inference_mode():
                s, a, r, co2, rtg, ctg, t, des, z, mask = self.buffer.batch_data(batch_size, k)
                rtg, ctg = self.policy.compute_goals(des, z, s, a, co2, r, t, mask,
                                                         self.limit_state, self.buffer,
                                                         env.step_count, simcase="uq")
                rtg_list.append(rtg)
                ctg_list.append(ctg)

        rtg_list = torch.cat(rtg_list, dim=0)
        ctg_list = torch.cat(ctg_list, dim=0)

        if not mu_profit_list:
            return rtg_list, ctg_list
        else:
            mu_profit_list = torch.cat(mu_profit_list, dim=0)
            std_profit_list = torch.cat(std_profit_list, dim=0)
            mu_ctg_list = torch.cat(mu_ctg_list, dim=0)
            std_ctg_list = torch.cat(std_ctg_list, dim=0)
            pfss_list = torch.cat(pfss_list, dim=0)

            mu_total_profit = mu_profit_list.mean()
            var_total_profit = (std_profit_list ** 2 + mu_profit_list ** 2).mean() - mu_total_profit ** 2
            mu_LCOX, var_LCOX = env.LCOX_calculation(mu_total_profit, var_total_profit)
            mu_total_ctg = mu_ctg_list.mean()
            var_total_ctg = (std_ctg_list ** 2 + mu_ctg_list ** 2).mean() - mu_total_ctg ** 2
            pfss_total = pfss_list.mean()

            return mu_LCOX, var_LCOX, pfss_total, mu_total_ctg, var_total_ctg, rtg_list, ctg_list

    def design_config_setting(self, x):
        if 'c_fax_fix' in self.args.design_option:
            self.args.env_config['LH2-cap'] = x[0]
            self.args.env_config['ESS-cap'] = x[1]
            self.args.env_config['PEM-ratio'] = x[2]
            self.args.env_config['X-flow'] = x[3]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap'] * 0.1
        else:
            self.args.env_config['c-tax'] = x[0]
            self.args.env_config['LH2-cap'] = x[1]
            self.args.env_config['ESS-cap'] = x[2]
            self.args.env_config['PEM-ratio'] = x[3]
            self.args.env_config['X-flow'] = x[4]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap'] * 0.1

    def scenario_sampling(self, netG, dataset, backupdataset):
        with torch.inference_mode():
            if netG is not None:
                noise = torch.randn(self.args.scenario_size, 205, device=self.args.device)
                weather_scenario = netG(noise).reshape(-1, 24 * 24)
                weather_scenario = torch.clamp(weather_scenario, min=0)
                weather_scenario = weather_scenario * (self.weather_max - self.weather_min) + self.weather_min
            else:
                print("netG is None. Real historial dataset is collected")
                num_iter = int(self.args.scenario_size/500)
                batch_size = 500
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                it = iter(loader)
                weather_scenario = []
                for i in range(num_iter):
                    weather_data, _ = next(it)
                    weather_scenario.append(weather_data)
                weather_scenario = torch.cat(weather_scenario, dim=0).to(device = self.args.device)
                weather_scenario = torch.clamp(weather_scenario, min=0)
                weather_scenario = weather_scenario * (self.weather_max - self.weather_min) + self.weather_min
            wind_power_scenario = self.wind_power_function(weather_scenario)
            rolling_mean_renew = F.avg_pool1d(wind_power_scenario.unsqueeze(1), kernel_size=24, stride=1).squeeze(1)
            noise = rolling_mean_renew[:, ::24].contiguous()
            backup_si = np.random.randint(0, backupdataset.__len__(), size=self.args.scenario_size)
            price_np = np.array([backupdataset.price_scaled[idx:idx + dataset.max_seq, 0] for idx in backup_si])
            price_scenario = dataset.price_scale.inverse_transform(price_np)
            price_scenario = torch.tensor(price_scenario, dtype=torch.float32, device=self.args.device)
        return noise, wind_power_scenario, price_scenario

    def scenario_sampling_simple(self, netG, dataset, backupdataset):
        with torch.inference_mode():
            if netG is not None:
                noise = torch.randn(self.args.scenario_size, 205, device=self.args.device)
                weather_scenario = netG(noise).reshape(-1, 24 * 24)
                weather_scenario = torch.clamp(weather_scenario, min=0)
                weather_scenario = weather_scenario * (self.weather_max - self.weather_min) + self.weather_min
            else:
                si = np.random.randint(0, dataset.__len__(), size=self.args.scenario_size)
                weather_np = np.array([dataset.weather_data_scaled.reshape(-1,1)[idx:idx + dataset.max_seq, 0] for idx in si])
                weather_scenario = dataset.weather_scale.inverse_transform(weather_np)
                weather_scenario = torch.tensor(weather_scenario, dtype=torch.float32, device=self.args.device)
            
            wind_power_scenario = self.wind_power_function(weather_scenario)
            rolling_mean_renew = F.avg_pool1d(wind_power_scenario.unsqueeze(1), kernel_size=24, stride=1).squeeze(1)
            noise = rolling_mean_renew[:, ::24].contiguous()

            backup_si = np.random.randint(0, backupdataset.__len__(), size=self.args.scenario_size)
            price_np = np.array([backupdataset.price_scaled[idx:idx + dataset.max_seq, 0] for idx in backup_si])
            price_scenario = backupdataset.price_scale.inverse_transform(price_np)
            price_scenario = torch.tensor(price_scenario, dtype=torch.float32, device=self.args.device)
        return noise, wind_power_scenario, price_scenario

    def wind_power_function(self, wind_speed: torch.Tensor) -> torch.Tensor:
        factor = (80.0 / 50.0) ** (1.0 / 7.0)
        w = wind_speed * factor
        cutin, rated, cutoff = 1.5, 12.0, 25.0
        denom = (rated ** 3) - (cutin ** 3)
        p = (w ** 3 - cutin ** 3) / denom
        p = p.clamp_(0.0, 1.0)
        p = torch.where(w > cutoff, torch.zeros((), dtype=p.dtype, device=p.device), p)
        return p