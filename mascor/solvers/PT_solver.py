# Import libraries
import os
import torch
import numpy as np
from mascor.utils import buffer
from botorch.utils.transforms import normalize
import math
import time
import torch.nn.functional as F
import time
from pathlib import Path
class solver():
    def __init__(self, args, env_class, policy, dataset, netG=None, netD=None):
        self.args = args
        self.netG = netG
        self.netD = netD
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

        self.device = self.args.device
        self.bounds = torch.stack((self.lb, self.ub)).to(device=self.device)
        self.normalize = normalize
        self.num_objectives = 2
        self.num_con = 1
        self.limit_state = 0
        self.prob_failure = args.prob_fail
        self.ref_point = torch.tensor(np.zeros(shape=self.num_objectives), dtype=torch.float64)

        # solver, buffer, env register
        self.policy = policy
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATASET_DIR = str(REPO_ROOT / "dataset")
        data_path = os.path.join(DATASET_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(
            country=args.target_country,
            region=args.region,
            option=args.design_option,
            sample=args.sample_size))
        self.buffer = buffer.RolloutBuffer(data_path, args.z_type, args.device)
        self.env_class = env_class
        self.infer_step = args.infer_step
        self.infer_action = args.infer_action
        self.candidate_num = args.candidate_num
        self.d_lamb = args.d_lambda

        # weather-price min-max value
        self.weather_min = torch.tensor(dataset.weather_scale.data_min_, dtype=torch.float32, device=self.device)
        self.weather_max = torch.tensor(dataset.weather_scale.data_max_, dtype=torch.float32, device=self.device)
        self.price_min = torch.tensor(dataset.price_scale.data_min_, dtype=torch.float32, device=self.device)
        self.price_max = torch.tensor(dataset.price_scale.data_max_, dtype=torch.float32, device=self.device)

    def planning(self, des, episode, noise_infer=True, save_option=True):
        self.design_config_setting(des.cpu().detach().numpy().astype(np.float32))
        wind_speed, grid, wind_speed_scaled = episode
        self.wind_speed = wind_speed_scaled.to(self.device, dtype=torch.float32)
        self.wind_speed.requires_grad_(False)
        renewable = self.args.env_config['scale'] * (self.wind_power_function_np(wind_speed))

        # env-reset
        env = self.env_class(self.args.env_config)
        states, _ = env.reset(renewable, grid)
        del renewable, grid
        self.noise_init()
        z_new, _, _ = self.noise_inference(self.netG, self.netD, env, True)
        states = torch.tensor(states[:4]).to(device=self.device, dtype=torch.float32).reshape(1, -1)
        self.buffer._init(states.repeat(self.candidate_num, 1), z_new, normalize(des.to(self.device), self.bounds))
        done = False
        is_terminal_record = []

        # trajectory history
        pred_rtg_dist = np.zeros((self.args.op_period, 2), dtype=np.float32)
        pred_rtg = np.zeros((self.args.op_period,), dtype=np.float32)
        pred_ctg_dist = np.zeros((self.args.op_period, 2), dtype=np.float32)
        pred_ctg = np.zeros((self.args.op_period,), dtype=np.float32)
        reward_list = np.zeros((self.args.op_period,), dtype=np.float32)
        co2_list = np.zeros((self.args.op_period,), dtype=np.float32)
        wind_forecast = torch.zeros((self.args.op_period // self.infer_step, self.candidate_num, self.args.op_period),
                                    device=self.device)
        power_forecast = torch.zeros((self.args.op_period // self.infer_step, self.candidate_num, 553),
                                     device=self.device)

        while not done:
            start = time.time()
            action = self.compute_actions()
            # fake step for action selection
            action_np = action.clone().detach().cpu().numpy()
            reward, co2 = env.fake_step(action_np, len(action_np))
            reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
            co2 = torch.as_tensor(co2, device=self.device, dtype=torch.float32)
            reward_norm = (reward - self.buffer.reward_mu) / self.buffer.reward_std
            co2_norm = (co2 - self.buffer.co2_mu) / self.buffer.co2_std
            self.buffer.insert_data(a=action, r=reward_norm.reshape(-1, 1), co2=co2_norm.reshape(-1, 1))

            mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg = self.compute_goals()
            action, mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg = self.select_actions(action, mu_rtg, std_rtg, mu_ctg,
                                                                                     std_ctg, rtg, ctg)

            # real-env-step
            state, reward, co2, done, _, _ = env.step(action.clone().detach().cpu().numpy()[0])
            reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32).reshape(-1, 1)
            co2 = torch.as_tensor(co2, device=self.device, dtype=torch.float32).reshape(-1, 1)
            reward_norm = (reward - self.buffer.reward_mu) / self.buffer.reward_std
            co2_norm = (co2 - self.buffer.co2_mu) / self.buffer.co2_std

            # data-storage
            self.buffer.insert_data(a=action.repeat(self.candidate_num, 1),
                                    r=reward_norm.reshape(-1, 1).repeat(self.candidate_num, 1),
                                    co2=co2_norm.reshape(-1, 1).repeat(self.candidate_num, 1))
            self.buffer.insert_data(rtg=rtg.repeat(self.candidate_num, 1),
                                    ctg=ctg.repeat(self.candidate_num, 1))
            self.buffer.rolling_data(s=True, a=True, r=True, co2=True, rtg=True, ctg=True, t=True, mask=True)

            # update next-token
            next_rtg = (rtg.reshape(-1, 1) * self.buffer.rtg_std + self.buffer.rtg_mu) - reward.reshape(-1, 1)
            next_rtg = (next_rtg - self.buffer.rtg_mu) / self.buffer.rtg_std
            next_ctg = (ctg.reshape(-1, 1) * self.buffer.ctg_std + self.buffer.ctg_mu) - co2.reshape(-1, 1)
            next_ctg = (next_ctg - self.buffer.ctg_mu) / self.buffer.ctg_std
            state = torch.tensor(state[:4]).to(device=self.device, dtype=torch.float32).reshape(1, -1)
            self.buffer.insert_data(s=state, mask=1, t=env.step_count % 576, rtg=next_rtg, ctg=next_ctg)

            # Save trajectory history
            pred_rtg_dist[env.step_count-1, 0] = mu_rtg.item()
            pred_rtg_dist[env.step_count-1, 1] = std_rtg.item()
            pred_ctg_dist[env.step_count-1, 0] = mu_ctg.item()
            pred_ctg_dist[env.step_count-1, 1] = std_ctg.item()
            pred_rtg[env.step_count-1] = rtg.item()
            pred_ctg[env.step_count-1] = ctg.item()
            reward_list[env.step_count-1] = reward.item()
            co2_list[env.step_count-1] = co2.item()
            is_terminal_record.append(done)
            end = time.time()
            # noise-vector inference
            if noise_infer:
                if env.step_count % self.infer_step == 0 and env.step_count < env.renewable.shape[0]:
                    if self.args.z_token:
                        z_new, fake_profile, rolling_mean_renew = self.noise_inference(self.netG, self.netD, env,
                                                                                       noise_infer)
                        wind_forecast[env.step_count // self.infer_step] = fake_profile
                        power_forecast[env.step_count // self.infer_step] = rolling_mean_renew
                    else:
                        z_new = self.noise
                    self.buffer.z = z_new.unsqueeze(1).repeat(1, self.buffer.max_seq, 1)

        if env.step_count == len(env.renewable):
            if self.policy.critic is not None:
                env.pred_rtg_dist = pred_rtg_dist
                env.pred_rtg = pred_rtg
                env.pred_ctg_dist = pred_ctg_dist
                env.pred_ctg = pred_ctg
                env.reward_list = reward_list
                env.co2_list = co2_list

                real_rtg = self.goal_calculation(reward_list, is_terminal_record)
                real_ctg = self.goal_calculation(co2_list, is_terminal_record)
                real_rtg, real_ctg = real_rtg.reshape(-1,1), real_ctg.reshape(-1,1)
                real_rtg, real_ctg = self.buffer.rtg_scaler.transform(real_rtg), self.buffer.ctg_scaler.transform(real_ctg)
                env.rtg = real_rtg
                env.ctg = real_ctg
                env.wind_forecast = wind_forecast.detach().cpu().numpy()
                env.power_forcast = power_forecast.detach().cpu().numpy()
                env.real_wind_speed =  self.wind_speed.detach().cpu().numpy()
        return env

    def noise_init(self):
        self.noise = torch.randn(self.candidate_num, 205, device=self.device, dtype=torch.float).requires_grad_()
        self.mse_erorr = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([self.noise], lr=0.01)
        self.z_infer_num = 5000
        for p in self.netG.parameters(): p.requires_grad_(False)
        for p in self.netD.parameters(): p.requires_grad_(False)

    def noise_inference(self, netG, netD, env, noise_infer):
        mask = torch.ones((self.candidate_num, 1, 24 * 24), dtype=torch.bool, device=self.device)
        if noise_infer:
            si = env.step_count % 576
            mask[:, 0, si:] = False
        mask_img = mask.reshape(-1, 1, 24, 24)
        si = env.step_count - env.step_count % 576
        #soft-reset of noise
        # self.noise = (self.noise.detach() * 0.5 + 0.5 * torch.randn_like(self.noise))
        # self.noise.requires_grad_()
        # self.optimizer = torch.optim.Adam([self.noise], lr=0.01)
        real_img = self.wind_speed[si:si + 576].view(1, 1, 24, 24).expand(self.candidate_num, -1, -1, -1)
        if not noise_infer:
            wind_speed = self.wind_speed * (self.weather_max - self.weather_min) + self.weather_min
            real_renew = self.wind_power_function(wind_speed)
            rolling_mean_renew = F.avg_pool1d(real_renew.unsqueeze(1), kernel_size=24, stride=1).squeeze(1)
            z_real = rolling_mean_renew[:, ::24].contiguous()
            # print(f"Z-real shape: {z_real.shape}") #checking
            return z_real.repeat(self.candidate_num, -1), None, rolling_mean_renew
        else:
            pass
        for j in range(self.z_infer_num):
            self.optimizer.zero_grad()
            fake_img = netG(self.noise)
            # masking only historical
            fake_img_mask = fake_img[mask_img]
            real_img_mask = real_img[mask_img]
            if env.step_count == 0:
                break
            mse_loss = self.mse_erorr(fake_img_mask, real_img_mask)
            loss = mse_loss - self.d_lamb * torch.mean(netD(fake_img))
            loss.backward()
            self.optimizer.step()
            if mse_loss.item() < 1e-3:
                break
        if self.args.z_type == 'default':
            z_best = self.noise.clone()
            fake_profile = fake_img.detach().cpu().numpy()
            fake_profile = fake_profile.reshape(-1, 24 * 24)
        else:
            fake_profile = fake_img.view(fake_img.size(0), -1)
            fake_profile_inv = fake_profile * (self.weather_max - self.weather_min) + self.weather_min
            fake_renew = self.wind_power_function(fake_profile_inv)
            rolling_mean_renew = F.avg_pool1d(fake_renew.unsqueeze(1), kernel_size=24, stride=1).squeeze(1)
            z_best = rolling_mean_renew[:, ::24].contiguous()

        return z_best, fake_profile, rolling_mean_renew

    @torch.inference_mode()
    def compute_actions(self, batch_size=10000):
        batch_iter = math.ceil(self.args.candidate_num / batch_size)
        outs = []
        for j in range(batch_iter):
            s, a, _, _, rtg, ctg, t, des, z, mask = self.buffer.batch_data(batch_size, j)
            action = self.policy.compute_actions(des, z, ctg, rtg, s, a, t, mask, mode=self.infer_action)
            outs.append(action)
        return torch.cat(outs, dim=0)

    @torch.inference_mode()
    def compute_goals(self, batch_size=10000):
        batch_iter = math.ceil(self.args.candidate_num / batch_size)
        mu_rtgs_list, std_rtgs_list = [], []
        mu_ctgs_list, std_ctgs_list = [], []
        rtgs_list, ctgs_list = [], []

        for k in range(batch_iter):
            s, a, r, co2, _, _, t, des, _, mask = self.buffer.batch_data(batch_size, k)
            mu_rtgs, std_rtgs, mu_ctgs, std_ctgs, rtgs, ctgs = self.goal_uq(des, s, a, co2, r, t, mask)
            mu_rtgs_list.append(mu_rtgs)
            std_rtgs_list.append(std_rtgs)
            mu_ctgs_list.append(mu_ctgs)
            std_ctgs_list.append(std_ctgs)
            rtgs_list.append(rtgs)
            ctgs_list.append(ctgs)

        mu_rtgs_list = torch.cat(mu_rtgs_list, dim=0)
        std_rtgs_list = torch.cat(std_rtgs_list, dim=0)
        mu_ctgs_list = torch.cat(mu_ctgs_list, dim=0)
        std_ctgs_list = torch.cat(std_ctgs_list, dim=0)
        if self.infer_action == "mu":
            rtgs_list = mu_rtgs_list.clone()
            ctgs_list = mu_ctgs_list.clone()
        else:
            rtgs_list = torch.cat(rtgs_list, dim=0)
            ctgs_list = torch.cat(ctgs_list, dim=0)

        return mu_rtgs_list, std_rtgs_list, mu_ctgs_list, std_ctgs_list, rtgs_list, ctgs_list

    @torch.inference_mode()
    def goal_uq(self, des_batch, s_batch, a_batch, co2_batch, r_batch, t_batch, mask_batch,
                batch_size=10000):  # z_set, a -->rtg, ctg
        batch_iter = math.ceil(self.args.candidate_num * len(des_batch) / batch_size)
        # repeating token for UQ
        des_repeat = self.token_expansion(des_batch, dim=self.buffer.des_dim)  # candidate_num**2 by 24 by 4
        s_repeat = self.token_expansion(s_batch,
                                        dim=self.buffer.state_dim)  # candidate_num**2 by 24 by 4 [s[0], s[0], ...,s[0], s[1], s[1], ...s[1],..]
        a_repeat = self.token_expansion(a_batch, dim=self.buffer.action_dim)
        co2_repeat = self.token_expansion(co2_batch, dim=1)
        r_repeat = self.token_expansion(r_batch, dim=1)
        t_repeat = self.token_expansion(t_batch, dim=0)
        mask_repeat = self.token_expansion(mask_batch, dim=0)
        z_repeat = self.buffer.z.repeat((len(des_batch), 1, 1))

        mu_rtg_list, std_rtg_list = [], []
        mu_ctg_list, std_ctg_list = [], []

        for k in range(batch_iter):
            si = k * batch_size
            ei = (k + 1) * batch_size
            des, s, a, co2, r, t, mask, z = (des_repeat[si:ei], s_repeat[si:ei], a_repeat[si:ei], co2_repeat[si:ei],
                                             r_repeat[si:ei], t_repeat[si:ei], mask_repeat[si:ei], z_repeat[si:ei])
            result = self.policy.compute_goals(des, z, s, a, co2, r, t, mask, None, None, None, simcase="online")
            mu_rtg_list.append(result[0]), std_rtg_list.append(result[1])
            mu_ctg_list.append(result[2]), std_ctg_list.append(result[3])

        mu_rtg, std_rtg = torch.cat(mu_rtg_list, dim=0), torch.cat(std_rtg_list, dim=0)
        mu_ctg, std_ctg = torch.cat(mu_ctg_list, dim=0), torch.cat(std_ctg_list, dim=0)

        # goal-distribution
        mu_rtg, std_rtg = mu_rtg.reshape(-1, self.args.candidate_num, 1), std_rtg.reshape(-1, self.args.candidate_num,
                                                                                          1)
        mu_rtg_mix = mu_rtg.mean(dim=1)
        var_rtg_mix = (std_rtg ** 2 + (mu_rtg - mu_rtg_mix[:, None, :]) ** 2).mean(dim=1)
        std_rtg_mix = var_rtg_mix.sqrt()
        rtg_dist = torch.distributions.Normal(loc=mu_rtg_mix, scale=std_rtg_mix)
        rtgs = rtg_dist.sample(()).clamp(mu_rtg_mix - 1.645 * std_rtg_mix, mu_rtg_mix + 1.645 * std_rtg_mix)

        # mu_rtg, std_rtg: (self.args.candidate_num*len(des_batch)) by 1
        mu_ctg, std_ctg = mu_ctg.reshape(-1, self.args.candidate_num, 1), std_ctg.reshape(-1, self.args.candidate_num,
                                                                                          1)
        mu_ctg_mix = mu_ctg.mean(dim=1)
        var_ctg_mix = (std_ctg ** 2 + (mu_ctg - mu_ctg_mix[:, None, :]) ** 2).mean(dim=1)
        std_ctg_mix = var_ctg_mix.sqrt()
        ctg_dist = torch.distributions.Normal(loc=mu_ctg_mix, scale=std_ctg_mix)
        ctgs = ctg_dist.sample(()).clamp(mu_ctg_mix - 1.645 * std_ctg_mix, mu_ctg_mix + 1.645 * std_ctg_mix)

        return mu_rtg_mix, std_rtg_mix, mu_ctg_mix, std_ctg_mix, rtgs, ctgs

    @torch.inference_mode()
    def token_expansion(self, tensor, dim):
        if dim == 0:
            tensor_repeat = tensor.unsqueeze(1).repeat(1, self.args.candidate_num, 1)
            return tensor_repeat.view(-1, self.buffer.max_seq)
        else:
            tensor_repeat = tensor.unsqueeze(1).repeat(1, self.args.candidate_num, 1, 1)
            return tensor_repeat.view(-1, self.buffer.max_seq, dim)

    @torch.inference_mode()
    def select_actions(self, action, mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg):
        ctg_unnorm = ctg * self.buffer.ctg_std + self.buffer.ctg_mu
        feas_mask = ctg_unnorm < 0
        if not torch.any(feas_mask):
            idx = torch.argmin(ctg)
        else:
            feas_mask = feas_mask.squeeze(1)
            feas_idx = feas_mask.nonzero(as_tuple=False).squeeze(1)
            feas_rtg = rtg[feas_idx]
            idx_local = torch.argmax(feas_rtg)
            idx = feas_idx[idx_local]
        return (action[idx].reshape(-1, 4), mu_rtg[idx].reshape(-1, 1), std_rtg[idx].reshape(-1, 1),
                mu_ctg[idx].reshape(-1, 1), std_ctg[idx].reshape(-1, 1), rtg[idx].reshape(-1, 1),
                ctg[idx].reshape(-1, 1))

    def goal_calculation(self, reward_list, is_terminal_record):
        discounted_reward = 0
        cum_reward_record = []

        for reward, is_terminal in zip(reversed(reward_list), reversed(is_terminal_record)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (discounted_reward)
            cum_reward_record.insert(0, discounted_reward)

        cum_reward_record = np.array(cum_reward_record)
        return cum_reward_record

    def design_config_setting(self, x):
        if 'c_fax_fix' in self.args.design_option:
            self.args.env_config['LH2-cap'] = x[0]
            self.args.env_config['ESS-cap'] = x[1]
            self.args.env_config['PEM-ratio'] = x[2]
            self.args.env_config['X-flow'] = x[3]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap'] * 0.1
            self.args.env_config['L-H2-init'] = 0
        else:
            self.args.env_config['c-tax'] = x[0]
            self.args.env_config['LH2-cap'] = x[1]
            self.args.env_config['ESS-cap'] = x[2]
            self.args.env_config['PEM-ratio'] = x[3]
            self.args.env_config['X-flow'] = x[4]
            self.args.env_config['SOC-init'] = self.args.env_config['ESS-cap'] * 0.1
            self.args.env_config['L-H2-init'] = 0

    def wind_power_function(self, wind_speed: torch.Tensor) -> torch.Tensor:
        factor = (80.0 / 50.0) ** (1.0 / 7.0)
        w = wind_speed * factor
        cutin, rated, cutoff = 1.5, 12.0, 25.0
        denom = (rated ** 3) - (cutin ** 3)
        p = (w ** 3 - cutin ** 3) / denom
        p = p.clamp_(0.0, 1.0)
        p = torch.where(w > cutoff, torch.zeros((), dtype=p.dtype, device=p.device), p)
        return p

    @torch.inference_mode()
    def wind_power_function_np(self, Wind_speed):
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