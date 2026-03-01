import os
import torch
import numpy as np
from mascor.solvers import ST_policy
from mascor.utils import buffer
from botorch.utils.transforms import normalize
import math
from pathlib import Path
class solver():
    def __init__(self, args, netG = None, netD = None, env_class = None):
        self.args = args
        self.netG = netG
        self.netD = netD
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
        self.ub = torch.tensor([self.c_tax_range[1], self.LH2_cap_range[1], self.ESS_cap_range[1], 1, self.X_flow_range[1]], dtype = torch.float64)
        
        if 'c_fax_fix' in self.args.design_option:
            self.lb = self.lb[1:] #c_tax is excluded
            self.ub = self.ub[1:] #c_tax is excluded

        self.device = self.args.device
        self.bounds = torch.stack((self.lb, self.ub)).to(device=self.device)
        self.normalize = normalize
        self.limit_state = 0
        self.prob_failure = args.prob_fail 
        self.candidate_num = args.candidate_num
        self.infer_action = args.infer_action
        
        #solver, buffer, env register
        self.policy = ST_policy.policy(self.args)
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATASET_DIR = str(REPO_ROOT / "dataset")
        data_path = os.path.join(DATASET_DIR, '{country}/{region}/oracle_dataset_{option}_sample_{sample}'.format(country = args.target_country, 
                                                                                                                  region = args.region,
                                                                                                                  option = args.design_option,
                                                                                                                  sample = args.sample_size))
        self.buffer = buffer.RolloutBuffer(data_path, args.z_type, args.device)
        self.env_class = env_class
        
    def planning(self, des, episode, save_option = True):
        self.design_config_setting(des.cpu().detach().numpy().astype(np.float64))
        wind_speed, grid, _ = episode
        renewable = self.args.env_config['scale']*(self.wind_power_function(wind_speed))
        #env-reset
        env = self.env_class(self.args.env_config)
        states, _ = env.reset(renewable, grid)
        del renewable, grid
        self.noise = torch.randn(self.candidate_num, 205, device = self.device, dtype=torch.float, requires_grad=False)
        states = torch.tensor(states[:4]).to(device=self.device, dtype=torch.float32).reshape(1, -1)
        self.buffer._init(states.repeat(self.candidate_num, 1), self.noise, normalize(des.to(self.device), self.bounds))
        done = False
        is_terminal_record = []
        
        #trajectory history
        if self.policy.critic is not None:
            pred_rtg_dist = np.zeros((self.args.op_period, 2), dtype=np.float32)
            pred_rtg = np.zeros((self.args.op_period,), dtype=np.float32)
            pred_ctg_dist = np.zeros((self.args.op_period, 2), dtype=np.float32)
            pred_ctg = np.zeros((self.args.op_period,), dtype=np.float32)
        reward_list = np.zeros((self.args.op_period,), dtype=np.float32)
        co2_list = np.zeros((self.args.op_period,), dtype=np.float32)
        while not done:
            action = self.compute_actions()
            action_np = action.clone().detach().cpu().numpy()
            if self.policy.critic is None: # Vanilla actor
                if env.step_count == 0: # Initial goal of rtg & ctg
                    rtg = torch.full((1, 1), 5.0, device=self.device, dtype = torch.float32)
                    ctg = torch.full((1, 1), -1.0, device=self.device, dtype=torch.float32)
                else:
                    rtg = next_rtg.clone().detach()
                    ctg = next_ctg.clone().detach()
            else: #Vannila actor-critic manner
                reward, co2 = env.fake_step(action_np, len(action_np))
                reward = torch.as_tensor(reward, device=self.device, dtype=torch.float32)
                co2 = torch.as_tensor(co2, device=self.device, dtype=torch.float32)
                # print(f"Fake step --> reward size: {reward.shape} & co2 size: {co2.shape}")
                reward_norm = (reward - self.buffer.reward_mu) / self.buffer.reward_std
                co2_norm = (co2 - self.buffer.co2_mu) / self.buffer.co2_std
                self.buffer.insert_data(a=action, r=reward_norm.reshape(-1, 1), co2=co2_norm.reshape(-1, 1))
                mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg = self.compute_goals()
                action, mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg = self.select_actions(action, mu_rtg, std_rtg,
                                                                                         mu_ctg, std_ctg, rtg, ctg)
            #real-env-step
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
            
            if self.policy.critic is not None:
                pred_rtg_dist[env.step_count-1, 0] = mu_rtg.item()
                pred_rtg_dist[env.step_count-1, 1] = std_rtg.item()
                pred_ctg_dist[env.step_count-1, 0] = mu_ctg.item()
                pred_ctg_dist[env.step_count-1, 1] = std_ctg.item()
                pred_rtg[env.step_count-1] = rtg.item()
                pred_ctg[env.step_count-1] = ctg.item()
            reward_list[env.step_count-1] = reward.item()
            co2_list[env.step_count-1] = co2.item()
            is_terminal_record.append(done)
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
        return env

    @torch.inference_mode()
    def compute_actions(self, batch_size=1000):
        batch_iter = math.ceil(self.args.candidate_num / batch_size)
        outs = []
        for j in range(batch_iter):
            s, a, _, _, rtg, ctg, t, des, z, mask = self.buffer.batch_data(batch_size, j)
            action = self.policy.compute_actions(des, z, ctg, rtg, s, a, t, mask, mode=self.infer_action)
            outs.append(action)
        return torch.cat(outs, dim=0)

    @torch.inference_mode()
    def compute_goals(self, batch_size = 1000):
        batch_iter = math.ceil(self.args.candidate_num/batch_size)
        mu_rtgs_list, std_rtgs_list = [], []
        mu_ctgs_list, std_ctgs_list = [], []
        if self.infer_action == "random":
            rtgs_list = []
            ctgs_list = []
        for k in range(batch_iter):
            s, a, r, co2, rtg, ctg, t, des, z, mask = self.buffer.batch_data(batch_size, k)
            mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg = self.policy.compute_goals(des, z, s, a, co2, r, t, mask, None, None, None, simcase="online")
            mu_rtgs_list.append(mu_rtg)
            std_rtgs_list.append(std_rtg)
            mu_ctgs_list.append(mu_ctg)
            std_ctgs_list.append(std_ctg)
            if self.infer_action == "random":
                rtgs_list.append(rtg)
                ctgs_list.append(ctg)
        mu_rtgs_list = torch.cat(mu_rtgs_list, dim=0)
        std_rtgs_list = torch.cat(std_rtgs_list, dim=0)
        mu_ctgs_list = torch.cat(mu_ctgs_list, dim=0)
        std_ctgs_list = torch.cat(std_ctgs_list, dim=0)
        if self.infer_action == "random":
            rtgs_list = torch.cat(rtgs_list, dim=0)
            ctgs_list = torch.cat(ctgs_list, dim=0)
        else:
            rtgs_list = mu_rtgs_list.clone()
            ctgs_list = mu_ctgs_list.clone()
        return mu_rtgs_list, std_rtgs_list, mu_ctgs_list, std_ctgs_list, rtgs_list, ctgs_list
    @torch.inference_mode()
    def select_actions(self, action, mu_rtg, std_rtg, mu_ctg, std_ctg, rtg, ctg):
        # print(f"ctg size: {ctg.shape}") #checking
        ctg_unnorm = ctg * self.buffer.ctg_std + self.buffer.ctg_mu
        feas_mask = ctg_unnorm < 0
        # print(f"feas-mask size: {feas_mask.shape}") #checking
        if not torch.any(feas_mask):
            idx = torch.argmin(ctg)
        else:
            feas_mask = feas_mask.squeeze(1)
            feas_idx = feas_mask.nonzero(as_tuple=False).squeeze(1)
            # print(f"feas-idx size: {feas_idx.shape}") #checking
            # print(f"rtg size: {rtg.shape}") #checking
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
    