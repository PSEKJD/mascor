import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import math

class RolloutBuffer(Dataset):
    def __init__(self, save_path, z_type = 'default', device = 'cuda:0'):
        self.save_path = save_path
        self.rtg_scaler = StandardScaler()
        self.ctg_scaler = StandardScaler()
        self.reward_scaler = StandardScaler()
        self.z_type = z_type
        self.device = torch.device(device)
        self.__read_data_()
        #self.clear()
        
    def __read_data_(self):
        #Loading the oracle dataset
        with open(os.path.join(self.save_path, 'scaler_package.pkl'), "rb") as file: 
              scaler_package = pickle.load(file)
        self.co2_scaler = scaler_package['co2']
        self.reward_scaler = scaler_package['reward']
        self.rtg_scaler = scaler_package['rtg']
        self.ctg_scaler = scaler_package['ctg']
        print("Scaler Statistics:")
        print(f"Reward Scaler - Mean: {self.reward_scaler.mean_[0]:.4f}, Std: {self.reward_scaler.scale_[0]:.4f}")
        print(f"CO2 Scaler - Mean: {self.co2_scaler.mean_[0]:.4f}, Std: {self.co2_scaler.scale_[0]:.4f}")
        print(f"RTG Scaler - Mean: {self.rtg_scaler.mean_[0]:.4f}, Std: {self.rtg_scaler.scale_[0]:.4f}")
        print(f"CTG Scaler - Mean: {self.ctg_scaler.mean_[0]:.4f}, Std: {self.ctg_scaler.scale_[0]:.4f}")
        self.state_dim = 4
        self.action_dim = 4
        
        if self.z_type == 'default':
            self.noise_dim = 205
        elif self.z_type == 'mv':
            self.noise_dim = 24
        elif self.z_type == 'fft':
            raise RuntimeError("Error: {} type z token is not existed".format(self.z_type))
        if 'c_fax_fix' in self.save_path:
            self.des_dim = 4
        else:
            self.des_dim = 5
        del scaler_package

        #extracting sta from scaler only for online-planning
        self.reward_mu = torch.tensor(self.reward_scaler.mean_, dtype=torch.float32, device=self.device)
        self.reward_std = torch.tensor(self.reward_scaler.scale_, dtype=torch.float32, device=self.device)
        self.co2_mu = torch.tensor(self.co2_scaler.mean_, dtype=torch.float32, device=self.device)
        self.co2_std = torch.tensor(self.co2_scaler.scale_, dtype=torch.float32, device=self.device)
        
        self.ctg_mu = torch.tensor(self.ctg_scaler.mean_, dtype=torch.float32, device=self.device)
        self.ctg_std = torch.tensor(self.ctg_scaler.scale_, dtype=torch.float32, device=self.device)
        self.rtg_mu = torch.tensor(self.rtg_scaler.mean_, dtype=torch.float32, device=self.device)
        self.rtg_std = torch.tensor(self.rtg_scaler.scale_, dtype=torch.float32, device=self.device)
       
    def clear(self):
        del self.state
        del self.action
        del self.reward
        del self.reward_norm
        del self.co2
        del self.return_to_go
        del self.return_to_go_norm
        del self.co2_to_go
        del self.co2_to_go_norm
        del self.co2_scale
        del self.noise
        del self.design
        
    def moment_est(self):
        self.reward_scaler.fit(self.reward.reshape(-1,1))
        self.rtg_scaler.fit(self.return_to_go.reshape(-1,1))
        self.ctg_scaler.fit(self.co2_to_go.reshape(-1,1))
        self.reward_norm = self.reward_scaler.transform(self.reward.reshape(-1,1)).reshape(-1,576)
        self.return_to_go_norm = self.rtg_scaler.transform(self.return_to_go.reshape(-1,1)).reshape(-1,576)
        self.co2_to_go_norm = self.ctg_scaler.transform(self.co2_to_go.reshape(-1,1)).reshape(-1,576)
        
        print("Scaler Statistics:")
        print(f"RTG Scaler - Mean: {self.rtg_scaler.mean_[0]:.4f}, Std: {self.rtg_scaler.scale_[0]:.4f}")
        print(f"CTG Scaler - Mean: {self.ctg_scaler.mean_[0]:.4f}, Std: {self.ctg_scaler.scale_[0]:.4f}")
    
    def _init(self, states, z, des):
        self.sample_n = len(states)
        self.max_seq = 24
        self.states = torch.zeros(size=(self.sample_n, self.max_seq, self.state_dim), dtype=torch.float32, device=self.device) # renew, grid, SOC, LH2
        self.states[:,-1,:] = states
        self.actions = torch.zeros(size=(self.sample_n, self.max_seq, self.action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(size=(self.sample_n, self.max_seq, 1), dtype=torch.float32, device=self.device)
        self.co2s = torch.zeros(size=(self.sample_n, self.max_seq, 1), dtype=torch.float32, device=self.device)
        self.rtgs = torch.zeros(size=(self.sample_n, self.max_seq, 1), dtype=torch.float32, device=self.device)
        self.ctgs = torch.zeros(size=(self.sample_n, self.max_seq, 1), dtype=torch.float32, device=self.device)
        self.timestamps = torch.zeros(size=(self.sample_n, self.max_seq), dtype=torch.long, device=self.device)
        self.z  = z.to(dtype=torch.float32, device=self.device).unsqueeze(1).repeat(1, self.max_seq, 1)
        self.des = des.to(dtype=torch.float32, device=self.device).repeat(self.sample_n, self.max_seq, 1)
        self.masks = torch.zeros(size=(self.sample_n, self.max_seq), dtype=torch.long, device=self.device)
        self.masks[:,-1] = 1

    def batch_data(self, batch_size, idx):
        batch_iter = math.ceil(len(self.states)/batch_size)
        si = idx*batch_size
        ei = (idx+1)*batch_size
        if si == batch_iter-1:
            return self.states[si:], self.actions[si:], self.rewards[si:], self.co2s[si:], self.rtgs[si:], self.ctgs[si:], self.timestamps[si:], self.des[si:], self.z[si:], self.masks[si:]
        else:
            return self.states[si:ei], self.actions[si:ei], self.rewards[si:ei], self.co2s[si:ei], self.rtgs[si:ei], self.ctgs[si:ei], self.timestamps[si:ei], self.des[si:ei], self.z[si:ei], self.masks[si:ei]
    
    def insert_data(self, s = None, a = None, r = None, co2 = None, rtg = None, ctg = None, t = None, mask = None):
        if s is not None:
            self.states[:,-1,:] = s 
        if a is not None:
            self.actions[:,-1,:] = a 
        if r is not None:
            self.rewards[:,-1,:] = r 
        if co2 is not None:
            self.co2s[:,-1,:] = co2 
        if rtg is not None:
            self.rtgs[:,-1,:] = rtg 
        if ctg is not None:
            self.ctgs[:,-1,:] = ctg 
        if t is not None:
            self.timestamps[:,-1] = t 
        if mask is not None:
            self.masks[:,-1] = torch.tensor(1).to(dtype = torch.long, device=self.device)
            
    def rolling_data(self, s = None, a = None, r = None, co2 = None, rtg = None, ctg = None, t = None, mask = None):
        if s is not None:
            self.states = torch.roll(self.states, dims = 1, shifts=-1)
        if a is not None:
            self.actions = torch.roll(self.actions, dims = 1, shifts=-1)
        if r is not None:
            self.rewards = torch.roll(self.rewards, dims = 1, shifts=-1)
        if co2 is not None:
            self.co2s = torch.roll(self.co2s, dims = 1, shifts=-1)
        if rtg is not None:
            self.rtgs = torch.roll(self.rtgs, dims = 1, shifts=-1)
        if ctg is not None:
            self.ctgs = torch.roll(self.ctgs, dims = 1, shifts=-1)
        if t is not None:
            self.timestamps = torch.roll(self.timestamps, dims = 1, shifts=-1)
        if mask is not None:
            self.masks = torch.roll(self.masks, dims = 1, shifts=-1)