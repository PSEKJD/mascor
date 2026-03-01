import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
import torch

class Dataset_global_solution(Dataset):
    def __init__(self, save_path, max_seq, flag='train',z_type = 'mv',scale=False):
        assert flag in ['train', 'test']
        self.scale = scale
        self.z_type = z_type
        self.max_seq = max_seq
        self.save_path = save_path
        self.rtg_scaler = StandardScaler()
        self.ctg_scaler = StandardScaler()
        self.reward_scaler = StandardScaler()
        self.co2_scaler = StandardScaler()
        self.__read_data_()
        
    def __read_data_(self):
        #Loading the oracle dataset
        with open(os.path.join(self.save_path, 'data_package.pkl'), "rb") as file: 
              data_package = pickle.load(file)
        print('Data-loading completed: {keys}'.format(keys = data_package.keys()))
        
        #trajectory info
        self.state = np.array(data_package['state-stack']) #sample x 576 x 4
        self.action = np.array(data_package['action-stack']) #sample x 576 x 4 ([-1,1;[0,1, 0,1]])
        self.reward = np.array(data_package['reward-stack']) #sample x 576, un-normalize
        self.co2 = np.array(data_package['co2-stack']) #sample x 576, un-normalize
        self.return_to_go = np.array(data_package['cum-reward-stack']) #sample x 576, un-normalize 
        self.co2_to_go = np.array(data_package['cum-co2-stack']) #sample x 576, un-normalize 
        self.co2_scale = np.array(data_package['co2-scale']) #sample X 2
 
        self.moment_est()
        self.state_dim = self.state.shape[-1]
        self.action_dim = self.action.shape[-1]
        
        #additional info
        self.converge = np.array(data_package['converge-idx']) # sample
        
        #replacing noise to trend data
        if self.z_type == 'default':
            self.noise = np.array(data_package['noise']) # sample x 205
        elif self.z_type =='mv':
            renew = self.state[:,:,0] #sample X 576
            window_size = 24
            renew_df = pd.DataFrame(renew)
            rolling_mean_renew = renew_df.rolling(window=window_size, axis=1).mean().values[:,23:] #first 24 is NaN
            self.noise = rolling_mean_renew[:,::24] #dowm-sample sample x 24
        elif self.z_type =='fft':
            raise RuntimeError("Error: {} type z token is not existed".format(self.z_type))
            
        self.noise_dim = self.noise.shape[-1]
        if 'c_fax_fix' in self.save_path:
            self.design = np.array(data_package['design-spec'])[:,:,1:] # sample x 576 x 5 (tax, LH2, ESS, PEM_ratio, X_flow), normalize
        else:
            self.design = np.array(data_package['design-spec'])
        self.des_dim = self.design.shape[-1]
            
        del data_package        
     
    def compute_to_go(self, reward):
        num_samples, horizon = reward.shape
        return_to_go = np.zeros_like(reward)

        for i in range(num_samples):
            discounted_reward = 0
            for t in reversed(range(horizon)):
                discounted_reward = reward[i, t] + discounted_reward  # discount = 1
                return_to_go[i, t] = discounted_reward

        return return_to_go
        
    def discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0]-1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
        return discount_cumsum
    
    def moment_est(self, save_option = True):
        self.reward_scaler.fit(self.reward.reshape(-1,1))
        self.co2_scaler.fit(self.co2.reshape(-1,1))
        self.rtg_scaler.fit(self.return_to_go.reshape(-1,1))
        self.ctg_scaler.fit(self.co2_to_go.reshape(-1,1))
        
        self.reward_norm = self.reward_scaler.transform(self.reward.reshape(-1,1)).reshape(-1,576)
        self.co2_norm = self.co2_scaler.transform(self.co2.reshape(-1,1)).reshape(-1,576)
        self.return_to_go_norm = self.rtg_scaler.transform(self.return_to_go.reshape(-1,1)).reshape(-1,576)
        self.co2_to_go_norm = self.ctg_scaler.transform(self.co2_to_go.reshape(-1,1)).reshape(-1,576)

        print("Scaler Statistics:")
        print(f"Reward Scaler - Mean: {self.reward_scaler.mean_[0]:.4f}, Std: {self.reward_scaler.scale_[0]:.4f}")
        print(f"CO2 Scaler - Mean: {self.co2_scaler.mean_[0]:.4f}, Std: {self.co2_scaler.scale_[0]:.4f}")
        print(f"RTG Scaler - Mean: {self.rtg_scaler.mean_[0]:.4f}, Std: {self.rtg_scaler.scale_[0]:.4f}")
        print(f"CTG Scaler - Mean: {self.ctg_scaler.mean_[0]:.4f}, Std: {self.ctg_scaler.scale_[0]:.4f}")
        
        if save_option:
            scaler_package = {}
            scaler_package['reward'] = self.reward_scaler
            scaler_package['co2'] = self.co2_scaler
            scaler_package['rtg'] = self.rtg_scaler
            scaler_package['ctg'] = self.ctg_scaler
        
            with open(os.path.join(self.save_path, 'scaler_package.pkl'), "wb") as file:
                pickle.dump(scaler_package, file)
        
    def __getitem__(self, index):
        
        ep_length = self.state.shape[1]
        ep_idx = index//24
        si = (index%24)*24
        
        if si+self.max_seq>ep_length: #padding
            s = np.concatenate([np.zeros((si+self.max_seq-ep_length, self.state_dim)), self.state[ep_idx,si:]], axis=0)
            a = np.concatenate([np.zeros((si+self.max_seq-ep_length, self.action_dim)), self.action[ep_idx,si:]], axis=0)
            r = np.concatenate([np.zeros((si+self.max_seq-ep_length, 1)), self.reward_norm[ep_idx,si:].reshape(-1,1)], axis=0)
            c = np.concatenate([np.zeros((si+self.max_seq-ep_length, 1)), self.co2_norm[ep_idx,si:].reshape(-1,1)], axis=0)
            rtg = np.concatenate([np.zeros((si+self.max_seq-ep_length, 1)), self.return_to_go_norm[ep_idx,si:].reshape(-1,1)], axis=0)
            ctg = np.concatenate([np.zeros((si+self.max_seq-ep_length, 1)), self.co2_to_go_norm[ep_idx,si:].reshape(-1,1)], axis=0)
            t = np.concatenate([np.zeros((si+self.max_seq-ep_length)), np.arange(si, ep_length)], axis=0)
            des = np.concatenate([np.zeros((si+self.max_seq-ep_length, self.des_dim)), self.design[ep_idx,si:]], axis=0)
            z = np.tile(self.noise[ep_idx][np.newaxis,:],(self.max_seq, 1))
            mask = np.concatenate([np.zeros((si+self.max_seq-ep_length)), np.ones((ep_length-si))], axis=0)
        else:
            s = self.state[ep_idx,si:si+self.max_seq]
            a = self.action[ep_idx,si:si+self.max_seq]
            r = self.reward_norm[ep_idx,si:si+self.max_seq].reshape(-1,1)
            c = self.co2_norm[ep_idx,si:si+self.max_seq].reshape(-1,1)
            rtg = self.return_to_go_norm[ep_idx,si:si+self.max_seq].reshape(-1,1)
            ctg = self.co2_to_go_norm[ep_idx,si:si+self.max_seq].reshape(-1,1)
            t = np.arange(si, si+self.max_seq)
            des = self.design[ep_idx,si:si+self.max_seq]
            z = np.tile(self.noise[ep_idx][np.newaxis,:],(self.max_seq, 1))
            mask = np.ones(shape = self.max_seq)
        return s, a, r, c, rtg, ctg, t,des, z, mask
    
    def valid_set(self, sample_size, device):
        val_idx = torch.randint(len(self.state), size=(sample_size,))
        # Evaluation step
        s = self.state[val_idx]
        a = self.action[val_idx]
        r = self.reward_norm[val_idx]
        c = self.co2_norm[val_idx]
        rtg = self.return_to_go_norm[val_idx]
        ctg = self.co2_to_go_norm[val_idx]
        t =  np.tile(np.arange(0, self.state.shape[1]), (sample_size,1))
        des = self.design[val_idx]
        z = np.tile(self.noise[val_idx][:,np.newaxis,:],(1,self.state.shape[1], 1))
        mask = np.ones(shape = (sample_size, self.state.shape[1]))
        
        s = torch.tensor(s, dtype = torch.float32, device = device).reshape(-1,self.max_seq,self.state_dim)
        a = torch.tensor(a, dtype = torch.float32, device = device).reshape(-1,self.max_seq,self.action_dim)
        r = torch.tensor(r, dtype = torch.float32, device = device).reshape(-1,self.max_seq,1)
        c = torch.tensor(c, dtype = torch.float32, device = device).reshape(-1,self.max_seq,1)
        rtg = torch.tensor(rtg, dtype = torch.float32, device = device).reshape(-1,self.max_seq,1)
        ctg = torch.tensor(ctg, dtype = torch.float32, device = device).reshape(-1,self.max_seq,1)
        t = torch.tensor(t, dtype = torch.long, device = device).reshape(-1,self.max_seq)
        des = torch.tensor(des, dtype = torch.float32, device = device).reshape(-1,self.max_seq, self.des_dim)
        z = torch.tensor(z, dtype = torch.float32, device = device).reshape(-1,self.max_seq, self.noise_dim)
        mask = torch.tensor(mask, dtype = torch.long, device = device).reshape(-1,self.max_seq)
        
        return s, a, r, c, rtg, ctg, t,des, z, mask
        
    def __len__(self):
        return self.state.shape[0]*(self.state.shape[1]//24) #49901*(576/24)