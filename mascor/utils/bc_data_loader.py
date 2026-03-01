import os
from torch.utils.data import Dataset
import pickle
import numpy as np

class Dataset_global_solution(Dataset):
    def __init__(self, save_path, obs_length=24, flatten = False):
        self.obs_length = obs_length
        self.save_path = save_path        
        self.flatten = flatten
        self.__read_data_()

    def __read_data_(self):
        #Loading the oracle dataset
        with open(os.path.join(self.save_path, 'data_package.pkl'), "rb") as file: 
            data_package = pickle.load(file)
        
        print('Data-loading completed: {keys}'.format(keys = data_package.keys()))
        
        #trajectory info
        self.state = np.array(data_package['state-stack']) #sample x 576 x 4
        self.action = np.array(data_package['action-stack']) #sample x 576 x 4 ([-1,1;[0,1, 0,1]])
        self.state_dim = self.state.shape[-1]
        self.action_dim = self.action.shape[-1]
        
        #additional info
        if 'c_fax_fix' in self.save_path:
            self.design = np.array(data_package['design-spec'])[:,:,1:] # sample x 576 x 5 (tax, LH2, ESS, PEM_ratio, X_flow), normalize
        else:
            self.design = np.array(data_package['design-spec'])
        self.des_dim = self.design.shape[-1]
            
        del data_package        
       
    def __getitem__(self, index):
        
        ep_idx = index//(self.state.shape[1] - self.obs_length)
        si = index % (self.state.shape[1] - self.obs_length)
 
        s = self.state[ep_idx,si:si+self.obs_length]
        action = self.action[ep_idx,si+self.obs_length-1]
        des = self.design[ep_idx,si+self.obs_length-1]
        
        if self.flatten:
            state = np.zeros(shape = (self.obs_length*2 + 2 + 4))
            state[:self.obs_length] = s[:,0]
            state[self.obs_length:self.obs_length*2] = s[:,1]
            state[self.obs_length*2:] = np.concatenate((s[-1,2:], des))
        else:
            state = np.zeros(shape = (2, self.obs_length + 2 + 4, 1))
            state[0,:self.obs_length,0] = s[:,0]
            state[1,:self.obs_length,0] = s[:,1]
            state[:,self.obs_length:,0] = np.concatenate((s[-1,2:], des))
            
        return state, action
    
    def __len__(self):
        return self.state.shape[0]*(self.state.shape[1]-self.obs_length) #49901*(576-obs_length)