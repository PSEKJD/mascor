import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import random
from pathlib import Path
random.seed(2024)

class Dataset(Dataset):
    def __init__(self, country, region, uni_seq, max_seq, data_type, flag='train'):
        
        assert flag in ['train', 'test']
        assert data_type in ['wind', 'wind-ele']
        self.data_type = data_type
        self.uni_seq = uni_seq
        self.max_seq = max_seq
        self.country = country
        self.region = region
        
        if "ele" in self.data_type:
            self.weather_scale = MinMaxScaler()
            self.price_scale = MinMaxScaler()
        else:
            self.weather_scale = MinMaxScaler()
        
        self.__read_data_('train') #for scaler fitting
        self.__read_data_(flag)
    
    def __read_data_(self, flag):
        REPO_ROOT = Path(__file__).resolve().parents[2]
        DATASET_DIR = str(REPO_ROOT / "dataset")
        #Loading the dataset
        print('Data loading of {region} in {country}'.format(region = self.region, country = self.country))
        print('//' * 100)
        weather_path = os.path.join(DATASET_DIR, '{country}/{region}/weather_unique_data.npy'.format(country = self.country, 
                                                                                                              region = self.region))
        grid_path = os.path.join(DATASET_DIR, 'european_electricity_price_data_hourly/{country}.csv'.format(country = self.country))
        
        self.weather_data = np.load(weather_path)
        self.demand_data = pd.read_csv(grid_path)
        self.price_data = np.array(self.demand_data['Price (EUR/MWhe)'].astype(str).values.tolist(), dtype=float)
        self.price_data = self.price_data[:87668]/1000*1.03 #EUR/MWhe --> $/kWhe 
        self.price_data[np.where(self.price_data<0)] = 0
        self.time_data = np.array(self.demand_data['Datetime (UTC)'].astype(str).values.tolist(), dtype=object)
        self.time_data = self.time_data[:87668]
        
        self.index_list = []
        if flag == 'train':
            for target_year in range(2015, 2023):
                indices = [i for i, timestamp in enumerate(self.time_data) if int(timestamp[:4]) == target_year]
                self.index_list += indices
        else:
            for target_year in range(2023, 2025):
                indices = [i for i, timestamp in enumerate(self.time_data) if int(timestamp[:4]) == target_year]
                self.index_list += indices

        self.moment_est(flag)
    
    def moment_est(self, flag):
        
        if flag == 'train':
            self.weather_data_scaled = self.weather_scale.fit_transform(self.weather_data[:,self.index_list].reshape(-1,1))
            self.weather_data_scaled = self.weather_data_scaled.reshape(-1,len(self.index_list))
            
            if "ele" in self.data_type:
                self.price_scaled = self.price_scale.fit_transform(self.price_data[self.index_list].reshape(-1,1))
                print("Scaler Statistics:")
                print(f"Weather Scaler - Min: {self.weather_scale.data_min_[0]:.4f}, Max: {self.weather_scale.data_max_[0]:.4f}")
                print(f"Price Scaler - Min: {self.price_scale.data_min_[0]:.4f}, Max: {self.price_scale.data_max_[0]:.4f}")
            else:
                print("Scaler Statistics:")
                print(f"Weather Scaler - Min: {self.weather_scale.data_min_[0]:.4f}, Max: {self.weather_scale.data_max_[0]:.4f}")
        else:
            self.weather_data_scaled = self.weather_scale.transform(self.weather_data[:,self.index_list].reshape(-1,1))
            self.weather_data_scaled = self.weather_data_scaled.reshape(-1,len(self.index_list))
            
            if "ele" in self.data_type:
                self.price_scaled = self.price_scale.transform(self.price_data[self.index_list].reshape(-1,1))
            else:
                pass
        
    def __getitem__(self, index):
        
        scen_idx = index//self.scenario_len
        index = index%self.scenario_len

        si = (index)*self.uni_seq
        weather = self.weather_data_scaled[scen_idx,si:si+self.max_seq]
        
        if "ele" in self.data_type:
            price = self.price_scaled[si:si+self.max_seq]
            return weather, price
        else:
            return weather

    def __len__(self):
        self.scenario_len = (len(self.index_list)-self.max_seq)//self.uni_seq
        return self.scenario_len*self.weather_data_scaled.shape[0]