import pandas as pd
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class SSVEPDataset(Dataset):
    def __init__(self, csvDataFrame, signal_length = 512, repeat = False):
        """
        csvDataFrame(DataFrame): DataFrame containing the info of the dataset 
        """
        self.csvDataFrame  = csvDataFrame 
        self.samples = []
        self.targets = []
        self.signal_length = signal_length
        self.initDataset()
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
    def read_data(self, dataframe):
        return torch.tensor(self.normalize(pd.read_csv(dataframe['path'], names = range(1, 16+1), dtype = float).values.T), dtype = torch.float)

    def normalize(self, x):
        mean = np.mean(x, axis = 1)
        std  = np.std(x, axis = 1)
        normalized = (x - mean[:, np.newaxis]) / std[:, np.newaxis]
        return normalized
    
    def initDataset(self):
        nRows = len(self.csvDataFrame)
        for dfIndex in range(nRows):
            rowInfo = self.csvDataFrame.loc[dfIndex]
            
            series = self.read_data(rowInfo)
            
            if not torch.isnan(series[0][0]):
                self.targets.append(rowInfo['state'] - 1)
                self.samples.append(series.unsqueeze(0))