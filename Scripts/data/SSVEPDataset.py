from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class SSVEPDataset(Dataset):
    def __init__(self, Project, preprocessing, csvDataFrame, signal_length):
        """
        csvDataFrame(DataFrame): DataFrame containing the info of the dataset 
        """
        self.csvDataFrame  = csvDataFrame 
        self.samples = []
        self.targets = []
        self.signal_length = signal_length
        self.preprocessing = preprocessing
        self.project_info = Project
        self.initDataset()
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
    def read_data(self, dataframe):
        subject = str(dataframe['subject'])
        state = str(dataframe['state'])
        file = dataframe['file']
        path = (self.project_info.base_series_dir / self.preprocessing / self.signal_length / subject / state / file).__str__()
        return torch.tensor(self.normalize(pd.read_csv(path, names = range(1, 16+1), dtype = float).values.T), dtype = torch.float)

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
                self.targets.append(int(rowInfo['state']) - 1)
                self.samples.append(series.unsqueeze(0))
                
            