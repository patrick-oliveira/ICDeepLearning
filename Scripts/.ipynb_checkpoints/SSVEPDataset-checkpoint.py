import pandas as pd
import torch
import numpy as n
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing

class SSVEPDataset(Dataset):
    def __init__(self, csvDataFrame, dataType = 512, repeat = False):
        """
        csvDataFrame(DataFrame): DataFrame containing the info of the dataset 
        """
        self.csvDataFrame  = csvDataFrame 
        self.samples = []
        self.targets = []
        self.initDataset(repeat, dataType)
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
    def read_data(self, dataframe):
        return torch.tensor(self.normalize(pd.read_csv(dataframe['path'], names = range(1, 16+1), dtype = float).values, alpha = 10**0), dtype = torch.float)

    def normalize(self, x, alpha = 1):
        x = preprocessing.normalize(x, norm = 'max', axis = 0)
        return x*alpha
    
    def repeat(self, X):
        while X.shape[1] < 512:
            X = torch.cat([X, X], 1)
        return X
    
    def initDataset(self, repeat, dataType):
        nRows = len(self.csvDataFrame)
        for dfIndex in range(nRows):
#             tempSample = {
#                 'series': None,
#                 'class' : None
#             }
            rowInfo = self.csvDataFrame.loc[dfIndex]
            
            series = self.read_data(rowInfo)
            
            if not torch.isnan(series[0][0]):
                if repeat:
                    series = self.repeat(series)
                    ydimension = dataType
                else:
                    ydimension = 16
                

                
#                 tempSample['series'] = series.reshape(1, 512, ydimension)
#                 tempSample['class'] = rowInfo['state'] - 1
#                 state = torch.tensor([0, 0, 0, 0, 0])
#                 state[rowInfo['state'] - 1] = 1
#                 tempSample['class'] = state
                self.targets.append(rowInfo['state'] - 1)
                self.samples.append(series.reshape(1, dataType, ydimension))
            
        
class SSVEPSignalDataset(Dataset):
    def __init__(self, csv_dataframe):
        super(SSVEPSignalDataset).__init__()
        self.samples = []
        self.initDataset(csv_dataframe)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def read_data(self, dataframe):
        return torch.tensor(self.normalize(pd.read_csv(dataframe['path'], header = None, dtype = float).values, alpha = 10**0)).T.squeeze()
    
    def normalize(self, x, alpha = 1):
        x = preprocessing.normalize(x, norm = 'max', axis = 0)
        return x*alpha
    
    def initDataset(self, csv_dataframe):
        nRows = len(csv_dataframe)
        for dfIndex in range(nRows):
            tempSample = {
                'signal': None,
                'class': None
            }
            rowInfo = csv_dataframe.loc[dfIndex]
            series = self.read_data(rowInfo)
            
            tempSample['signal'] = series
            tempSample['class'] = rowInfo['state'] - 1

            self.samples.append(deepcopy(tempSample))