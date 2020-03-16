import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SSVEPDataset(Dataset):
    def __init__(self, csvDataFrame, rootPath, repeat = False):
        """
        csvDataFrame(DataFrame): DataFrame containing the info of the dataset 
        rootPath(string): directory where the files are
        """
        self.rootPath = rootPath
        self.csvDataFrame  = csvDataFrame 
        self.samples = []
        self.initDataset(repeat)
        
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def read_data(self, fileName, path, ind):
        return self.normalize(torch.tensor(pd.read_csv(path/fileName,  sep = ' ', usecols = range(16), names = range(1, 17), dtype = float).values), alpha = 10**0)

    def normalize(self, x, alpha = 1):
        x = (x - torch.max(x) + x - torch.min(x))/(torch.max(x) - torch.min(x))
        return x*alpha
    
    def repeat(self, X):
        while X.shape[1] < 512:
            X = torch.cat([X, X], 1)

        return X
    
    def initDataset(self, repeat):
        nRows = len(self.csvDataFrame)
        for dfIndex in range(nRows):
            tempSample = {
                'series': None,
                'class' : None
            }
            rowInfo = self.csvDataFrame.loc[dfIndex]
            
            series = self.read_data(rowInfo['path'], self.rootPath, rowInfo.name)
            
            if not torch.isnan(series[0][0]):
                if repeat:
                    series = self.repeat(series)
                    ydimension = 512
                else:
                    ydimension = 16
                
                tempSample['series'] = series.reshape(1, 512, ydimension)
                tempSample['class'] = rowInfo['state']

                self.samples.append(tempSample)
            