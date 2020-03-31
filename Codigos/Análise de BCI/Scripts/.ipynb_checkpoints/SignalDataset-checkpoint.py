import torch
import numpy as np
import os
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader

class SignalDataset(Dataset):
    def __init__(self):
        super(SignalDataset).__init__()
        self.samples = []
        self.initDataset()
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def read_data(self, filetitle):
        return np.loadtxt(filetitle)
    
    def initDataset(self):
        path = 'C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\Signals'
        frequencies = list(range(10, 20+1))
        sample = {
            'signal': None,
            'class': None
        }
        
        for frequency in frequencies:
            new_path = path+'\\'+str(frequency)
            files = os.listdir(new_path)
            for file in files:
                signal = self.read_data(new_path+'\\'+file)
                sample['signal'] = signal
                sample['class'] = frequency - 10
                self.samples.append(deepcopy(sample))
                