from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from itertools import product

class BETADataset(Dataset):
    def __init__(self, Project, classes = list(range(1, 40 + 1)), individuals = list(range(1, 70 + 1))):
        base_path = Project.tsinghua
        self.data_path = base_path / 'beta' / '{}'.format(sequence_length)
        self.samples = []=
        self.classes = (lambda L: [str(c) for c in L])(classes)
        self.individuals = (lambda L: [str(i) for i in L])(individuals)
        self.segments = [*range(int(750/sequence_length))]
        self.init_dataset()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def normalize(self, X):
        mean = np.mean(X, axis = 1)
        std = np.std(X, axis = 1)
        return (X - mean[:, np.newaxis]) / std[:, np.newaxis]
        
    def read_data(self, file_path):
        return torch.tensor(self.normalize(pd.read_csv(file_path, names = range(1, 64 + 1), dtype = float, sep = ' ').values.T))
                
    def init_dataset(self):
        file_names = ["SS{}_{}_{}.csv".format(i, b, s) for (i, b, s) in product(self.individuals, range(1, 4 + 1), self.segments)]
        self.samples = [(self.read_data(self.data_path / c / file), int(c) - 1) for c, file in product(self.classes, file_names)]
