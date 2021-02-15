from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from itertools import product

class BETADataset(Dataset):
    def __init__(self, Project, classes = list(range(1, 40 + 1)), individuals = list(range(1, 70 + 1))):
        self.data_path = Project.tsinghua_raw_dir / 'beta' / 'separated'
        self.samples = []
        self.targets = []
        self.classes = (lambda L: [str(c) for c in L])(classes)
        self.individuals = (lambda L: [str(i) for i in L])(individuals)
        self.init_dataset()
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.targets[idx]
    
#     def __getitem__(self, idx):
#         return self.read_data(self.samples[idx]), self.targets[idx]
    
    def normalize(self, X):
        mean = np.mean(X, axis = 1)
        std = np.std(X, axis = 1)
        return (X - mean[:, np.newaxis]) / std[:, np.newaxis]
        
    def read_data(self, file_path):
        return torch.tensor(self.normalize(pd.read_csv(file_path, names = range(1, 64 + 1), dtype = float, sep = ' ').values.T))
        
        
    def init_dataset(self):
        file_names = ["S{}_{}_{}.csv".format(i, b, p) for (i, b, p) in product(self.individuals, range(1, 4 + 1), range(1, 2 + 1))] # Define this correctly later
        for c in self.classes:
            for file in file_names:
                file_path = self.data_path / c / file
                self.samples.append(self.read_data(file_path))
                self.targets.append(int(c) - 1)
          
        
# In case it's not possible to load everything into memory
#     def init_dataset(self):
#         file_names = ["S{}_{}.csv".format(c, b) for (c, b) in product(self.classes, range(1, 4 + 1))]
#         for c in self.classes:
#             for file in file_names:
#                 file_path = self.data_path / c / file
#                 self.samples.append(file_path.__str__())
#                 self.targets.append(int(c) - 1)