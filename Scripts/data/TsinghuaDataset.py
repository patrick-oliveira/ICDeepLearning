from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
from itertools import product

class TsinghuaDataset(Dataset):
    '''
    
    Args:
        dataset (str):
        Project:
        sequence_length (int):
        classes (List[int]):
        individuals (List[int]):
    ''' 
    def __init__(self, dataset: str, Project, sequence_length: int, classes = list(range(1, 40 + 1)), individuals = None):
        self.classes = (lambda L: [str(c) for c in L])(classes)
        
        base_path = Project.tsinghua
        if dataset == 'benchmark':
            self.data_path = base_path / 'benchmark' / '{}'.format(sequence_length)
            if individuals == None:
                self.individuals = list(map(lambda k: str(k), list(range(1, 35 + 1))))
                self.individuals.remove('5') # This should be removed later
            else:
                self.individuals = individuals
            self.segments = [*range(int(1500/sequence_length))]
            self.blocks = range(1, 6 + 1)
        else:
            self.data_path = base_path / 'beta' / '{}'.format(sequence_length)
            if individuals == None:
                self.individuals = list(map(lambda k: str(k), list(range(1, 70 + 1))))
            self.segments = [*range(int(750/sequence_length))]
            self.blocks = range(1, 4 + 1)
            
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
        return torch.tensor(self.normalize(pd.read_csv(file_path, names = range(1, 64 + 1), dtype = float, sep = ' ').values.T), dtype = torch.float).unsqueeze(0)
        
    def init_dataset(self):
        file_names = ["SS{}_{}_{}.csv".format(i, b, s) for (i, b, s) in product(self.individuals, self.blocks, self.segments)]
        self.samples = [(self.read_data(self.data_path / c / file), int(c) - 1) for c, file in product(self.classes, file_names)]
        
class GroupedDataset(Dataset):
    def __init__(self, Project, sequence_length, classes = list(range(1, 40 + 1)), benchmark_subjects = None, beta_subjects = None):
        '''
        
        Args:
            Project:
            sequence_length:
            classes:
            benchmark_subjects:
            beta_subjects:
        '''
        self.classes = (lambda L: [str(c) for c in L])(classes)
        