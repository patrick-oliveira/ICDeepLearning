import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

def generate_time_series(frequency=11, size=512, mode='sin', theta = 0):
    if mode=='cos':
        return np.array([np.cos(2*np.pi*frequency*(val/size) + theta) for val in range(size)])
    return np.array([np.sin(2*np.pi*frequency*(val/size)) for val in range(size)])
    

def read_data(dataframe):
    return pd.read_csv(dataframe['path'], header = None, dtype = np.float64)