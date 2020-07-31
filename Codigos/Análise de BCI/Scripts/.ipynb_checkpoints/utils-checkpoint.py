import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, utils, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler

class path():
    def __init__(self, directory):
        self.path = directory
        self.verify()
    
    def verify(self):
        if(not os.path.isdir(self.path)):
            try:
                os.mkdir(self.path)
            except:
                print("Creation of the directory %s failed." % self.path)
        
    
    def __add__(self, other, att = False):
        """
        type(other) = string or path
        """
        if(type(other) == str):
            new = self.path + other
        else:
            new = self.path + other.path
        
        if att:
            self.path = new
            self.verify()
            
        return new
    
    def __truediv__(self, other, att = False):
        """
        type(other) = string or path
        """
        if(type(other) == str):
            new = self.path + "/" + other
        else:
            new = self.path + "/" + other.path
        
        if att:
            self.path = new
            self.verify()
        
        return new
    
    def __str__(self):
        return self.path
    
    def __repr__(self):
        return self.path
    
    def __call__(self):
        return self.path

def generate_time_series(frequency=11, size=512, mode='sin', theta = 0):
    if mode=='cos':
        return np.array([np.cos(2*np.pi*frequency*(val/size) + theta) for val in range(size)])
    return np.array([np.sin(2*np.pi*frequency*(val/size)) for val in range(size)])
    

def read_data(dataframe):
    return pd.read_csv(dataframe['path'], header = None, dtype = np.float64)

def getData(figPath, image_size, train_split, validation_split, batch_size = 8, shuffle_dataset = True, random_seed = 42, workers = 1, ngpu = 1):
    dataset = datasets.ImageFolder(root = figPath,
                                  transform = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                  ]))

    class_names = dataset.classes

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    trainsplit = int(np.floor(train_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[trainsplit:], indices[:trainsplit]
    trainset_size = len(train_indices)
    valsplit = int(np.floor(validation_split * trainset_size))
    train_indices, val_indices = train_indices[valsplit:], train_indices[:valsplit]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(train_indices)

    train_loader = DataLoader(dataset, batch_size = batch_size,
                                        sampler = train_sampler)
    validation_loader = DataLoader(dataset, batch_size = batch_size,
                                        sampler = valid_sampler)
    test_loader = DataLoader(dataset, batch_size = batch_size,
                                        sampler = test_sampler)


    dataloaders = {"train": train_loader, "val": validation_loader, "test": test_loader}

    dataset_sizes = {"train": len(train_indices),
                     "val": len(val_indices),
                     "test": len(test_indices)}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                             shuffle = True, num_workers = workers)

    
    return dataloaders, dataset_sizes, dataloader, class_names