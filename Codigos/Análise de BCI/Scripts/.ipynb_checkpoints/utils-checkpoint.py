import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

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
            else:
                print("Directory %s created successfully." % self.path)
        else:
            print("Directory %s loaded successfully." % self.path)
        
    
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
    
    
# class SSVEPDataset(Dataset):
#     """SSVEP Dataset"""
    
#     def __init__(self, csv_file, root_dir, transform = None):
#         """
#         Args:
#             csv_file(string): Path ot the csv file.
#             root_dir(string): Directory with all the images.
#             transform(callable, optional): Optional transform to be applied on a sample.
#         """
#         self.root_dir = root_dir
#         self.data = pd.read_csv(csv_file)
#         self.transform = transform
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         series = pd.read_csv(self.root_dir+'\\'+self.data.loc[idx]['path'], sep = ' ', usecols = range(16), names = range(1, 17), dtype = float)
        
#         series = torch.tensor([series.values], dtype = torch.float)
#         Class = self.data.loc[idx]['state'] - 1
        
#         for col in range(16):
#             x = series[0][:, col]
#             series[0][:, col] = (x - torch.max(x) + x - torch.min(x))/(torch.max(x) - torch.min(x))
        
#         sample = {'series' : series,
#                   'class'  : Class}
#         return sample
    
def batchVisualization(series = None, dataloader = None):
    if(dataloader != None):
        sample = iter(dataloader).next()
        series = sample['series'][0].squeeze(0)
    
    fig, ax = plt.subplots(nrows = 16, ncols = 1, figsize = (10, 50))

    for i, axis in enumerate(ax):
        axis.plot(range(512), series[:, i], color = 'k')
        axis.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0))
        
def createInputFigure(inp):
    """
    Creates a 512x512 matrix.
    input:
        input: a dictionary; {"series": 4 tensors, "class": list of size 4}
    """
    def increaseSize(X):
        output = X
        for i in range(31):
            output = torch.cat((output, X), 0)
        return output
    
    return { "series": torch.stack([increaseSize(X) for X in inp['series']]),
             "class": inp['class'] }


def ConvArithR1(i, k):
    """
        s = 1
        p = 0
    """
    o = (i - k) + 1
    return o

def ConvArithR2(i, k, p):
    """
        s = 1
    """
    o = (i - k) + 2*p + 1
    return o

def ConvArithR3(i, k):
    """
        s = 1
        k = 2*n + 1
        p = k//2 = n
    """
    o = i
    return o

def ConvArithR4(i, k):
    """
        p = k - 1
        s = 1
    """
    o = i + (k - 1)
    return o

def ConvArithR5(i, k, s): 
    """
        p = 0
    """
    o = (i - k)//s + 1
    return o

def ConvArithR6(i, k, p, s):
    o = (i + 2*p - k)//s + 1
    return o

def ConvArithP(i, k, s):
    o = (i - k)//s + 1
    return o
    

def ConvArithSelect(relationship, *args, **kwargs):
    functions = {
        'R1': ConvArithR1,
        'R2': ConvArithR2,
        'R3': ConvArithR3,
        'R4': ConvArithR4,
        'R5': ConvArithR5,
        'R6': ConvArithR6,
        'P' : ConvArithP
    }
    return functions[relationship](*args, **kwargs)


    
