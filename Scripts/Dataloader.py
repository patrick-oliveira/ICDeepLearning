import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def Dataloader(Dataset, batch_size = 4, validation_split = .2, train_split = .2):
    random_seed = 42
    
    dataset_size = len(Dataset)
    indices = list(range(dataset_size))
    trainsplit = int(np.floor(train_split * dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    train_indices, test_indices = indices[trainsplit:], indices[:trainsplit]
    trainset_size = len(train_indices)
    
    validsplit = int(np.floor(validation_split * trainset_size))
    train_indices, valid_indices = train_indices[validsplit:], train_indices[:validsplit]
    
    dataloaders = {
        'train': DataLoader(Dataset, batch_size = batch_size, sampler = SubsetRandomSampler(train_indices)),
        'val': DataLoader(Dataset, batch_size = batch_size, sampler = SubsetRandomSampler(valid_indices)),
        'test': DataLoader(Dataset, batch_size = batch_size, sampler = SubsetRandomSampler(test_indices))
    }
    
    dataset_sizes = {
        'train': len(train_indices),
        'val': len(valid_indices),
        'test': len(test_indices)
    }
    
    return dataloaders, dataset_sizes
    
    