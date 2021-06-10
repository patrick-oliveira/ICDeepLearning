import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def Dataloader(Dataset, batch_size, validation_split = .2, shuffle_dataset = True, random_seed = 42):
    dataset_size = len(Dataset)
    indices = list(range(dataset_size))
    train_split = int(np.floor(validation_split*dataset_size))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    validation_indices, train_indices = indices[:train_split], indices[train_split:]
    
    train_sampler, validation_sampler = [SubsetRandomSampler(train_indices),
                                         SubsetRandomSampler(validation_indices)]
    
    train_loader, validation_loader = [DataLoader(Dataset, batch_size = batch_size, sampler = train_sampler),
                                       DataLoader(Dataset, batch_size = batch_size, sampler = validation_sampler)]
    
    dataloaders = {"train": train_loader, "val": validation_loader}
    datasets_sizes = {"train": len(train_indices), "val": len(validation_indices)}
    
    return dataloaders, datasets_sizes