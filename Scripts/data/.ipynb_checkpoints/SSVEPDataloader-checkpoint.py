import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

def SSVEPDataloader(SSVEPDataset, batchSize, validation_split = .2):
    shuffle_dataset = True
    random_seed = 42
    
    datasetSize = len(SSVEPDataset)
    indices = list(range(datasetSize))
    trainsplit = int(np.floor(validation_split * datasetSize))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    trainIndices, validIndices = indices[trainsplit:], indices[:trainsplit]
    trainSetSize = len(trainIndices)
    
    trainSampler, validSampler = [SubsetRandomSampler(trainIndices), 
                                  SubsetRandomSampler(validIndices)]
    
    trainLoader, validationLoader = [DataLoader(SSVEPDataset, batch_size = batchSize, sampler = trainSampler), 
                                    DataLoader(SSVEPDataset, batch_size = batchSize, sampler = validSampler)]
            
    dataloaders = {"train": trainLoader, 
                   "val": validationLoader}
    datasetsSizes = {"train": len(trainIndices),
                    "val": len(validIndices)}
    
    return dataloaders, datasetsSizes