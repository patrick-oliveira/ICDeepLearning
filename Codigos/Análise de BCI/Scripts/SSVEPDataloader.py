import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def SSVEPDataloaders(SSVEPDataset, batchSize):
    validation_split = .2
    train_split = .1
    shuffle_dataset = True
    random_seed = 42
    
    datasetSize = len(SSVEPDataset)
    indices = list(range(datasetSize))
    trainsplit = int(np.floor(train_split * datasetSize))
    
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    trainIndices, testIndices = indices[trainsplit:], indices[:trainsplit]
    trainSetSize = len(trainIndices)
    validSplit = int(np.floor(validation_split * trainSetSize))
    trainIndices, validIndices = trainIndices[validSplit:], trainIndices[:validSplit]
    
    trainSampler, validSampler, testSampler = [SubsetRandomSampler(trainIndices), 
                                               SubsetRandomSampler(validIndices), 
                                               SubsetRandomSampler(testIndices)]
    
    trainLoader, validationLoader, testLoader = [DataLoader(SSVEPDataset, batch_size = batchSize, sampler = trainSampler), 
                                                 DataLoader(SSVEPDataset, batch_size = batchSize, sampler = validSampler), 
                                                 DataLoader(SSVEPDataset, batch_size = batchSize, sampler = testSampler)]
            
    dataloaders = {"train": trainLoader, 
                   "val": validationLoader, 
                   "test": testLoader}
    datasetsSizes = {"train": len(trainIndices),
                    "val": len(validIndices),
                    "test": len(testIndices)}
    
    return dataloaders, datasetsSizes