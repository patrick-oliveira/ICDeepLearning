import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')

from sklearn.model_selection import KFold

import copy
import time

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def visualize_model(model, num_images = 6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])
                
                if images_so_far == num_images:
                    model.train(mode = was_training)
                    return
        model.train(mode = was_training)

        
def get_dataset(path, image_size):
    dataset = datasets.ImageFolder(root = path,
                                  transform = transforms.Compose([
                                      transforms.Resize(image_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
                                  ]))
    return dataset

def getData(data_type, folder, data_address = 'C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data',
            random_seed = 42, shuffle_dataset = True, batch_size = 4, image_size = 224, workers = 1,
            test_split = .2):
    
    figPath = data_address+'\\'+folder+'\\'+data_type

    dataset = get_dataset(figPath, image_size)
    class_names = dataset.classes
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    test_split_id = int(np.floor(test_split * dataset_size))
    train_indices, test_indices = indices[test_split_id:], indices[:test_split_id]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size = batch_size,
                                        sampler = train_sampler)
    test_loader = DataLoader(dataset, batch_size = batch_size,
                                        sampler = test_sampler)


    dataloaders = {"train": train_loader, "test": test_loader}

    dataset_sizes = {"train": len(train_indices),
                     "test": len(test_indices)}
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                             shuffle = True, num_workers = workers)

    
    return dataloader, dataloaders, dataset_sizes, class_names

def train(model, dataloaders, dataset_size, criterion, optimizer, scheduler, num_epochs = 10):
    since = time.time()
    
    best_model_wgts = copy.deepcopy(model.state_dict())
    best_acc = .0
    trainAcc = []; trainLoss = []
    testAcc  = []; testLoss  = []
    
    for epoch in range(num_epochs):
        print('-'*2)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-'*5)
        
        # Each epoch as a training and testing phase
        for phase in ['train', 'test']:
            print('Stage: {}'.format(phase))
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            cumulative_loss = 0.0
            cumulative_hits  = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                cumulative_loss += loss.item()*inputs.size(0)
                cumulative_hits += torch.sum(preds == labels.data)
                del(inputs); del(labels)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = cumulative_loss / dataset_size[phase]
            epoch_acc  = cumulative_hits.double() / dataset_size[phase]
            
            if phase == 'train':
                trainAcc.append(epoch_acc); trainLoss.append(epoch_loss)
            else:
                testAcc.append(epoch_acc); testLoss.append(epoch_loss)
                
            print('Loss: {:.4f} - Accuracy: {:.4f}'.format(epoch_loss, epoch_acc))
            
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
    time_elapsed = time.time() - since
    print('Best validation accuracy: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wgts)
    return model, best_acc, time_elapsed
                    
def kfold_train(model, dataset, criterion, optimizer, scheduler, num_folds = 5, batch_size = 4, shuffle = False, save_model = False, num_epochs = 10):
    kfold = KFold(num_folds, shuffle)
    cumulative_best_acc = 0
    cumulative_time = 0
    
    for fold, (train_index, test_index) in enumerate(kfold.split(dataset, dataset.targets)):
        print('Fold {}'.format(fold))
        train_subset = torch.utils.data.Subset(dataset, train_index)
        test_subset  = torch.utils.data.Subset(dataset, test_index)
        dataset_size = {}
        dataset_size['train'] = len(train_subset) 
        dataset_size['test'] = len(test_subset)
        dataloaders  = {}
        dataloaders['train'] = torch.utils.data.DataLoader(train_subset, batch_size = batch_size, shuffle = shuffle)
        dataloaders['test']  = torch.utils.data.DataLoader(test_subset, batch_size = batch_size, shuffle = shuffle)
        
        _, fold_best_acc, time_elapsed = train(model, dataloaders, dataset_size, criterion, optimizer, scheduler, num_epochs = num_epochs)
            
        cumulative_time += time_elapsed    
        cumulative_best_acc += fold_best_acc
        print(cumulative_best_acc.item())
        
        print('')
        
    mean_acc = cumulative_best_acc.item() / num_folds
    
    print('Training complete in {:.0f}m {:.0f}s'.format(cumulative_time // 60, cumulative_time % 60))
    print('Mean acc: {}'.format(mean_acc))
    if save_model:
        torch.save(model.state_dict, 'model.pth')
    
    return model, cumulative_best_acc, mean_acc
        