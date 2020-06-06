import sys
sys.path.append('Scripts')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

from utils import path
from utils import createInputFigure

from sklearn import preprocessing
import os
import time
import copy

image_size = 224
batch_size = 64
workers = 1
ngpu = 1
validation_split = .2
train_split = 0
shuffle_dataset = True
random_seed = 42
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

def getData(dataType, batchSize):
    folder = 'GAF_Combinados'
    figPath = path('C:\\Users\\Patrick\\Documents\\GitHub\\ICDeepLearning\\Data\\'+folder+'\\'+dataType)

    dataset = datasets.ImageFolder(root = figPath(),
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
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size = batchSize,
                                        sampler = train_sampler)
    validation_loader = DataLoader(dataset, batch_size = batchSize,
                                        sampler = valid_sampler)
    test_loader = DataLoader(dataset, batch_size = batchSize,
                                        sampler = test_sampler)


    dataloaders = {"train": train_loader, "val": validation_loader, "test": test_loader}

    dataset_sizes = {"train": len(train_indices),
                     "val": len(val_indices),
                     "test": len(test_indices)}
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize,
                                             shuffle = True, num_workers = workers)

    
    return dataloaders, dataset_sizes, dataloader, class_names, dataset

def trainLoop(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, dataloader, num_epochs = 25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    trainAcc = []
    trainLoss = []
    validAcc = []
    validLoss = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                del(inputs)
                del(labels)
                
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                trainAcc.append(epoch_acc)
                trainLoss.append(epoch_loss)
            else:
                validAcc.append(epoch_acc)
                validLoss.append(epoch_loss)
                
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, [trainAcc, trainLoss], [validAcc, validLoss]

def visualizeModel(model, num_images = 6):
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
        
def trainModel(model, modelName, criterion, optimizer, scheduler, dataType, figSize, batchSize):
#     dataTypes = ['512', '768', 'shifted']
    stats = {
        '512': [],
        '768': [],
        'shifted': []
    }
    
    print("Training stage:{}".format(dataType))
    dataloaders, dataset_sizes, dataloader, _, _ = getData(dataType, batchSize)
    model, *stats[dataType] = trainLoop(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, dataloader, num_epochs = 10)
    torch.save(model.state_dict(), 'Models\\'+modelName+'\\cca'+figSize+"\\model.pth")
    torch.save(stats, 'Models\\'+modelName+'\\cca'+figSize+'\\stats'+dataType+'.pth')
        
    return model, stats

modelDict = {
    'resnet18': [models.resnet18(pretrained = True), 512],
    'resnet34': [models.resnet34(pretrained = True), 512],
    'resnet50': [models.resnet50(pretrained = True), 2048],
    'resnet101': [models.resnet101(pretrained = True), 2048],
    'resnet152': [models.resnet152(pretrained = True), 2048],
    'densenet121': [models.densenet121(pretrained = True), 1024],
    'densenet161': [models.densenet161(pretrained = True), 2208],
    'densenet169': [models.densenet169(pretrained = True), 1664],
    'densenet201': [models.densenet201(pretrained = True), 1920],
    'alexnet': [models.alexnet(pretrained = True), 4096],
    'googlenet': [models.googlenet(pretrained = True), 1024],
    'vgg11': [models.vgg11(pretrained = True), 4096],
    'vgg13': [models.vgg13(pretrained = True), 4096],
    'vgg16': [models.vgg16(pretrained = True), 4096],
    'vgg19': [models.vgg19(pretrained = True), 4096],
    'mnasnet0_5': [models.mnasnet0_5(pretrained = True), 1280],
    'mnasnet1_0': [models.mnasnet1_0(pretrained = True), 1280],
    'resnext101_32x8d': [models.resnext101_32x8d(pretrained = True), 2048],
    'resnext50_32x4d': [models.resnext50_32x4d(pretrained = True), 2048],
    'shufflenet_v2_x0_5': [models.shufflenet_v2_x0_5(pretrained = True), 1024]
}

def lastLayer(modelType, model):
    Dict = {
        'resnet': lambda model: model.fc,
        'densenet': lambda model: model.classifier,
        'alexnet': lambda model: model.classifier[6],
        'vgg': lambda model: model.classifier[6],
        'googlenet': lambda model: model.fc,
        'mnasnet': lambda model: model.classifier[1],
        'resnext': lambda model: model.fc,
        'shufflenet': lambda model: model.fc
    }
    return Dict[modelType](model)

def autoTraining(modelType, modelSize, dataType, figSize, batchSize):
    modelName = modelType+modelSize
    numClasses = 5
    model = modelDict[modelName][0]
    numFeatures = modelDict[modelName][1]
    classifier = lastLayer(modelType, model)
    classifier = nn.Linear(numFeatures, numClasses)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)
    model, stats = trainModel(model, modelName, criterion, optimizer, scheduler, dataType, figSize, batchSize)
#     del(model)
    return model