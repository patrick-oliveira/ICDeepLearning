import time
import copy
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


def visualize_model(model, num_images=6):
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
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def training_logger(project):
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y - %H_%M', t)
    file_name = (project.output_dir / ("Training - "+timestamp)).__str__()
    file_mode = 'w'
    logger = logging.getLogger("training_logger")
    logger.setLevel(logging.INFO)
    f_handler = logging.FileHandler(file_name+'.log')
    f_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(f_handler)

    return logger

def close_log_handlers(logger):
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
        
def train_autoencoder(project, model, dataloaders, dataset_size, criterion, optimizer, scheduler = None, num_epochs = 10,
                      logger = None, kfold = False):
    if logger == None: logger = training_logger(project)
    
    logger.info("{:7}  {:10}  {:6}\n".format("Epoch", "Stage", "Loss"))
    
    best_model_wgts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    train_loss = []
    validation_loss  = []
    stats = {"Train": train_loss,
             "Validation": validation_loss}
    
    since = time.time()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            cumulative_loss = 0.0
            for inputs, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, inputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                cumulative_loss += loss.item()*inputs.size(0)
                del(inputs); del(_)
            if phase == 'train' and scheduler != None:
                scheduler.step()
            epoch_loss = cumulative_loss / dataset_size[phase]
            train_loss.append(epoch_loss) if phase == 'train' else validation_loss.append(epoch_loss)
            logger.info("{:7}  {:10}  {:<6.2f}".format("{}/{}".format(epoch + 1, num_epochs) if phase == "train" else " ",
                                                        "Training" if phase == "train" else "Validation",
                                                        epoch_loss))
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wgts = copy.deepcopy(model.state_dict())     
    time_elapsed = time.time() - since
    
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info("Best Validation Loss: {:.2f}\n".format(best_loss))
    
    close_log_handlers(logger)
    
    stats["BestLoss"] = best_loss
    model.load_state_dict(best_model_wgts)
    return model, stats

def train(project, model, dataloaders, dataset_size, criterion, optimizer, scheduler=None, num_epochs=10,
          logger = None, kfold = False):
    if logger == None: logger = training_logger(project)
    
    logger.info("{:7}  {:10}  {:6}  {:8}\n".format("Epoch", "Stage", "Loss", "Accuracy"))

    best_model_wgts = copy.deepcopy(model.state_dict())
    best_acc = .0; best_loss = np.inf
    train_stats = {"Accuracy": [], "Loss": []}
    validation_stats = {"Accuracy":[], "Loss":[]}
    stats = {"Train": train_stats,
             "Validation": validation_stats}

    since = time.time()
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            cumulative_loss = .0
            cumulative_hits = 0
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
            if phase == 'train' and scheduler != None:
                scheduler.step()
            epoch_loss = cumulative_loss / dataset_size[phase]
            epoch_acc  = cumulative_hits.double() / dataset_size[phase]
            if phase == 'train':
                train_stats['Accuracy'].append(epoch_acc)
                train_stats['Loss'].append(epoch_loss)
            else:
                validation_stats['Accuracy'].append(epoch_acc)
                validation_stats['Loss'].append(epoch_loss)
            logger.info("{:7}  {:10}  {:<6.2f}  {:<8.2f}".format("{}/{}".format(epoch + 1, num_epochs) if phase == "train" else " ",
                                                                 "Training" if phase == "train" else "Validation",
                                                                 epoch_loss, epoch_acc))
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wgts = copy.deepcopy(model.state_dict())
    time_elapsed = time.time() - since
    
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    logger.info("Best Validation Accuracy: {:.2f}".format(best_acc))
    logger.info("Best Validation Loss: {:.2f}\n".format(best_loss))
    if not kfold: close_log_handlers(logger)   
    stats["BestAccuracy"] = best_acc
    stats["BestLoss"]     = best_loss
    model.load_state_dict(best_model_wgts)
    return model, stats


def kfold_train(project, model, dataset, criterion, optimizer, scheduler=None, num_folds=5, batch_size=4, save_model=False, num_epochs=10):
    """
    Para fazer:
        - Extrair os valores ótimos de validação de cada fold e a média.
        - Passar log para um arquivo.
    """
    logger = training_logger(project)
    kfold_stats = {}
    kfold = KFold(num_folds, shuffle = True)
    cumulative_time = 0
    cumulative_best_acc = .0
    backup_weights = copy.deepcopy(model.state_dict())
    init_state_opt = copy.deepcopy(optimizer.state_dict())

    for fold, (train_index, test_index) in enumerate(kfold.split(dataset, dataset.targets)):
        logger.info('Fold {}'.format(fold))
        train_subset = torch.utils.data.Subset(dataset, train_index)
        test_subset = torch.utils.data.Subset(dataset, test_index)
        dataset_size = {}
        dataset_size['train'] = len(train_subset)
        dataset_size['val'] = len(test_subset)
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_subset, batch_size=batch_size, shuffle=True)
        dataloaders['val'] = torch.utils.data.DataLoader(
            test_subset, batch_size=batch_size, shuffle=True)

        model.load_state_dict(backup_weights)
        optimizer.load_state_dict(init_state_opt)

        since = time.time()
        model, stats = train(project, model, dataloaders, dataset_size, criterion, optimizer, scheduler, num_epochs, logger = logger, kfold = True)
        time_elapsed = time.time() - since

        kfold_stats['Fold{fold}'] = stats
        
        cumulative_time += time_elapsed
        cumulative_best_acc += stats["BestAccuracy"]

    mean_acc = cumulative_best_acc / num_folds
    kfold_stats['MeanAccuracy'] = mean_acc

    logger.info('Training complete in {:.0f}m {:.0f}s'.format(cumulative_time // 60, cumulative_time % 60))
    logger.info('Mean acc: {:.2f}'.format(mean_acc))
    close_log_handlers(logger)

    return model, kfold_stats
