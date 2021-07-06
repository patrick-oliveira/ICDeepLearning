import time
import logging
import sys

import copy
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import pandas as pd
import numpy as np
import contextlib

ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

@contextlib.contextmanager
def log_training(log_path: str, log_name: str):
    logging.basicConfig(stream = sys.stdout, 
                        level = logging.INFO, 
                        format = '%(message)s')
    logger = logging.getLogger('Training Logger')
    
    timestamp = time.strftime('%b-%d-%Y', time.localtime())
    
    if log_name != None:
        handler = logging.FileHandler(filename = log_path + '/' + log_name + ' - ' + timestamp + '.log',
                                      mode = 'w')
        logger.addHandler(handler)
    
    try:
        yield logger
    finally:
        if log_name != None:
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
class Trainer:
    def __init__(self, problem_type: str,
                       training_routine,
                       validation_routine,
                       path: str = None,
                       log_name = None):
        '''
        Classe de gerenciamento de treinamento de redes neurais. A classe foi criada para que seja possível treinar redes de diferentes tipos (classificação, regressão), com especificidades quanto ao algoritmo de treinamento, sem que seja necessário copiar as funções de treinamento que, de modo geral, são similares. Também permite o log do treinamento com facilidade.
        A classe é criada especificando o tipo de problema (regressão ou classificação), que permite a escolha da função geral de treinamento, e recebe também duas funções com rotinas de treinamento especificas do modelo. Essas funções são passadas para a função geral.
        
        Args:
           problem_type (str): Deve ser "regression" ou "classification"; permite selecionar a função geral de treinamento.
           training_routine (function): Função a ser utilizada na etapa de treinamento do modelo.
           validation_routine (function): Função a ser utilizada na etapa de validação do modelo.
           path (string): Endereço onde os logs serão salvos, se for o caso.
           log_name (string): Nome do log, se for o caso.
        '''
        self.problem_type = problem_type
        self.training_function = {'regression': train_regression,
                                  'classification': train_classification}[problem_type]
        self.training_routine  = training_routine
        self.validation_routine = validation_routine
        
        # Colocar esses dois como @attribute
        self.path = path
        self.log_name = log_name
        
    def train(self, model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10):
        '''
        Este método faz a chamada da função geral de treinamento, passando os objetos necessários mais as funções especificas de treinamento e validação. O log é feito automaticamente, salvo em um arquivo caso 'log_name' no diretório 'path', caso estes tenham sido especificados na instanciação da classe.
        
        Args:
            model: Um modelo de rede neural.
            dataloaders (dict): Um dicionário com os dataloaders de treinamento e validação.
            criterion (pytorch.criterion): Um critério de avaliação do modelo.
            optimizer (pytorch.optimizer): O otimizador do modelo.
            scheduler (pytorch.scheduler): Classe de variação automática da taxa de aprendizado.
            num_epochs (int): Número de épocas
        '''
        
        with log_training(self.path, self.log_name) as logger:
            model, training_info = self.training_function(model, dataloaders, criterion, optimizer, scheduler, num_epochs, logger, self.training_routine, self.validation_routine)
       
        return model, training_info
    
    def kfold_train(self, model_initializer, optimizer_initializer, criterion,
                    dataset, num_folds: int = 5, batch_size: int = 5, num_epochs:int = 20,
                    scheduler_initializer = None,
                    save_model = False,
                    *args, **kwargs):
        '''
        Este método faz a chamada da função geral de treinamento no contexto do kfold cross-validation.
        
        Args:
            model_initializer:
            optimizer_initializer:
            criterion:
            dataset:
            num_folds (int):
            batch_size (int):
            num_epochs (int):
            scheduler_initializer:
            save_model (bool):
        '''
        kfold_stats = {}
        kfold = KFold(num_folds, shuffle = True)
        cumulative_time = 0
        cumulative_best = .0
        
        with log_training(self.path, self.log_name) as logger:
            for fold, (train_index, test_index) in enumerate(kfold.split(dataset, dataset.targets)):
                logger.info(f'Fold {fold}')
                
                train_subset = torch.utils.data.Subset(dataset, train_index)
                test_subset = torch.utils.data.Subset(dataset, test_index)
                dataset_size = {}
                dataset_size['train'] = len(train_subset)
                dataset_size['val'] = len(test_subset)
                dataloaders = {}
                dataloaders['train'] = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                dataloaders['val'] = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True)
                
                model = model_initializer(kwargs['classifier'](), kwargs['autoencoder'](), device = device)
                optimizer = optimizer_initializer(model, lr = 1e-3, momentum = 0.9)
                scheduler = scheduler_initializer(optimizer) if scheduler_initializer != None else None
                
                since = time.time()
                model, training_info = self.training_function(model, dataloaders, criterion, optimizer, scheduler,
                                                              num_epochs, logger, self.training_routine, 
                                                              self.validation_routine)
                time_elapsed = time.time() - since
                
                kfold_stats[f'Fold {fold}'] = training_info
                
                cumulative_time += time_elapsed
                if self.problem_type == 'regression':
                    cumulative_best += training_info['Best Loss']
                elif self.problem_type == 'classification':
                    cumulative_best += training_info['Best Accuracy']
                
            kfold_stats['Mean Best'] = (cumulative_best / num_folds)
            logger.info('Training complete in {:.0f}m {:.0f}s'.format(cumulative_time // 60,
                                                                      cumulative_time %  60))
            logger.info('Mean Best Validation {} = {}'.format('Loss' if self.problem_type == 'regression' else 'Accuracy', 
                                                              kfold_stats['Mean Best']))
            
        return model, kfold_stats
            
            
def train_classification(model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10, logger = None,
                         training_routine = None, validation_routine = None):
    '''
    
    Args:
        model:
        dataloaders:
        criterion:
        optimizer:
        scheduler:
        num_epochs:
        logger:
        training_routine:
        validation_routine:
    '''
    
    assert training_routine != None and validation_routine != None, 'Must provide training and validation routines.'
    
    header = "{:7}  {:10}  {:6}  {:8}\n".format("Epoch", "Stage", "Loss", "Accuracy")
    logger.info(header)
        
    best_model = copy.deepcopy(model.state_dict())
    training_info = {
        'Best Accuracy' : .0,
        'Best Loss' : np.inf,
        'Training Stats' : {'Accuracy' : [], 'Loss': []},
        'Validation Stats' : {'Accuracy' : [], 'Loss': []}
    }

    since = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        cumulative_loss = .0
        cumulative_hits = 0
        for inputs, labels in dataloaders['train']:
            c_loss_temp, c_hits_temp = training_routine(model, inputs, labels, criterion, optimizer, scheduler)
            cumulative_loss += c_loss_temp
            cumulative_hits += c_hits_temp
            
        training_accuracy = cumulative_hits / len(dataloaders['train'].dataset)
        training_loss     = cumulative_loss / len(dataloaders['train'].dataset)
        training_info['Training Stats']['Accuracy'].append(training_accuracy)
        training_info['Training Stats']['Loss'].append(training_loss)
        
        
        
        # Validation Phase
        model.eval()
        cumulative_loss = .0
        cumulative_hits = 0
        for inputs, labels in dataloaders['val']:
            c_loss_temp, c_hits_temp = validation_routine(model, inputs, labels, criterion)
            cumulative_loss += c_loss_temp
            cumulative_hits += c_hits_temp
            
        validation_accuracy = cumulative_hits / len(dataloaders['val'].dataset)
        validation_loss     = cumulative_loss / len(dataloaders['val'].dataset)
        training_info['Validation Stats']['Accuracy'].append(validation_accuracy)
        training_info['Validation Stats']['Loss'].append(validation_loss)
        
        
        
        # Model evaluation and logging
        if training_info['Best Accuracy'] < validation_accuracy:
            training_info['Best Accuracy'] = validation_accuracy
            training_info['Best Loss']     = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            
        training_log = "{:7}  {:10}  {:<6.2f}  {:<8.2f}".format("{}/{}".format(epoch + 1, num_epochs), "Training", training_loss, training_accuracy)
        validation_log = "{:7}  {:10}  {:<6.2f}  {:<8.2f}".format(" ", "Validation", validation_loss, validation_accuracy)
        logger.info(training_log); logger.info(validation_log)

    time_elapsed = time.time() - since
    conclusion_log = 'Training complete in {:.0f}m {:.0f}s \nBest Validation Accuracy: {:.2f} \nBest Validation Loss: {:.2f}'.format(time_elapsed // 60, time_elapsed % 60, training_info['Best Accuracy'], training_info['Best Loss'])
    logger.info(conclusion_log)
    model.load_state_dict(best_model)
    return model, training_info

def train_regression(model, dataloaders, criterion, optimizer, scheduler = None, num_epochs = 10, logger = None,
                     training_routine = None, validation_routine = None):
    '''
    
    Args:
        model:
        dataloaders:
        criterion:
        optimizer:
        scheduler:
        num_epochs:
        logger:
        training_routine:
        validation_routine:
    '''
    
    assert (training_routine != None and validation_routine != None), 'Must provide training and validation routines.'
    
    header = "{:7}  {:10}  {:6}\n".format("Epoch", "Stage", "Loss")
    logger.info(header)
        
    best_model = copy.deepcopy(model.state_dict())
    training_info = {
        'Best Loss' : np.inf,
        'Training Loss' : [],
        'Validation Loss' : []
    }

    since = time.time()
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        cumulative_loss = .0
        for inputs, labels in dataloaders['train']:
            c_loss_temp = training_routine(model, inputs, labels, criterion, optimizer, scheduler)
            cumulative_loss += c_loss_temp
            
        training_loss     = cumulative_loss / len(dataloaders['train'].dataset)
        training_info['Training Loss'].append(training_loss)
        
        
        
        # Validation Phase
        model.eval()
        cumulative_loss = .0
        for inputs, labels in dataloaders['val']:
            c_loss_temp = validation_routine(model, inputs, labels, criterion)
            cumulative_loss += c_loss_temp
            
        validation_loss     = cumulative_loss / len(dataloaders['val'].dataset)
        training_info['Validation Loss'].append(validation_loss)
        
        
        
        # Model evaluation and logging
        if training_info['Best Loss'] > validation_loss:
            training_info['Best Loss']     = validation_loss
            best_model = copy.deepcopy(model.state_dict())
            
        training_log = "{:7}  {:10}  {:<6.2f}".format("{}/{}".format(epoch + 1, num_epochs), "Training", training_loss)
        validation_log = "{:7}  {:10}  {:<6.2f}".format(" ", "Validation", validation_loss)
        logger.info(training_log); logger.info(validation_log)
    
    time_elapsed = time.time() - since
    conclusion_log = 'Training complete in {:.0f}m {:.0f}s \nBest Validation Loss: {:.2f}'.format(time_elapsed // 60, time_elapsed % 60, training_info['Best Loss'])
    logger.info(conclusion_log)
    model.load_state_dict(best_model)
    return model, training_info


def basic_training_routine(model, inputs, labels, criterion, optimizer, scheduler = None):
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    
    # Compute the loss
    loss = criterion(outputs, labels)
    
    # backward pass
    loss.backward()
    
    # Optimize
    optimizer.step()
    
    # Decrease learning rate
    if scheduler != None:
        scheduler.step()
        
    # Compute cumulative loss (to return)
    cumulative_loss = loss.data.item() * inputs.size(0)
    cumulative_hits = torch.sum(preds == labels.data)
    
    del(inputs); del(labels)
    return cumulative_loss, cumulative_hits

def basic_validation_routine(model, inputs, labels, criterion):
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)
    cumulative_loss = loss.data.item() * inputs.size(0)
    cumulative_hits = torch.sum(preds == labels.data)
    del(inputs); del(labels)
    return cumulative_loss, cumulative_hits


def autoencoder_basic_training_routine(model, inputs, labels, criterion, optimizer, scheduler = None):
    inputs = inputs.to(device)
    # Zero the parameter gradients
    optimizer.zero_grad()
    
    # Forward pass
    outputs = model(inputs)
    
    # Compute the loss
    loss = criterion(outputs, inputs)
    
    # Backward pass
    loss.backward()
    
    # Optimize
    optimizer.step()
    
    # Decrease learning rate
    if scheduler != None:
        scheduler.step()
        
    # Compute cumulative loss (to return)
    cumulative_loss = loss.data.item() * inputs.size(0)
    
    del(inputs); del(labels)
    return cumulative_loss

def autoencoder_basic_validation_routine(model, inputs, labels, criterion):
    inputs = inputs.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, inputs)
    cumulative_loss = loss.data.item() * inputs.size(0)
    del(inputs); del(labels)
    return cumulative_loss



