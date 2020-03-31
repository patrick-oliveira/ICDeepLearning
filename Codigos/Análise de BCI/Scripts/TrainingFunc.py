import time
import copy
import numpy as np
import torch

def train(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device,
          num_epochs = 10, learning_rate = 1e-2, temp = 0):
    since = time.time()
    best_model = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_acc = 0.0
    trainAcc = []
    trainLoss = []
    validAcc = []
    validLoss = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs = sample['signal'].to(device)
                labels = sample['class'].to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Foward stage
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                del(sample)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_corrects.double() / dataset_sizes[phase]
            
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

def test(model, dataloader, dataset_size, criterion, device):
    running_loss = 0.0
    running_corrects = 0
    for sample in dataloader:
        inputs = sample['signal'].to(device)
        labels = sample['class'].to(device)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        del(sample)
    
    loss = running_loss / dataset_size
    acc  = running_corrects.double() / dataset_size
    
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', loss, acc))