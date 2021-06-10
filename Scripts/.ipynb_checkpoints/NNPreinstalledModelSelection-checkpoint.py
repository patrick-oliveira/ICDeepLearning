from torchvision import models
import torch.nn as nn

def resnet18(numClasses, pretrained = True):
    model = models.resnet18(pretrained = pretrained)
    model.fc = nn.Linear(512, numClasses)
    return model

def resnet34(numClasses, pretrained = True):
    model = models.resnet34(pretrained = pretrained)
    model.fc = nn.Linear(512, numClasses)
    return model

def resnet50(numClasses, pretrained = True):
    model = models.resnet50(pretrained = pretrained)
    model.fc = nn.Linear(2048, numClasses)
    return model

def resnet101(numClasses, pretrained = True):
    model = models.resnet101(pretrained = pretrained)
    model.fc = nn.Linear(2048, numClasses)
    return model

def resnet152(numClasses, pretrained = True):
    model = models.resnet152(pretrained = pretrained)
    model.fc = nn.Linear(2048, numClasses)
    return model

def densenet121(numClasses, pretrained = True):
    model = models.densenet121(pretrained = pretrained)
    model.classifier = nn.Linear(1024, numClasses)
    return model

def densenet161(numClasses, pretrained = True):
    model = models.densenet161(pretrained = pretrained)
    model.classifier = nn.Linear(2208, numClasses)
    return model

def densenet169(numClasses, pretrained = True):
    model = models.densenet169(pretrained = pretrained)
    model.classifier = nn.Linear(1664, numClasses)
    return model

def densenet201(numClasses, pretrained = True):
    model = models.densenet201(pretrained = pretrained)
    model.classifier = nn.Linear(1920, numClasses)
    return model

def alexnet(numClasses, pretrained = True):
    model = models.alexnet(pretrained = pretrained)
    model.classifier[6] = nn.Linear(4096, numClasses)
    return model

def googlenet(numClasses, pretrained = True):
    model = models.googlenet(pretrained = pretrained)
    model.fc = nn.Linear(1024, numClasses)
    return model

def model_selection(modelName, numClasses, pretrained = True):
    model_dict = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'densenet121': densenet121,
        'densenet161': densenet161,
        'densenet169': densenet169,
        'densenet201': densenet201,
        'alexnet': alexnet,
        'googlenet': googlenet
    }
    return model_dict[modelName](numClasses, pretrained)