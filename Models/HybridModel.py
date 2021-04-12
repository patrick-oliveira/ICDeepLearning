import torch
import torch.nn as nn


class HybridModel(nn.Module):
    def __init__(self, classifier, encoder):
        super(HybridModel, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.encoder.eval()
        
    def forward(self, x):
        with torch.set_grad_enabled(False):
            x = self.encoder(x)
            
        x = self.classifier(x)
        
        return x
    
