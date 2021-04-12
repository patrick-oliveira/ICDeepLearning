import torch
import torch.nn as nn


class Relation(nn.Module):
    def __init__(self, weight = None, size_average = None):
        super(Relation, self).__init__()
        
    def forward(self, inputs: torch.tensor) -> torch.float:
        return self.function(inputs)
    
    def function(self, inputs: torch.tensor, *args, **kwargs) -> torch.float:
        raise NotImplementedError()
        
class RelationalLoss(nn.Module):
    def __init__(self, loss, relation: Relation, alpha: float = None, weight = None, size_average = None):
        super(RelationalLoss, self).__init__()
        self.loss = loss
        self.relation = relation
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        alpha = self.alpha if self.alpha != None else 1
        reconstruction = self.loss(inputs, targets)
        relation = self.loss(self.relation(inputs), self.relation(targets))
        
        return (1 - alpha)*reconstruction + alpha*relation, reconstruction