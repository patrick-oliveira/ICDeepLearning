import torch
import torch.nn as nn

def ConvBlocks(inSizes, outSizes, middleActivationFunc = 'relu', *args, **kwargs):
    return nn.Sequential(*[ConvBlock(inputSize, outputSize, middleActivationFunc, *args, **kwargs) for
                                     inputSize, outputSize in zip(inSizes, outSizes)]) 
    
class DenseBlock(nn.Module):
    def __init__(self, firstChannels, fixedSizeOutChannels, layerSize, kernel_size):
        super().__init__()
        
        inSizes = [firstChannels, fixedSizeOutChannels]
        outSizes = [fixedSizeOutChannels]*layerSize
        self.relu = nn.ReLU()
        for i in range(layerSize - 2):
            inSizes.append(inSizes[-1] + fixedSizeOutChannels)
            
        self.BN = nn.BatchNorm2d(num_features = firstChannels)
        if len(kernel_size) > 1:
            padding = (kernel_size[0]//2, kernel_size[1]//2)
        else:
            padding = kernel_size//2
        self.ConvLayers = ConvBlocks(inSizes, outSizes, 
                                     kernel_size = kernel_size, stride = 1, padding = kernel_size//2)
    
    def forward(self, X):
        X = self.BN(X)
        output = [self.ConvLayers[0](X)]
        dense  = self.relu(torch.cat(output, 1))
        for conv in self.ConvLayers[1:]:
            Y = conv(dense)
            output.append(Y)
            dense = nn.ReLU(torch.cat(output, 1))
            
        return dense    
    
class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.relu = nn.ReLU(inplace = True)
        self.bn   = nn.BatchNorm2d(num_features = out_channels)
        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size = 1, bias = False)
        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)
        
    def forward(self, X):
        bn = self.bn(self.relu(self.conv(X)))
        out = self.avg_pool(bn)
        
        return out
    
    
class DenseNet(nn.Module):
    def __init__(self, nr_classes):
        super(DenseNet, self).__init__()
        
        self.FirstConv = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, bias = False)
        )
        
        self.DenseLayer1 = nn.Sequential( # 160 = 32 * 5
                DenseBlock(64, 32, 5, 3),
                TransitionLayer(160, 128) 
        )
        self.DenseLayer2 = nn.Sequential(
                DenseBlock(128, 32, 5, 3),
                TransitionLayer(160, 128)
        )
        self.DenseLayer3 = nn.Sequential(
                DenseBlock(128, 32, 5, 3),
                TransitionLayer(160, 64)
        )
        self.BN = nn.BatchNorm2d(num_features = 64)
        self.Classifier = nn.Sequential(
                nn.Linear(64*4*4, 512),
                nn.Linear(512, nr_classes)
        )
        
    def forward(self, X):
        X = self.FirstConv(X)
        X = self.DenseLayer1(X)
        X = self.DenseLayer2(X)
        X = self.DenseLayer3(X)
        X = self.BN(X)
        x.view(-1, 64*4*4)
        X = self.Classifier(X)
        return X