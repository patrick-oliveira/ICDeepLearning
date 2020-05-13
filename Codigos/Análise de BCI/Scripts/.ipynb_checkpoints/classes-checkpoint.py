import torch
import torch.nn as nn

# class ConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, kernel, stride, pad_size = 0):
#         super(ConvBlock, self).__init__()
#         if pad_size == None: pad_size = kernel // 2
#         self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride = stride,
#                                              padding = pad_size, bias = False),
#                                    nn.BatchNorm2d(out_size),
#                                    nn.ReLU(inplace = True))
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs
        
# class DeConvBlock(nn.Module):
#     def __init__(self, in_size, out_size, kernel, pad_size = 0):
#         super(DeConvBlock, self).__init__()
#         if pad_size == None: pad_size = kernel // 2
#         self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, stride = 2,
#                                                       padding = pad_size, output_padding = 1, bias = False),
#                                    nn.BatchNorm2d(out_size),
#                                    nn.ReLU(inplace = True),)
#     def forward(self, inputs):
#         return outputs
    
    
# class ConvBlock_last(nn.Module):
#     def __init__(self, in_size, out_size, kernel, pad_size = 0):
#         super(ConvBlock_last, self).__init__()
#         if pad_size == None: pad_size = kernel // 2
#         self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, padding = pad_size, bias = False))
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs
    
# class DeConvBlock_last(nn.Module):
#     def __init__(self, in_size, out_size, kernel, pad_size = 0):
#         super(DeConvBlock_last, self).__init__()
#         if pad_size == None: pad_size = kernel // 2
#         self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_size, out_size, kernel, padding = pad_soze, bias = False))
            
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs
    
    
# class ConvPoolBlock(nn.Module):
#     def __init__(self, in_size, out_size, kernel, stride, pad_size = 0, pool_kernel, pool_stride):
#         super(ConvPoolBlock, self).__init__()
#         if pad_size == None: pad_size = kernel // 2
#         self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride = stride, 
#                                              padding = pad_size),
#                                              nn.ReLU(inplace = True),
#                                              nn.MaxPool2d(pool_kernel, pool_stride))
        
#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         return outputs
    
# class DoubleConvPoolBlock(nn.Module):
#     def __init__(self, in_size1, out_size1, kernel1, stride1, pad_size1 = 0,
#                        in_size2, out_size2, kernel2, stride2, pad_size2 = 0,
#                        pool_kernel, pool_stride):
#         super(DoubleConvPoolBlock, self).__init__()
#         if pad_size1 == None: pad_size1 = kernel1 // 2
#         if pad_size2 == None: pad_size2 = kernel2 // 2
#         self.conv1 = nn.Sequential(nn.Conv2d(in_size1, out_size1, kernel1, stride = stride1,
#                                              padding = pad_size1),
#                                    nn.ReLU(inplace = True),
#                                    nn.Conv2d(in_size2, out_size2, kernel2, stride = stride2,
#                                              padding = pad_size2),
#                                    nn.ReLU(inplace = True),
#                                    nn.MaxPool2d(pool_kernel, pool_stride))
        
# #     def forward(self, inputs):
        
        
# class FCReLU(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(FCReLU, self).__init__()
#         self.FC = nn.Sequential(nn.Linear(in_size, out_size),
#                                 nn.ReLU(inplace = True))
        
#     def forward(self, inputs):
#         outputs = self.FC(inputs)
#         return outputs


def activation(activationFunction):
    activations = nn.ModuleDict([
        ['lrelu', nn.LeakyReLU(negative_slope = 0.01, inplace = True)],
        ['relu', nn.ReLU(inplace = True)],
        ['none', nn.Identity()],
        ['selu', nn.SELU(inplace = True)],
        ['sigmoid', nn.Sigmoid()],
    ])
    return activations[activationFunction]

# def activation_func(activation):
#     return nn.ModuleDict([
#         ['relu', nn.ReLU(inplace = True)],
#         ['leaky_relu', nn.LeakyReLU(negative_slope = 0.01, inplace = True)],
#         ['selu', nn.SELU(inplace = True)],
#         ['none', nn.Identity()]
#     ])

def ConvBlock(in_size, out_size, activFunc = 'relu', *args, **kwargs):
#     if padding == None: padding = kernel // 2
    return nn.Sequential(nn.Conv2d(in_size, out_size, *args, **kwargs),
                         nn.BatchNorm2d(out_size),
                         activation(activFunc))
    

def DeconvBlock(in_size, out_size, activFunc = 'relu', *args, **kwargs):
#     if padding == None: padding = kernel // 2
    return nn.Sequential(nn.ConvTranspose2d(in_size, out_size, *args, **kwargs),
                         nn.BatchNorm2d(out_size),
                         activation(activFunc))

# Faça funções para blocos de convolução especificando funções intermediárias

# def ConvBlocks(encSizes, middleActivationFunc = 'relu', *args, **kwargs):
#     return [ConvBlock(inputSize, outputSize, middleActivationFunc, *args, **kwargs) for
#                                      inputSize, outputSize in zip(encSizes, encSizes[1:])]

def DeconvBlocks(decSizes, middleActivationFunc = 'relu', *args, **kwargs):
    return [ConvBlock(inputSize, outputSize, middleActivationFunc, *args, **kwargs) for
                                     inputSize, outputSize in zip(encSizes, encSizes[1:])]


# class ConvBlocks(encSizes, middleActivationFunc = 'relu', endActivationFunc = 'identity' *args, **kwargs):
#     return nn.Sequential(*[ConvBlock(inputSize, outputSize, middleActivationFunc, *args, **kwargs) for
#                                      inputSize, outputSize in zip(encSizes, encSizes[1:])],
#                          activation(endActivationFunc))

        
# class DeconvBlocks(decSizes, middleActivationFunc = 'relu', endActivationFunc = 'identity', *args, **kwargs):
#     return nn.Sequential(*[ConvBlock(inputSize, outputSize, middleActivationFunc, *args, **kwargs) for
#                                      inputSize, outputSize in zip(encSizes, encSizes[1:])],
#                          activation(endActivationFunc))

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