import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, pad_size = 0):
        super(ConvBlock, self).__init__()
        if pad_size == None: pad_size = kernel // 2
        self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, kernel, stride = stride,
                                             padding = pad_size, bias = False),
                                   nn.BatchNorm2d(out_size),
                                   nn.ReLU(inplace = True))
        
    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs
        
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
        ['lrelu', nn.LeakyReLU()],
        ['relu', nn.ReLU(inplace = True)],
        ['identity', nn.Identity()],
        ['sigmoid', nn.Sigmoid()]
    ])
    return activations[activationFunction]

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

class ConvBlocks(nn.Module):
    def __init__(self, encSizes, *args, **kwargs):
        super().__init__()
        self.convblocks = nn.Sequential(*[ConvBlock(inputSize, outputSize, *args, **kwargs) for
                                          inputSize, outputSize in zip(encSizes, encSizes[1:])])
    
    def forward(self, X):
        return self.convblocks(X)
        
class DeconvBlocks(nn.Module):
    def __init__(self, decSizes, *args, **kwargs):
        super().__init__()
        self.decovblocks = nn.Sequential(*[DeconvBlock(inputSize, outputSize, *args, **kwargs) for
                                          inputSize, outputSize in zip(decSizes, decSizes[1:])])