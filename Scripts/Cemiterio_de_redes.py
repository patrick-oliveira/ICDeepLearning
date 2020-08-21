# Tentativa 1

# class Autoencoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             ConvBlock(1, 16, kernel_size = [5,5], padding = [0, 2]),
#         )
        
#         self.decoder = nn.Sequential(
#             DeconvBlock(16, 1, kernel_size = [5,5], padding = [0, 2])
#         )
        
#     def forward(self, X):
#         X = self.encoder(X)
#         X = self.decoder(X)
#         return X


# Tentativa 2

# class _DenseBlock(nn.Module):
#     def __init__(self, firstChannels, fixedSizeOutChannels, layerSize, kernel_size):
#         super().__init__()
        
#         inSizes = [firstChannels, fixedSizeOutChannels]
#         outSizes = [fixedSizeOutChannels]*layerSize
#         self.relu = nn.ReLU()
#         for i in range(layerSize - 2):
#             inSizes.append(inSizes[-1] + fixedSizeOutChannels)
            
#         self.BN = nn.BatchNorm2d(num_features = firstChannels)
#         self.ConvLayers = ConvBlocks(inSizes, outSizes, 
#                                      kernel_size = (kernel_size, 1), stride = 1, padding = (kernel_size//2, 0))
    
#     def forward(self, X):
#         X = self.BN(X)
#         output = [self.ConvLayers[0](X)]
#         dense  = self.relu(torch.cat(output, 1))
#         for conv in self.ConvLayers[1:]:
#             Y = conv(dense)
#             output.append(Y)
#             dense = self.relu(torch.cat(output, 1))

#         return dense    
    
# class _TransitionLayer(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
        
#         self.relu = nn.ReLU(inplace = True)
#         self.bn   = nn.BatchNorm2d(num_features = out_channels)
#         self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
#                               kernel_size = 1, bias = False)
#         self.avg_pool = nn.AvgPool2d(kernel_size = (2, 1), stride = 2, padding = 0)
        
#     def forward(self, X):
#         bn = self.bn(self.relu(self.conv(X)))
#         out = bn
        
#         return out
    
# class Resize(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self, X):
#         size = X.shape[0]
#         return X.view(size, 1, 512, 16)

# class Autoencoder(nn.Module):
#     def __init__(self,):
#         super(Autoencoder, self).__init__()
        
#         self.encoder = nn.Sequential(
#             _DenseBlock(1, 32, 3, 11),
#             _TransitionLayer(3*32, 32),
#             ConvBlock(32, 32, kernel_size = (77, 1))
#         )
#         self.decoder = nn.Sequential(
#             DeconvBlock(32, 1, kernel_size = (77, 1)),
#         )
        
#     def forward(self, X):
#         X = self.encoder(X)
#         X = self.decoder(X)
#         return X
