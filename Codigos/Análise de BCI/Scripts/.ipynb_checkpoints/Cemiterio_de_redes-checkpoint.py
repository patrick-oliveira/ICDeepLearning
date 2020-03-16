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