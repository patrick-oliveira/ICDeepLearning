import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, stride, kernel_n):
        super(Autoencoder, self).__init__()
        n1 = kernel_n
        s = stride
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size = (1, 2*n1 + 1), padding = (0, n1), stride = (1, s)),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace = True)
        )
        
        self.pool    = nn.MaxPool2d(kernel_size = (1, 2), stride = (1, 2), return_indices = True)
        self.unpool  = nn.MaxUnpool2d(kernel_size = (1, 2), stride = (1, 2))
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(3, 1, kernel_size = (1, 2*n1 + 1), padding = (0, n1), stride = (1, s), output_padding = (0, s - 1))
        )
        
    
    def forward(self, x):
        x = self.encoder(x)
        x, _2 = self.pool(x)
        x = self.unpool(x, _2)
        x = self.decoder(x)
        
        return x
    
    def encode(self, x):
        with torch.set_grad_enabled(False):
            x = self.encoder(x)
            x, _ = self.pool(x)
            
        return x
