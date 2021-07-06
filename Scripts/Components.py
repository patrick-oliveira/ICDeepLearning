from torch import nn

from typing import Tuple

class Autoencoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Autoencoder, self).__init__()
        self.encoder = self.build_encoder(*args, **kwargs)
        self.decoder = self.build_decoder(*args, **kwargs)
        
    def forward(self, X):
        return self.decoder(self.encoder(X))
    
    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError()
    
    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError()
    


class DepthwiseConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size: Tuple[int, int], 
                 padding:Tuple[int, int] = (0, 0), stride:Tuple[int, int] = (1, 1), bias: bool = False):
        '''
        

        Parameters
        ----------
        in_channels : int
            DESCRIPTION.
        out_channels : int
            DESCRIPTION.
        kernel_size : Tuple[int, int]
            DESCRIPTION.
        padding : int, optional
            DESCRIPTION. The default is 0.
        stride : int, optional
            DESCRIPTION. The default is 1.
        bias : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        assert out_channels%in_channels == 0, '#out_channels is not divisible by #in_channels'
        
        super(DepthwiseConv2D, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, out_channels,
                                   kernel_size = kernel_size, padding = padding, stride = stride,
                                   groups = in_channels, bias = bias)
        
    def forward(self, X):
        return self.depthwise(X)
    
    
class DepthwiseSeparableConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Tuple[int, int],
                 depth:int = 1, padding: Tuple[int, int] = (0, 0), stride:Tuple[int, int] = (1, 1), bias:bool = False):
        '''
        

        Parameters
        ----------
        in_channels : int
            DESCRIPTION.
        out_channels : int
            DESCRIPTION.
        kernel_size : Tuple[int, int]
            DESCRIPTION.
        depth : int, optional
            DESCRIPTION. The default is 1.
        padding : int, optional
            DESCRIPTION. The default is 0.
        stride : int, optional
            DESCRIPTION. The default is 1.
        bias : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        super(DepthwiseSeparableConv2D, self).__init__()
            
        depthwise = nn.Conv2d(in_channels, in_channels, 
                              kernel_size = kernel_size,  padding = padding, stride = stride, 
                              groups = in_channels, bias = bias)
        pointwise = nn.Conv2d(in_channels, depth * out_channels, kernel_size = 1, bias = bias)
        
        self.depthwise_separable_convolution = nn.Sequential(depthwise,
                                                             pointwise)
        
    def forward(self, X):
        return self.depthwise_separable_convolution(X)
    
    
class SeparableConv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size: Tuple[int, int], 
                 padding: Tuple[int, int] = (0, 0), stride: Tuple[int, int] = (1, 1), bias: bool = False):
        '''
        

        Parameters
        ----------
        in_channels : int
            DESCRIPTION.
        out_channels : int
            DESCRIPTION.
        kernel_size : Tuple[int, int]
            DESCRIPTION.
        padding : Tuple(int, int), optional
            DESCRIPTION. The default is (0, 0).
        stride : Tuple(int, int), optional
            DESCRIPTION. The default is (1, 1).
        bias : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        super(SeparableConv2D, self).__init__()

        horizontal_convolution = nn.Conv2d(in_channels, out_channels, 
                                           kernel_size = (1, kernel_size[1]), padding = (0, padding[1]), stride = (1, stride[1]),
                                           groups = in_channels, bias = bias)
        vertical_convolution = nn.Conv2d(out_channels, out_channels, 
                                         kernel_size = (kernel_size[0], 1), padding = (padding[0], 0), stride = (stride[0], 1),
                                         groups = out_channels, bias = bias)
        
        self.separable_convolution = nn.Sequential(horizontal_convolution,
                                                   vertical_convolution,
                                                   nn.BatchNorm2d(out_channels)
                                                  )
        
    def forward(self, X):
        return self.separable_convolution(X)
    
    
class DepthwiseSeparableDeconv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size:Tuple[int, int],
                 padding: Tuple[int, int] = (0, 0), stride:Tuple[int, int] = (1, 1), bias:bool = False):
        '''
        

        Parameters
        ----------
        in_channels : int
            DESCRIPTION.
        out_channels : int
            DESCRIPTION.
        kernel_size : Tuple[int, int]
            DESCRIPTION.
        depth : int, optional
            DESCRIPTION. The default is 1.
        padding : Tuple[int, int], optional
            DESCRIPTION. The default is (0, 0).
        stride : Tuple[int, int], optional
            DESCRIPTION. The default is (1, 1).
        bias : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        super(DepthwiseSeparableDeconv2D, self).__init__()
        pointwise = nn.Conv2d(in_channels, out_channels, kernel_size = (1, 1), padding = (0, 0), bias = bias)
        depthwise = nn.ConvTranspose2d(out_channels, out_channels, 
                                       kernel_size = kernel_size, padding = padding, stride = stride, 
                                       groups = out_channels, bias = bias, output_padding = (stride[0] - 1, stride[1] - 1))
        
        self.depthwise_separable_deconvolution = nn.Sequential(pointwise,
                                                               depthwise)
        
    def forward(self, X):
        return self.depthwise_separable_deconvolution(X)
    
class SeparableDeconv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size: Tuple[int, int], 
                 padding: Tuple[int, int] = (0, 0), stride: Tuple[int, int] = (1, 1), bias: bool = False):
        '''
        

        Parameters
        ----------
        in_channels : int
            DESCRIPTION.
        out_channels : int
            DESCRIPTION.
        kernel_size : Tuple[int, int]
            DESCRIPTION.
        padding : Tuple[int, int], optional
            DESCRIPTION. The default is (0, 0).
        stride : Tuple[int, int], optional
            DESCRIPTION. The default is (1, 1).
        bias : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        '''
        super(SeparableDeconv2D, self).__init__()
        
        vertical_deconvolution = nn.ConvTranspose2d(in_channels, in_channels, kernel_size = (kernel_size[0], 1), padding = (padding[0], 0), 
                                                    stride = (stride[0], 1), output_padding = (stride[0] - 1, 0), groups = in_channels)
        horizontal_deconvolution = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = (1, kernel_size[1]), padding = (0, padding[1]),
                                                     stride = (1, stride[1]), output_padding = (0, stride[1] - 1))
        
        self.separable_deconvolution = nn.Sequential(vertical_deconvolution,
                                                     horizontal_deconvolution)
        
    def forward(self, X):
        return self.separable_deconvolution(X)
    
    
class DepthwiseDeconv2D(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size: Tuple[int, int], 
                 padding:Tuple[int, int] = (0, 0), stride:Tuple[int, int] = (1, 1), bias: bool = False):
        assert out_channels%in_channels == 0, '#out_channels is not divisible by #in_channels'
        super(DepthwiseDeconv2D, self).__init__()
        
        self.depthwise_deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                                   kernel_size = kernel_size, padding = padding, stride = stride, output_padding = (stride[0] - 1, stride[1] - 1),
                                                   groups = in_channels, bias = bias)
        
    def forward(self, X):
        return self.depthwise_deconv(X)
    
    
class TemporalPadding(nn.Module):
    def __init__(self, padding: Tuple[int, int]):
        super(TemporalPadding, self).__init__()
        self.padding = padding
        
    def forward(self, X):
        return nn.functional.pad(X, self.padding)