# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 15:12:57 2021

@author: olipp
"""
import sys
sys.path.append('Scripts')

from torch import nn
from Components import SeparableDeconv2D, DepthwiseConv2D, SeparableConv2D, Autoencoder, DepthwiseSeparableConv2D, DepthwiseSeparableDeconv2D

class SimpleAutoencoder(Autoencoder):
    def build_encoder(self, n_filters:int, kernel_size:int, stride:int):
        assert kernel_size %2 != 0, 'kernel_size must be an odd number in order to auto padding to work'
        encoder = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size = (1, kernel_size), padding = (0, kernel_size//2), stride = (1, stride)),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(inplace = True),
            nn.Conv2d(n_filters, n_filters, kernel_size = (1, 3), padding = (0, 1), stride = (1, 2))
        )
        
        return encoder
        
    def build_decoder(self, n_filters:int, kernel_size:int, stride:int):
        assert kernel_size %2 != 0, 'kernel_size must be an odd number in order to auto padding to work'
        decoder = nn.Sequential(
            nn.ConvTranspose2d(n_filters, n_filters, kernel_size = (1, 3), padding = (0, 1), stride = (1, 2), output_padding = (0, 1)),
            nn.ReLU(inplace = True),
            nn.BatchNorm2d(n_filters),
            nn.ConvTranspose2d(n_filters, 1, kernel_size = (1, kernel_size), padding = (0, kernel_size//2), stride = (1, stride), output_padding = (0, stride - 1)),
        )
        
        return decoder

class SimpleSepDepthAutoencoder(Autoencoder):
    def build_encoder(self, n_filters:int, kernel_size:int, depth:int):
        assert kernel_size % 2 != 0, 'kernel_size must be an odd number in order to auto padding to work'
        separable_convolution = SeparableConv2D(1, n_filters, kernel_size = (16, kernel_size), padding = (0, kernel_size//2))
        depthwise_convolution = DepthwiseConv2D(n_filters, depth*n_filters, kernel_size = (1, 5), padding = (0, 5//2), stride = (1, 2))
        
        return nn.Sequential(separable_convolution,
                             depthwise_convolution,
                             Transpose_1_2())
        
    def build_decoder(self, n_filters:int, kernel_size:int, depth:int):
        depthwise_deconvolution = nn.ConvTranspose2d(depth*n_filters, n_filters, kernel_size = (1, 5), padding = (0, 5//2), stride = (1, 2), output_padding = (0, 1))
        separable_deconvolution = SeparableDeconv2D(n_filters, 1, kernel_size = (16, kernel_size), padding = (0, kernel_size//2))
        
        return nn.Sequential(Transpose_1_2(),
                             depthwise_deconvolution,
                             separable_deconvolution)
    
    
class Transpose_1_2(nn.Module):
    def __init__(self):
        super(Transpose_1_2, self).__init__()
        
    def forward(self, X):
        return X.transpose(1, 2)
    
    
class PicSepDepthAutoencoder(Autoencoder):
    def build_encoder(self, n_filters:int, kernel_size:int):
        assert kernel_size % 2 != 0, 'kernel_size must be an odd number in order to auto padding to work'
        
        separable_convolution = SeparableConv2D(1, n_filters, kernel_size = (16, kernel_size), padding = (0, kernel_size // 2), stride = (1, 1))
        conv_size_reduction = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size = (1, 5), padding = (0, 2), stride = (1, 2)),
            nn.Conv2d(n_filters, n_filters, kernel_size = (1, 5), padding = (0, 2), stride = (1, 2))
        )
        make_pic = Transpose_1_2()        
            
        return nn.Sequential(separable_convolution,
                             conv_size_reduction,
                             make_pic)
        
    def build_decoder(self, n_filters:int, kernel_size:int):
        unmake_pic = Transpose_1_2()
        conv_size_increase = nn.Sequential(
            nn.ConvTranspose2d(n_filters, n_filters, kernel_size = (1, 5), padding = (0, 2), stride = (1, 2), output_padding = (0, 1)),
            nn.ConvTranspose2d(n_filters, n_filters, kernel_size = (1, 5), padding = (0, 2), stride = (1, 2), output_padding = (0, 1))
        )
        separable_deconvolution = SeparableDeconv2D(n_filters, 1, kernel_size = (16, kernel_size), padding = (0, kernel_size//2), stride = (1, 1))
        
        return nn.Sequential(unmake_pic,
                             conv_size_increase,
                             separable_deconvolution)
    
class SepConvAutoencoder(Autoencoder):
    def build_encoder(self, n_filters, kernel_size):
        first_conv = nn.Conv2d(1, n_filters, kernel_size = (1, kernel_size), padding = (0, kernel_size//2))
        separable_convolution = SeparableConv2D(n_filters, n_filters, kernel_size = (64, kernel_size), padding = (0, kernel_size//2), stride = (1, 2))
        
        return nn.Sequential(first_conv,
                             separable_convolution)
    
    def build_decoder(self, n_filters, kernel_size):
        separable_deconvolution = SeparableDeconv2D(n_filters, n_filters, kernel_size = (64, kernel_size), padding = (0, kernel_size//2), stride = (1, 2))
        deconv = nn.ConvTranspose2d(n_filters, 1, kernel_size = (1, kernel_size), padding = (0, kernel_size//2))
        
        return nn.Sequential(separable_deconvolution,
                             deconv)
    
class mAutoencoder(Autoencoder):
    def build_encoder(self, depth:int, kernel_size, n_filters: int):
        sep_conv = SeparableConv2D(1, n_filters, kernel_size = (16, kernel_size), padding =  (0, kernel_size//2))
        # High-level feature maps, time compression
        high_level_features = DepthwiseSeparableConv2D(n_filters, depth * n_filters, depth = depth, kernel_size = (1, kernel_size), padding = (0, kernel_size//2),
                                                       stride = (1, 2))
        # Pic Encoding
        pic_encoding = Transpose_1_2()
        
        return nn.Sequential(
            nn.Dropout(0.25),
            sep_conv,
            nn.ELU(),
            nn.Dropout(0.25),
            high_level_features,
            nn.BatchNorm2d(depth * n_filters),
            nn.ELU(),
            pic_encoding,
        )
    
    def build_decoder(self, depth:int, kernel_size:int, n_filters:int):
        # Pic Decoding
        pic_decoding = Transpose_1_2()
        # Info reconstruction (really necessary?)
        high_level_features_decoding = DepthwiseSeparableDeconv2D(depth * n_filters, n_filters, kernel_size = (1, kernel_size), padding = (0, kernel_size // 2),
                                                                  stride = (1, 2))
        sep_deconv = SeparableDeconv2D(n_filters, 1, kernel_size = (16, kernel_size), padding = (0, kernel_size//2))
        
        return nn.Sequential(
            pic_decoding,
            high_level_features_decoding,
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            sep_deconv,
            nn.ELU(),
        )

class MiddleFlow(nn.Module):
    def __init__(self, num_maps, kernel_size):
        super(MiddleFlow, self).__init__()
        self.flow = nn.Sequential(
            nn.ReLU(),
            DepthwiseSeparableConv2D(num_maps, num_maps, kernel_size, padding = 1),
            nn.ReLU(),
            DepthwiseSeparableConv2D(num_maps, num_maps, kernel_size, padding = 1),
            nn.ReLU(),
            DepthwiseSeparableConv2D(num_maps, num_maps, kernel_size, padding = 1)
        )
    
    def forward(self, X):
        return X + self.flow(X)
    
class ResidualDepthwiseSepBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDepthwiseSepBlock, self).__init__()
        self.flow = nn.Sequential(
            DepthwiseSeparableConv2D(out_channels, out_channels, (1, 3), padding = (0, 1)),
            nn.ELU(),
            nn.Conv2d(out_channels, out_channels, (1, 3), stride = (1, 2), padding = (0, 1), groups = out_channels)
        )
        self.residue = nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = (1, 2))
        
    def forward(self, X):
        return self.flow(X) + self.residue(X)
    
class ResidualDepthwiseSepDeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualDepthwiseSepDeconvBlock, self).__init__()
        self.flow = nn.Sequential(
            DepthwiseSeparableDeconv2D(out_channels, out_channels, kernel_size = (1, 3), padding = (0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(out_channels, out_channels, (1, 3), stride = (1, 2), padding = (0, 1), output_padding = (0, 1), groups = out_channels)
        )
        self.residue = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 1, stride = (1, 2), output_padding = (0, 1))
        
    def forward(self, X):
        return self.flow(X) + self.residue(X)
    
class DeepXAutoencoder(Autoencoder):
    def build_encoder(self, depth:int, kernel_size, n_filters: int):
        
        separable_conv = SeparableConv2D(1, n_filters, kernel_size = (16, kernel_size), padding = (0, kernel_size//2))
        
        high_level_features = ResidualDepthwiseSepBlock(n_filters, depth*n_filters)
        # Pic Encoding
        pic_encoding = Transpose_1_2()
        
        return nn.Sequential(
            nn.Dropout(0.25),
            separable_conv,
            nn.ELU(),
            nn.Dropout(0.25),
            high_level_features,
            nn.BatchNorm2d(depth * n_filters),
            nn.ELU(),
            pic_encoding,
        )
    
    def build_decoder(self, depth:int, kernel_size, n_filters: int):
        # Pic Decoding
        pic_decoding = Transpose_1_2()
        # Info reconstruction (really necessary?)
        high_level_features_decoding = ResidualDepthwiseSepDeconvBlock(depth * n_filters, n_filters)
        
        separable_deconv = SeparableDeconv2D(n_filters, 1, kernel_size = (16, kernel_size), padding = (0, kernel_size//2))
        
        return nn.Sequential(
            pic_decoding,
            high_level_features_decoding,
            nn.BatchNorm2d(n_filters),
            nn.ELU(),
            separable_deconv,
        )