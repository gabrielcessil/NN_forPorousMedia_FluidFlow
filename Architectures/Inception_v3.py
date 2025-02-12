import torch
import torch.nn as nn
import numpy as np

# Author: Gabriel Cesar Silveira
# Date: 14/01/2025
# Inception: https://arxiv.org/pdf/1409.4842
# Inception implementation: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

class PoolingBlock(nn.Module):
    def __init__(self, input_size, kernel_size, stride=1, padding=None, dilation=1):
        super(PoolingBlock,self).__init__()
        
        
        if not (isinstance(input_size,int)): raise Exception("ConvBlock parameter input_size must be integer.")
        if not (isinstance(kernel_size,int)): raise Exception("ConvBlock parameter kernel_size must be integer.")
        if not (isinstance(stride,int)): raise Exception("ConvBlock parameter stride must be integer.")
        if not (isinstance(dilation,int)): raise Exception("ConvBlock parameter dilation must be integer.")
        if not (isinstance(padding,int)) and padding is not None: raise Exception("ConvBlock parameter stride must be integer or None.")


        
        
        if padding is None: padding = int((input_size*(stride-1) + dilation*(kernel_size-1))//2)
        # branch4 : 
        self.max_p = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        
        self.output_size = self.get_output_size(
            input_size=input_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    def get_output_size(self, input_size, kernel_size, stride, padding, dilation): 
        return int((input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
        
    def forward(self,x): 
        return self.max_p(x)
        
        

class ConvBlock(nn.Module):

    def __init__(self, input_size, in_channels, out_channels , kernel_size , stride=1 , padding=None, bias=False):
        super(ConvBlock,self).__init__()
        

        if not (isinstance(input_size,int)): raise Exception("ConvBlock parameter input_size must be integer.")
        if not (isinstance(in_channels,int)): raise Exception("ConvBlock parameter in_channels must be integer.")
        if not (isinstance(out_channels,int)): raise Exception("ConvBlock parameter out_channels must be integer.")
        if not (isinstance(kernel_size,int)): raise Exception("ConvBlock parameter kernel_size must be integer.")
        if not (isinstance(stride,int)): raise Exception("ConvBlock parameter stride must be integer.")
        if not (isinstance(padding,int)) and padding is not None: raise Exception("ConvBlock parameter stride must be integer or None.")
            
        if padding==None: padding = int(((stride-1)*input_size -stride +kernel_size)//2)

        # 2d convolution
        self.conv2d = nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                bias=False )

        # batchnorm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        #### ADAPTED FROM RELU TO CELU, FOLLOWING PREVIOUS LITERATURE
        self.celu = nn.Tanh()
        
        self.output_size = int(((input_size - kernel_size + 2*padding)/stride) + 1)
        self.output_channels = out_channels
        self.output_shape = (self.output_size,self.output_size,out_channels)
        
        self.n_conv_operations = int(input_size**2 * in_channels * kernel_size**2 * out_channels)
                             
        self.n_trainable_parameters = int((input_size**2 * in_channels +1)*out_channels)
        
    def forward(self,x):
        return self.celu(self.batchnorm2d(self.conv2d(x)))


class InceptionBlock(nn.Module):
   
    def __init__(self , input_size, in_channels , b1_out_channels , b2_mid_channels , b2_out_channels , b3_mid_channels , b3_out_channels , b4_out_channels):
        super(InceptionBlock,self).__init__()

        if not (isinstance(input_size,int)): raise Exception("ConvBlock parameter input_size must be integer.")
        if not (isinstance(in_channels,int)): raise Exception("ConvBlock parameter in_channels must be integer.")
        if not (isinstance(b1_out_channels,int)): raise Exception("ConvBlock parameter b1_out_channels must be integer.")
        if not (isinstance(b2_mid_channels,int)): raise Exception("ConvBlock parameter b2_mid_channels must be integer.")
        if not (isinstance(b2_out_channels,int)): raise Exception("ConvBlock parameter b2_out_channels must be integer.")
        if not (isinstance(b3_mid_channels,int)): raise Exception("ConvBlock parameter b3_mid_channels must be integer.")
        if not (isinstance(b3_out_channels,int)): raise Exception("ConvBlock parameter b3_out_channels must be integer.")
        if not (isinstance(b4_out_channels,int)): raise Exception("ConvBlock parameter b4_out_channels must be integer.")

        
        # ConvBlock: in_channels , out_channels , kernel_size , stride , padding 
        # branch1 : 
        conv_1 = ConvBlock( 
            input_size=input_size,
            in_channels=in_channels,
            out_channels=b1_out_channels,
            kernel_size=1
            )
        self.branch1 = conv_1

        # branch2 : 
        conv_2_1 = ConvBlock(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=b2_mid_channels,
            kernel_size=1
            )
        
        conv_2_2 = ConvBlock(
            input_size=conv_2_1.output_size,
            in_channels=b2_mid_channels,
            out_channels=b2_out_channels,
            kernel_size=3
            )
        self.branch2 = nn.Sequential(conv_2_1,conv_2_2)
                                     

        # branch3 :
        conv_3_1=ConvBlock(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=b3_mid_channels,
            kernel_size=1
            )
        conv_3_2=ConvBlock(
            input_size=conv_3_1.output_size,
            in_channels=b3_mid_channels,
            out_channels=b3_out_channels,
            kernel_size=5
            )
        self.branch3 = nn.Sequential(conv_3_1,conv_3_2)

        max_p = PoolingBlock(
            input_size=input_size,
            kernel_size=3)
        
        
        conv_4 =  ConvBlock(
            input_size=max_p.output_size,
            in_channels=in_channels,
            out_channels=b4_out_channels,
            kernel_size=1
            )

        self.branch4 = nn.Sequential(max_p,conv_4)
        
                
        if not (conv_1.output_size==conv_2_2.output_size==conv_3_2.output_size==conv_4.output_size):
            raise Exception(f"Inception block branches with unmatched output sizes, please verify:\nconv_1.output_size: {conv_1.output_size}, conv_2.output_size: {conv_2_2.output_size}, conv_3.output_size: {conv_3_2.output_size}, conv_4.output_size: {conv_4.output_size}")
        
        
        self.output_size = conv_1.output_size
        n_channels = b1_out_channels + b2_out_channels + b3_out_channels + b4_out_channels
        self.output_shape = (self.output_size,self.output_size,n_channels)
        self.output_channels = n_channels
        self.n_conv_operations = conv_1.n_conv_operations +conv_2_1.n_conv_operations +conv_2_2.n_conv_operations +conv_3_1.n_conv_operations +conv_3_2.n_conv_operations +conv_4.n_conv_operations
        self.n_trainable_parameters = conv_1.n_trainable_parameters +conv_2_1.n_trainable_parameters +conv_2_2.n_trainable_parameters +conv_3_1.n_trainable_parameters +conv_3_2.n_trainable_parameters +conv_4.n_trainable_parameters
        
    def forward(self,x):

        # Parallel signals 
        # concatenation from dim=1 as dim=0 represents batchsize
        
        return torch.cat(
            [self.branch1(x),
             self.branch2(x),
             self.branch3(x),
             self.branch4(x)],dim=1)




# Main model block 
# output_masks: list of callable functions with size of output channels.
# Each listed function will be applied to the channel for prediction mode
# If None (default), no mask is applied
class MODEL(nn.Module):

    def __init__(self, in_shape, out_shape, output_masks=None):
        super(MODEL,self).__init__()

        model1 = InceptionBlock(
            input_size=in_shape[1],
            in_channels=in_shape[0],
            b1_out_channels=10,
            b2_mid_channels=5,
            b2_out_channels=10,
            b3_mid_channels=5,
            b3_out_channels=10,
            b4_out_channels=10
            )
        
        model2 = ConvBlock(
            input_size=model1.output_size,
            in_channels=model1.output_channels, 
            out_channels=out_shape[0],
            kernel_size=1
            )
                
        tail =  nn.Tanh() #nn.Sigmoid()
        
        self.model = nn.Sequential(model1, model2, tail)
        
        self.output_masks = output_masks
        
    def forward(self,x):
        return self.model(x)
    
    def predict(self, x):

        self.model.eval()
        y = self.forward(x).detach()
        
        if self.output_masks is not None:
            # Ensure the number of functions matches the number of channels
            if y.shape[1] != len(self.output_masks): raise Exception(f"Output masks must match the number of channels, but {len(self.output_masks)} masks for {y.shape[1]} channels happened in output with shape {y.shape}")
                
            # Apply each function to its corresponding channel
            processed_channels = [func(y[:, ch, :, :]) for ch, func in enumerate(self.output_masks)]
            
            # Stack back into a single tensor
            y = torch.stack(processed_channels, dim=1)
        
        return y
    