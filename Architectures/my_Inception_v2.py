import torch
import torch.nn as nn
import numpy as np

# Author: Gabriel Cesar Silveira
# Date: 14/01/2025
# Inception: https://arxiv.org/pdf/1409.4842
# Inception implementation: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

def maxpool2d_output_shape(input_size, kernel_size, stride, padding, dilation=1):
    out_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    return out_size

class ConvBlock(nn.Module):
    """
    Creates a convolutional layer followed by batchNorm and relu. Bias is False as batchnorm will nullify it anyways.

    Args:
        in_channels (int) : input channels of the convolutional layer
        out_channels (int) : output channels of the convolutional layer
        kernel_size (int) : filter size
        stride (int) : number of pixels that the convolutional filter moves
        padding (int) : extra zero pixels around the border which affects the size of output feature map


    Attributes:
        Layer consisting of conv->batchnorm->relu

    """
    def __init__(self, in_channels , out_channels , kernel_size , stride , padding , input_size, bias=False):
        super(ConvBlock,self).__init__()

        # 2d convolution
        self.conv2d = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding , bias=False )

        # batchnorm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        #### ADAPTED FROM RELU TO CELU, FOLLOWING PREVIOUS LITERATURE
        self.celu = nn.CELU()
        
        self.output_size = ((input_size - kernel_size + 2*padding)/stride) + 1
        self.output_shape = (self.output_size,self.output_size,out_channels)
        
        self.n_conv_operations = input_size**2 * in_channels * kernel_size**2 * out_channels
                             
        self.n_trainable_parameters = (input_size**2 * in_channels +1)*out_channels
        
    def forward(self,x):
        return self.celu(self.batchnorm2d(self.conv2d(x)))
    
class InceptionBlock(nn.Module):
   
    def __init__(self , in_channels , b1_out_channels , b2_mid_channels , b2_out_channels , b3_mid_channels , b3_out_channels , b4_out_channels, input_size):
        super(InceptionBlock,self).__init__()

        # ConvBlock: in_channels , out_channels , kernel_size , stride , padding 
        # branch1 : 
        conv_1 = ConvBlock( in_channels,
                            out_channels=b1_out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            input_size=input_size)
        self.branch1 = conv_1

        # branch2 : 
        conv_2_1 = ConvBlock(in_channels,
                             out_channels=b2_mid_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             input_size=input_size)
        conv_2_2 = ConvBlock(b2_mid_channels,
                              out_channels=b2_out_channels,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              input_size=conv_2_1.output_size)
        self.branch2 = nn.Sequential(conv_2_1,conv_2_2)
                                     

        # branch3 :
        conv_3_1=ConvBlock(in_channels,
                            out_channels=b3_mid_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            input_size=input_size)
        conv_3_2=ConvBlock(b3_mid_channels,
                            out_channels=b3_out_channels,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                            input_size=conv_3_1.output_size)
        self.branch3 = nn.Sequential(conv_3_1,conv_3_2)
                                     

        # branch4 : 
        max_p = nn.MaxPool2d(kernel_size=3,
                            stride=1,
                            padding=1)
        
        maxp_out_size = maxpool2d_output_shape(input_size=input_size, 
                                               kernel_size=3,
                                               stride=1,
                                               padding=1)
        
        conv_4 =  ConvBlock(in_channels,
                            out_channels=b4_out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            input_size=maxp_out_size)
        
        self.branch4 = nn.Sequential(max_p,conv_4)
        
                
        if not (conv_1.output_size==conv_2_2.output_size==conv_3_2.output_size==conv_4.output_size):
            raise Exception("Inception block branches with unmatched output sizes, please verify")
        self.output_size = conv_1.output_size
        n_channels = b1_out_channels + b2_out_channels + b3_out_channels + b4_out_channels
        self.output_shape = (self.output_size,self.output_size,n_channels)

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


class Parallel_Inception_Block(nn.Module):
    '''
    step-by-step building the inceptionv1 architecture. Using testInceptionv1 to evaluate the dimensions of output after each layer and deciding the padding number.

    Args:
        in_channels (int) : input channels. 3 for RGB image
        out_channels : number of classes of training dataset

    Attributes:
        inceptionv1 model

    For conv2 2 layers with first having 1x1 conv
    '''

    def __init__(self , in_channels, in_size, out_size):
        super(Parallel_Inception_Block,self).__init__()
        
        # Conv Block: in_channels , out_channels , kernel_size , stride , padding
        # InceptionBlock:  in_channels , b1_out_channels , b2_mid_channels , b2_out_channels , b3_mid_channels , b3_out_channels , b4_out_channels
        
        print("\n\n  Parallel_Inception_Block Initialization:")
        
        print(" - Input shape: ", (in_size,in_size,in_channels))
        # HEAD
        conv_0_1 = ConvBlock(in_channels,
                          out_channels=32,
                          kernel_size=7,
                          stride=1,
                          padding=0,
                          input_size=in_size)
        
        print(" - head_1 output shape: ", conv_0_1.output_shape)
        conv_0_2 = ConvBlock(in_channels=32,
                          out_channels=64,
                          kernel_size=7,
                          stride=1,
                          padding=0,
                          input_size=conv_0_1.output_size)
        print(" - head_2 output shape: ", conv_0_2.output_shape)
        self.head = nn.Sequential(conv_0_1,conv_0_2)

        # FIRST BRANCH
        incp_1 = InceptionBlock(in_channels=64,
                                b1_out_channels=64,
                                b2_mid_channels=64,
                                b2_out_channels=128,
                                b3_mid_channels=32,
                                b3_out_channels=16,
                                b4_out_channels=128,
                                input_size=conv_0_2.output_size)
        print(" - incp_1 output shape: ", incp_1.output_shape)
        maxp_1 =  nn.MaxPool2d(kernel_size=3,
                      stride=2,
                      padding=1)
        maxp_1_out_size = maxpool2d_output_shape(input_size=incp_1.output_size, 
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1)
        print(" - maxp_1 output shape: ", (maxp_1_out_size,maxp_1_out_size,incp_1.output_shape[2]))
        self.branch_1 = nn.Sequential(incp_1,maxp_1)

        # SECOND BRANCH
        incp_2 = InceptionBlock(in_channels=64,
                                b1_out_channels=64,
                                b2_mid_channels=64,
                                b2_out_channels=128,
                                b3_mid_channels=32,
                                b3_out_channels=16,
                                b4_out_channels=128,
                                input_size=conv_0_2.output_size)
        print(" - incp_2 output shape: ", incp_2.output_shape)
        maxp_2 = nn.MaxPool2d(kernel_size=3,
                     stride=2,
                     padding=1)
        maxp_2_out_size = maxpool2d_output_shape(input_size=incp_2.output_size, 
                                                   kernel_size=3,
                                                   stride=2,
                                                   padding=1)
        print(" - maxp_2 output shape: ", (maxp_2_out_size,maxp_2_out_size,incp_2.output_shape[2]))
        self.branch_2 = nn.Sequential(incp_2,maxp_2)
                                      
        
        
        # TAIL 
        branches_concat_size = int(maxp_1_out_size**2*incp_1.output_shape[2] + maxp_2_out_size**2*incp_2.output_shape[2])
        
        print(" - concat output shape: ", branches_concat_size)
        self.tail =  nn.Sequential(nn.Linear(branches_concat_size,out_size**2),
                                   nn.Dropout(p=0.4))
        print(" - linear_tail output shape: ", out_size**2)
                
        self.output_layer = nn.Linear(out_size**2, out_size**2)
        print(" - linear_output shape: ", out_size**2)
        
        self.output_size = out_size**2
        self.output_shape = (out_size**2,out_size**2,1) 
        self.n_conv_operations = conv_0_1.n_conv_operations +conv_0_2.n_conv_operations +incp_1.n_conv_operations +incp_2.n_conv_operations
        self.n_trainable_parameters = conv_0_1.n_trainable_parameters +conv_0_2.n_trainable_parameters +incp_1.n_trainable_parameters +incp_2.n_trainable_parameters + (branches_concat_size*out_size) + (out_size*out_size)
        

    # 0 elements must be rock, and 1 void space
    def forward(self,x):
        
        #print("\nForward operation: ")
        #print("Input: ",x.shape)
        y = self.head(x)
        #print("Head output: ",x.shape)
        
        y = torch.cat([self.branch_1(y),self.branch_2(y)],dim=1)
        #print("Branches concat output: ",x.shape)
        
        y = torch.flatten(y, start_dim=1)
        #print("Concat flatten output: ",x.shape)

        y = self.tail(y)
        #print("Tail output: ",x.shape)
        
        y = self.output_layer(y)
        #print("Output: ", x.shape)
        
        return torch.mul(torch.flatten(x, start_dim=1), y)