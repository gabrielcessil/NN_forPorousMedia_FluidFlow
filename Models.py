import torch
import torch.nn as nn
import numpy as np

# Author: Gabriel Cesar Silveira
# Date: 14/01/2025
# Inception: https://arxiv.org/pdf/1409.4842
# Inception implementation: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

import Architectures.FunctionalModels as fm





#######################################################
#************ FULL MODELS ****************************#
# Models that can be directly used to this problem,   #
# providing the proper input and output.              #
# However, there are no restrictions on using them    #
# to build other models                               #
#######################################################

# Main model block 
# output_masks: list of callable functions with size of output channels.
# Each listed function will be applied to the channel for prediction mode
# If None (default), no mask is applied
class INCEPTION_MODEL(fm.BASE_MODEL):

    def __init__(self, 
                 in_shape, 
                 out_shape, 
                 output_masks=None,
                 tail=None, 
                 b1_out_channels=10,
                 b2_mid_channels=5,
                 b2_out_channels=10,
                 b3_mid_channels=5,
                 b3_out_channels=10,
                 b4_out_channels=10,
                 tail_kernel_size = 1):
        
        super(INCEPTION_MODEL,self).__init__(in_shape, out_shape, output_masks)
        
        # Creating model's blocks
        model1 = fm.InceptionBlock(
            input_size=in_shape[1],
            in_channels=in_shape[0],
            b1_out_channels=b1_out_channels,
            b2_mid_channels=b2_mid_channels,
            b2_out_channels=b2_out_channels,
            b3_mid_channels=b3_mid_channels,
            b3_out_channels=b3_out_channels,
            b4_out_channels=b4_out_channels
            )
        
        model2 = fm.ConvBlock(
            input_size=model1.output_size,
            in_channels=model1.out_channels, 
            out_channels=out_shape[0],
            kernel_size=tail_kernel_size
            )
                
        if tail is None:
            self.tail = nn.Sigmoid()
        else:
            self.tail = tail
        
        self.model = nn.Sequential(model1, model2, self.tail)
        
        # Update metadata dictionary with inception model parameters
        self.metadata.update({
            "Inception Block": {
                "b1_out_channels": b1_out_channels,
                "b2_mid_channels": b2_mid_channels,
                "b2_out_channels": b2_out_channels,
                "b3_mid_channels": b3_mid_channels,
                "b3_out_channels": b3_out_channels,
                "b4_out_channels": b4_out_channels
            },
            "Tail Conv Kernel Size": tail_kernel_size
        })
        
        
        

        
    
    def encoder(self, rock, estimative=None):
        # Rock branch
        rock_pooling = self.rock_pooling(rock)
        # Encoder branch
        # - Rock Branch
        y1 = self.enc_rock_estimator(rock)

        # - Estimation Branch
        if self.estimative_signal:
            y2 = self.enc_estimator(rock)
            self.encoder_upscaled_output = self.enc_output_concat(y1,y2)
        else: 
            self.encoder_upscaled_output = y1

        # - Output scaling
        processed_estimate = self.enc_pooling(self.encoder_upscaled_output)
        return rock_pooling, processed_estimate
        
    def decoder(self, upcoming_processed_estimate):
        y = self.dec_upsample(upcoming_processed_estimate)
        y = self.dec_concat(y, self.encoder_upscaled_output)
        y = self.dec_conv(y)
        return y
    
"""  
class BLOCK_2(BASE_MODEL):
    def __init__(self, in_shape, out_shape, output_masks=None, enc_decay=2, add_channels=2, estimative_signal=False):
        super(BLOCK_2,self).__init__(in_shape, out_shape, output_masks)
        
        self.estimative_signal = estimative_signal
        
        # ENCODING:
        # Rock branch:
        self.rock_pooling = fm.PoolingBlock(
            input_size=self.input_size,
            in_channels=self.in_channels,
            kernel_size=3,
            output_size=self.output_size//enc_decay,
            stride=2
            )
        self.rock_output_shape = self.rock_pooling.out_shape


        # Estimation branch (Upcoming feature: receive the inception block Pre-trained):
        self.enc_estimator = INCEPTION_MODEL(in_shape, out_shape, output_masks)
        
        # Encoding processed info branch:
        if estimative_signal:    
            print("::Estimative branch enabled")
            self.enc_input_concat = fm.ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= self.in_channels + self.out_channels
                )
            self.enc_conv = fm.ConvBlock(
                input_size=self.input_size, 
                in_channels=self.enc_input_concat.out_channels, 
                out_channels=self.in_channels +self.out_channels +add_channels, 
                kernel_size=3, 
                stride=1, 
                )
            self.enc_output_concat = fm.ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= self.enc_estimator.out_channels + self.enc_conv.out_channels
                )
            self.enc_pooling = fm.PoolingBlock(
                input_size=self.input_size,
                in_channels=self.enc_output_concat.out_channels,
                kernel_size=3,
                output_size=self.output_size//enc_decay,
                stride=2
                )
        else:
            print("::Estimative branch disabled")
            self.enc_pooling = fm.PoolingBlock(
                input_size=self.input_size,
                in_channels=self.enc_estimator.out_channels,
                kernel_size=3,
                output_size=self.output_size//enc_decay,
                stride=2
                )
        
        self.encoding_output_shape = ( self.enc_pooling.out_channels ,self.output_size//enc_decay, self.output_size//enc_decay)

        # DECODING:
        self.dec_upsample = fm.UpSampleBlock(
            input_size=self.enc_pooling.output_size, 
            in_channels=self.enc_pooling.out_channels,
            output_size=self.enc_pooling.input_size, 
            )
        
        
        if estimative_signal: 
            self.dec_concat = fm.ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= 2*self.dec_upsample.out_channels
                )
            print(":::: DECODER: Concat output  ", self.dec_concat.out_shape)
            
            self.dec_conv_1 = fm.ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_concat.out_channels, 
                out_channels=self.dec_concat.out_channels//2, 
                kernel_size=3, 
                stride=1, 
                )
            print(":::: DECODER: Conv1 output  ", self.dec_conv_1.out_shape)
            
            self.dec_conv_2 = fm.ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_conv_1.out_channels, 
                out_channels=self.dec_conv_1.out_channels -add_channels, 
                kernel_size=5, 
                stride=1, 
                )
            print(":::: DECODER: Conv2 output  ", self.dec_conv_2.out_shape)

            self.dec_conv_3 = fm.ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_conv_2.out_channels, 
                out_channels=self.out_channels, 
                kernel_size=7, 
                stride=1, 
                )
            print(":::: DECODER: Conv3 output  ", self.dec_conv_3.out_shape)

            self.dec_conv = nn.Sequential(self.dec_conv_1, self.dec_conv_2, self.dec_conv_3)
        else:
            self.dec_conv = fm.ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_upsample.out_channels, 
                out_channels=self.out_channels, 
                kernel_size=7, 
                stride=1, 
                )
            
        self.tail = nn.Sigmoid()
        
        # METADATA TRACKING:
        self.metadata.update({
            "Encoder": {
                "Rock Pooling": {
                    "Kernel Size": self.rock_pooling.kernel_size,
                    "Stride": self.rock_pooling.stride,
                    "Padding": self.rock_pooling.padding,
                },
                "Encoding Estimator Output Shape": self.enc_estimator.out_shape,
                "Encoding Pooling": {
                    "Kernel Size": self.enc_pooling.kernel_size,
                    "Stride": self.enc_pooling.stride,
                    "Padding": self.enc_pooling.padding,
                }
            },
            "Decoder": {
                "Upsampling": {
                    "Method": "Nearest",
                    "Output Shape": self.dec_upsample.out_shape
                },
                "Decoder Convolutions": {
                    "Output channels": {
                        "Conv1": self.dec_conv_1.out_channels if estimative_signal else self.dec_conv.out_channels,
                        "Conv2": self.dec_conv_2.out_channels if estimative_signal else None,
                        "Conv3": self.dec_conv_3.out_channels if estimative_signal else None
                        },
                    "Structures":[
                    {
                        "Kernel Size": self.dec_conv_1.kernel_size if estimative_signal else self.dec_conv.kernel_size,
                        "Stride": self.dec_conv_1.stride if estimative_signal else self.dec_conv.stride,
                        "Padding": self.dec_conv_1.padding if estimative_signal else self.dec_conv.padding,
                    },
                    {
                        "Kernel Size": self.dec_conv_2.kernel_size if estimative_signal else None,
                        "Stride": self.dec_conv_2.stride if estimative_signal else None,
                        "Padding": self.dec_conv_2.padding if estimative_signal else None,
                    },
                    {
                        "Kernel Size": self.dec_conv_3.kernel_size if estimative_signal else None,
                        "Stride": self.dec_conv_3.stride if estimative_signal else None,
                        "Padding": self.dec_conv_3.padding if estimative_signal else None,
                    }]
                },
                "Final Output Shape": self.out_shape
            }
        })
        
        
        
    # Short-Circuit propagation 
    def forward(self, rock, estimation=None):
        rock_pooling, processed_estimate = self.encoder(rock, estimation)
        output = self.decoder(processed_estimate)
        return self.tail(output)
    
    def encoder(self, rock, estimative=None):
        # Rock branch
        rock_pooling = self.rock_pooling(rock)
        # Encoder branch
        # - Estimation
        y2 = self.enc_estimator(rock)
        # - Processing
        if self.estimative_signal:
            y1 = self.enc_input_concat(rock, estimative)
            y1 = self.enc_conv(y1)
            self.encoder_upscaled_output = self.enc_output_concat(y1,y2)

        else: 
            self.encoder_upscaled_output = y2

        # - Output scaling
        processed_estimate = self.enc_pooling(self.encoder_upscaled_output)
        return rock_pooling, processed_estimate
        
    def decoder(self, upcoming_processed_estimate):
        y = self.dec_upsample(upcoming_processed_estimate)
        if self.estimative_signal:
            y = self.dec_concat(y, self.encoder_upscaled_output)
        y = self.dec_conv(y)
        return y
"""


class MULTI_BLOCK_MODEL(fm.BASE_MODEL):
    
    def __init__(self, in_shape, out_shape, min_size=2, output_masks=None, enc_decay=2, add_channels=2, estimative_signal=False):
        super(MULTI_BLOCK_MODEL, self).__init__(in_shape, out_shape, output_masks)
        
        self.blocks = nn.ModuleList()
        
        N_blocks = int(np.ceil(np.log2(in_shape[1]/min_size)))
        
        print(f"\n creating Multi-block model with {N_blocks}:\n")
        block_metadata = []
        for i in range(N_blocks):
            print(f"Block {i+1}:\n")
            
            if i == 0: # If is first block, use initial in/out shapes
                block_in_shape = in_shape
                block_out_shape = out_shape
                block_estimative_signal = estimative_signal
            else: # If folowwing block, use the shape of previous block and enable estimative blocks
                block_in_shape = self.blocks[-1].rock_output_shape
                block_out_shape = self.blocks[-1].encoding_output_shape
                block_estimative_signal = True 
            
            # Create a list of blocks
            block = fm.BLOCK_3(block_in_shape, block_out_shape, output_masks, enc_decay, add_channels, block_estimative_signal)
            self.blocks.append(block)
            
            print(f"- in: {self.blocks[-1].in_shape},\n"
                  f"--- rock out: {self.blocks[-1].rock_output_shape}\n"
                  f"--- processed out: {self.blocks[-1].encoding_output_shape}\n"
                  f"- out: {self.blocks[-1].out_shape}\n\n")
            
            # Store the block's metadata 
            block_metadata.append({
                "Block Index": i + 1,
                "Input Shape": block_in_shape,
                "Output Shape": block_out_shape,
                "Rock Output Shape": block.rock_output_shape,
                "Encoded Output Shape": block.encoding_output_shape,
                "Estimative Signal": block_estimative_signal,
                "Details": block.metadata
            })
            
        self.tail = nn.Sigmoid()
        
        # Save model-wide metadata
        self.metadata.update({
            "Number of Blocks": N_blocks,
            "Encoder Decay Factor": enc_decay,
            "Additional Channels": add_channels,
            "Estimative Signal Enabled": estimative_signal,
            "Block Details": block_metadata
        })
        
        
    def forward(self, rock, estimation=None):
        
        print("Encoding: ")
        # Encoding 
        for block in self.blocks:
            rock, estimation = block.encoder(rock, estimation)
            
        print("Decoding: ")
        # Decoding
        for block in reversed(self.blocks):
            estimation = block.decoder(estimation)
            
        return estimation







        
        

    








    

