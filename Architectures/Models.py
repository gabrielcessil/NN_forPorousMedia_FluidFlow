import torch
import torch.nn as nn
import numpy as np

# Author: Gabriel Cesar Silveira
# Date: 14/01/2025
# Inception: https://arxiv.org/pdf/1409.4842
# Inception implementation: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005

import Architectures.FunctionalBlocks as fm





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

class RockAware_UNet(fm.BASE_MODEL):
    
    def __init__(self, in_shape, out_shape, min_size=2, output_masks=None, enc_decay=2, add_channels=2, estimative_signal=False):
        super(RockAware_UNet, self).__init__(in_shape, out_shape, output_masks)
        
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
        
        # Encoding 
        for block in self.blocks:
            rock, estimation = block.encoder(rock, estimation)
            
        # Decoding
        for block in reversed(self.blocks):
            estimation = block.decoder(estimation)
            
        return estimation







        
        

    








    

