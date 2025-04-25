import torch
import torch.nn as nn
import numpy as np

# Author: Gabriel Cesar Silveira
# Date: 14/01/2025
# Inception: https://arxiv.org/pdf/1409.4842
# Inception implementation: https://medium.com/@karuneshu21/implement-inception-v1-in-pytorch-66bdbb3d0005



#######################################################
#************ BASE MODEL *****************************#
# The basic information that any proposed model to    #
# this image-to-image problem needs to carry          #
#######################################################

class BASE_MODEL(nn.Module):
    def __init__(self, in_shape, out_shape, output_masks=None):
        super(BASE_MODEL,self).__init__()
        
        self.output_masks = None
        self.model = None
        self.metadata = {}
        
        
        # Saving infos        
        self.in_shape=in_shape
        self.out_shape=out_shape
        self.output_masks = output_masks
        self.output_size = out_shape[1]
        self.input_size = in_shape[1]
        self.out_channels = out_shape[0]
        self.in_channels = in_shape[0]
        
        # Initialize metadata with general model info
        self.metadata["Model Name"] = self.__class__.__name__
        self.metadata["Input Shape"] = in_shape
        self.metadata["Output Shape"] = out_shape

        
    def predict(self, x):
        if self.model is None: self.eval()
        else: self.model.eval()
        
        with torch.no_grad():
            y = self.forward(x)
        
        if self.output_masks is not None:
            # Ensure the number of functions matches the number of channels
            if y.shape[1] != len(self.output_masks): raise Exception(f"Output masks must match the number of channels, but {len(self.output_masks)} masks for {y.shape[1]} channels happened in output with shape {y.shape}")
                
            # Apply each function to its corresponding channel
            processed_channels = [func(y[:, ch, :, :]) for ch, func in enumerate(self.output_masks)]
            
            # Stack back into a single tensor
            y = torch.stack(processed_channels, dim=1)
        
        return y
    
    def forward(self,x):
        return self.model(x)



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
class INCEPTION_MODEL(BASE_MODEL):

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
        model1 = InceptionBlock(
            input_size=in_shape[1],
            in_channels=in_shape[0],
            b1_out_channels=b1_out_channels,
            b2_mid_channels=b2_mid_channels,
            b2_out_channels=b2_out_channels,
            b3_mid_channels=b3_mid_channels,
            b3_out_channels=b3_out_channels,
            b4_out_channels=b4_out_channels
            )
        
        model2 = ConvBlock(
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
        
        
        

        
class BLOCK_2(BASE_MODEL):
    def __init__(self, in_shape, out_shape, output_masks=None, enc_decay=2, add_channels=2, estimative_signal=False):
        super(BLOCK_2,self).__init__(in_shape, out_shape, output_masks)
        
        self.estimative_signal = estimative_signal
        
        # ENCODING:
        # Rock branch:
        self.rock_pooling = PoolingBlock(
            input_size=self.input_size,
            in_channels=self.in_channels,
            kernel_size=3,
            output_size=self.output_size//enc_decay,
            stride=2
            )
        self.rock_output_shape = self.rock_pooling.out_shape


        # Estimation branch (Upcoming feature: Pre-trained):
        self.enc_estimator = INCEPTION_MODEL(in_shape, out_shape, output_masks)
        
        # Encoding processed info branch:
        if estimative_signal:    
            print("::Estimative branch enabled")
            self.enc_input_concat = ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= self.in_channels + self.out_channels
                )
            self.enc_conv = ConvBlock(
                input_size=self.input_size, 
                in_channels=self.enc_input_concat.out_channels, 
                out_channels=self.in_channels +self.out_channels +add_channels, 
                kernel_size=3, 
                stride=1, 
                )
            self.enc_output_concat = ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= self.enc_estimator.out_channels + self.enc_conv.out_channels
                )
            self.enc_pooling = PoolingBlock(
                input_size=self.input_size,
                in_channels=self.enc_output_concat.out_channels,
                kernel_size=3,
                output_size=self.output_size//enc_decay,
                stride=2
                )
        else:
            print("::Estimative branch disabled")
            self.enc_pooling = PoolingBlock(
                input_size=self.input_size,
                in_channels=self.enc_estimator.out_channels,
                kernel_size=3,
                output_size=self.output_size//enc_decay,
                stride=2
                )
        
        self.encoding_output_shape = ( self.enc_pooling.out_channels ,self.output_size//enc_decay, self.output_size//enc_decay)

        # DECODING:
        self.dec_upsample = UpSampleBlock(
            input_size=self.enc_pooling.output_size, 
            in_channels=self.enc_pooling.out_channels,
            output_size=self.enc_pooling.input_size, 
            )
        
        
        if estimative_signal: 
            self.dec_concat = ChannelConcat_Block(
                input_size= self.input_size,
                in_channels= 2*self.dec_upsample.out_channels
                )
            print(":::: DECODER: Concat output  ", self.dec_concat.out_shape)
            
            self.dec_conv_1 = ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_concat.out_channels, 
                out_channels=self.dec_concat.out_channels//2, 
                kernel_size=3, 
                stride=1, 
                )
            print(":::: DECODER: Conv1 output  ", self.dec_conv_1.out_shape)
            
            self.dec_conv_2 = ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_conv_1.out_channels, 
                out_channels=self.dec_conv_1.out_channels -add_channels, 
                kernel_size=5, 
                stride=1, 
                )
            print(":::: DECODER: Conv2 output  ", self.dec_conv_2.out_shape)

            self.dec_conv_3 = ConvBlock(
                input_size=self.input_size, 
                in_channels=self.dec_conv_2.out_channels, 
                out_channels=self.out_channels, 
                kernel_size=7, 
                stride=1, 
                )
            print(":::: DECODER: Conv3 output  ", self.dec_conv_3.out_shape)

            self.dec_conv = nn.Sequential(self.dec_conv_1, self.dec_conv_2, self.dec_conv_3)
        else:
            self.dec_conv = ConvBlock(
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
        


class MULTI_BLOCK_MODEL_2(BASE_MODEL):
    
    def __init__(self, in_shape, out_shape, min_size=2, output_masks=None, enc_decay=2, add_channels=2, estimative_signal=False):
        super(MULTI_BLOCK_MODEL_2, self).__init__(in_shape, out_shape, output_masks)
        
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
            block = BLOCK_2(block_in_shape, block_out_shape, output_masks, enc_decay, add_channels, block_estimative_signal)
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
        for block in self.blocks:
            rock, estimation = block.encoder(rock, estimation)

        for block in reversed(self.blocks):
            estimation = block.decoder(estimation)

        return estimation





#######################################################
#****** FUNCTIONAL BLOCKS TO BUILD MODELS ************#
# Models that are mainly functional to build other    #
# models, since their input and output do not have    #
# the same properties as the original problem domain  # 
#######################################################

class UpSampleBlock(nn.Module):
    def __init__(self,input_size, in_channels, output_size, mode='nearest'):
        super(UpSampleBlock,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_channels = in_channels
        self.out_channels = in_channels 
        self.out_shape = (self.out_channels, self.output_size, self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)
        
        self.upscale = nn.Upsample(size=output_size, mode='nearest')
        
    def forward(self,x): 
        return self.upscale(x)
            
class PoolingBlock(nn.Module):
    def __init__(self, input_size, in_channels, kernel_size, stride=1, padding=None, dilation=1, output_size=None):
        super(PoolingBlock,self).__init__()
        
        
        if not (isinstance(input_size,int)): raise Exception("ConvBlock parameter input_size must be integer.")
        if not (isinstance(kernel_size,int)): raise Exception("ConvBlock parameter kernel_size must be integer.")
        if not (isinstance(stride,int)): raise Exception("ConvBlock parameter stride must be integer.")
        if not (isinstance(dilation,int)): raise Exception("ConvBlock parameter dilation must be integer.")
        if not (isinstance(padding,int)) and padding is not None: raise Exception("ConvBlock parameter stride must be integer or None.")

        
        padding = self.calculate_padding(input_size, kernel_size, stride, padding, dilation, output_size)
            
        # branch4 : 
        self.max_p = nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.output_size = self.calculate_output_size(input_size, kernel_size, stride, padding, dilation)
        self.out_channels = in_channels
        self.out_shape = (self.out_channels, self.output_size, self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        
    def forward(self,x): 
        return self.max_p(x)
    
    def get_output_size(self, input_size, kernel_size, stride, padding, dilation): 
        return int(np.floor(  (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1 ))
        

    
    def calculate_output_size(self, input_size, kernel_size, stride, padding, dilation):

        # Calculate output size using the formula
        output_size = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        output_size = np.floor((output_size / stride) + 1)

        # Handle edge cases
        if output_size <= 0:
            raise ValueError(
                f"Invalid parameters: output size cannot be <= 0. Calculated output size = {output_size}."
            )

        return int(output_size)
        
    def calculate_padding(self, input_size, kernel_size, stride, padding, dilation, output_size):

        if output_size is None:
            output_size = input_size

        if padding is None:
            padding = int(np.ceil(((output_size - 1) * stride +1 - input_size + dilation * (kernel_size - 1)) / 2))

        if padding < 0 or padding > kernel_size / 2:
            suggested_stride_1 = np.floor((input_size-1-dilation*(kernel_size-1))/(output_size-1))
            suggested_dilation_1 = np.ceil( (input_size-1-stride*(output_size-1))/(kernel_size-1) )
            suggested_kernel_1 = np.ceil((input_size-1-(output_size-1)*stride)/dilation + 1)

            
            suggested_stride_2 = np.floor( (kernel_size-1+input_size-dilation*(kernel_size-1))/(output_size-1) )
            suggested_dilation_2 = np.ceil( (kernel_size-1+input_size-stride*(output_size-1))/(dilation+1))
            suggested_kernel_2 = np.ceil( ((output_size-1)*stride - input_size+1+dilation)/(dilation+1))

            raise ValueError(
                f"Invalid parameters: padding cannot be {padding} to reduce {input_size} to {output_size}).\n"
                f"To keep input dimensions, use one of the suggested fixes based on padding restrictions ( K/2 >= P >= 0 ):\n"
                f"   - Use: {suggested_stride_2} > stride > {suggested_stride_1}\n"
                f"   - Use: {suggested_dilation_2} > dilation > {suggested_dilation_1}\n"
                f"   - Use: kernel_size > {suggested_kernel_1} and kernel_size > {suggested_kernel_2}\n"
            )
        
        return padding
        

class ConvBlock(nn.Module):

    def __init__(self, input_size, in_channels, out_channels , kernel_size , stride=1, dilation=1, padding=None, bias=False, output_size=None):
        super(ConvBlock,self).__init__()
        

        if not (isinstance(input_size,int)): raise Exception("ConvBlock parameter input_size must be integer.")
        if not (isinstance(in_channels,int)): raise Exception("ConvBlock parameter in_channels must be integer.")
        if not (isinstance(out_channels,int)): raise Exception("ConvBlock parameter out_channels must be integer.")
        if not (isinstance(kernel_size,int)): raise Exception("ConvBlock parameter kernel_size must be integer.")
        if not (isinstance(stride,int)): raise Exception("ConvBlock parameter stride must be integer.")
        if not (isinstance(padding,int)) and padding is not None: raise Exception("ConvBlock parameter stride must be integer or None.")
         
        padding = self.calculate_padding(input_size, kernel_size, stride, padding, dilation, output_size)
        

        
                
        # 2d convolution
        self.conv2d = nn.Conv2d(in_channels = in_channels,
                                out_channels = out_channels,
                                kernel_size = kernel_size,
                                stride = stride,
                                padding = padding,
                                dilation=dilation,
                                bias=False)

        # batchnorm
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        #### ADAPTED FROM RELU TO CELU, FOLLOWING PREVIOUS LITERATURE
        self.actv = nn.CELU()
        
        
        self.output_size = self.calculate_output_size(input_size, kernel_size, stride, padding, dilation)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        self.out_shape = (self.out_channels, self.output_size, self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        
        self.n_conv_operations = int(input_size**2 * in_channels * kernel_size**2 * out_channels)
        self.n_trainable_parameters = int((input_size**2 * in_channels +1)*out_channels)
        
        
    def forward(self,x):
        return self.actv(self.batchnorm2d(self.conv2d(x)))
    
    def calculate_output_size(self, input_size, kernel_size, stride, padding, dilation):

        # Calculate output size using the formula
        output_size = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
        output_size = np.floor((output_size / stride) + 1)

        # Handle edge cases
        if output_size <= 0:
            raise ValueError(
                f"Invalid parameters: output size cannot be <= 0. Calculated output size = {output_size}."
            )

        return int(output_size)
    
    def calculate_padding(self,input_size, kernel_size, stride, padding, dilation, output_size):

        if output_size is None:
            output_size = input_size

        if padding is None:
            padding = int(np.ceil(((output_size - 1) * stride +1 - input_size + dilation * (kernel_size - 1)) / 2))

        if padding < 0 or padding > kernel_size / 2:
            suggested_stride_1 = np.floor((input_size-1-dilation*(kernel_size-1))/(output_size-1))
            suggested_dilation_1 = np.ceil( (input_size-1-stride*(output_size-1))/(kernel_size-1) )
            suggested_kernel_1 = np.ceil((input_size-1-(output_size-1)*stride)/dilation + 1)

            
            suggested_stride_2 = np.floor( (kernel_size-1+input_size-dilation*(kernel_size-1))/(output_size-1) )
            suggested_dilation_2 = np.ceil( (kernel_size-1+input_size-stride*(output_size-1))/(dilation+1))
            suggested_kernel_2 = np.ceil( ((output_size-1)*stride - input_size+1+dilation)/(dilation+1))

            raise ValueError(
                f"Invalid parameters: padding cannot be {padding} to reduce {input_size} to {output_size}).\n"
                f"To keep input dimensions, use one of the suggested fixes based on padding restrictions ( K/2 >= P >= 0 ):\n"
                f"   - Use: {suggested_stride_2} > stride > {suggested_stride_1}\n"
                f"   - Use: {suggested_dilation_2} > dilation > {suggested_dilation_1}\n"
                f"   - Use: kernel_size > {suggested_kernel_1} and kernel_size > {suggested_kernel_2}\n"
            )
        
        return padding


class ConvTransposeBlock(nn.Module):
    def __init__(self, input_size, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, output_padding=None, bias=False, output_size=None):
        super(ConvTransposeBlock, self).__init__()

        # Validate input parameters
        if not isinstance(input_size, int): raise Exception("ConvTransposeBlock parameter input_size must be integer.")
        if not isinstance(in_channels, int): raise Exception("ConvTransposeBlock parameter in_channels must be integer.")
        if not isinstance(out_channels, int): raise Exception("ConvTransposeBlock parameter out_channels must be integer.")
        if not isinstance(kernel_size, int): raise Exception("ConvTransposeBlock parameter kernel_size must be integer.")
        if not isinstance(stride, int): raise Exception("ConvTransposeBlock parameter stride must be integer.")
        if not isinstance(padding, int) and padding is not None: raise Exception("ConvTransposeBlock parameter padding must be integer or None.")


        # Calculate output_padding to match the desired output_size
        output_padding = self.calculate_output_padding(input_size, kernel_size, stride, dilation, padding ,output_padding, output_size)
        
        # Transposed 2D convolution
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=bias
        )

        # Batch normalization
        self.batchnorm2d = nn.BatchNorm2d(out_channels)

        # Activation function
        self.actv = nn.CELU()

        # Calculate output size
        self.output_size = self.calculate_output_size_transpose(input_size, kernel_size, stride, padding, dilation, output_padding)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.input_size = input_size
        self.out_shape = (self.out_channels, self.output_size, self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        
    def forward(self, x):
        return self.actv(self.batchnorm2d(self.conv_transpose2d(x)))
    
    def calculate_output_size_transpose(self, input_size, kernel_size, stride, padding, dilation, output_padding):

        output_size = (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        return int(output_size)
    
    def calculate_output_padding(self, input_size, kernel_size, stride, dilation, padding ,output_padding, output_size):
        
        if output_padding is None:
            output_padding = output_size-1 - (input_size-1)*stride + 2*padding - dilation*(kernel_size-1)
            
        print(f"OUTPUT PADDING {output_padding}, output_size {output_size}, input_size {input_size}, kernel_size {kernel_size}, stride {stride}, dilation {dilation}, padding {padding}")
        
        # Checks and help
        stride_help_1 = np.ceil((output_size-1+2*padding-dilation*(kernel_size-1))//input_size)
        dilatation_help_1 = np.ceil( (output_size-1+2*padding-stride*(input_size-1))//kernel_size )
        padding_help_1 = (1-output_size+dilation*(kernel_size-1)+stride*(input_size-1))//2
        
        stride_help_2 = np.floor( (output_size-1+2*padding-dilation*(kernel_size-1))//(input_size-1) )
        dilatation_help_2 = np.floor( (output_size-1+2*padding-stride*(input_size-1))//(kernel_size-1))
        padding_help_2 = (stride*input_size+dilation*(kernel_size-1)+1-output_size)//2
        
        if output_padding < 0:
            raise Exception("Negative output padding is not feasible.\n"
                            f"Consider using one of the below:\n"
                            f"-- {stride_help_2} >= stride > {stride_help_1} (currently {stride})\n"
                            f"-- {dilatation_help_2}>= dilation > {dilatation_help_1} (currently {dilation})\n"
                            f"-- {padding_help_2}>= padding > {padding_help_1} (currently {padding})\n"
                            )
        elif output_padding > stride or output_padding > dilation:
            raise Exception(f"Output padding must smaller than stride or dilation.\n"
                            f"Consider using one of the below:\n"
                            f"-- {stride_help_2} >= stride > {stride_help_1} (currently {stride})\n"
                            f"-- {dilatation_help_2}>= dilation > {dilatation_help_1} (currently {dilation})\n"
                            f"-- {padding_help_2}>= padding > {padding_help_1} (currently {padding})\n"
                            ) 
            
        return int(output_padding)


    
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
            in_channels=in_channels,
            kernel_size=3)
        

        conv_4 =  ConvBlock(
            input_size=max_p.output_size,
            in_channels=max_p.out_channels,
            out_channels=b4_out_channels,
            kernel_size=1
            )

        self.branch4 = nn.Sequential(max_p,conv_4)
        
                
        if not (conv_1.output_size==conv_2_2.output_size==conv_3_2.output_size==conv_4.output_size):
            raise Exception(f"Inception block branches with unmatched output sizes, please verify:\nconv_1.output_size: {conv_1.output_size}, conv_2.output_size: {conv_2_2.output_size}, conv_3.output_size: {conv_3_2.output_size}, conv_4.output_size: {conv_4.output_size}")
        
        self.input_size = input_size
        self.in_channels = in_channels
        self.output_size = conv_1.output_size
        self.out_channels = b1_out_channels + b2_out_channels + b3_out_channels + b4_out_channels
        self.out_shape = (self.out_channels, self.output_size,self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)
        
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

class ChannelConcat_Block(nn.Module):
    """
    A container for concatenating multiple tensors along the channel dimension.
    Usage:
        concat = ChannelConcat_Block()
        output = concat(tensor1, tensor2, ...)
    Tensors must have the same batch size, height, and width.
    """

    def __init__(self, input_size, in_channels):
        super(ChannelConcat_Block, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.output_size = input_size
        self.out_channels = in_channels
        self.out_shape = (self.out_channels, self.output_size, self.output_size)
        self.in_shape = (self.in_channels, self.input_size, self.input_size)
        
    def forward(self, *tensors):
        """
        Concatenates tensors along the channel dimension.
        Args:
            *tensors: Variable number of tensors to concatenate. Must have the same spatial dimensions.
        Returns:
            torch.Tensor: Concatenated tensor along the channel dimension.
        """
        if len(tensors) < 2:
            raise ValueError("At least two tensors are required for concatenation.")

        # Ensure all tensors have the same batch size, height, and width
        shapes = [t.shape for t in tensors]
        base_shape = shapes[0][0], shapes[0][2], shapes[0][3]  # (Batch, H, W)
        
        for i, t in enumerate(tensors):
            if t.shape[0] != base_shape[0] or t.shape[2] != base_shape[1] or t.shape[3] != base_shape[2]:
                raise ValueError(f"Tensor {i} has mismatched shape: {t.shape}. Expected batch, height, width: {base_shape}")

        # Concatenate along channel dimension (dim=1)
        return torch.cat(tensors, dim=1)






        
        

    








    

