import torch.nn as nn
import torch
import json
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import loader_creator as lc
from Architectures import Inception_v3 as incp
from Utilities import dataset_reader as dr
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
import numpy as np



#######################################################
#************ USER INPUTS: Hyperparameters ***********#
#######################################################

max_samples = 3 # Total samples loaded
val_ratio = 0.15 # Fraction of max_samples used to validate
train_ratio = 0.7 # Fraction of max_samples used to train 
batch_size = 10 # Group size of train samples that influence one update on weights
learning_rate = 0.15 # 
N_epochs = 800 # Number of times that all the samples are visited
model_name = "Model" # The desired model name, avoid overwritting previous models
loss_functions = {
    "MSE":  {"obj":     nn.MSELoss(),"Thresholded": False},
    "L1":   {"obj":     nn.L1Loss(), "Thresholded": False},
    #"Masked MSE (Neg. 0's target cells)": {"obj": lf.Mask_LossFunction(nn.L1Loss()), "Thresholded": False},
    "Binary Cross-Entropy": {"obj": lf.Custom_BCE(), "Thresholded": False},
                             
    "Complementary MIOU": {"obj":     lf.CustomLoss_MIOU(),"Thresholded": True},
    "Complementary Accuracy": {"obj": lf.CustomLoss_Accuracy(),"Thresholded": True},
}
    
earlyStopping_loss = "Complementary MIOU" # Which listed loss_function is used to stop trainning
backPropagation_loss = "MSE" # Which listed loss_function is used to calculate weighs



#######################################################
#************ LOADING CONFIGS ************************#
#######################################################

# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
examples_shape = config_loaded["Rock_shape"]
NN_dataset_folder = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
model_full_name = NN_model_weights_folder+model_name
metadata_file_name = "/home/gabriel/Desktop/Dissertacao/NN_Results/Metadata/"



#######################################################
#************ LOADING DATA ***************************#
#######################################################
        
# Segmentation Benchmarks:https://paperswithcode.com/task/semantic-segmentation


## Example 1: FOREST SEGMENTATION
"""
#import kagglehub
#path = kagglehub.dataset_download("quadeer15sh/augmented-forest-segmentation")
#print("Path to dataset files:", path)
path = "/home/gabriel/.cache/kagglehub/datasets/quadeer15sh/augmented-forest-segmentation/versions/2/"
dataset_path = path+"/Forest Segmented/Forest Segmented/"
examples_shape = [3, 256, 256]
output_shape = [1, 256, 256]
def make_binary_int(a): return (a > 0.5).int() # Make it binary
output_masks = [make_binary_int]
data = dr.ForestSegmentationData(dataset_path, examples_shape)
"""


## Example 2: INRIA SEGMENTATION
# Original content: https://project.inria.fr/aerialimagelabeling/files/
# Original dataset's paper: https://inria.hal.science/hal-01468452
# Dataset's Benchmark: https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image
# Related works: 
# - U-net: http://arxiv.org/pdf/1912.09216v1
# - Transformers: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10108049
dataset_path = "/home/gabriel/Downloads/AerialLabelling/AerialImageDataset_01/train/"
examples_shape = [3, 1000, 1000]
output_shape = [1, 1000, 1000]
data = dr.AerialSegmentationData(dataset_path, examples_shape)
def make_binary_int(a): return (a > 0.5).int() # Make it binary
output_masks = [make_binary_int] # Mask for binary classification

# CREATING DATALOADER
input_tensors = data.inputs
output_tensors = data.outputs

dataset = lc.Data_Loader(input_tensors, output_tensors)
train_loader, val_loader, test_loader = dataset.get_splitted(train_ratio=train_ratio,
                                                            val_ratio=val_ratio,
                                                            batch_size=batch_size,
                                                            max_samples=max_samples)

# Print dataset statistics
print(" Dataset Split Summary:")
print(f"  Total samples: {input_tensors.shape[0]}")  # Number of samples (N)
print(f"  - Total Train samples: {len(train_loader.dataset)}")
print(f"  - Total Validation samples: {len(val_loader.dataset)}")
print(f"  - Total Test samples: {len(test_loader.dataset)}")
print(f"  Number of training batches: {len(train_loader)}")

# Fetch a batch and print correct shapes
batch_inputs, batch_outputs = next(iter(train_loader))
input_shape = batch_inputs[0].shape
output_shape = batch_outputs[0].shape

print("\n Batch Shape Details: (batch_size, channels, height, weight)")
print(f" - Train Batch Input Shape: {input_shape}")  # (batch_size, C, H, W)
print(f" - Train Batch Output Shape: {output_shape}")  # (batch_size, ...)
print(f" - Training samples per batch: {batch_inputs.shape[0]}")  # Correct batch size




#######################################################
#************ CREATING MODEL *************************#
#######################################################

# CREATING MODEL

"""
model = incp.MULTI_BLOCK_MODEL_2(
            in_shape=input_shape, # input shape 
            out_shape=output_shape, # output shape
            min_size=2, # the depth of the architecture, equal to the number of image contractions
            output_masks=output_masks, # mask not included to train weighs (ex: binarization)
            enc_decay=2, # the rate of image contraction in image length
            add_channels=6, # the number of channels added from block to block
            estimative_signal=False) # if estimative is provided in 
"""
"""
model = incp.BLOCK_2(
    in_shape=input_shape,
    out_shape=output_shape,
    output_masks=output_masks)
"""

model = incp.INCEPTION_MODEL(
    in_shape=input_shape,
    out_shape=output_shape,
    output_masks=output_masks,
    b1_out_channels=10,
    b2_mid_channels=5,
    b2_out_channels=10,
    b3_mid_channels=5,
    b3_out_channels=10,
    b4_out_channels=10,
    tail_kernel_size = 1)




# RUNNING EXAMPLE WITHOUT TRAINNING
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample (without trainning)")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample (without trainning)")




#######################################################
#************ COMPUTATIONS ***************************#
#######################################################


    
#### MODEL TRAINNING
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler_period = N_epochs/2
lr_lambda = lambda epoch: 1 + 0.1 * np.sin(2 * np.pi * epoch / scheduler_period) # Multiplier of initial learning rate
scheduler = LambdaLR(optimizer, lr_lambda) # Optimizer handler, being able to update/schedule learning rates

model_paths, train_ch, val_ch, metadata = nnt.full_train(
                                                model,
                                                train_loader,
                                                val_loader,                    
                                                loss_functions,                                      
                                                earlyStopping_loss,
                                                backPropagation_loss,
                                                optimizer,
                                                scheduler=None,
                                                N_epochs=N_epochs,
                                                weights_file_name=model_full_name,
                                                metadata_file_name=metadata_file_name,
                                                )

nnt.Plot_Validation(train_ch, val_ch)



#### MODEL TEST
model_path = model_paths[0]
model.load_state_dict(torch.load(model_path)) # Load model 
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample")

### DELETE MODEL AFTER USING IT
mh.delete_model(model)
