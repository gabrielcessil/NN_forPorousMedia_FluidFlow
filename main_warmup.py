

import torch.nn as nn
import torch
import json
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import loader_creator as lc
from Architectures import Models as md
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
learning_rate = 0.015 # 
momentum = 0.9 
N_epochs = 100 # Number of times that all the samples are visited
model_name = "Inception" # The desired model name, avoid overwritting previous models
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


## Example: INRIA SEGMENTATION
# Original content: https://project.inria.fr/aerialimagelabeling/files/
# Original dataset's paper: https://inria.hal.science/hal-01468452
# Dataset's Benchmark: https://paperswithcode.com/sota/semantic-segmentation-on-inria-aerial-image
# Related works: 
# - U-net: http://arxiv.org/pdf/1912.09216v1
# - Transformers: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10108049
dataset_path = "/home/gabriel/Downloads/AerialLabelling/AerialImageDataset_01/train/"


def make_binary_int(a): return (a > 0.5).int() # Make it binary
output_masks = [make_binary_int] # Mask for binary classification





#######################################################
#************ CREATING MODEL *************************#
#######################################################

# CREATING MODELS
model_1 = md.INCEPTION_MODEL(
        in_shape=(3, 2048, 2048),
        out_shape=(1, 2048, 2048),
        output_masks=output_masks,
        b1_out_channels=16,
        b2_mid_channels=8,
        b2_out_channels=16,
        b3_mid_channels=8,
        b3_out_channels=16,
        b4_out_channels=16,
        tail_kernel_size = 5)

model_2 = md.INCEPTION_MODEL(
        in_shape=(3, 1024, 1024),
        out_shape=(1, 1024, 1024),
        output_masks=output_masks,
        b1_out_channels=16,
        b2_mid_channels=8,
        b2_out_channels=16,
        b3_mid_channels=8,
        b3_out_channels=16,
        b4_out_channels=16,
        tail_kernel_size = 5)

model_3 = md.INCEPTION_MODEL(
        in_shape=(3, 512, 512),
        out_shape=(1, 512, 512),
        output_masks=output_masks,
        b1_out_channels=32,
        b2_mid_channels=16,
        b2_out_channels=32,
        b3_mid_channels=16,
        b3_out_channels=32,
        b4_out_channels=32,
        tail_kernel_size = 5)



#######################################################
#************ COMPUTATIONS ***************************#
#######################################################

models = [model_1,model_2,model_3]

for i, model in enumerate(models):
    
    model_full_name_i = model_full_name + f"_warm{i}"
    
    # CREATING DATALOADER
    examples_shape = model.in_shape
    output_shape = model.out_shape
    data = dr.AerialSegmentationData(dataset_path, examples_shape)
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
    
    #### MODEL TRAINNING
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,  momentum=momentum)
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
                                                    weights_file_name=model_full_name_i,
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
