import pyvista as pv
import torch
import numpy as np
import json
from scipy import ndimage

from Utilities import LBPM_runner 
from Utilities import plotter 
from Utilities import domain_creator

#######################################################
#************ USER INPUTS                  ***********#
#######################################################

dataset_save_name = "50_SimpleTube_1x250x250"
void_cell_default_value = 0

    
#######################################################
#************ COMPUTATIONS                 ***********#
#######################################################  
def Normalize_Channels(data, dataset_file_path,dataset_save_name):
    norm_info = {
        "inputs": [],
        "targets": []
    }
    
    # Convert data list to tensors for normalization
    inputs = np.stack([pair[0] for pair in data], axis=0)   # (N, C_in, D, H, W)
    targets = np.stack([pair[1] for pair in data], axis=0)  # (N, C_out, D, H, W)

    print("\n\nInput (Pre-Normalization)")
    for c in range(inputs.shape[1]):
        channel = inputs[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        print(f"-- Channel {c}, min {min_val}, max {max_val}")
    print("Target (Pre-Normalization)")
    for c in range(targets.shape[1]):
        channel = targets[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        print(f"-- Channel {c}, min {min_val}, max {max_val}")
    
    
    # NORMALIZATION
    for c in range(inputs.shape[1]):
        channel = inputs[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        inputs[:, c, :, :] = (channel) / (max_val - min_val)
        norm_info["inputs"].append({"channel": c, "min": float(min_val), "max": float(max_val)})
    for c in range(targets.shape[1]):
        channel = targets[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        targets[:, c, :, :] = (channel) / (max_val - min_val)
        norm_info["targets"].append({"channel": c, "min": float(min_val), "max": float(max_val)})
    
    
    print("\nInput (Pos-Normalization)")
    for c in range(inputs.shape[1]):
        channel = targets[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        print(f"-- Channel {c}, min {min_val}, max {max_val}")
    print("Target (Pos-Normalization)")
    for c in range(targets.shape[1]):
        channel = targets[:, c, :, :]
        min_val = channel.min()
        max_val = channel.max()
        print(f"-- Channel {c}, min {min_val}, max {max_val}")
            
            
    # Repack into original format
    data = [[inputs[i], targets[i]] for i in range(len(data))]        

    # Save to .json file
    with open(f"{dataset_file_path}{dataset_save_name}_normalization.json", "w") as f:
        json.dump(norm_info, f, indent=4)
        
    return data
        
def crop_to_original_shape(array, original_shape):
    """
    Crops a 3D or 4D volume to the original shape (D, H, W) by removing padding from the end.

    Assumes reflection padding was added to the end (not center).

    Parameters:
        array (np.ndarray): Input volume. Shape can be (D, H, W) or (C, D, H, W)
        original_shape (tuple): Desired shape (d0, h0, w0)

    Returns:
        np.ndarray: Cropped volume
    """
    d0, h0, w0 = original_shape

    if array.ndim == 3:
        return array[:d0, :h0, :w0]
    
    elif array.ndim == 4:
        return array[:, :d0, :h0, :w0]
    
    else:
        raise ValueError("Volume must be 3D or 4D (D,H,W) or (C,D,H,W)")

# Directory configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
simulations_main_folder = config_loaded["LBPM_IO_folder"]
dataset_file_path = config_loaded["NN_dataset_folder"]
rock_shape = config_loaded["Rock_shape"]

simulated_rock_shape = domain_creator.get_reflected_shape(rock_shape)

# Any folder inside simulations_main_folder must be name a example and the example's files must be follow the folder name 
folder_names = LBPM_runner.get_folder_names(simulations_main_folder, full_path=True)

data = []
for example_folder in folder_names:
    
    
    # DATASET targetS: velocity field from LBPM simulation
    print("\nGetting IO from ", example_folder)
    vis_folder = LBPM_runner.find_highest_vis_folder(example_folder)
    print("Vis folder: ", vis_folder)
    vti_filename = LBPM_runner.get_vti_files(vis_folder, full_path=True)
    print("Vti file:", vti_filename)
    
    if vti_filename:
        
        # TREAT DATASET TARGET DATA
        grid = pv.read(vti_filename)
        keys = grid.cell_data.keys()
        uX = grid["Velocity_x"].reshape(simulated_rock_shape)
        uY = grid["Velocity_y"].reshape(simulated_rock_shape)

        # If the domain is reflect, crop it to unreflect
        target_array_channel1 = crop_to_original_shape(uX, rock_shape)
        target_array_channel2 = crop_to_original_shape(uY, rock_shape)
        
        target_array = np.concatenate([target_array_channel1, 
                                        target_array_channel2], axis=0)
        
        # TREAT DATASET INPUTS DATA: rock array + distance transform
        raw_filename = LBPM_runner.get_raw_files(example_folder, full_path=True)
        raw_data = np.fromfile(raw_filename, dtype=np.uint8).reshape(simulated_rock_shape)
        input_array_channel1 = crop_to_original_shape(raw_data, rock_shape)
        
        # Get Distance transform
        void_mask = input_array_channel1 != void_cell_default_value  
        input_array_channel2 = ndimage.distance_transform_edt(void_mask)
        
        # Stack raw and distance as two input channels
        input_array = np.concatenate([input_array_channel1, 
                                        input_array_channel2], axis=0)
        
        
        # ORGANIZE INPUTS AND targetS
        example_data = [input_array, target_array]
        data.append(example_data)
        
        # Add fields plot inside the example folder
        plotter.plot_heatmap(target_array_channel1.reshape(rock_shape), folder_name=vis_folder, simulation_name="uX_Field", vmax=None, save=True)
        plotter.plot_heatmap(target_array_channel2.reshape(rock_shape), folder_name=vis_folder, simulation_name="uY_Field", vmax=None, save=True)
        plotter.plot_heatmap(input_array_channel2.reshape(rock_shape), folder_name=example_folder, simulation_name="dist_map", vmax=None, save=True)
        

    

data = Normalize_Channels(data, dataset_file_path, dataset_save_name, epsilon=1e-8)
    
# Save as dataset
torch.save(data, f"{dataset_file_path}{dataset_save_name}.pt")

print()
print(f"Dataset created: {dataset_file_path}{dataset_save_name}.pt")
print(f"Dataset sample's format: [{', '.join(str(data.shape) for data in example_data)}]")
print(f"Total number of samples in dataset: {len(data)}")