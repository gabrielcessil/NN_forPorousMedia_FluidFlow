import pyvista as pv
from Utilities import LBPM_runner 
import torch
import numpy as np
import json
from Utilities import plotter 

#######################################################
#************ USER INPUTS                  ***********#
#######################################################

dataset_save_name = "Examples_TUBES"



    
#######################################################
#************ COMPUTATIONS                 ***********#
#######################################################  
  
# Directory configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
simulations_main_folder = config_loaded["LBPM_IO_folder"]
dataset_file_path = config_loaded["NN_dataset_folder"]
rock_shape = config_loaded["Rock_shape"]


# Any folder inside simulations_main_folder must be name a example and the example's files must be follow the folder name 
folder_names = LBPM_runner.get_folder_names(simulations_main_folder, full_path=True)

data = []
for examples_name in folder_names:
    
    
    # DATASET OUTPUTS: velocity field from LBPM simulation
    print("\nGetting IO from ", examples_name)
    vis_folder = LBPM_runner.find_highest_vis_folder(examples_name)
    print("Vis folder: ", vis_folder)
    vti_filename = LBPM_runner.get_vti_files(vis_folder, full_path=True)
    print("Vti file:", vti_filename)
    grid = pv.read(vti_filename)
    keys = grid.cell_data.keys()
    uX = grid["Velocity_x"]
    uY = grid["Velocity_y"]
    output_array_channel1 = uX.reshape(rock_shape)
    output_array_channel2 = uY.reshape(rock_shape)
    output_array = np.concatenate([output_array_channel1, 
                                    output_array_channel2], axis=0)
    
    
    # DATASET INPUTS: rock data .raw
    raw_filename = LBPM_runner.get_raw_files(examples_name, full_path=True)
    raw_data = np.fromfile(raw_filename, dtype=np.uint8).reshape(rock_shape)
    input_array = raw_data.flatten().reshape(rock_shape)
    

    # ORGANIZE INPUTS AND OUTPUTS
    example_data = [input_array, output_array]
    data.append(example_data)
    
    
    # Add fields plot inside the example folder
    plotter.plot_heatmap(uX.reshape(rock_shape), folder_name=vis_folder, simulation_name="uX_Field", vmax=None, save=True)
    plotter.plot_heatmap(uY.reshape(rock_shape), folder_name=vis_folder, simulation_name="uY_Field", vmax=None, save=True)
    
# Save as dataset
torch.save(data, f"{dataset_file_path}{dataset_save_name}.pt")
print(f"Dataset created: {dataset_file_path}{dataset_save_name}.pt")
print(f"Dataset sample's format: [{', '.join(str(data.shape) for data in example_data)}]")


