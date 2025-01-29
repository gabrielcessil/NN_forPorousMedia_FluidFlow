import pyvista as pv
from Utilities import LBPM_runner 
import torch
import numpy as np
import json
from Utilities import plotter 


# INPUTS

dataset_save_name = "Examples"


    
# COMPUTATIONS
    
# Directory configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
simulations_main_folder = config_loaded["LBPM_IO_folder"]
file_path_silo2vti_exe = config_loaded["silo2vti_exe_folder"]
lbm_file_path = config_loaded["LBPM_installation_folder"]
dataset_file_path = config_loaded["NN_dataset_folder"]
raw_shape = config_loaded["raw_shape"]


# Any folder inside simulations_main_folder must be name a example and the example's files must be follow the folder name 
folder_names = LBPM_runner.get_folder_names(simulations_main_folder, full_path=True)

data = []
for examples_name in folder_names:

    # Find the highest .vti folder
    vis_folder = LBPM_runner.find_highest_vis_folder(examples_name)
    vti_filenames = LBPM_runner.get_vti_files(vis_folder, full_path=True)
    if len(vti_filenames) != 1: raise Exception(f"More than one .vti file inside {vis_folder}")
    # Dataset outputs: velocity field from LBPM simulation
    grid = pv.read(vti_filenames[0])
    keys = grid.cell_data.keys()
    uX = grid["Velocity_x"]
    uY = grid["Velocity_y"]
    
    # Find the .raw data 
    raw_filenames = LBPM_runner.get_raw_files(examples_name, full_path=True)
    if len(vti_filenames) != 1: raise Exception(f"More than one .raw file inside {raw_filenames}")
    # Dataset input: solid rock image
    raw_data = np.fromfile(raw_filenames[0], dtype=np.uint8).reshape(raw_shape)
    rock = raw_data.flatten().tolist()
    
    # Organize data for dataset
    example_data = [rock ,uX.tolist(), uY.tolist()]
    data.append(example_data)
    
    # Add fields plot inside the example folder
    plotter.plot_heatmap(uX.reshape(raw_shape), folder_name=examples_name, simulation_name="uX_Field", vmax=None)
    plotter.plot_heatmap(uY.reshape(raw_shape), folder_name=examples_name, simulation_name="uY_Field", vmax=None)
    
# Save as dataset
torch.save(data, f"{dataset_file_path}/{dataset_save_name}.pt")

