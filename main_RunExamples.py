from Utilities import LBPM_runner 
import json


# Load configs
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
    
# INPUTS
simulations_main_folder = config_loaded["LBPM_IO_folder"]
file_path_silo2vti_exe = config_loaded["silo2vti_exe_folder"]
lbm_file_path = config_loaded["LBPM_installation_folder"]



# Any folder inside simulations_main_folder must be name a example and the example's files must be follow the folder name 
examples_names = LBPM_runner.get_folder_names(simulations_main_folder)

# Run every folder example
for simulation_name in examples_names:
    print("\n\nRunning: ", simulation_name)
    LBPM_runner.Run_Example(simulations_main_folder,
                             simulation_name,
                             file_path_silo2vti_exe,
                             lbm_file_path)