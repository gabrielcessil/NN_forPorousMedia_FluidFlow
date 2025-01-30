import json


# INPUTS

# Where the LBPM inputs (raw, db, etc) and outputs (silo, vti, etc) are locatte or must be saved
LBPM_IO_folder = r"/home/gabriel/Desktop/Dissertacao/"
# Where the converter of silo to vti is installed (exe file)
silo2vti_exe_folder = r"/home/gabriel/Documents/LBPM_Install/silo2vti/silo2vti-master/silo2vti"
# Where the LBPM is installed (main folder)
LBPM_installation_folder = r"/home/gabriel/Documents/LBPM_Install"
# Where the resulting pytorch dataset must be saved
NN_dataset_folder = r"/home/gabriel/Desktop/Dissertacao/Code/NN_Datasets/"
# The fixed shape of examples
raw_shape = (50, 50)
# Where the NN model must be saved
NN_model_weights_folder = r"/home/gabriel/Desktop/Dissertacao/Code/NN_Model_Weights/"


# COMPUTATIONS

# Define the dictionary with the configurations
config = {
    "LBPM_IO_folder":LBPM_IO_folder,
    "NN_dataset_folder": NN_dataset_folder,
    "NN_model_weights_folder":NN_model_weights_folder,
    "raw_shape": [raw_shape[0], raw_shape[1]],
    "LBPM_installation_folder": LBPM_installation_folder,
    "silo2vti_exe_folder":silo2vti_exe_folder
}

# Save as JSON file
json_file_path = "config.json"  # Change the name if needed
with open(json_file_path, "w") as json_file:
    json.dump(config, json_file, indent=4)

print(f"Configuration saved to {json_file_path}")