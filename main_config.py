import json


# INPUTS

# Where the LBPM inputs (raw, db, etc) and outputs (silo, vti, etc) are locatte or must be saved
LBPM_IO_folder = r"/media/gabriel/a91864d5-f0d3-4994-921d-b2238c6ce60e/labcc/Área de Trabalho/LBPM_Results"
# Where the converter of silo to vti is installed (exe file)
silo2vti_exe_folder = r"/home/gabriel/Documents/LBPM_Install/silo2vti/silo2vti-master/silo2vti"
# Where the LBPM is installed (main folder)
LBPM_installation_folder = "/home/gabriel/Documents/LBPM_Install"
# Where the resulting pytorch dataset must be saved
NN_dataset_folder = "/home/gabriel/Downloads/Dissertacao/Github/My_Inception_NNArchitecture/NN_Datasets"
# The fixed shape of examples
raw_shape = (50, 50)


# COMPUTATIONS

# Define the dictionary with the configurations
config = {
    "LBPM_IO_folder":LBPM_IO_folder,
    "NN_dataset_folder": NN_dataset_folder,
    "raw_shape": [raw_shape[0], raw_shape[1]],
    "LBPM_installation_folder": LBPM_installation_folder,
    "silo2vti_exe_folder":silo2vti_exe_folder
}

# Save as JSON file
json_file_path = "config.json"  # Change the name if needed
with open(json_file_path, "w") as json_file:
    json.dump(config, json_file, indent=4)

print(f"Configuration saved to {json_file_path}")