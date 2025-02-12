import torch.nn as nn
import torch
import json
from Utilities import loss_functions as lf
from Utilities import nn_trainner as nnt
from Utilities import model_handler as mh
from Utilities import loader_creator as lc
from Architectures import Inception_v3 as incp
from Utilities import dataset_reader as dr


#######################################################
#************ USER INPUTS: Hyperparameters ***********#
#######################################################

batch_size = 5
max_samples = 100
train_ratio = 0.7
val_ratio = 0.15
learning_rate = 0.1
N_epochs = 300
NN_dataset_name = "Examples_TUBES.pt" # The desired dataset to train the model
model_name = "Model" # The desired model name, avoid overwritting previous models
loss_functions = {
    "MSE":  {"obj":     lf.Mask_LossFunction(nn.MSELoss()),"Thresholded": False},
    "L1":   {"obj":     lf.Mask_LossFunction(nn.L1Loss()), "Thresholded": False},
    #"Complementary MIOU": {"obj":     lf.CustomLoss_MIOU(),"Thresholded": True},
    #"Complementary Accuracy": {"obj": lf.CustomLoss_Accuracy(),"Thresholded": True}
    }
earlyStopping_loss = "MSE"
backPropagation_loss = "MSE"




#######################################################
#************ COMPUTATIONS ***************************#
#######################################################


# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
rock_shape = config_loaded["Rock_shape"]
NN_dataset_folder = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
model_full_name = NN_model_weights_folder+model_name

# LOADING DATASET - Change 'LBM_Dataset_Processor' to collect inputs and outputs
# customized to other datasets that differ from the original ones
dataset_path = NN_dataset_folder+NN_dataset_name
data = dr.LBM_Dataset_Processor(dataset_path)


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




# CREATING MODEL

model = incp.MODEL(
    in_shape=input_shape,  # (C, H, W)
    out_shape=output_shape  # (C, H, W)
)


# RUNNING EXAMPLE WITHOUT TRAINNING
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample (without trainning)")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample (without trainning)")


#### MODEL TRAINNING
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model_paths, train_costs_history, val_costs_history = nnt.full_train(
                                                            model,
                                                            train_loader,
                                                            val_loader,                    
                                                            loss_functions,                                      
                                                            earlyStopping_loss,
                                                            backPropagation_loss,
                                                            optimizer,
                                                            N_epochs=N_epochs,
                                                            file_name=model_full_name)

nnt.Plot_Validation(train_costs_history, val_costs_history)



#### MODEL TEST
model_path = model_paths[0]
model.load_state_dict(torch.load(model_path)) # Load model 
nnt.Run_Example(model, train_loader,loss_functions, earlyStopping_loss, i_example=0, title="Train sample")
nnt.Run_Example(model, test_loader,loss_functions, earlyStopping_loss, i_example=0, title="Test sample")

### DELETE MODEL AFTER USING IT
mh.delete_model(model)
