from Utilities.plotter import plot_heatmap
from Utilities.dataset_creator import Data_from_Simulation, get_input_output_tensors, TensorsDataset
from Utilities.usage_metrics import get_ProcessingTime, check_memory, delete_model, estimate_memory
from Architectures.my_Inception_v2 import Parallel_Inception_Block
from Utilities.nn_trainner import full_train, Plot_Validation
import torch.nn as nn
import torch
import json

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        if output.size() != target.size():
            raise ValueError(f"CustomLoss forward: Tensors have different sizes ({output.size()} vs {target.size()})")
        
        # Remove solid cells (=0). 
        # The loss function used must be a mean across the tensor lenght, 
        # so that the quantity of solid cells do not affect the loss
        mask = target != 0

        target = target[mask]
        output = output[mask]

        mse = nn.MSELoss() #nn.MSELoss() nn.L1Loss, nn.HuberLoss
        return mse(output, target)
    
# USER INPUTS: Hyperparameters

batch_size = 10
train_ratio=0.7
val_ratio=0.15
learning_rate = 0.01
N_epochs = 10

NN_dataset_name = "Examples.pt" # The desired dataset to train the model
model_name = "Model" # The desired model name, avoid overwritting previous models


# LOADING CONFIGS
with open("config.json" , "r") as json_file:
    config_loaded = json.load(json_file)
examples_shape = config_loaded["raw_shape"]
NN_dataset_folder = config_loaded["NN_dataset_folder"]
NN_model_weights_folder = config_loaded["NN_model_weights_folder"]
model_path =  NN_model_weights_folder+model_name


# LOADING DATASET 
print(NN_dataset_folder)
input_tensors, output_tensors = get_input_output_tensors(NN_dataset_folder+NN_dataset_name, examples_shape)
dataset = TensorsDataset(input_tensors, output_tensors) # Initialize the dataset
train_loader, val_loader, test_loader = dataset.get_dataloaders(train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size)
input_example, output_example = next(iter(test_loader))
input_example, output_example = input_example[0], output_example[0]
vmin, vmax = torch.min(output_example), torch.max(output_example)
print(f"  Input shape: {input_example.shape}")  # Shape of the input
print(f"  Output shape: {output_example.shape}\n")  # Shape of the output
print(f"  Total samples: {len(input_tensors)}")
print(f"  - Total Test samples: {len(test_loader.dataset)}")
print(f"  - Total Validation samples: {len(val_loader.dataset)}")
print(f"  - Total Train samples {len(train_loader.dataset)}")
print(f"  Number of trainning batches: {len(train_loader)}")
print(f"  Trainning samples per batch: {train_loader.batch_size}")



# CREATING MODEL
in_shape = input_example.shape
model = Parallel_Inception_Block(in_channels=in_shape[0], in_size=in_shape[1], out_size=in_shape[1])
loss_function = CustomLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("\n  Conv Operations: ", model.n_conv_operations)
print("  Trainable Parameters: ", model.n_trainable_parameters)
# Real Input example
inp = input_example.squeeze(0).numpy()
plot_heatmap(inp, show=True,  save=False, vmin=None, vmax=None)
# Real Output example
plot_heatmap(output_example.numpy().reshape(examples_shape), show=True, save=False, vmin=None, vmax=None)
pred_example = model.forward(input_example.unsqueeze(0)).detach()
loss_example = loss_function.forward(pred_example, output_example.unsqueeze(0))
pred_example = pred_example.numpy().reshape(examples_shape)
# Model's Initial Output (random) example
plot_heatmap(pred_example,  show=True, save=False, vmin=torch.min(output_example), vmax=torch.max(output_example))
print(f"  Loss Example: {loss_example}")





#### MODEL TRAINNING
model_paths, train_cost_history, val_cost_history = full_train(
                                                    model,
                                                    train_loader,
                                                    val_loader,
                                                    loss_function,
                                                    optimizer,
                                                    N_epochs=N_epochs,
                                                    file_name=model_path)

Plot_Validation(train_cost_history, val_cost_history)



#### MODEL TEST
model.load_state_dict(torch.load(model_path))
pred = model.forward(input_example.unsqueeze(0)).detach()
pred = pred.numpy().reshape(examples_shape)
plot_heatmap(pred, show=True, save=False, vmin=torch.min(output_example), vmax=torch.max(output_example))



### DELETE MODEL AFTER USING IT
delete_model(model)


# Incluir canal com distancia euclidiana na entrada, estudar outros possiveis canais
#assim, nao afeta loss e o sinal tbm pode ser passado adiante carregando a informação
# Modularizar e aumentar rede neural com pelo menos 3 Inceptions e aumentar seus canais até formar 500MB


