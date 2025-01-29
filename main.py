from Utilities.plotter import plot_heatmap
from Utilities.dataset_creator import Data_from_Simulation, get_input_output_tensors, TensorsDataset
from Utilities.usage_metrics import get_ProcessingTime, check_memory, delete_model, estimate_memory
from Architectures.my_Inception_v2 import Parallel_Inception_Block
from Utilities.nn_trainner import full_train, Plot_Validation
#from Utilities.array_handler import *
#from Utilities.nn_trainner import *
import torch.nn as nn
import torch


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
    
# INPUTS
batch_size = 50
train_ratio=0.7
val_ratio=0.15
learning_rate = 0.01
N_epochs = 10



# CREATING DATASET FOR TRAIN/ VALIDATION/ TEST
#Data_from_Simulation(["./LBM_Simulations/LBM.mat"], save_name="grouped_250_images_50x50", subshape=(50,50,50))

# LOADING DATASET 
input_tensors, output_tensors = get_input_output_tensors("./NN_Datasets/grouped_250_images_50x50.pt")
dataset = TensorsDataset(input_tensors, output_tensors) # Initialize the dataset
train_loader, val_loader, test_loader = dataset.get_dataloaders(train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size)
input_example, output_example = next(iter(test_loader))
input_example, output_example = input_example[0], output_example[0]
vmin, vmax = torch.min(output_example), torch.max(output_example)
print(f"  Input shape: {input_example.shape}")  # Shape of the input
print(f"  Output shape: {output_example.shape}\n")  # Shape of the output
print(f"  Total Test samples: {len(test_loader.dataset)}")
print(f"  Total Validation samples: {len(val_loader.dataset)}")
print(f"  Total Train samples {len(train_loader.dataset)}")
print(f"  Number of trainning batches: {len(train_loader)}")
print(f"  Trainning samples per batch: {train_loader.batch_size}")

# CREATING MODEL
in_shape = input_example.shape
model = Parallel_Inception_Block(in_channels=in_shape[0], in_size=in_shape[1], out_size=in_shape[1])
loss_function = CustomLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print("\n  Conv Operations: ", model.n_conv_operations)
print("  Trainable Parameters: ", model.n_trainable_parameters)


# EXAMPLE
"""
# Real Input example
plot_heatmap(input_example.numpy().reshape((50,50)),  filename="input_heatmap", vmin=vmin, vmax=vmax)
plot_heatmap(output_example.numpy().reshape((50,50)),  filename="lbm_heatmap", vmin=vmin, vmax=vmax)
pred_example = model.forward(input_example.unsqueeze(0)).detach()
loss_example = loss_function.forward(pred_example, output_example.unsqueeze(0))
print(f"Loss Example: {loss_example}")
pred_example = pred_example.numpy().reshape((50,50))
plot_heatmap(pred_example, filename="initial_pred_heatmap", vmin=vmin, vmax=vmax)
"""






#### MODEL TRAINNING
model_paths, train_cost_history, val_cost_history = full_train(
                                                    model,
                                                    train_loader,
                                                    val_loader,
                                                    loss_function,
                                                    optimizer,
                                                    N_epochs=N_epochs)

Plot_Validation(train_cost_history, val_cost_history)



#### MODEL TEST
"""
model.load_state_dict(torch.load("model_weights_ProgressTracking_95.0.pth"))
pred = model.forward(input_example.unsqueeze(0)).detach()
pred = pred.numpy().reshape((50,50))
plot_heatmap(pred, filename="final_pred_heatmap", vmin=vmin, vmax=vmax)
"""

delete_model(model)


# Na saida forward da rede neural, corrigir itens de rocha.
# Incluir canal com distancia euclidiana na entrada, estudar outros possiveis canais
#assim, nao afeta loss e o sinal tbm pode ser passado adiante carregando a informação
# Modularizar e aumentar rede neural com pelo menos 3 Inceptions e aumentar seus canais até formar 500MB


