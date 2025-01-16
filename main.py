from Utilities.plotter import plot_heatmap
from Utilities.dataset_creator import Data_from_Simulation, get_input_output_tensors, TensorsDataset
from Utilities.usage_metrics import get_ProcessingTime, get_MemoryUsage_MB, check_memory, delete_model
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
        print(f"output size: {output.size()}, target size: {target.size()}")

        # Remove solid cells (=0). 
        # The loss function used must be a mean across the tensor lenght, 
        # so that the quantity of solid cells do not affect the loss
        mask = target != 0

        target = target[mask]
        output = output[mask]

        mse = nn.MSELoss()
        return mse(output, target)
    
# INPUTS
batch_size = 50
train_ratio=0.7
val_ratio=0.15
learning_rate = 0.00001
N_epochs = 2
loss_function = CustomLoss()


# CREATING DATASET FOR TRAIN/ VALIDATION/ TEST
#Data_from_Simulation(["./LBM_Simulations/LBM.mat"], save_name="grouped_250_images_50x50", subshape=(50,50,50))
input_tensors, output_tensors = get_input_output_tensors("./NN_Datasets/grouped_250_images_50x50.pt")
dataset = TensorsDataset(input_tensors, output_tensors) # Initialize the dataset
train_loader, val_loader, test_loader = dataset.get_dataloaders(train_ratio=train_ratio, val_ratio=val_ratio, batch_size=batch_size)
input_example, output_example = next(iter(test_loader))
input_example, output_example = input_example[0], output_example[0]
print(f"  Input shape: {input_example.shape}")  # Shape of the input
print(f"  Output shape: {output_example.shape}\n")  # Shape of the output
print(f"  Total Test samples: {len(test_loader.dataset)}")
print(f"  Total Validation samples: {len(val_loader.dataset)}")
print(f"  Total Train samples {len(train_loader.dataset)}")
print(f"  Number of trainning batches: {len(train_loader)}")
print(f"  Trainning samples per batch: {train_loader.batch_size}")
in_shape = input_example.shape
model = Parallel_Inception_Block(in_channels=in_shape[0], in_size=in_shape[1], out_size=in_shape[1])
print("\nConv Operations: ", model.n_conv_operations)
print("Trainable Parameters: ", model.n_trainable_parameters)
print(f"Memory Usage estimation: {int(get_MemoryUsage_MB(model.n_trainable_parameters, 4))} MB")

# EXAMPLE
#"""
# Real Input example
plot_heatmap(input_example.numpy().reshape((50,50)))
plot_heatmap(output_example.numpy().reshape((50,50)))
pred_example = model.forward(input_example.unsqueeze(0)).detach()
loss_example = loss_function.forward(pred_example, output_example.unsqueeze(0))
print(f"Loss Example: {loss_example}")

pred_example = pred_example.numpy().reshape((50,50))
plot_heatmap(pred_example)
#"""

#loss_function = nn.L1Loss() #nn.MSELoss() nn.L1Loss, nn.HuberLoss




optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#"""
#### MODEL TRAINNING
check_memory(device="cpu")

model_path, train_cost_history, val_cost_history = full_train(
                                                    model,
                                                    train_loader,
                                                    val_loader,
                                                    loss_function,
                                                    optimizer,
                                                    N_epochs=N_epochs)

#### MODEL EVALUATION
Plot_Validation(train_cost_history, val_cost_history)

pred = model.forward(input_example.unsqueeze(0)).detach()
plot_heatmap(input_example.numpy().reshape((50,50)))
plot_heatmap(output_example.numpy().reshape((50,50)))
pred = pred.numpy().reshape((50,50))
plot_heatmap(pred)
#"""

delete_model(model)


# Na saida forward da rede neural, corrigir itens de rocha.
# Incluir canal com distancia euclidiana na entrada, estudar outros possiveis canais
#assim, nao afeta loss e o sinal tbm pode ser passado adiante carregando a informação
# Modularizar e aumentar rede neural com pelo menos 3 Inceptions e aumentar seus canais até formar 500MB


