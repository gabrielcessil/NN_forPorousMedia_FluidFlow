import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os.path
import json
from Utilities import usage_metrics as um
from Utilities import plotter
        

def train_one_epoch(model, train_loader, loss_function, optimizer, scheduler):
    model.train()

    # Para cada batch
    for batch_inputs, batch_labels in train_loader:
        
        # REALIZA BATCH:
        optimizer.zero_grad() # Reinicia os gradientes para calculo do passo dos pesos
        batch_outputs = model(batch_inputs) # Calcula Saídas
        loss = loss_function["obj"](batch_outputs, batch_labels) # Calcula Custo
        loss.backward() # Calcula gradiente em relacao do Custo
        optimizer.step() # Realiza um passo dos pesos
        if not scheduler is None: scheduler.step() # Realiza um passo no learning rate
    return  model

def get_loader_loss(model, loader, loss_functions):
    
    results = {loss_name: 0.0 for loss_name in loss_functions}
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
        
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      
      for inputs, labels in loader:
         
          for loss_name, loss_function in loss_functions.items():
              
              
              # Gera a saida do modelo atual
              if loss_function["Thresholded"]: outputs = model.predict(inputs)
              
              else: outputs = model(inputs)
              
              aux  = loss_function["obj"](outputs, labels).item()
                            
              results[loss_name] += aux
    
      # Compute average loss
      num_batches = len(loader)
      for loss_name in results: results[loss_name] /= num_batches

    return results

def validate_one_epoch(model, train_loader, valid_loader, loss_functions):
    return get_loader_loss(model, train_loader, loss_functions), get_loader_loss(model, valid_loader, loss_functions)
    
    
def verify_memory(model, in_shape, batch_size, device, dtype=torch.float32):
    current_memory = um.check_memory(device=device)
    estimative_memory = um.estimate_memory(model, input_size=in_shape, batch_size=batch_size, dtype=torch.float32)
    
    etimative = estimative_memory["Total"]
    current_free_memory = current_memory["Free"]
    if etimative > current_free_memory: 
        raise ValueError("Free memory is not sufficient for trainning. Change device, reduce batch size or review architecture")
    else:
        print(f"The current free memory is {current_free_memory}, the trainning usage estimative is {etimative}")
        
# valid_loss_functions: dict of loss functions, where the key identify them
# train_loss_function: must be the name of the loss function used as key inside valid_loss_functions
def full_train(model, 
               train_loader,
               valid_loader,
               loss_functions,
               earlyStopping_loss,
               backPropagation_loss,
               optimizer,
               scheduler=None,
               N_epochs=1000,
               weights_file_name = "model_weights",
               metadata_file_name = "metadata",
               include_time=False, 
               device="cpu"):
    
    inputs, targets = next(iter(train_loader))    
    in_shape = (inputs.shape[1],inputs.shape[2],inputs.shape[3]) #(C,H,W)
    batch_size = inputs.shape[0]
    
    verify_memory(model, in_shape, batch_size, device, dtype=torch.float32)
    
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    train_costs_history = []
    val_costs_history = []
    best_valid_loss = np.inf    
    best_model_path = weights_file_name+"lowerValLoss.pth"
    model_paths = [best_model_path]

    

    progress_points = set(int(N_epochs * i / 100) for i in range(5, 101, 5))  # Define os pontos de 5% a 100%
    
    for epoch_index in range(N_epochs):
        
    
        model = train_one_epoch( 
            model=model,
            train_loader=train_loader,
            loss_function=loss_functions[backPropagation_loss], 
            optimizer=optimizer,
            scheduler=scheduler)
        
        train_avg_loss, valid_avg_loss = validate_one_epoch(
                                              model=model, 
                                              train_loader=train_loader,
                                              valid_loader=valid_loader,
                                              loss_functions=loss_functions)
        
        train_costs_history.append(train_avg_loss)
        val_costs_history.append(valid_avg_loss)
        
        # Track best performance, and save the model's state
        if valid_avg_loss[earlyStopping_loss]< best_valid_loss:
            best_valid_loss = valid_avg_loss[earlyStopping_loss]
            # Define the path name
            torch.save(model.state_dict(), best_model_path)
        
        #current_lr = scheduler.get_lr()[0]
        #print(f"--> LR = {current_lr}")
        
        if epoch_index in progress_points:
            percent = round((epoch_index / N_epochs) * 100,2)  # Calcula o percentual relativo
            print(f"\nExecutando epoca {epoch_index} / {N_epochs} ({percent:.1f}%)")
            
            #print(f"--> LR = {current_lr:.6f}")
            print("--> Allocated memory {} (MB) ".format(round(um.get_memory_usage())))
            
            model_path = weights_file_name+"_ProgressTracking_{}.pth".format(round((epoch_index / N_epochs) * 100))
            torch.save(model.state_dict(), model_path)
            model_paths.append(model_path)
      
        
    # Save training metadata
    metadata = {
        "Timestamp": timestamp,
        "Best Validation Loss": best_valid_loss,
        "Early Stopping Loss": earlyStopping_loss,
        "Backpropagation Loss": backPropagation_loss,
        "Total Epochs": N_epochs,
        "Input Shape": in_shape,
        "Batch Size": batch_size,
        "Device": device,
        "Optimizer": optimizer.__class__.__name__,
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Model Details": model.metadata
    }
    save_training_metadata(metadata_file_name+"training_metadata_"+timestamp, metadata)
    return model_paths, train_costs_history, val_costs_history, metadata

def save_training_metadata(filename, data):
    """
    Saves training metadata as a JSON file, appending to an existing JSON file if it exists.

    Args:
        filename (str): The name of the JSON file.
        data (dict): A dictionary containing the training metadata.
    """
    # Ensure the filename has the .json extension
    if not filename.endswith(".json"):
        filename += ".json"

    # Check if the file exists
    if os.path.isfile(filename):
        with open(filename, "r") as file:
            existing_data = json.load(file)  # Load existing data
    else:
        existing_data = []

    # Append new metadata
    existing_data.append(data)

    # Save back to JSON
    with open(filename, "w") as file:
        json.dump(existing_data, file, indent=4)

    
def Plot_Validation(train_cost_history, val_cost_history, normalize=False):
    # Extract training history
    train_history_dicts = {loss_name: [] for loss_name in train_cost_history[0]}
    for epoch_dict in train_cost_history:
        for loss_name in epoch_dict:
            train_history_dicts[loss_name].append(epoch_dict[loss_name])
    
    # Extract validation history
    valid_history_dicts = {loss_name: [] for loss_name in val_cost_history[0]}
    for epoch_dict in val_cost_history:
        for loss_name in epoch_dict:
            valid_history_dicts[loss_name].append(epoch_dict[loss_name])
    
    
    
    num_plots = len(train_history_dicts)  # Number of loss functions to plot
    fig, axes = plt.subplots(num_plots, 2, figsize=(12, 5 * num_plots))  # Create subplots (Nx2 grid)

    if num_plots == 1:  # Ensure axes are always iterable
        axes = np.array([axes])
        
    def scale(data):
        return data / np.max(data)


    for idx, loss_name in enumerate(train_history_dicts.keys()):
        
        # Collet the histories of certain loss function
        train_loss_history = train_history_dicts[loss_name]
        valid_loss_history = valid_history_dicts[loss_name]
        
        # Scale data to fit MinMax
        if normalize: train_loss_history = scale(train_loss_history)
        if normalize: valid_loss_history = scale(valid_loss_history)
        
        # Train and validation cost history
        x = range(len(train_cost_history))
        axes[idx, 0].plot(x, train_loss_history, label='train', color='blue')
        axes[idx, 0].plot(x, valid_loss_history, label='validation', color='red')
        axes[idx, 0].set_xlabel('Epochs')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].set_title(f'Cost History ({loss_name})')
        axes[idx, 0].legend()
        axes[idx, 0].set_ylim(bottom=0.0)

        # Cost rate history (differences between epochs)
        train_cost_rate_history = np.diff(train_loss_history).tolist()
        val_cost_rate_history = np.diff(valid_loss_history).tolist()

        x = range(len(val_cost_rate_history))
        axes[idx, 1].plot(x, train_cost_rate_history, label='train', color='cyan')
        axes[idx, 1].plot(x, val_cost_rate_history, label='validation', color='orange')
        axes[idx, 1].set_xlabel('Epochs')
        axes[idx, 1].set_ylabel('Loss Rate')
        axes[idx, 1].set_title(f'Cost Rate History ({loss_name})')
        axes[idx, 1].legend()
        axes[idx, 1].set_ylim(bottom=0.0)

    plt.tight_layout()  # Adjust layout for readability
    plt.show()
        
def get_example(loader, i_batch, i_example):
    input_example, target_example = next(itertools.islice(loader, i_batch, None)) 
    input_example, target_example = input_example[i_example], target_example[i_example]
    
    print(f"  Input shape: {input_example.shape}")  # Shape of the in_shapeinput
    print(f"  Output shape: {target_example.shape}\n")  # Shape of the output

    return input_example, target_example
    
def Run_Example(model, loader, loss_functions, main_loss, title="Example", i_batch=0, i_example=0):
    
    input_example, target_example = get_example(loader, i_batch, i_example)
    
    batch_example_input = input_example.unsqueeze(0) # Add batch dimension in 3D tensor for forward
    pred_example = model.predict(batch_example_input) # Disable graphs 
    
    pred_example = pred_example.squeeze(0) # Remove batch dimension from predictions (1,C,H,W)->(C,H,W)
    loss_example = loss_functions[main_loss]["obj"](pred_example, target_example) # Compute the example of loss
    print(f"{title},  {main_loss} Loss: {loss_example}")
    
    plotter.display_image_tensor(input_example, title+"_input")
    plotter.display_image_tensor(target_example, title+"_target")
    plotter.display_image_tensor(pred_example, title+"_pred")
    plotter.display_example(input_example, pred_example, target_example, title=title)

def Plot_DataloaderBatch(dataloader, num_images=5):
    """
    Displays a batch of images and masks from a DataLoader with filenames as titles.

    Args:
        dataloader: The DataLoader to retrieve the batch from.
        num_images: The number of images to display (default: 5).
    """
    
    
    data_iter = iter(dataloader)
    images, masks, img_names, mask_names = next(data_iter)  # Get filenames too
    
    largura_base = 10  # Largura base da figura
    incremento_por_imagem = 3  # Incremento na largura por imagem
    largura_total = largura_base + incremento_por_imagem * num_images
    altura = 6  # Altura fixa da figura
    fig, axes = plt.subplots(2, num_images, figsize=(largura_total, altura))


    for i in range(num_images):  # Display num_images samples
        img = np.transpose(images[i].numpy(), (1, 2, 0))  # Convert tensor format
        mask = masks[i].squeeze().numpy()  # Remove extra dimensions

        axes[0, i].imshow(img)
        axes[0, i].set_title(img_names[i])  # Set title with image filename
        axes[0, i].axis("off")

        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(mask_names[i])  # Set title with mask filename
        axes[1, i].axis("off")

    plt.show()
    


def save_metadata(filename, data, header=None):
    """Appends a row (dictionary) to a CSV file, creating it if necessary.

    Args:
        filename: The name of the CSV file.
        data: A dictionary representing the row to append.
        header: An optional list representing the header row.
    """
    file_exists = os.path.isfile(filename)

    if file_exists:
        df = pd.read_csv(filename)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_csv(filename, index=False, header=header)
