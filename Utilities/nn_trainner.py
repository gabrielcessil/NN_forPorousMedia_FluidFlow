import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from Utilities.usage_metrics import get_memory_usage, check_memory, estimate_memory

#### NEURAL NETWORK UTILITIES 

def train_one_epoch(model, train_loader, loss_function, optimizer):
    model.train()

    running_loss = 0.0    
    # Para cada batch
    for batch_inputs, batch_labels in train_loader:
        
        # REALIZA BATCH:
        optimizer.zero_grad() # Reinicia os gradientes para calculo do passo dos pesos
        batch_outputs = model(batch_inputs) # Calcula Saídas
        loss = loss_function(batch_outputs, batch_labels) # Calcula Custo
        loss.backward() # Calcula gradiente em relacao do Custo
        optimizer.step() # Realiza um passo dos pesos
        
        running_loss += loss.item() # Atualizar a perda acumulada da epoca

    # Perda média para a última época
    avg_loss = running_loss / len(train_loader)

    return avg_loss, model

def validate_one_epoch(model, valid_loader, loss_function):
    
    with torch.no_grad(): # Desativa a computação do gradiente e a construção do grafo computacional durante a avaliação da nova rede
    
      model.eval() # Entre em modo avaliacao, desabilitando funcoes exclusivas de treinamento (ex:Dropout)
      valid_summed_loss = 0.0

      # Para cada sample de validacao
      for valid_input, valid_label in valid_loader:
          # Gera a saida do modelo atual
          valid_output = model(valid_input)
          # Calcula o loss entre output gerado e label
          valid_loss = loss_function(valid_output, valid_label)
          valid_summed_loss += valid_loss.item()
         
      valid_avg_loss = valid_summed_loss / len(valid_loader)
      
      return valid_avg_loss
    
def verify_memory(model, in_shape, batch_size, device, dtype=torch.float32):
    current_memory = check_memory(device=device)
    estimative_memory = estimate_memory(model, input_size=in_shape, batch_size=batch_size, dtype=torch.float32)
    if estimative_memory["Parameters"] > current_memory["Free"]: 
        raise ValueError("Free memory is not sufficient for trainning. Change device, reduce batch size or review architecture")

def full_train(model, train_loader, valid_loader, loss_function, optimizer,N_epochs=1000, file_name = "model_weights", include_time=False, device="cpu"):
    inputs, targets = next(iter(train_loader))    
    in_shape = (inputs.shape[1],inputs.shape[2],inputs.shape[3]) #(C,H,W)
    batch_size = inputs.shape[0]
    verify_memory(model, in_shape, batch_size, device, dtype=torch.float32)
  
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_cost_history = []
    val_cost_history = []
    best_valid_loss = np.inf    
    model_paths = []

    progress_points = set(int(N_epochs * i / 100) for i in range(5, 101, 5))  # Define os pontos de 5% a 100%
    
    for epoch_index in range(N_epochs):
    
      train_avg_loss, model = train_one_epoch( 
          model=model,
          train_loader=train_loader,
          loss_function=loss_function, 
          optimizer=optimizer)
      
      valid_avg_loss = validate_one_epoch(
          model=model, 
          valid_loader=valid_loader,
          loss_function=loss_function)
      
      train_cost_history.append(train_avg_loss)
      val_cost_history.append(valid_avg_loss)
      
      # Track best performance, and save the model's state
      if valid_avg_loss < best_valid_loss:
          best_valid_loss = valid_avg_loss
          # Define the path name
          if include_time: model_path = file_name+"lowerValLoss_{}.pth".format(timestamp)
          else: model_path = file_name
          torch.save(model.state_dict(), model_path)
         
      if epoch_index in progress_points:
          percent = round((epoch_index / N_epochs) * 100,2)  # Calcula o percentual relativo
          print(f"\nExecutando epoca {epoch_index} / {N_epochs} ({percent:.1f}%)")
          print("Allocated memory {} (MB) ".format(round(get_memory_usage())))
          
          model_path = file_name+"_ProgressTracking_{}.pth".format(round((epoch_index / N_epochs) * 100))
          torch.save(model.state_dict(), model_path)
          model_paths.append(model_path)
      
    return model_paths, train_cost_history, val_cost_history


def Plot_Validation(train_cost_history, val_cost_history):
    
    # Trainning Visualization
    x = range(len(train_cost_history))
    plt.plot(x, train_cost_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cost History')
    plt.legend()
    plt.show()
    
    train_cost_rate_history = [train_cost_history[i+1]-train_cost_history[i] for i in range(len(train_cost_history)-1)]
    val_cost_rate_history = [val_cost_history[i+1]-val_cost_history[i] for i in range(len(val_cost_history)-1)]
    x = range(len(val_cost_rate_history))
    plt.plot(x, train_cost_rate_history, label='train', color='blue', marker='none')  # Line 1
    plt.plot(x, val_cost_rate_history, label='validation', color='red', marker='none')     # Line 2
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Cost Rate History')
    plt.legend()
    plt.show()
